"""Retrieval-Augmented Generation service using Gemini."""

from __future__ import annotations

from functools import lru_cache

from google import genai

from app.config import Settings, get_settings_or_raise, resolve_retrieval_depth
from app.schemas import ChatResponse, SourceCitation
from app.services.embeddings import EmbeddingService
from app.services.retrieval import (
    hybrid_retrieve,
    select_display_sources,
    should_use_fallback,
)
from app.services.vector_store import SearchResult, VectorStore
from app.utils.logging import logger

FALLBACK_ANSWER = (
    "I could not find enough information in the provided documents to answer this "
    "confidently. Please rephrase the question or contact the relevant Hong Kong "
    "authority for confirmation."
)

PROMPT_TEMPLATE = """You are PoBot, an AI assistant for migrant worker support in Hong Kong.

You answer questions using ONLY the provided context from official/reliable documents.

Rules:
1. Do not invent facts.
2. If the retrieved context contains a direct or partial answer, answer using that context.
3. Do not say the documents lack information when the context contains relevant rules, obligations, prohibitions, examples, or consent requirements.
4. For yes/no legal questions, start with a clear yes/no answer when supported, then explain the condition.
5. Keep answers clear, practical, and supportive.
6. For legal or regulatory questions, explain that this is informational support, not legal advice.
7. Always cite the source file and page number when available.
8. If the user asks in Arabic, answer in Arabic.
9. If the user asks in English, answer in English.
10. If the user asks in Tagalog, answer in simple Tagalog if possible; otherwise answer in English and mention that multilingual support is limited.
11. Only say the provided documents do not contain enough information when the context truly has no relevant rules or facts.

Context:
{context}

User question:
{question}

Answer:"""


class RAGService:
    """End-to-end RAG pipeline for PoBot."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings_or_raise()
        self.embedding_service = embedding_service or EmbeddingService(self.settings)
        self.vector_store = vector_store or VectorStore(self.settings.vector_store_dir)
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        self._ensure_vector_store_loaded()

    def _ensure_vector_store_loaded(self) -> None:
        """Load the vector store from disk if available."""
        if not self.vector_store.is_loaded:
            self.vector_store.load()

    def reload_vector_store(self) -> None:
        """Reload the vector store after ingestion."""
        self.vector_store = VectorStore(self.settings.vector_store_dir)
        self.vector_store.load()

    def _build_context(self, results: list[SearchResult]) -> str:
        """Format retrieved chunks for the LLM prompt."""
        blocks: list[str] = []
        for index, result in enumerate(results, start=1):
            blocks.append(
                f"[Source {index}] {result.source_file} "
                f"(pages {result.page_start}-{result.page_end}, score={result.score:.3f})\n"
                f"{result.text}"
            )
        return "\n\n".join(blocks)

    def _to_citations(self, results: list[SearchResult]) -> list[SourceCitation]:
        """Convert search results to API citations."""
        return [
            SourceCitation(
                source_file=result.source_file,
                page_start=result.page_start,
                page_end=result.page_end,
                score=round(result.score, 4),
            )
            for result in results
        ]

    def _generate_answer(self, question: str, context: str) -> str:
        """Call Gemini chat model with the grounded prompt."""
        prompt = PROMPT_TEMPLATE.format(context=context, question=question.strip())
        response = self.client.models.generate_content(
            model=self.settings.gemini_chat_model,
            contents=prompt,
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini chat model returned an empty response.")
        return text.strip()

    def ask(self, question: str, top_k: int | None = None) -> ChatResponse:
        """
        Answer a user question using retrieval-augmented generation.

        Returns a grounded answer with source citations and confidence score.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        retrieval_k, display_k = resolve_retrieval_depth(top_k, self.settings)
        if not self.vector_store.is_loaded:
            logger.warning("Vector store is not loaded; returning fallback response.")
            return ChatResponse(answer=FALLBACK_ANSWER, sources=[], confidence=0.0)

        results = hybrid_retrieve(
            question=question,
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            top_k=retrieval_k,
        )

        if not results:
            return ChatResponse(answer=FALLBACK_ANSWER, sources=[], confidence=0.0)

        top_score = results[0].score
        if should_use_fallback(
            results,
            question,
            self.settings.min_confidence_score,
        ):
            logger.info(
                "Retrieval score %.3f below threshold %.3f with no direct evidence",
                top_score,
                self.settings.min_confidence_score,
            )
            return ChatResponse(answer=FALLBACK_ANSWER, sources=[], confidence=top_score)

        context = self._build_context(results)
        answer = self._generate_answer(question, context)
        display_results = select_display_sources(
            results,
            question,
            display_k,
            answer=answer,
        )

        return ChatResponse(
            answer=answer,
            sources=self._to_citations(display_results),
            confidence=round(top_score, 4),
        )


@lru_cache
def get_rag_service() -> RAGService:
    """Return a cached RAG service instance."""
    return RAGService()
