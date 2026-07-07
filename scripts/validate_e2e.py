"""End-to-end validation script for PoBot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings_or_raise
from app.services.chunker import chunk_documents, save_chunks
from app.services.document_loader import load_documents
from app.services.ingestion import run_ingestion
from app.services.rag_service import RAGService
from app.utils.logging import logger

QUESTIONS = [
    "What are the rights of foreign domestic helpers in Hong Kong?",
    "What are the rules for recruitment agencies?",
    "Can an employer keep a helper's passport?",
    "When should wages be paid to a domestic helper?",
    "What happens if a helper is injured at work?",
    "Can a foreign domestic helper work part-time for another employer?",
    "What should an employment agency charge a job seeker?",
    "هل يمكن لصاحب العمل الاحتفاظ بجواز سفر العاملة؟",
    "Pwede bang kunin ng employer ang passport ng helper?",
]


def validate_documents(settings) -> dict:
    """Validate document loading without calling Gemini."""
    pages = load_documents(settings.data_dir)
    files = sorted({page.source_file for page in pages})
    return {
        "files_discovered": len(files),
        "files": files,
        "pages_extracted": len(pages),
    }


def validate_chunks(settings, pages) -> dict:
    """Validate chunk generation and JSONL output."""
    chunks = chunk_documents(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    save_chunks(chunks, settings.chunks_path)
    return {
        "chunks_created": len(chunks),
        "chunks_path": str(settings.chunks_path),
        "chunks_file_exists": settings.chunks_path.exists(),
    }


def validate_ingestion() -> dict:
    """Run full ingestion including embeddings and FAISS."""
    result = run_ingestion()
    settings = get_settings_or_raise()
    return {
        "files_processed": result.files_processed,
        "pages_processed": result.pages_processed,
        "chunks_created": result.chunks_created,
        "index_path": result.index_path,
        "chunks_path": result.chunks_path,
        "index_exists": settings.faiss_index_path.exists(),
        "metadata_exists": settings.metadata_path.exists(),
    }


def validate_chat() -> list[dict]:
    """Run sample chat questions and return responses."""
    rag = RAGService()
    rag.reload_vector_store()
    outputs: list[dict] = []
    for question in QUESTIONS:
        logger.info("Validating question: %s", question[:80])
        response = rag.ask(question)
        outputs.append(
            {
                "question": question,
                "answer": response.answer,
                "confidence": response.confidence,
                "sources": [source.model_dump() for source in response.sources],
            }
        )
    return outputs


def main() -> int:
    """Run validation and print a JSON summary."""
    try:
        settings = get_settings_or_raise()
    except Exception as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        return 1

    summary: dict = {"status": "ok", "phases": {}}

    doc_stats = validate_documents(settings)
    summary["phases"]["documents"] = doc_stats

    pages = load_documents(settings.data_dir)
    chunk_stats = validate_chunks(settings, pages)
    summary["phases"]["chunks"] = chunk_stats

    try:
        ingest_stats = validate_ingestion()
        summary["phases"]["ingestion"] = ingest_stats
        chat_outputs = validate_chat()
        summary["phases"]["chat"] = {"responses": chat_outputs}
        output_path = PROJECT_ROOT / "validation_results.json"
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        summary["status"] = "partial"
        summary["phases"]["ingestion_error"] = str(exc)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
