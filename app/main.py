"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from app.config import get_settings_or_raise
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentSourceInfo,
    HealthResponse,
    IngestResponse,
    SourcesResponse,
)
from app.services.ingestion import run_ingestion
from app.services.rag_service import RAGService, get_rag_service
from app.utils.logging import logger


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize services on startup."""
    try:
        settings = get_settings_or_raise()
        logger.info("PoBot starting with data_dir=%s", settings.data_dir)
        rag = get_rag_service()
        if rag.vector_store.is_loaded:
            logger.info("Vector store loaded with %d chunks", rag.vector_store.chunk_count)
        else:
            logger.warning(
                "Vector store not loaded. Run POST /ingest or `python scripts/ingest.py`."
            )
    except Exception as exc:
        logger.error("Startup configuration error: %s", exc)
    yield


app = FastAPI(
    title="PoBot Expansion – AI Assistant for Migrant Support",
    description=(
        "RAG chatbot for Hong Kong labour regulations, migrant worker protections, "
        "and related official guidance."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health and vector store status."""
    try:
        rag = get_rag_service()
        return HealthResponse(
            status="ok",
            vector_store_loaded=rag.vector_store.is_loaded,
            chunks_count=rag.vector_store.chunk_count,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {exc}",
        ) from exc


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    """Ingest documents from the data directory and rebuild the vector index."""
    try:
        result = run_ingestion()
        rag = get_rag_service()
        rag.reload_vector_store()
        return IngestResponse(
            files_processed=result.files_processed,
            pages_processed=result.pages_processed,
            chunks_created=result.chunks_created,
            index_path=result.index_path,
            chunks_path=result.chunks_path,
        )
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question using retrieved document context."""
    try:
        rag = get_rag_service()
        return rag.ask(request.question, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/sources", response_model=SourcesResponse)
def sources() -> SourcesResponse:
    """List indexed documents and chunk counts."""
    try:
        rag = get_rag_service()
        if not rag.vector_store.is_loaded:
            return SourcesResponse(documents=[], total_chunks=0)

        summary = rag.vector_store.document_summary()
        documents = [
            DocumentSourceInfo(source_file=name, chunk_count=count)
            for name, count in sorted(summary.items())
        ]
        return SourcesResponse(
            documents=documents,
            total_chunks=rag.vector_store.chunk_count,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
