"""Document ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings_or_raise
from app.services.chunker import chunk_documents, save_chunks
from app.services.document_loader import load_documents
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.utils.logging import logger


@dataclass
class IngestionResult:
    """Summary of an ingestion run."""

    files_processed: int
    pages_processed: int
    chunks_created: int
    index_path: str
    chunks_path: str


def run_ingestion(settings: Settings | None = None) -> IngestionResult:
    """
    Run the full ingestion pipeline:

    load documents -> clean/chunk -> embed -> build FAISS index.
    """
    settings = settings or get_settings_or_raise()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)

    pages = load_documents(settings.data_dir)
    if not pages:
        raise RuntimeError(
            f"No documents were loaded from {settings.data_dir}. "
            "Add supported files (.pdf, .txt, .md, .html) and try again."
        )

    files_processed = len({page.source_file for page in pages})
    pages_processed = len(pages)

    chunks = chunk_documents(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError("Chunking produced no usable chunks.")

    save_chunks(chunks, settings.chunks_path)
    logger.info("Saved %d chunks to %s", len(chunks), settings.chunks_path)

    embedding_service = EmbeddingService(settings)
    texts = [chunk.text for chunk in chunks]
    logger.info("Generating embeddings for %d chunks...", len(texts))
    embeddings = embedding_service.embed_texts(texts)

    vector_store = VectorStore(settings.vector_store_dir)
    vector_store.build(chunks, embeddings)
    vector_store.save()

    return IngestionResult(
        files_processed=files_processed,
        pages_processed=pages_processed,
        chunks_created=len(chunks),
        index_path=str(settings.faiss_index_path),
        chunks_path=str(settings.chunks_path),
    )
