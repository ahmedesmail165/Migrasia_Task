"""FAISS vector store for chunk retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.services.chunker import Chunk
from app.utils.logging import logger


@dataclass
class SearchResult:
    """A chunk returned from vector search."""

    chunk_id: str
    text: str
    source_file: str
    page_start: int
    page_end: int
    score: float


class VectorStore:
    """FAISS-backed vector store with JSON metadata."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_path = index_dir / "index.faiss"
        self.metadata_path = index_dir / "metadata.json"
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, object]] = []

    @property
    def is_loaded(self) -> bool:
        """Return True when an index and metadata are available."""
        return self.index is not None and bool(self.metadata)

    @property
    def chunk_count(self) -> int:
        """Return number of indexed chunks."""
        return len(self.metadata)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for cosine similarity via inner product."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def build(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Build a new FAISS index from chunks and embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length.")
        if not chunks:
            raise ValueError("Cannot build vector store without chunks.")

        vectors = np.array(embeddings, dtype=np.float32)
        vectors = self._normalize(vectors)
        dimension = vectors.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)

        self.index = index
        self.metadata = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_file": chunk.source_file,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "text_hash": chunk.text_hash,
            }
            for chunk in chunks
        ]
        logger.info("Built FAISS index with %d vectors (dim=%d)", len(chunks), dimension)

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No FAISS index to save.")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, ensure_ascii=False, indent=2)
        logger.info("Saved vector store to %s", self.index_dir)

    def load(self) -> bool:
        """Load an existing index if present."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning("Vector store files not found in %s", self.index_dir)
            return False

        self.index = faiss.read_index(str(self.index_path))
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                "FAISS index and metadata are out of sync: "
                f"{self.index.ntotal} vectors vs {len(self.metadata)} metadata rows."
            )

        logger.info("Loaded FAISS index with %d chunks", len(self.metadata))
        return True

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        """Search for the most similar chunks to a query embedding."""
        if self.index is None or not self.metadata:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        query = self._normalize(query)
        scores, indices = self.index.search(query, min(top_k, len(self.metadata)))

        results: list[SearchResult] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0:
                continue
            row = self.metadata[index]
            results.append(
                SearchResult(
                    chunk_id=str(row["chunk_id"]),
                    text=str(row["text"]),
                    source_file=str(row["source_file"]),
                    page_start=int(row["page_start"]),
                    page_end=int(row["page_end"]),
                    score=float(max(0.0, min(1.0, score))),
                )
            )
        return results

    def document_summary(self) -> dict[str, int]:
        """Return chunk counts grouped by source file."""
        summary: dict[str, int] = {}
        for row in self.metadata:
            source_file = str(row["source_file"])
            summary[source_file] = summary.get(source_file, 0) + 1
        return summary
