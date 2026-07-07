"""Tests for the RAG service and API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.services.rag_service import FALLBACK_ANSWER, RAGService
from app.services.vector_store import SearchResult


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def test_settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "vector_store"))
    get_settings.cache_clear()
    return get_settings()


def test_fallback_when_no_chunks_retrieved(test_settings: Settings) -> None:
    rag = RAGService(settings=test_settings)
    rag.vector_store = MagicMock()
    rag.vector_store.is_loaded = True
    rag.embedding_service = MagicMock()

    with patch("app.services.rag_service.hybrid_retrieve", return_value=[]):
        response = rag.ask("Can an employer keep a passport?")

    assert response.answer == FALLBACK_ANSWER
    assert response.sources == []
    assert response.confidence == 0.0


def test_fallback_when_confidence_below_threshold(test_settings: Settings) -> None:
    rag = RAGService(settings=test_settings)
    rag.vector_store = MagicMock()
    rag.vector_store.is_loaded = True
    rag.embedding_service = MagicMock()

    low_results = [
        SearchResult(
            chunk_id="abc",
            text="Unrelated licensing content only.",
            source_file="other.pdf",
            page_start=1,
            page_end=1,
            score=0.1,
        )
    ]

    with patch("app.services.rag_service.hybrid_retrieve", return_value=low_results):
        response = rag.ask("Random unrelated question about zoning permits?")

    assert response.answer == FALLBACK_ANSWER
    assert response.confidence == 0.1


@patch("app.main.get_rag_service")
def test_health_endpoint(mock_get_rag_service: MagicMock, test_settings: Settings) -> None:
    mock_rag = MagicMock()
    mock_rag.vector_store.is_loaded = True
    mock_rag.vector_store.chunk_count = 42
    mock_get_rag_service.return_value = mock_rag

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["vector_store_loaded"] is True
    assert payload["chunks_count"] == 42
