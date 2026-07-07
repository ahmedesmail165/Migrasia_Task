"""Regression tests for passport retention answer quality."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings, get_settings
from app.services.rag_service import FALLBACK_ANSWER, RAGService
from app.services.retrieval import hybrid_retrieve, lexical_scan, should_use_fallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
ENV_FILE = PROJECT_ROOT / ".env"

PASSPORT_QUESTIONS = [
    "Can an employer keep a helper's passport?",
    "Can an employment agency hold my passport?",
    "Who should keep the helper's passport?",
    "هل يمكن لصاحب العمل الاحتفاظ بجواز سفر العاملة؟",
    "Pwede bang kunin ng employer ang passport ng helper?",
]

GOOD_ANSWER = (
    "No. A foreign domestic helper should keep their own passport and Hong Kong "
    "Identity Card. No other person, including the employer or employment agency "
    "staff, should keep these documents without the helper's consent. If an "
    "employment agency needs to temporarily keep a passport or identification "
    "document, it must explain the reason, obtain written consent, provide a "
    "written acknowledgement, and return the document without delay. This is "
    "informational support, not legal advice."
)


def _load_api_key_from_env_file() -> str | None:
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("GEMINI_API_KEY="):
            value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            if value and value != "your_gemini_api_key_here":
                return value
    return None


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def live_settings(monkeypatch) -> Settings:
    api_key = os.environ.get("GEMINI_API_KEY") or _load_api_key_from_env_file()
    if not api_key:
        pytest.skip("GEMINI_API_KEY required for live regression tests")

    monkeypatch.setenv("GEMINI_API_KEY", api_key)
    monkeypatch.setenv("VECTOR_STORE_DIR", str(VECTOR_STORE_DIR))
    monkeypatch.setenv("PROCESSED_DIR", str(PROJECT_ROOT / "processed"))
    monkeypatch.setenv("DATA_DIR", str(PROJECT_ROOT / "data"))
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def live_rag(live_settings: Settings) -> RAGService:
    rag = RAGService(settings=live_settings)
    if not rag.vector_store.is_loaded:
        pytest.skip("Vector store not available for regression tests")
    return rag


def _assert_passport_results(results: list, question: str) -> None:
    assert results, f"No chunks retrieved for: {question}"
    sources = {r.source_file for r in results}
    assert sources & {"FDHguideEnglish.pdf", "CoP_Eng.pdf"}, (
        f"Expected FDHguideEnglish.pdf or CoP_Eng.pdf in sources, got {sources}"
    )
    combined = " ".join(r.text.lower() for r in results[:10])
    assert "passport" in combined
    assert "consent" in combined


@pytest.mark.parametrize("question", PASSPORT_QUESTIONS)
def test_passport_lexical_scan_finds_authoritative_sources(
    live_rag: RAGService,
    question: str,
) -> None:
    results = lexical_scan(live_rag.vector_store, question, limit=10)
    _assert_passport_results(results, question)


@pytest.mark.parametrize("question", PASSPORT_QUESTIONS)
def test_passport_retrieval_finds_authoritative_sources(
    live_rag: RAGService,
    question: str,
) -> None:
    results = hybrid_retrieve(
        question=question,
        vector_store=live_rag.vector_store,
        embedding_service=live_rag.embedding_service,
        top_k=10,
    )
    _assert_passport_results(results, question)


@pytest.mark.parametrize("question", PASSPORT_QUESTIONS)
def test_passport_retrieval_should_not_fallback(
    live_rag: RAGService,
    question: str,
) -> None:
    results = hybrid_retrieve(
        question=question,
        vector_store=live_rag.vector_store,
        embedding_service=live_rag.embedding_service,
        top_k=10,
    )
    assert not should_use_fallback(
        results,
        question,
        live_rag.settings.min_confidence_score,
    )


@pytest.mark.parametrize("question", PASSPORT_QUESTIONS)
@patch.object(RAGService, "_generate_answer")
def test_passport_ask_does_not_return_fallback(
    mock_generate: MagicMock,
    live_rag: RAGService,
    question: str,
) -> None:
    mock_generate.return_value = GOOD_ANSWER
    response = live_rag.ask(question, top_k=10)

    assert FALLBACK_ANSWER not in response.answer
    assert response.sources
    source_files = {s.source_file for s in response.sources}
    assert source_files & {"FDHguideEnglish.pdf", "CoP_Eng.pdf"}

    combined_answer = response.answer.lower()
    assert "passport" in combined_answer
    assert "consent" in combined_answer


@pytest.mark.parametrize("question", PASSPORT_QUESTIONS)
@patch.object(RAGService, "_generate_answer")
def test_passport_ask_cites_expected_documents(
    mock_generate: MagicMock,
    live_rag: RAGService,
    question: str,
) -> None:
    mock_generate.return_value = GOOD_ANSWER
    response = live_rag.ask(question, top_k=10)
    cited = {s.source_file for s in response.sources}
    assert "FDHguideEnglish.pdf" in cited or "CoP_Eng.pdf" in cited
