"""Regression tests for low user top_k with safe internal retrieval depth."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings, get_settings, resolve_retrieval_depth
from app.services.rag_service import FALLBACK_ANSWER, RAGService
from app.services.retrieval import hybrid_retrieve, should_use_fallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
ENV_FILE = PROJECT_ROOT / ".env"

PASSPORT_QUESTION = "Can an employer keep a helper's passport?"

LOW_TOP_K_CASES = [
    {
        "question": PASSPORT_QUESTION,
        "internal_markers": ("passport", "consent"),
        "evidence_sources": {"FDHguideEnglish.pdf", "CoP_Eng.pdf"},
    },
    {
        "question": "When should wages be paid to a domestic helper?",
        "internal_markers": ("wage", "pay"),
        "evidence_sources": {"FDHguideEnglish.pdf"},
    },
    {
        "question": "Can a foreign domestic helper work part-time for another employer?",
        "internal_markers": ("immigration", "foreign domestic helper"),
        "evidence_sources": {"FDHguideEnglish.pdf"},
    },
    {
        "question": "What should an employment agency charge a job seeker?",
        "internal_markers": ("commission", "job seeker", "10%"),
        "evidence_sources": {"CoP_Eng.pdf"},
    },
]

GOOD_PASSPORT_ANSWER = (
    "No. A foreign domestic helper should keep their own passport and Hong Kong "
    "Identity Card. No other person, including the employer or employment agency "
    "staff, should keep these documents without the helper's consent."
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
    monkeypatch.setenv("DEFAULT_TOP_K", "8")
    monkeypatch.setenv("MIN_RETRIEVAL_K", "8")
    monkeypatch.setenv("MAX_RETRIEVAL_K", "20")
    monkeypatch.setenv("MAX_DISPLAY_SOURCES", "10")
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def live_rag(live_settings: Settings) -> RAGService:
    rag = RAGService(settings=live_settings)
    if not rag.vector_store.is_loaded:
        pytest.skip("Vector store not available for regression tests")
    return rag


def test_resolve_retrieval_depth_clamps_low_user_top_k(live_settings: Settings) -> None:
    retrieval_k, display_k = resolve_retrieval_depth(1, live_settings)
    assert retrieval_k == live_settings.min_retrieval_k
    assert display_k == 1


def test_resolve_retrieval_depth_caps_display_sources(live_settings: Settings) -> None:
    retrieval_k, display_k = resolve_retrieval_depth(15, live_settings)
    assert retrieval_k == 15
    assert display_k == live_settings.max_display_sources


@patch("app.services.rag_service.hybrid_retrieve")
def test_ask_top_k_1_uses_min_retrieval_depth(
    mock_retrieve: MagicMock,
    live_rag: RAGService,
) -> None:
    mock_retrieve.return_value = [
        MagicMock(
            chunk_id="x",
            text="passport consent",
            source_file="FDHguideEnglish.pdf",
            page_start=5,
            page_end=6,
            score=0.8,
        )
    ]

    with patch.object(RAGService, "_generate_answer", return_value=GOOD_PASSPORT_ANSWER):
        live_rag.ask(PASSPORT_QUESTION, top_k=1)

    assert mock_retrieve.call_args.kwargs["top_k"] == live_rag.settings.min_retrieval_k


@pytest.mark.parametrize("case", LOW_TOP_K_CASES, ids=[c["question"][:40] for c in LOW_TOP_K_CASES])
@patch.object(RAGService, "_generate_answer")
def test_low_top_k_does_not_fallback_with_internal_evidence(
    mock_generate: MagicMock,
    live_rag: RAGService,
    case: dict,
) -> None:
    question = case["question"]
    captured: list = []

    def retrieve_and_capture(**kwargs: object) -> list:
        results = hybrid_retrieve(
            question=question,
            vector_store=live_rag.vector_store,
            embedding_service=live_rag.embedding_service,
            top_k=int(kwargs["top_k"]),
        )
        captured.append(results)
        return results

    mock_generate.side_effect = lambda _question, context: (
        "No. " + context[:500] if "passport" in question.lower() else context[:500]
    )

    with patch("app.services.rag_service.hybrid_retrieve", side_effect=retrieve_and_capture):
        response = live_rag.ask(question, top_k=1)

    assert captured, "hybrid_retrieve was not called"
    internal_results = captured[0]
    assert internal_results, f"No internal chunks retrieved for: {question}"
    assert len(internal_results) >= live_rag.settings.min_retrieval_k

    internal_sources = {r.source_file for r in internal_results}
    assert internal_sources & case["evidence_sources"], (
        f"Expected one of {case['evidence_sources']} in internal retrieval, got {internal_sources}"
    )

    combined_internal = " ".join(r.text.lower() for r in internal_results)
    assert any(marker in combined_internal for marker in case["internal_markers"]), (
        f"Internal context missing markers {case['internal_markers']}"
    )
    assert not should_use_fallback(
        internal_results,
        question,
        live_rag.settings.min_confidence_score,
    )

    assert FALLBACK_ANSWER not in response.answer
    assert len(response.sources) == 1

    if "passport" in question.lower():
        first = response.sources[0]
        assert not (first.source_file == "CoP_Eng.pdf" and first.page_start == 9), (
            f"Weak CoP glossary page returned: {first}"
        )


@patch.object(RAGService, "_generate_answer")
def test_passport_top_k_1_answer_mentions_consent(
    mock_generate: MagicMock,
    live_rag: RAGService,
) -> None:
    mock_generate.return_value = GOOD_PASSPORT_ANSWER
    response = live_rag.ask(PASSPORT_QUESTION, top_k=1)

    answer_lower = response.answer.lower()
    assert "no" in answer_lower
    assert "passport" in answer_lower or "personal identification" in answer_lower
    assert "consent" in answer_lower
    assert len(response.sources) == 1
