"""Regression tests for citation-aware displayed sources."""

from __future__ import annotations

import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings, get_settings
from app.services.rag_service import RAGService
from app.services.retrieval import (
    COP_FILE,
    FDH_GUIDE_FILE,
    hybrid_retrieve,
    is_weak_display_source,
    select_display_sources,
)
from app.services.vector_store import SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
ENV_FILE = PROJECT_ROOT / ".env"

CITATION_CASES = [
    {
        "question": "Can an employer keep a helper's passport?",
        "mock_answer": (
            "No. A foreign domestic helper should keep their own passport and Hong Kong "
            "Identity Card. No other person should keep these documents without the "
            "helper's consent. Source: FDHguideEnglish.pdf (page 6)"
        ),
        "preferred_sources": {FDH_GUIDE_FILE, COP_FILE},
        "forbidden_pages": {(COP_FILE, 9)},
        "evidence_terms": ("passport", "consent"),
    },
    {
        "question": "When should wages be paid to a domestic helper?",
        "mock_answer": (
            "Wages should be paid within 7 days after the end of the wage period. "
            "Source: FDHguideEnglish.pdf (page 14)"
        ),
        "preferred_sources": {FDH_GUIDE_FILE},
        "forbidden_pages": {(COP_FILE, 9), (FDH_GUIDE_FILE, 1)},
        "evidence_terms": ("wage", "pay", "7 day"),
    },
    {
        "question": "What happens if a helper is injured at work?",
        "mock_answer": (
            "The employer must have employees' compensation insurance. "
            "Source: ImportantInformationForEmployersAndEmployees_Eng.pdf (page 1)"
        ),
        "preferred_sources": {
            "ImportantInformationForEmployersAndEmployees_Eng.pdf",
            FDH_GUIDE_FILE,
            "Handy_Guide_for_Employers_of_FDHs_English_version_Web_version.pdf",
        },
        "forbidden_pages": {(COP_FILE, 9)},
        "evidence_terms": ("injur", "compensation", "insurance"),
    },
    {
        "question": "What should an employment agency charge a job seeker?",
        "mock_answer": (
            "The prescribed commission is no more than 10% of the first month's wages. "
            "Source: CoP_Eng.pdf (page 9)"
        ),
        "preferred_sources": {COP_FILE},
        "forbidden_pages": set(),
        "evidence_terms": ("commission", "10%", "job seeker"),
    },
]


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
        pytest.skip("GEMINI_API_KEY required for citation consistency tests")

    monkeypatch.setenv("GEMINI_API_KEY", api_key)
    monkeypatch.setenv("VECTOR_STORE_DIR", str(VECTOR_STORE_DIR))
    monkeypatch.setenv("PROCESSED_DIR", str(PROJECT_ROOT / "processed"))
    monkeypatch.setenv("DATA_DIR", str(PROJECT_ROOT / "data"))
    monkeypatch.setenv("DEFAULT_TOP_K", "8")
    monkeypatch.setenv("MIN_RETRIEVAL_K", "8")
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def live_rag(live_settings: Settings) -> RAGService:
    rag = RAGService(settings=live_settings)
    if not rag.vector_store.is_loaded:
        pytest.skip("Vector store not available for citation consistency tests")
    return rag


def _sources_mentioned_in_answer(answer: str) -> list[tuple[str, int]]:
    citations: list[tuple[str, int]] = []
    for match in re.finditer(
        r"([\w\(\)\-\.]+\.pdf)\s*\(pages?\s*(\d+)",
        answer,
        flags=re.IGNORECASE,
    ):
        citations.append((match.group(1), int(match.group(2))))
    for match in re.finditer(
        r"Source:\s*([\w\(\)\-\.]+\.pdf)\s*,?\s*page\s*(\d+)",
        answer,
        flags=re.IGNORECASE,
    ):
        citations.append((match.group(1), int(match.group(2))))
    return citations


@pytest.mark.parametrize("case", CITATION_CASES, ids=[c["question"][:42] for c in CITATION_CASES])
def test_select_display_sources_prefers_direct_evidence(
    live_rag: RAGService,
    case: dict,
) -> None:
    results = hybrid_retrieve(
        question=case["question"],
        vector_store=live_rag.vector_store,
        embedding_service=live_rag.embedding_service,
        top_k=live_rag.settings.min_retrieval_k,
    )
    assert results

    rank_first = results[0]
    display = select_display_sources(
        results,
        case["question"],
        display_k=1,
        answer=case["mock_answer"],
    )
    assert len(display) == 1

    selected = display[0]
    assert not is_weak_display_source(selected, case["question"]), (
        f"Weak citation selected: {selected.source_file} p.{selected.page_start}; "
        f"rank-1 was {rank_first.source_file} p.{rank_first.page_start}"
    )
    assert selected.source_file in case["preferred_sources"]
    assert (selected.source_file, selected.page_start) not in case["forbidden_pages"]

    text_lower = selected.text.lower()
    assert any(term in text_lower for term in case["evidence_terms"]), (
        f"Selected chunk lacks evidence terms {case['evidence_terms']}"
    )


@pytest.mark.parametrize("case", CITATION_CASES[:3], ids=[c["question"][:42] for c in CITATION_CASES[:3]])
@patch.object(RAGService, "_generate_answer")
def test_ask_top_k_1_display_sources_match_answer_citations(
    mock_generate: MagicMock,
    live_rag: RAGService,
    case: dict,
) -> None:
    mock_generate.return_value = case["mock_answer"]
    response = live_rag.ask(case["question"], top_k=1)

    assert len(response.sources) == 1
    source = response.sources[0]
    assert (source.source_file, source.page_start) not in case["forbidden_pages"]

    for cited_file, cited_page in _sources_mentioned_in_answer(response.answer):
        assert any(
            s.source_file == cited_file
            and (
                s.page_start <= cited_page <= s.page_end
                or not any(
                    r.source_file == cited_file
                    and r.page_start <= cited_page <= r.page_end
                    for r in hybrid_retrieve(
                        question=case["question"],
                        vector_store=live_rag.vector_store,
                        embedding_service=live_rag.embedding_service,
                        top_k=live_rag.settings.min_retrieval_k,
                    )
                )
            )
            for s in response.sources
        ), f"Answer cites {cited_file} p.{cited_page} but sources are {response.sources}"


@patch.object(RAGService, "_generate_answer")
def test_passport_top_k_1_not_cop_page_9_toc(
    mock_generate: MagicMock,
    live_rag: RAGService,
) -> None:
    question = "Can an employer keep a helper's passport?"
    mock_generate.return_value = (
        "No. Helpers should keep their own passport. "
        "Source: FDHguideEnglish.pdf (page 6)"
    )
    response = live_rag.ask(question, top_k=1)

    assert response.sources
    first = response.sources[0]
    assert not (first.source_file == COP_FILE and first.page_start == 9), (
        "Displayed source must not be generic CoP page 9"
    )
    assert first.source_file in {FDH_GUIDE_FILE, COP_FILE}
    assert "no" in response.answer.lower()
