"""Unit tests for hybrid retrieval and query expansion."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.services.retrieval import (
    COP_FILE,
    FDH_GUIDE_FILE,
    HANDY_GUIDE_FILE,
    detect_intents,
    expand_query,
    has_direct_evidence,
    hybrid_retrieve,
    is_passport_question,
    is_weak_display_source,
    select_display_sources,
    should_use_fallback,
    source_title_relevance,
)
from app.services.vector_store import SearchResult

FDH_CHUNK = SearchResult(
    chunk_id="fdh-passport",
    text=(
        "You should keep your own personal identification documents (e.g. identity "
        "card, passport, etc.). No other person, including your employer or staff of "
        "the employment agency, should keep these documents for you without your consent."
    ),
    source_file="FDHguideEnglish.pdf",
    page_start=5,
    page_end=6,
    score=0.55,
)

COP_CHUNK = SearchResult(
    chunk_id="cop-passport",
    text=(
        "Job seekers' passports or personal identification documents should be kept "
        "by themselves. In cases where EAs need to temporarily keep the passports, "
        "EAs must explain the reason, seek written consent, provide written "
        "acknowledgement, return without delay, and must not withhold documents."
    ),
    source_file="CoP_Eng.pdf",
    page_start=28,
    page_end=30,
    score=0.52,
)

UNRELATED = SearchResult(
    chunk_id="unrelated",
    text="Employment agency licensing requirements under Part XII of the EO.",
    source_file="CoP_Eng.pdf",
    page_start=5,
    page_end=5,
    score=0.75,
)

COP_TOC_CHUNK = SearchResult(
    chunk_id="cop-toc",
    text=(
        '"foreign domestic helper" means a person admitted into Hong Kong. '
        '"prescribed commission" means the maximum commission which may be charged '
        'by an EA, which is no more than 10% of the first-month wages received '
        "by a job seeker."
    ),
    source_file="CoP_Eng.pdf",
    page_start=9,
    page_end=9,
    score=0.78,
)


@pytest.mark.parametrize(
    "question",
    [
        "Can an employer keep a helper's passport?",
        "Can an employment agency hold my passport?",
        "هل يمكن لصاحب العمل الاحتفاظ بجواز سفر العاملة؟",
        "Pwede bang kunin ng employer ang passport ng helper?",
    ],
)
def test_passport_question_detection(question: str) -> None:
    assert is_passport_question(question)


def test_expand_query_adds_passport_variants() -> None:
    variants = expand_query("Can an employer keep a helper's passport?")
    assert variants[0] == "Can an employer keep a helper's passport?"
    assert "helper passport employer keep" in variants
    assert "withhold passport employment agency helper" in variants
    assert len(variants) >= 5


def test_has_direct_evidence_for_passport_chunks() -> None:
    question = "Can an employer keep a helper's passport?"
    assert has_direct_evidence([FDH_CHUNK], question)
    assert has_direct_evidence([COP_CHUNK], question)
    assert has_direct_evidence([FDH_CHUNK, COP_CHUNK], question)
    assert not has_direct_evidence([UNRELATED], question)


def test_should_not_fallback_when_passport_evidence_exists() -> None:
    question = "Can an employer keep a helper's passport?"
    results = [FDH_CHUNK, COP_CHUNK]
    assert not should_use_fallback(results, question, min_confidence_score=0.9)


def test_hybrid_retrieve_merges_and_boosts_passport_chunks() -> None:
    vector_store = MagicMock()
    vector_store.is_loaded = True

    def fake_search(embedding: list[float], top_k: int) -> list[SearchResult]:
        # First variant (original) returns unrelated high-vector chunk.
        if embedding == [1.0]:
            return [UNRELATED]
        # Expanded variants surface passport chunks.
        return [FDH_CHUNK, COP_CHUNK]

    vector_store.search.side_effect = fake_search

    embedding_service = MagicMock()
    embedding_service.embed_text.side_effect = (
        lambda text: [1.0] if "employer keep" in text else [2.0]
    )

    results = hybrid_retrieve(
        "Can an employer keep a helper's passport?",
        vector_store,
        embedding_service,
        top_k=5,
    )

    chunk_ids = {r.chunk_id for r in results}
    assert "fdh-passport" in chunk_ids
    assert "cop-passport" in chunk_ids
    top_sources = {r.source_file for r in results[:3]}
    assert "FDHguideEnglish.pdf" in top_sources or "CoP_Eng.pdf" in top_sources


def test_detect_intents_for_fdh_wages_question() -> None:
    intents = detect_intents("When should wages be paid to a domestic helper?")
    assert "fdh_rights" in intents
    assert "agency" not in intents


def test_source_boost_prefers_fdh_guide_for_wages() -> None:
    question = "When should wages be paid to a domestic helper?"
    assert source_title_relevance(FDH_GUIDE_FILE, question) > source_title_relevance(
        COP_FILE, question
    )


def test_source_boost_prefers_cop_for_agency_rules() -> None:
    question = "What are the rules for recruitment agencies?"
    assert source_title_relevance(COP_FILE, question) > source_title_relevance(
        FDH_GUIDE_FILE, question
    )


def test_source_boost_prefers_handy_guide_for_live_in() -> None:
    question = "Can a foreign domestic helper live outside the employer's home?"
    assert source_title_relevance(HANDY_GUIDE_FILE, question) >= source_title_relevance(
        COP_FILE, question
    )


def test_select_display_sources_prefers_passport_chunk_over_cop_toc() -> None:
    question = "Can an employer keep a helper's passport?"
    results = [COP_TOC_CHUNK, UNRELATED, FDH_CHUNK]
    display = select_display_sources(results, question, display_k=1)
    assert display[0].chunk_id == "fdh-passport"
    assert not is_weak_display_source(display[0], question)


def test_select_display_sources_aligns_with_answer_citation() -> None:
    question = "Can an employer keep a helper's passport?"
    answer = "No. Source: FDHguideEnglish.pdf (page 6)"
    results = [COP_TOC_CHUNK, FDH_CHUNK, COP_CHUNK]
    display = select_display_sources(results, question, display_k=1, answer=answer)
    assert display[0].source_file == FDH_GUIDE_FILE
