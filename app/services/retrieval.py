"""Hybrid retrieval with query expansion and lexical re-ranking."""

from __future__ import annotations

import re
from dataclasses import replace

from app.services.embeddings import EmbeddingService
from app.services.vector_store import SearchResult, VectorStore
from app.utils.debug_session_log import debug_session_log

# Lexical boost terms for passport / identification document questions.
PASSPORT_LEXICAL_TERMS: tuple[str, ...] = (
    "passport",
    "personal identification documents",
    "personal identification document",
    "hong kong identity card",
    "identity card",
    "withhold",
    "kept by themselves",
    "keep these documents",
    "keep your passport",
    "keep your own",
    "consent",
    "employer",
    "employment agency",
    "job seekers",
    "written acknowledgement",
    "written consent",
    "without delay",
)

PASSPORT_SOURCE_FILES: frozenset[str] = frozenset(
    {"FDHguideEnglish.pdf", "CoP_Eng.pdf"}
)

COP_FILE = "CoP_Eng.pdf"
FDH_GUIDE_FILE = "FDHguideEnglish.pdf"
HANDY_GUIDE_FILE = (
    "Handy_Guide_for_Employers_of_FDHs_English_version_Web_version.pdf"
)
ID969_FILE = "ID(E)969.pdf"
IMPORTANT_INFO_FILE = "ImportantInformationForEmployersAndEmployees_Eng.pdf"

EMPLOYMENT_AGENCY_MARKERS: tuple[str, ...] = (
    "recruitment agency",
    "recruitment agencies",
    "employment agency",
    "employment agencies",
    "job seeker",
    "job seekers",
    "commission",
    "licence",
    "license",
    "service agreement",
    "overcharg",
    "code of practice for employment",
)

FDH_RIGHTS_MARKERS: tuple[str, ...] = (
    "foreign domestic helper",
    "domestic helper",
    "fdh",
    "helper's rights",
    "rights of foreign domestic",
    "wage",
    "wages",
    "salary",
    "rest day",
    "rest days",
    "leave",
    "holiday",
    "annual leave",
    "sick leave",
    "medical",
    "termination",
    "minimum allowable wage",
)

EMPLOYER_OBLIGATION_MARKERS: tuple[str, ...] = (
    "live-in",
    "live in",
    "accommodation",
    "employer's home",
    "employers home",
    "outside the employer",
    "part-time",
    "part time",
    "another employer",
    "employer obligation",
    "employer obligations",
)

INJURY_MARKERS: tuple[str, ...] = (
    "injured",
    "injury",
    "work injury",
    "compensation",
    "insurance",
    "employees' compensation",
    "employees compensation",
)

FDH_LEXICAL_TERMS: tuple[str, ...] = (
    "foreign domestic helper",
    "domestic helper",
    "wage",
    "wages",
    "salary",
    "rest day",
    "rest days",
    "annual leave",
    "statutory holiday",
    "sick leave",
    "sickness allowance",
    "medical",
    "termination",
    "minimum allowable wage",
    "employment contract",
    "standard employment contract",
)

AGENCY_LEXICAL_TERMS: tuple[str, ...] = (
    "employment agency",
    "recruitment agency",
    "job seeker",
    "commission",
    "licence",
    "license",
    "service agreement",
    "code of practice",
    "prescribed commission",
    "overcharg",
)

EMPLOYER_LEXICAL_TERMS: tuple[str, ...] = (
    "accommodation",
    "live-in",
    "live in",
    "employer",
    "part-time",
    "part time",
    "another employer",
    "schedule of accommodation",
    "domestic duties",
)

INJURY_LEXICAL_TERMS: tuple[str, ...] = (
    "injury",
    "injured",
    "compensation",
    "insurance",
    "employees' compensation",
    "employees compensation",
    "occupational",
    "accident",
)

PASSPORT_EVIDENCE_TERMS: tuple[str, ...] = (
    "passport",
    "personal identification",
    "identity card",
    "hong kong identity card",
    "consent",
    "withhold",
    "kept by themselves",
    "keep these documents",
    "keep your passport",
    "keep your own",
)

PASSPORT_QUERY_MARKERS: tuple[str, ...] = (
    "passport",
    "جواز",
    "identification document",
    "identity card",
    "hkid",
)

# Weights for hybrid score (sum to 1.0).
VECTOR_WEIGHT = 0.50
LEXICAL_WEIGHT = 0.35
SOURCE_WEIGHT = 0.15

# Per-variant vector retrieval depth before merge.
CANDIDATES_PER_VARIANT = 12


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_passport_question(question: str) -> bool:
    """Return True when the question is about passport / ID document retention."""
    q = _normalize(question)
    return any(marker in q for marker in PASSPORT_QUERY_MARKERS)


def _matches_any(question: str, markers: tuple[str, ...]) -> bool:
    q = _normalize(question)
    return any(marker in q for marker in markers)


def detect_intents(question: str) -> set[str]:
    """Classify query intent for source-aware retrieval boosting."""
    intents: set[str] = set()
    if is_passport_question(question):
        intents.add("passport")
    if _matches_any(question, EMPLOYMENT_AGENCY_MARKERS):
        intents.add("agency")
    if _matches_any(question, FDH_RIGHTS_MARKERS):
        intents.add("fdh_rights")
    if _matches_any(question, EMPLOYER_OBLIGATION_MARKERS):
        intents.add("employer_obligations")
    if _matches_any(question, INJURY_MARKERS):
        intents.add("injury")
    return intents


def boost_terms_for_question(question: str) -> tuple[str, ...]:
    """Return lexical boost terms matched to query intent."""
    if is_passport_question(question):
        return PASSPORT_LEXICAL_TERMS

    intents = detect_intents(question)
    terms: list[str] = []
    if "agency" in intents:
        terms.extend(AGENCY_LEXICAL_TERMS)
    if "fdh_rights" in intents:
        terms.extend(FDH_LEXICAL_TERMS)
    if "employer_obligations" in intents:
        terms.extend(EMPLOYER_LEXICAL_TERMS)
    if "injury" in intents:
        terms.extend(INJURY_LEXICAL_TERMS)

    if terms:
        return tuple(dict.fromkeys(terms))

    q_tokens = [
        token
        for token in re.findall(r"[a-z]{4,}", _normalize(question))
        if token not in {"what", "when", "should", "happens", "entitled"}
    ]
    return tuple(q_tokens[:8])


def expand_query(question: str) -> list[str]:
    """
    Expand a user question into retrieval variants.

    Always includes the original question. For passport-related legal
    yes/no questions, adds targeted keyword variants.
    """
    variants: list[str] = [question.strip()]
    if not is_passport_question(question):
        return variants

    variants.extend(
        [
            "helper passport employer keep",
            "personal identification documents passport consent",
            "withhold passport employment agency helper",
            "FDH passport Hong Kong Identity Card keep consent",
            "job seekers passports personal identification documents",
            "foreign domestic helper keep own passport employer consent",
            "employment agency temporarily keep passport written consent",
        ]
    )

    # Preserve order while deduplicating.
    seen: set[str] = set()
    unique: list[str] = []
    for variant in variants:
        key = _normalize(variant)
        if key and key not in seen:
            seen.add(key)
            unique.append(variant)
    return unique


def _term_hits(text: str, terms: tuple[str, ...]) -> int:
    lowered = _normalize(text)
    return sum(1 for term in terms if term in lowered)


def lexical_score(
    text: str,
    source_file: str,
    question: str,
    boost_terms: tuple[str, ...] | None = None,
) -> float:
    """Score chunk text and metadata for keyword overlap (0.0–1.0)."""
    terms = boost_terms or PASSPORT_LEXICAL_TERMS
    text_hits = _term_hits(text, terms)
    question_hits = _term_hits(question, terms)
    source_hits = _term_hits(source_file, terms)

    # Normalize: up to 6 text hits saturates the score.
    text_component = min(text_hits / 6.0, 1.0)
    question_component = min(question_hits / 3.0, 1.0) * 0.3
    source_component = min(source_hits / 2.0, 1.0) * 0.1
    return min(text_component + question_component + source_component, 1.0)


def source_title_relevance(source_file: str, question: str) -> float:
    """Boost chunks from sources likely to contain the answer (0.0–1.0)."""
    intents = detect_intents(question)

    if "passport" in intents:
        if source_file in PASSPORT_SOURCE_FILES:
            return 1.0
        return 0.0

    if "agency" in intents and "fdh_rights" not in intents and "employer_obligations" not in intents:
        if source_file == COP_FILE:
            return 1.0
        if source_file == FDH_GUIDE_FILE:
            return 0.25
        return 0.0

    score = 0.0

    if "agency" in intents and source_file == COP_FILE:
        score = max(score, 0.85)

    if "fdh_rights" in intents:
        if source_file == FDH_GUIDE_FILE:
            score = max(score, 1.0)
        elif source_file == HANDY_GUIDE_FILE:
            score = max(score, 0.45)
        elif source_file == COP_FILE:
            score = max(score, 0.12)

    if "employer_obligations" in intents:
        if source_file == HANDY_GUIDE_FILE:
            score = max(score, 1.0)
        if source_file == ID969_FILE:
            score = max(score, 0.9)
        if source_file == FDH_GUIDE_FILE:
            score = max(score, 0.65)
        if source_file == COP_FILE:
            score = max(score, 0.1)

    if "injury" in intents:
        if source_file == IMPORTANT_INFO_FILE:
            score = max(score, 1.0)
        if source_file == FDH_GUIDE_FILE:
            score = max(score, 0.75)
        if source_file == COP_FILE:
            score = max(score, 0.1)

    if score > 0:
        return score

    # Mild boost when filename tokens appear in the question.
    q = _normalize(question)
    name = _normalize(source_file.replace(".pdf", "").replace("_", " "))
    overlap = sum(1 for token in name.split() if len(token) > 3 and token in q)
    return min(overlap / 3.0, 0.35)


def hybrid_score(
    vector_score: float,
    text: str,
    source_file: str,
    question: str,
) -> float:
    """Combine vector, lexical, and source-title signals."""
    boost_terms = boost_terms_for_question(question)
    lex = lexical_score(text, source_file, question, boost_terms)
    src = source_title_relevance(source_file, question)
    combined = (
        VECTOR_WEIGHT * vector_score
        + LEXICAL_WEIGHT * lex
        + SOURCE_WEIGHT * src
    )
    return min(max(combined, 0.0), 1.0)


def merge_search_results(
    result_lists: list[list[SearchResult]],
) -> dict[str, SearchResult]:
    """Merge results from multiple queries, keeping the best vector score per chunk."""
    merged: dict[str, SearchResult] = {}
    for results in result_lists:
        for result in results:
            existing = merged.get(result.chunk_id)
            if existing is None or result.score > existing.score:
                merged[result.chunk_id] = result
    return merged


def lexical_scan(
    vector_store: VectorStore,
    question: str,
    limit: int = 15,
) -> list[SearchResult]:
    """
    Scan indexed chunk metadata for strong lexical matches.

    Provides a keyword retrieval path that does not require extra embeddings.
    """
    if not vector_store.metadata:
        return []

    boost_terms = boost_terms_for_question(question)
    candidates: list[SearchResult] = []

    for row in vector_store.metadata:
        text = str(row["text"])
        source = str(row["source_file"])
        lex = lexical_score(text, source, question, boost_terms)
        src = source_title_relevance(source, question)

        if is_passport_question(question):
            if "passport" not in _normalize(text):
                continue
            if source not in PASSPORT_SOURCE_FILES and lex < 0.45:
                continue
        elif src < 0.2 and lex < 0.35:
            continue

        combined = LEXICAL_WEIGHT * lex + SOURCE_WEIGHT * src
        if combined < 0.25:
            continue

        candidates.append(
            SearchResult(
                chunk_id=str(row["chunk_id"]),
                text=text,
                source_file=source,
                page_start=int(row["page_start"]),
                page_end=int(row["page_end"]),
                score=combined,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:limit]


def hybrid_retrieve(
    question: str,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    top_k: int,
) -> list[SearchResult]:
    """
    Retrieve chunks using query expansion, vector search, and hybrid re-ranking.
    """
    if not vector_store.is_loaded:
        return []

    variants = expand_query(question)
    per_variant = max(CANDIDATES_PER_VARIANT, top_k)

    result_lists: list[list[SearchResult]] = []
    for variant in variants:
        embedding = embedding_service.embed_text(variant)
        result_lists.append(vector_store.search(embedding, top_k=per_variant))

    # Lexical scan ensures keyword-critical chunks (e.g. passport rules) are included.
    result_lists.append(lexical_scan(vector_store, question, limit=per_variant))

    merged = merge_search_results(result_lists)
    if not merged:
        return []

    reranked: list[SearchResult] = []
    for result in merged.values():
        final_score = hybrid_score(
            vector_score=result.score,
            text=result.text,
            source_file=result.source_file,
            question=question,
        )
        reranked.append(replace(result, score=final_score))

    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked[:top_k]


def has_direct_evidence(results: list[SearchResult], question: str) -> bool:
    """
  Return True when retrieved chunks contain direct legal support for the question.

    Used to block false insufficient-information fallbacks.
    """
    if not results:
        return False

    if is_passport_question(question):
        combined_text = _normalize(" ".join(r.text for r in results[:10]))
        sources = {r.source_file for r in results[:10]}

        passport_hit = "passport" in combined_text
        id_hit = any(
            term in combined_text
            for term in (
                "personal identification",
                "identity card",
                "hong kong identity card",
            )
        )
        policy_hit = any(
            term in combined_text
            for term in (
                "consent",
                "withhold",
                "kept by themselves",
                "keep these documents",
                "keep your passport",
                "keep your own",
                "no other person",
            )
        )
        relevant_source = bool(sources & PASSPORT_SOURCE_FILES)
        return relevant_source and passport_hit and (id_hit or policy_hit)

    intents = detect_intents(question)
    if intents:
        combined_text = _normalize(" ".join(r.text for r in results[:12]))
        sources = {r.source_file for r in results[:12]}

        if "agency" in intents:
            agency_sources = sources & {COP_FILE, FDH_GUIDE_FILE}
            agency_terms = (
                "commission",
                "job seeker",
                "prescribed",
                "10%",
                "service charge",
                "overcharg",
            )
            if agency_sources and any(term in combined_text for term in agency_terms):
                return True

        if "fdh_rights" in intents:
            fdh_sources = sources & {FDH_GUIDE_FILE, HANDY_GUIDE_FILE, ID969_FILE}
            rights_terms = (
                "wage",
                "salary",
                "rest day",
                "annual leave",
                "statutory holiday",
                "sick leave",
                "leave",
                "minimum allowable wage",
            )
            if fdh_sources and any(term in combined_text for term in rights_terms):
                return True

        if "injury" in intents:
            injury_sources = sources & {
                IMPORTANT_INFO_FILE,
                FDH_GUIDE_FILE,
                HANDY_GUIDE_FILE,
            }
            injury_terms = (
                "injur",
                "compensation",
                "employees' compensation",
                "employees compensation",
                "insurance",
                "sick leave pay",
            )
            if injury_sources and any(term in combined_text for term in injury_terms):
                return True

        if "employer_obligations" in intents:
            obligation_sources = sources & {
                HANDY_GUIDE_FILE,
                ID969_FILE,
                FDH_GUIDE_FILE,
            }
            obligation_terms = (
                "part-time",
                "part time",
                "another employer",
                "live-in",
                "live in",
                "accommodation",
                "employer's residence",
                "immigration ordinance",
            )
            if obligation_sources and any(
                term in combined_text for term in obligation_terms
            ):
                return True

    # Generic: meaningful lexical overlap with the question tokens.
    q_tokens = {t for t in re.findall(r"[a-z\u0600-\u06ff]{4,}", _normalize(question))}
    if not q_tokens:
        return False

    for result in results[:8]:
        chunk_tokens = set(re.findall(r"[a-z\u0600-\u06ff]{4,}", _normalize(result.text)))
        overlap = len(q_tokens & chunk_tokens)
        if overlap >= 2 and result.score >= 0.25:
            return True
    return False


_GENERIC_COP_MARKERS: tuple[str, ...] = (
    "table of contents",
    "chapter 1",
    "means a person who operates",
    "means the maximum commission",
    "means a record maintained",
    "means the person appointed",
    "same meaning assigned",
    "abbreviation",
    "commissioner for labour",
    "code of practice for employment agencies",
)

_FD_H_INTRO_MARKERS: tuple[str, ...] = (
    "practical guide for employment",
    "what foreign domestic helpers and their employers should know",
    "on first employment",
    "table of contents",
)

_AGENCY_EVIDENCE_TERMS: tuple[str, ...] = (
    "commission",
    "job seeker",
    "prescribed",
    "10%",
    "service charge",
    "first-month",
    "first month",
)

_WAGE_EVIDENCE_TERMS: tuple[str, ...] = (
    "wage",
    "salary",
    "pay",
    "payment",
    "within 7 days",
    "within seven days",
    "monthly",
)

_PASSPORT_POLICY_PHRASES: tuple[str, ...] = (
    "keep your own",
    "keep these documents",
    "without your consent",
    "without their consent",
    "without explicit consent",
    "no other person",
    "should not withhold",
    "must not withhold",
    "return the passport",
    "written consent",
    "written acknowledgement",
)

_INJURY_EVIDENCE_TERMS: tuple[str, ...] = (
    "injur",
    "compensation",
    "employees' compensation",
    "employees compensation",
    "insurance",
    "sick leave",
    "accident",
)


def _has_passport_retention_policy(text: str) -> bool:
    """True when chunk states who may keep passports / ID documents."""
    normalized = _normalize(text)
    return any(phrase in normalized for phrase in _PASSPORT_POLICY_PHRASES)


def _is_generic_cop_intro(text: str) -> bool:
    """True for CoP glossary/TOC/abbreviation pages without substantive rules."""
    normalized = _normalize(text)
    return any(marker in normalized for marker in _GENERIC_COP_MARKERS)


def _is_fdh_intro_page(text: str) -> bool:
    """True for FDH guide cover/intro pages without substantive Q&A content."""
    normalized = _normalize(text)
    if not any(marker in normalized for marker in _FD_H_INTRO_MARKERS):
        return False
    substantive = (
        "passport",
        "wage",
        "salary",
        "injur",
        "compensation",
        "rest day",
        "leave",
    )
    return not any(term in normalized for term in substantive)


def _count_term_hits(text: str, terms: tuple[str, ...]) -> int:
    normalized = _normalize(text)
    return sum(1 for term in terms if term in normalized)


def chunk_direct_evidence_score(result: SearchResult, question: str) -> float:
    """
    Score how directly a single chunk supports the answer (0-1).

    Used for citation display — independent of hybrid retrieval rank.
    """
    text = _normalize(result.text)
    boost_terms = boost_terms_for_question(question)
    lex = lexical_score(result.text, result.source_file, question, boost_terms)
    src = source_title_relevance(result.source_file, question)
    score = 0.40 * lex + 0.30 * src + 0.10 * min(result.score, 1.0)

    intents = detect_intents(question)

    if is_passport_question(question):
        if not _has_passport_retention_policy(text):
            score *= 0.10
        else:
            hits = _count_term_hits(text, PASSPORT_EVIDENCE_TERMS)
            score += min(hits / 6.0, 0.25)
            score += 0.20
        if result.source_file == FDH_GUIDE_FILE and _has_passport_retention_policy(text):
            score += 0.18
        if _is_generic_cop_intro(text) and not _has_passport_retention_policy(text):
            score *= 0.05

    elif "agency" in intents:
        hits = _count_term_hits(text, _AGENCY_EVIDENCE_TERMS)
        if hits:
            score += min(hits / 4.0, 0.30)
        else:
            score *= 0.20
        if _is_generic_cop_intro(text) and hits == 0:
            score *= 0.08

    elif "fdh_rights" in intents:
        hits = _count_term_hits(text, _WAGE_EVIDENCE_TERMS)
        if hits:
            score += min(hits / 4.0, 0.30)
        if result.source_file == FDH_GUIDE_FILE:
            score += 0.08
        if _is_fdh_intro_page(text):
            score *= 0.08

    elif "injury" in intents:
        hits = _count_term_hits(text, _INJURY_EVIDENCE_TERMS)
        if hits:
            score += min(hits / 4.0, 0.30)
        if result.source_file in {IMPORTANT_INFO_FILE, FDH_GUIDE_FILE, HANDY_GUIDE_FILE}:
            score += 0.08
        if _is_fdh_intro_page(text):
            score *= 0.10

    elif "employer_obligations" in intents:
        obligation_hits = _count_term_hits(text, EMPLOYER_LEXICAL_TERMS)
        if obligation_hits:
            score += min(obligation_hits / 4.0, 0.25)

    if _is_generic_cop_intro(text) and score < 0.35:
        score *= 0.10

    return min(max(score, 0.0), 1.0)


def is_weak_display_source(result: SearchResult, question: str) -> bool:
    """True when a chunk is a poor citation for the question (e.g. CoP TOC)."""
    text = _normalize(result.text)
    evidence = chunk_direct_evidence_score(result, question)

    if is_passport_question(question):
        if result.source_file == COP_FILE and result.page_start == 9:
            return True
        if not _has_passport_retention_policy(text):
            return True

    if _is_generic_cop_intro(text) and evidence < 0.30:
        return True

    return evidence < 0.15


def strip_answer_citations(answer: str) -> str:
    """Remove inline source citations from model answer prose."""
    text = re.sub(
        r"\[Source\s+\d+[^\]]*\]\s*[\w\(\)\-\.]+\.pdf\s*\(pages?\s*\d+[^)]*\)",
        "",
        answer,
    )
    text = re.sub(r"\[Source\s+\d+[^\]]*\]", "", text)
    patterns = [
        r"\s*Sources?:\s*[\w\(\)\-\.]+\.pdf\s*\(pages?\s*\d+[^)]*\)",
        r"\s*Sources?:\s*[\w\(\)\-\.]+\.pdf\s*,?\s*page\s*\d+",
        r"\s*Sources?:\s*[\w\(\)\-\.]+\.pdf\s*$",
        r"\s*\[Source\s+\d+[^\]]*\]",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


def _extract_answer_citations(answer: str) -> list[tuple[str, int | None]]:
    """Parse intentional source citations from model answer prose."""
    cleaned = re.sub(r"\[Source\s+\d+\][^\n]*", "", answer)
    citations: list[tuple[str, int | None]] = []
    patterns = [
        r"Source:\s*([\w\(\)\-\.]+\.pdf)\s*\(pages?\s*(\d+)",
        r"Source:\s*([\w\(\)\-\.]+\.pdf)\s*,?\s*page\s*(\d+)",
        r"Sources?:\s*([\w\(\)\-\.]+\.pdf)\s*\(pages?\s*(\d+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
            citations.append((match.group(1), int(match.group(2))))

    file_only = re.findall(
        r"Source:\s*([\w\(\)\-\.]+\.pdf)\s*(?:\(|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    for source_file in file_only:
        if not any(file_name == source_file for file_name, _ in citations):
            citations.append((source_file, None))

    return citations


def _extract_bracket_source_refs(answer: str) -> list[tuple[int, int | None]]:
    """Parse numbered context refs like [Source 7, page 6]."""
    refs: list[tuple[int, int | None]] = []
    for match in re.finditer(
        r"\[Source\s+(\d+)(?:,\s*page\s*(\d+))?\]",
        answer,
        flags=re.IGNORECASE,
    ):
        page = int(match.group(2)) if match.group(2) else None
        refs.append((int(match.group(1)), page))
    return refs


def _chunk_from_bracket_ref(
    results: list[SearchResult],
    source_index: int,
    page: int | None,
    exclude: set[str],
) -> SearchResult | None:
    """Map a 1-based [Source N] tag to the retrieval chunk shown in LLM context."""
    if source_index < 1 or source_index > len(results):
        return None
    candidate = results[source_index - 1]
    if candidate.chunk_id in exclude:
        return None
    if page is not None and not _chunk_matches_citation(candidate, candidate.source_file, page):
        same_file = [
            result
            for result in results
            if result.chunk_id not in exclude and result.source_file == candidate.source_file
        ]
        if same_file:
            return max(same_file, key=lambda result: result.score)
    return candidate


def _apply_answer_citation_boosts(
    scored: list[tuple[float, SearchResult]],
    answer: str,
) -> None:
    """Boost chunks referenced in the answer; fall back to same-file best evidence."""
    for source_file, page in _extract_answer_citations(answer):
        matched = False
        for index, (ev_score, result) in enumerate(scored):
            if _chunk_matches_citation(result, source_file, page):
                scored[index] = (ev_score + 0.55, result)
                matched = True

        if matched or page is None:
            continue

        same_file = [
            (index, ev_score, result)
            for index, (ev_score, result) in enumerate(scored)
            if result.source_file == source_file
        ]
        if not same_file:
            continue

        best_index, best_score, best_result = max(same_file, key=lambda item: item[1])
        scored[best_index] = (best_score + 0.35, best_result)


def _chunk_matches_citation(
    result: SearchResult,
    source_file: str,
    page: int | None,
) -> bool:
    if result.source_file != source_file:
        return False
    if page is None:
        return True
    return result.page_start <= page <= result.page_end


def _best_chunk_for_citation(
    results: list[SearchResult],
    source_file: str,
    page: int | None,
    exclude: set[str],
) -> SearchResult | None:
    """Pick the best retrieval chunk for a citation parsed from the answer."""
    matches = [
        result
        for result in results
        if result.chunk_id not in exclude and _chunk_matches_citation(result, source_file, page)
    ]
    if matches:
        return max(matches, key=lambda result: result.score)

    same_file = [
        result
        for result in results
        if result.chunk_id not in exclude and result.source_file == source_file
    ]
    if same_file:
        return max(same_file, key=lambda result: result.score)
    return None


def _best_passport_evidence_chunk(
    results: list[SearchResult],
    question: str,
    exclude: set[str],
) -> SearchResult | None:
    """Prefer FDH passport-policy chunks when answering passport questions."""
    fdh_candidates = [
        result
        for result in results
        if result.chunk_id not in exclude
        and result.source_file == FDH_GUIDE_FILE
        and _has_passport_retention_policy(result.text)
    ]
    if fdh_candidates:
        return max(
            fdh_candidates,
            key=lambda result: chunk_direct_evidence_score(result, question),
        )

    policy_candidates = [
        result
        for result in results
        if result.chunk_id not in exclude and _has_passport_retention_policy(result.text)
    ]
    if not policy_candidates:
        return None
    return max(
        policy_candidates,
        key=lambda result: chunk_direct_evidence_score(result, question),
    )


def select_display_sources(
    results: list[SearchResult],
    question: str,
    display_k: int,
    answer: str | None = None,
) -> list[SearchResult]:
    """
    Choose citation sources from internal retrieval context.

    Prioritizes chunks cited in the model answer, then direct-evidence chunks
    over raw hybrid rank.
    """
    if not results or display_k <= 0:
        return []

    selected: list[SearchResult] = []
    seen: set[str] = set()

    if is_passport_question(question):
        passport_chunk = _best_passport_evidence_chunk(results, question, seen)
        # region agent log
        debug_session_log(
            "F",
            "retrieval.py:select_display_sources",
            "passport_evidence_pick",
            {
                "matched": passport_chunk is not None,
                "match_file": passport_chunk.source_file if passport_chunk else None,
                "match_page": passport_chunk.page_start if passport_chunk else None,
            },
        )
        # endregion
        if passport_chunk:
            selected.append(passport_chunk)
            seen.add(passport_chunk.chunk_id)
            if len(selected) >= display_k:
                return selected

    if answer:
        citations = _extract_answer_citations(answer)
        bracket_refs = _extract_bracket_source_refs(answer)
        # region agent log
        debug_session_log(
            "C",
            "retrieval.py:select_display_sources",
            "citation_extraction",
            {
                "citations": [{"file": f, "page": p} for f, p in citations],
                "bracket_refs": [{"index": i, "page": p} for i, p in bracket_refs],
                "result_files": list({r.source_file for r in results}),
                "display_k": display_k,
            },
        )
        # endregion
        for source_index, page in bracket_refs:
            match = _chunk_from_bracket_ref(results, source_index, page, seen)
            # region agent log
            debug_session_log(
                "E",
                "retrieval.py:select_display_sources",
                "bracket_match_attempt",
                {
                    "source_index": source_index,
                    "cited_page": page,
                    "matched": match is not None,
                    "match_file": match.source_file if match else None,
                    "match_page": match.page_start if match else None,
                },
            )
            # endregion
            if match:
                selected.append(match)
                seen.add(match.chunk_id)
            if len(selected) >= display_k:
                return selected
        for source_file, page in citations:
            match = _best_chunk_for_citation(results, source_file, page, seen)
            # region agent log
            debug_session_log(
                "D",
                "retrieval.py:select_display_sources",
                "citation_match_attempt",
                {
                    "cited_file": source_file,
                    "cited_page": page,
                    "matched": match is not None,
                    "match_file": match.source_file if match else None,
                    "match_page": match.page_start if match else None,
                },
            )
            # endregion
            if match:
                selected.append(match)
                seen.add(match.chunk_id)
            if len(selected) >= display_k:
                return selected

    scored: list[tuple[float, SearchResult]] = [
        (chunk_direct_evidence_score(result, question), result) for result in results
    ]

    if answer:
        _apply_answer_citation_boosts(scored, answer)

    scored.sort(key=lambda item: (item[0], item[1].score), reverse=True)

    for _, result in scored:
        if result.chunk_id in seen:
            continue
        if is_weak_display_source(result, question):
            continue
        seen.add(result.chunk_id)
        selected.append(result)
        if len(selected) >= display_k:
            return selected

    cited_files = {source_file for source_file, _ in _extract_answer_citations(answer or "")}
    if len(selected) < display_k:
        for _, result in scored:
            if result.chunk_id in seen:
                continue
            if cited_files and result.source_file not in cited_files:
                continue
            if is_weak_display_source(result, question):
                continue
            seen.add(result.chunk_id)
            selected.append(result)
            if len(selected) >= display_k:
                break

    return selected


def should_use_fallback(
    results: list[SearchResult],
    question: str,
    min_confidence_score: float,
) -> bool:
    """Decide whether to return the insufficient-information fallback."""
    if not results:
        return True
    if has_direct_evidence(results, question):
        return False
    return results[0].score < min_confidence_score
