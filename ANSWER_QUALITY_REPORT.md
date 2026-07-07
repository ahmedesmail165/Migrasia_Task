# RAG Answer Quality Hardening Report

**Date:** 2026-07-07  
**Issue:** Passport retention questions returned false “insufficient information” despite direct corpus support in `FDHguideEnglish.pdf` and `CoP_Eng.pdf`.

---

## Retrieval changes

New module: `app/services/retrieval.py`

| Feature | Implementation |
|---------|----------------|
| **Query expansion** | Passport/legal yes-no questions expand to 8 variants (original + keyword phrases like `helper passport employer keep`, `withhold passport employment agency helper`, etc.) |
| **Multi-query merge** | Each variant is embedded and searched; results merged by `chunk_id`, best vector score kept |
| **Lexical scan** | Metadata scan boosts chunks containing `passport`, `consent`, `personal identification documents`, `withhold`, `kept by themselves`, etc. |
| **Hybrid re-ranking** | Final score = 50% vector + 35% lexical + 15% source-title relevance |
| **Source boosting** | Passport: `FDHguideEnglish.pdf` + `CoP_Eng.pdf`. Agency: `CoP_Eng.pdf`. FDH rights / employer obligations / injury: intent-based boosts with CoP demotion |
| **Answerability gate** | `has_direct_evidence()` blocks fallback when retrieved chunks contain passport + consent/keep rules from authoritative sources |
| **Default `top_k`** | Increased from 5 → **8** (`TOP_K` in config / `.env.example`) |
| **Prompt hardening** | Instructs model to answer when context supports it; start yes/no legal answers with clear yes/no; avoid false “insufficient info” |

`RAGService.ask()` now calls `hybrid_retrieve()` instead of single-query FAISS search.

---

## Before / after — passport question

**Question:** `Can an employer keep a helper's passport?`

### Before (broken)

> The provided documents do not contain enough information to answer whether an employer can keep a helper's passport.

Sources were generic `CoP_Eng.pdf` table-of-contents chunks (pages 1–5); **no** `FDHguideEnglish.pdf`.

### After (fixed)

**Live `/chat` response (top_k=10):**

> No, an employer should not keep a helper's passport without their consent. Helpers should keep their own personal identification documents, and no other person, including their employer or staff of the employment agency, should keep these documents for them without their consent.
>
> This is informational support, not legal advice.
>
> Source: FDHguideEnglish.pdf (pages 5-6)

**Citations:** `FDHguideEnglish.pdf` (pages 5-6) + multiple `CoP_Eng.pdf` pages (19, 30, 124-125, etc.)  
**Confidence:** 0.75

---

## Multilingual passport tests

| Language | Question | Pass? | Notes |
|----------|----------|-------|-------|
| English | Can an employer keep a helper's passport? | **Yes** | Clear “No”; mentions consent; cites FDHguide + CoP |
| English | Can an employment agency hold my passport? | **Yes** | Regression tests pass (retrieval + mocked ask) |
| English | Who should keep the helper's passport? | **Yes** | Regression tests pass |
| Arabic | هل يمكن لصاحب العمل الاحتفاظ بجواز سفر العاملة؟ | **Yes** | Arabic “No” + consent; cites FDHguideEnglish.pdf p.6 |
| Tagalog | Pwede bang kunin ng employer ang passport ng helper? | **Yes** | Tagalog “Hindi”; consent + withholding rules; cites CoP_Eng.pdf |

All passport regression tests: **no insufficient-information fallback**.

---

## Tests added

| File | Coverage |
|------|----------|
| `tests/test_retrieval.py` | Query expansion, evidence detection, hybrid merge/boost, intent detection, source-title boosting (unit) |
| `tests/test_passport_regression.py` | 5 passport questions × lexical scan, hybrid retrieve, ask path (live vector store) |

**pytest result:** `64 passed` (full suite including citation consistency tests)

---

## Low `top_k` guardrail (display vs retrieval depth)

**Bug:** `POST /chat` with `{"question": "Can an employer keep a helper's passport?", "top_k": 1}` returned a false “documents do not contain information” answer while citing only one weak `CoP_Eng.pdf` chunk.

**Root cause:** User `top_k` was passed directly into `hybrid_retrieve()`, shrinking both retrieval and LLM context to a single chunk.

**Fix:** `resolve_retrieval_depth()` in `app/config.py` separates:

| Variable | Purpose |
|----------|---------|
| `retrieval_k` | Internal hybrid search, answerability gate, Gemini context (`max(requested, MIN_RETRIEVAL_K)`) |
| `display_k` | Source citations returned to client (`min(requested, MAX_DISPLAY_SOURCES)`) |

New env settings (`.env.example`):

```
DEFAULT_TOP_K=8
MIN_RETRIEVAL_K=8
MAX_RETRIEVAL_K=20
MAX_DISPLAY_SOURCES=10
```

`TOP_K` remains a backward-compatible alias for `DEFAULT_TOP_K`.

### Before / after — `/chat` with `top_k=1`

**Before (broken):**

```json
{
  "answer": "The provided documents do not contain information on whether an employer can keep a helper's passport.",
  "sources": [{"source_file": "CoP_Eng.pdf", "page_start": 9, "page_end": 9, "score": 0.75}],
  "confidence": 0.75
}
```

**After (fixed — answer + citation consistency):**

```json
{
  "answer": "No, an employer should not keep a helper's passport without the helper's consent. ...",
  "sources": [{"source_file": "FDHguideEnglish.pdf", "page_start": 5, "page_end": 6, "score": 0.75}],
  "confidence": 0.75
}
```

Only **one** source is returned (`display_k=1`). The model receives **8** internal chunks, and the displayed source is the **best direct-evidence** chunk (not raw hybrid rank #1).

**Tests added:** `tests/test_retrieval_depth.py`, `tests/test_low_top_k_regression.py`, `tests/test_citation_consistency.py`

---

## Citation / source consistency (display vs evidence)

**Bug:** With `top_k=1`, the answer was correct (grounded on `FDHguideEnglish.pdf` passport/consent rules) but `sources[0]` was still `CoP_Eng.pdf` page 9 — a glossary/definitions page with no passport-retention rule.

**Root cause:** `RAGService.ask()` returned `results[:display_k]` — the first N chunks by **hybrid retrieval score**, not by **citation/evidence quality**.

**Fix:** `select_display_sources()` in `app/services/retrieval.py`:

| Signal | Weight / rule |
|--------|----------------|
| Per-chunk `chunk_direct_evidence_score()` | Intent-aware lexical + source relevance + passport/consent/wage/injury/agency term hits |
| Generic CoP intro penalty | TOC, glossary, abbreviation pages demoted when they lack direct rule text |
| Answer prose alignment | Parses `Source: …pdf (page N)` from Gemini answer and boosts matching chunks |
| `top_k=1` | Returns single **best evidence** chunk from internal context, not hybrid rank #1 |

`RAGService.ask()` now calls `select_display_sources(results, question, display_k, answer=answer)` **after** answer generation so displayed citations match grounded prose.

### Before / after — passport `top_k=1` displayed source

| | `sources[0]` | Evidence quality |
|---|--------------|------------------|
| **Before** | `CoP_Eng.pdf` p.9 (FDH/job-seeker definitions) | Weak — no passport-retention rule |
| **After** | `FDHguideEnglish.pdf` p.5–6 (or CoP p.19/30 with passport+consent) | Strong — direct “keep your own passport … without consent” |

Regression coverage: `tests/test_citation_consistency.py` (passport, wages, injury, agency charge × `top_k=1`).

---

## Source-balance sanity check

**Date:** 2026-07-07  
**Concern:** `CoP_Eng.pdf` holds 499/841 chunks (~59% of corpus). Pre-fix retrieval returned CoP for 8–9 of 10 sources on almost every question, including FDH rights, wages, and leave.

### Root cause

Hybrid retrieval applied passport/agency lexical terms (`employer`, `employment agency`, etc.) to **all** questions. Combined with a generic filename `"guide"` token match, `CoP_Eng.pdf` dominated vector + lexical re-ranking even when questions were unrelated to employment agencies.

### Fix (intent-based source boosting)

Extended `app/services/retrieval.py` with:

| Intent | Detected by | Preferred sources | CoP treatment |
|--------|-------------|-------------------|---------------|
| `agency` | recruitment agency, commission, licence, job seeker, … | `CoP_Eng.pdf` = 1.0 | Full boost |
| `passport` | passport, identity card, … | `FDHguideEnglish.pdf` + `CoP_Eng.pdf` | Both boosted |
| `fdh_rights` | wages, leave, rest days, domestic helper, … | `FDHguideEnglish.pdf` = 1.0 | Demoted to 0.12 |
| `employer_obligations` | live-in, part-time, accommodation, … | `Handy_Guide…pdf`, `ID(E)969.pdf` | Demoted to 0.10 |
| `injury` | injured, compensation, insurance, … | `ImportantInformation…pdf` | Demoted to 0.10 |

Additional changes:
- Intent-specific lexical terms replace blanket `PASSPORT_LEXICAL_TERMS` for non-passport queries
- Lexical scan skips low-relevance CoP chunks when `src < 0.2` and `lex < 0.35`
- Unit tests for `detect_intents()` and `source_title_relevance()` in `tests/test_retrieval.py`
- Script: `scripts/source_balance_check.py` → `source_balance_results.json`

### Before / after — CoP share (top 10 retrieved sources)

| Question | Pre-fix CoP | Post-fix CoP | Verdict |
|----------|-------------|--------------|---------|
| Rules for recruitment agencies | 9/10 | **10/10** | Expected — CoP is authoritative |
| Employment agency charge to job seeker | 9/10 | **10/10** | Expected |
| Can employer keep helper's passport? | 9/10 | **9/10** | Relevant — both CoP and FDH guide cover passport rules |
| Rights of FDHs in Hong Kong | 9/10 | **0/10** | Fixed — `FDHguideEnglish.pdf` only |
| When should wages be paid? | 9/10 | **0/10** | Fixed — `FDHguideEnglish.pdf` only |
| What leave is a domestic helper entitled to? | 9/10 | **0/10** | Fixed — `FDHguideEnglish.pdf` only |
| Helper injured at work | 8/10 | **0/10** | Fixed — `ImportantInformation…pdf` (5), Handy Guide (2), FDH guide (3) |
| FDH work part-time for another employer | 9/10 | **0/10** | Fixed — `FDHguideEnglish.pdf` only |
| FDH live outside employer's home | 9/10 | **0/10** | Fixed — `FDHguideEnglish.pdf` only |

### Per-question results (post-fix, live `/chat`, top_k=10)

| Question | Confidence | Top sources | CoP assessment |
|----------|------------|-------------|----------------|
| Recruitment agency rules | 0.75 | CoP (10) | **Relevant** — correct dominance |
| Agency charge to job seeker | 0.75 | CoP (10) | **Relevant** — 10% commission rule cited |
| Employer keep passport? | 0.75 | CoP (9), FDH guide (1) | **Relevant** — clear No + consent; both authoritative |
| FDH rights | 0.75 | FDH guide (10) | **Not selected** — correct |
| Wages timing | 0.75 | FDH guide (10) | **Not selected** — correct |
| Leave entitlement | 0.75 | FDH guide (10) | **Not selected** — rest days, holidays, annual leave |
| Work injury | 0.75 | ImportantInfo (5), Handy Guide (2), FDH guide (3) | **Not selected** — ECO compensation answered |
| Part-time for another employer | 0.75 | FDH guide (10) | **Not selected** — Immigration Ordinance offence |
| Live outside employer's home | 0.75 | FDH guide (10) | **Not selected** — live-in requirement from SEC |

### Verdict

**PASS** — CoP dominance on unrelated FDH employment-rights questions was a retrieval bug, not corpus size alone. After intent-based boosting:

- Agency questions correctly prefer `CoP_Eng.pdf`
- FDH rights, wages, leave, injury, part-time, and live-in questions no longer over-select CoP
- Passport questions still cite CoP alongside `FDHguideEnglish.pdf` (both documents contain direct rules)

**Note:** Re-run `/chat` checks against a single uvicorn instance. Duplicate servers on port 8000 served stale pre-fix code during initial validation.

---

## Updated artifacts

- `sample_outputs.md` — regenerated with live Gemini answers (passport Q3, Q8, Q9 now grounded)
- `source_balance_results.json` — post-fix retrieval source counts
- `scripts/source_balance_check.py` — repeatable 9-question sanity check
- `.env.example` — `TOP_K=8`

---

## Demo commands

```powershell
# From project root
.venv\Scripts\activate

pytest tests/ -q
python scripts\source_balance_check.py
python scripts\generate_sample_outputs.py
uvicorn app.main:app --reload

python -c "import urllib.request,json; b=json.dumps({\"question\": \"Can an employer keep a helper's passport?\", \"top_k\": 10}).encode(); r=urllib.request.Request('http://127.0.0.1:8000/chat',data=b,headers={'Content-Type':'application/json'},method='POST'); print(urllib.request.urlopen(r).read().decode())"
```
