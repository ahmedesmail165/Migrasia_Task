# PoBot Validation Report

Generated during final delivery validation.

## Summary

| Check | Status | Details |
|-------|--------|---------|
| PDF discovery | **PASS** | 7/7 PDFs found in `data/` |
| Page extraction | **PASS** | 402 pages extracted |
| Text cleaning + chunking | **PASS** | 841 unique chunks |
| `processed/chunks.jsonl` | **PASS** | Created (~1.2 MB) |
| FAISS index build | **PASS** | `vector_store/index.faiss` + `metadata.json` |
| Live chat validation | **PASS** | See `LIVE_VALIDATION_REPORT.md` and `sample_outputs.md` |
| Unit tests (`pytest`) | **PASS** | **64/64** tests passed |
| Missing API key error | **PASS** | Clear configuration error returned |

## Documents discovered (7 files)

| Source file | Pages loaded | Chunks |
|-------------|-------------:|-------:|
| CoP_Eng.pdf | 223 | 499 |
| FDHguideEnglish.pdf | 50 | 76 |
| Handy_Guide_for_Employers_of_FDHs_English_version_Web_version.pdf | 84 | 155 |
| ID(E)969.pdf | 20 | 53 |
| ImportantInformationForEmployersAndEmployees_Eng.pdf | 2 | 5 |
| Letter_to_EA_d.d.30.7.2018.pdf | 21 | 52 |
| PointToNotesForEmployersOnEmployment_English.pdf | 2 | 1 |

**Totals:** 402 pages extracted, 841 chunks written to `processed/chunks.jsonl`.

## Production readiness review

| Requirement | Status |
|-------------|--------|
| No hardcoded API keys | OK – keys loaded from `.env` only |
| Clear error if `GEMINI_API_KEY` missing | OK – validated |
| Clean logging | OK – structured `pobot` logger |
| Exception handling | OK – API/CLI return clear errors |
| No absolute paths in code | OK – relative paths via config |
| README accuracy | OK – Windows + Linux commands documented |
| Embedding model fallback | OK – auto-fallback to `gemini-embedding-001`, then `text-embedding-004` |
| Windows commands | OK – `.venv\Scripts\activate`, `copy`, `python scripts\ingest.py` |
| Citation-aware displayed sources | OK – `select_display_sources()` in `retrieval.py` |
| Low `top_k` guardrail | OK – internal retrieval ≥ `MIN_RETRIEVAL_K` |

## Live validation (2026-07-07)

| Check | Status | Details |
|-------|--------|---------|
| Live ingestion (embeddings) | **PASS** | 841/841 chunks embedded |
| FAISS index build | **PASS** | `index.faiss` + `metadata.json` written |
| Live chat / API tests | **PASS** | `/health`, `/sources`, `/chat` — see `LIVE_VALIDATION_REPORT.md` |
| Answer quality hardening | **PASS** | Passport, source balance, citation consistency — see `ANSWER_QUALITY_REPORT.md` |
| `pytest -v` | **PASS** | **64 passed** |

> **Historical note:** An early validation attempt failed with `API_KEY_INVALID` before a valid key was configured in `.env`. That blocker was resolved; the final pipeline completed successfully.

## Re-run validation (optional)

```powershell
.venv\Scripts\activate
pytest -v
python scripts\source_balance_check.py
python scripts\test_top_k_one.py
python -m app.cli ask "Can an employer keep a helper's passport?"
```

## Expected API results

**GET /health**

```json
{
  "status": "ok",
  "vector_store_loaded": true,
  "chunks_count": 841
}
```

**GET /sources** – returns all 7 documents with per-file chunk counts.

## Notes

- One PDF page was empty after extraction and was skipped during chunking (401 non-empty pages → 841 chunks).
- First full ingestion may take 20–40 minutes for 841 chunks due to sequential Gemini embedding calls.
- Re-run `python scripts\generate_sample_outputs.py` after ingestion to refresh live chat examples in `sample_outputs.md`.
- Restart uvicorn after code changes; stale servers on port 8000 can serve outdated behavior.
