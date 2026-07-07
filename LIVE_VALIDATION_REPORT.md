# Live Validation Report — PoBot RAG Chatbot

**Date:** 2026-07-07 (UTC)  
**Environment:** Windows, Python 3.14.4, project virtualenv (`.venv`)  
**Status:** **PASSED** — full live pipeline completed successfully (post answer-quality hardening)

---

## 1. Ingestion

| Item | Result |
|------|--------|
| **Outcome** | **PASSED** (exit code 0) |
| **Duration** | ~25 minutes (13:11:23 – 13:36:54 local) |
| **PDFs processed** | 7 files, 402 pages |
| **Chunks created** | **841** unique chunks |
| **Chunks embedded** | **841** |
| **Configured embedding model** | `gemini-embedding-2` (`.env`) |
| **Actual embedding model used** | **`gemini-embedding-001`** (fallback after rate limit on primary) |
| **Embedding dimension** | 768 |
| **Chat model** | `gemini-2.5-flash` |

### Vector store artifacts

| File | Created | Size |
|------|---------|------|
| `vector_store/index.faiss` | Yes | 2,583,597 bytes |
| `vector_store/metadata.json` | Yes | 1,266,411 bytes |
| `processed/chunks.jsonl` | Yes | ~1.2 MB |

### Per-document chunk counts

| Source file | Chunks |
|-------------|--------|
| CoP_Eng.pdf | 499 |
| Handy_Guide_for_Employers_of_FDHs_English_version_Web_version.pdf | 155 |
| FDHguideEnglish.pdf | 76 |
| ID(E)969.pdf | 53 |
| Letter_to_EA_d.d.30.7.2018.pdf | 52 |
| ImportantInformationForEmployersAndEmployees_Eng.pdf | 5 |
| PointToNotesForEmployersOnEmployment_English.pdf | 1 |
| **Total** | **841** |

---

## 2. Warnings, fallbacks, and rate limits

During ingestion:

1. **`gemini-embedding-2`** hit **429 RESOURCE_EXHAUSTED** (`global_embed_content_requests_per_minute_per_base_model`) on the first chunk batch. The service fell back to **`gemini-embedding-001`** after 2 attempts.
2. **`gemini-embedding-001`** had **3 transient retry warnings** (chunks ~100, ~475, ~575) — each recovered on retry; no ingestion failure.
3. **Embedding fallback stickiness fix applied** before this run: once `gemini-embedding-001` succeeded, all 841 chunks used that model (confirmed in progress logs every 25 chunks).
4. **Throttling:** `EMBEDDING_REQUEST_DELAY=1.5` seconds between embed calls (default) to reduce 429s.
5. **No auth errors** — API key is valid.
6. **No chat/API errors** during sample generation or endpoint tests.

**Recommendation for faster re-ingestion:** Set `GEMINI_EMBEDDING_MODEL=gemini-embedding-001` in `.env` to skip the initial rate-limited primary model attempt.

---

## 3. API endpoint tests

Server: `uvicorn app.main:app --reload` on `http://127.0.0.1:8000`

### `GET /health`

```json
{"status":"ok","vector_store_loaded":true,"chunks_count":841}
```

### `GET /sources` (summary)

7 documents indexed, **841 total chunks** (see table in §1).

### `POST /chat` (post-hardening example)

**Request:**
```json
{"question": "Can an employer keep a helper's passport?", "top_k": 5}
```

**Response (representative, after retrieval + citation fixes):**
```json
{
  "answer": "No, an employer should not keep a helper's passport without the helper's consent. ...",
  "confidence": 0.75,
  "sources": [
    {"source_file": "FDHguideEnglish.pdf", "page_start": 5, "page_end": 6, "score": 0.75},
    {"source_file": "CoP_Eng.pdf", "page_start": 19, "page_end": 19, "score": 0.75}
  ]
}
```

> **Note (PowerShell):** `curl` with apostrophes in JSON can fail. Use the Python one-liner in §6 or `Invoke-RestMethod` with a here-string.  
> **Note (stale server):** If `/chat` still returns old “insufficient information” or CoP page 9 only, stop all uvicorn processes on port 8000 and restart `uvicorn app.main:app --reload`.

---

## 4. Sample outputs (`sample_outputs.md`)

Generated live via `python scripts/generate_sample_outputs.py` at **2026-07-07 11:14 UTC** (regenerated after answer-quality hardening). All 9 questions received real Gemini answers with source citations.

| # | Language | Question (summary) | Confidence | Answer summary |
|---|----------|-------------------|------------|----------------|
| 1 | English | FDH rights in HK | 0.53 | Grounded rights answer with passport/consent rules |
| 2 | English | Recruitment agency rules | 0.75 | **Grounded answer** — EO/EAR licensing, Code of Practice, 10% commission cap |
| 3 | English | Passport retention | 0.75 | **Clear “No”** + consent; cites FDHguide + CoP |
| 4 | English | Wage payment timing | 0.75 | Within 7 days after wage period |
| 5 | English | Work injury | 0.75 | Employees' compensation insurance and sick leave pay |
| 6 | English | Part-time for another employer | 0.75 | No — Immigration Ordinance offence |
| 7 | English | Agency charges | 0.75 | Prescribed commission ≤ 10% of first month's wages |
| 8 | **Arabic** | هل يمكن لصاحب العمل الاحتفاظ بجواز سفر العاملة؟ | 0.75 | Arabic “No” + consent; cites FDHguide |
| 9 | **Tagalog** | Pwede bang kunin ng employer ang passport ng helper? | 0.75 | Tagalog “Hindi” + consent; cites CoP/FDHguide |

Full answers with sources and raw JSON: see **`sample_outputs.md`**.

---

## 5. API key security

| Check | Result |
|-------|--------|
| Key printed in ingestion logs | **No** — logs show model names and errors only |
| Key in `sample_outputs.md` | **No** |
| Key in `ingest_log.txt` / reports | **No** (`AIza` pattern not found in project output files) |
| `.env` in `.gitignore` | **Yes** (line 5) |
| `.env` committed to git | **No** — excluded by `.gitignore` |
| Key exposed in API responses | **No** |

---

## 6. Demo commands for delivery

```powershell
# From project root (adjust path to your clone)
.venv\Scripts\activate

# One-time / after PDF updates (~25 min for 841 chunks with current rate limits)
python scripts\ingest.py

# Generate sample_outputs.md with live Gemini answers
python scripts\generate_sample_outputs.py

# Start API
uvicorn app.main:app --reload

# Health & sources
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/sources

# Chat (PowerShell-safe — use Python for apostrophes)
python -c "import urllib.request,json; b=json.dumps({\"question\": \"Can an employer keep a helper's passport?\", \"top_k\": 5}).encode(); r=urllib.request.Request('http://127.0.0.1:8000/chat',data=b,headers={'Content-Type':'application/json'},method='POST'); print(urllib.request.urlopen(r).read().decode())"

# Run unit tests
pytest tests/ -q
```

---

## 7. Summary

The **full live pipeline passed**: ingestion embedded 841 chunks (via `gemini-embedding-001` fallback), FAISS index and metadata were written, `sample_outputs.md` contains real Gemini answers including **English, Arabic, and Tagalog** test questions, and all three API endpoints respond correctly with the vector store loaded. Post-hardening, passport and related labour questions return grounded answers with evidence-prioritized citations (`ANSWER_QUALITY_REPORT.md`). **`pytest -v`: 64 passed.** Minor embedding rate-limit warnings occurred but did not block completion. The API key is not exposed in logs or output files and is excluded from version control.
