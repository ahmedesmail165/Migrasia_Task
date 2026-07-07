# Final Delivery Summary — PoBot Expansion

**Project:** PoBot Expansion – AI Assistant for Migrant Support  
**Delivery date:** 2026-07-07  
**Status:** Ready for demo and handoff — **live validation passed**

---

## Approach (summary)

PoBot ingests seven official Hong Kong labour/FDH PDFs from `data/`, extracts and chunks text, embeds chunks with Gemini, and stores vectors in a local FAISS index. User questions go through **hybrid retrieval** (query expansion, semantic search, lexical boosting, intent-aware re-ranking) before Gemini generates a grounded answer with **source citations**. A separate **display-source** step returns evidence-prioritized citations (not raw retrieval rank), including a low-`top_k` guardrail that keeps at least eight internal chunks for answering while honoring the requested citation count. The same RAG pipeline is exposed via **FastAPI** (`/chat`, `/ingest`, `/health`) and a **CLI**, with regression tests and live sample outputs for English, Arabic, and Tagalog passport questions.

---

## Objective

Build a production-style **RAG chatbot** that helps migrant workers and support staff in Hong Kong answer questions about labour regulations, foreign domestic helpers (FDHs), employment agencies, wages, leave, recruitment rules, and related official guidance — with **grounded answers**, **source citations**, and **multilingual support** (English, Arabic, Tagalog).

---

## Tech stack

| Layer | Technology |
|-------|------------|
| API | FastAPI + Uvicorn |
| LLM & embeddings | Google Gemini (`google-genai`) |
| Chat model | `gemini-2.5-flash` |
| Embedding model (configured) | `gemini-embedding-2` |
| Embedding model (live ingestion) | **`gemini-embedding-001`** (automatic fallback) |
| Vector store | FAISS (`IndexFlatIP`, 768-dim, L2-normalized) |
| Retrieval | Hybrid: query expansion + FAISS + lexical scan + re-ranking |
| Config | `pydantic-settings` + `.env` |
| PDF parsing | PyMuPDF |
| Tests | pytest (**64 tests**) |

---

## Data sources

Seven official Hong Kong labour / FDH PDFs in `data/`:

| Document | Chunks |
|----------|--------|
| CoP_Eng.pdf | 499 |
| Handy_Guide_for_Employers_of_FDHs_English_version_Web_version.pdf | 155 |
| FDHguideEnglish.pdf | 76 |
| ID(E)969.pdf | 53 |
| Letter_to_EA_d.d.30.7.2018.pdf | 52 |
| ImportantInformationForEmployersAndEmployees_Eng.pdf | 5 |
| PointToNotesForEmployersOnEmployment_English.pdf | 1 |

**Corpus totals:** 7 documents · **402 pages** · **841 unique chunks**

---

## Ingestion status

| Item | Status |
|------|--------|
| Ingestion | **PASSED** (live run completed) |
| Chunks file | `processed/chunks.jsonl` |
| Vectors embedded | **841 / 841** |
| Actual embedding model | **`gemini-embedding-001`** (fell back from `gemini-embedding-2` due to 429 rate limits) |
| Embedding dimension | 768 |
| Duration | ~25 minutes (with throttling) |

---

## Vector store status

| Artifact | Status | Size |
|----------|--------|------|
| `vector_store/index.faiss` | Present | ~2.58 MB |
| `vector_store/metadata.json` | Present | ~1.27 MB |
| `/health` | `vector_store_loaded: true`, `chunks_count: 841` | — |

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health + vector store status |
| `GET` | `/sources` | Indexed documents and chunk counts |
| `POST` | `/ingest` | Re-ingest `data/` and rebuild index |
| `POST` | `/chat` | RAG Q&A with source citations |

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## CLI commands

```powershell
# From project root (adjust path to your clone)
.venv\Scripts\activate

# Ingest documents
python scripts\ingest.py
python -m app.cli ingest

# Single question
python -m app.cli ask "Can an employer keep a helper's passport?"

# Interactive chat
python -m app.cli chat
```

---

## Sample questions and outputs

Nine live-validated questions in `sample_outputs.md` (generated 2026-07-07):

1. Rights of foreign domestic helpers (English)
2. Recruitment agency rules (English) — strong grounded answer
3. **Passport retention (English)** — clear “No” + consent rules
4. Wage payment timing (English)
5. Work injury (English)
6. Part-time work for another employer (English)
7. Employment agency charges (English)
8. **Passport retention (Arabic)** — Arabic answer with FDHguide citation
9. **Passport retention (Tagalog)** — Tagalog answer with CoP_Eng citation

---

## Multilingual support status

| Language | Status |
|----------|--------|
| English | Full — primary corpus language |
| Arabic | Working — answers in Arabic when question is in Arabic |
| Tagalog | Working — simple Tagalog answers; English gloss when needed |

Retrieval uses English query expansion for passport topics, which improves cross-language recall.

---

## Known limitations

- Answers depend on documents in `data/`; topics not covered will return a low-confidence fallback.
- PDF extraction may miss scanned images, tables, and complex layouts.
- First ingestion takes ~25+ minutes for 841 chunks (Gemini embedding rate limits).
- Hybrid retrieval uses multiple embeddings per question — `/chat` latency is higher than single-query RAG.
- Informational support only — **not legal advice**.
- Regression tests for passport questions require a valid API key and built vector index.

---

## Future improvements

- Incremental ingestion (add/update documents without full rebuild)
- OCR for scanned PDFs
- Conversation memory and follow-up questions
- Admin UI for document management
- Query caching for frequent questions
- Broader evaluation harness with labelled Q&A pairs
- Set `GEMINI_EMBEDDING_MODEL=gemini-embedding-001` by default to avoid rate-limit delays on `gemini-embedding-2`

---

## Demo commands

```powershell
# From project root (adjust path to your clone)
.venv\Scripts\activate

# Verify tests
pytest -v

# Start API (already running if you started it earlier)
uvicorn app.main:app --reload

# Health check
curl http://127.0.0.1:8000/health

# List sources
curl http://127.0.0.1:8000/sources

# Chat — passport question (PowerShell-safe)
python -c "import urllib.request,json; b=json.dumps({\"question\": \"Can an employer keep a helper's passport?\", \"top_k\": 10}).encode(); r=urllib.request.Request('http://127.0.0.1:8000/chat',data=b,headers={'Content-Type':'application/json'},method='POST'); print(urllib.request.urlopen(r).read().decode())"

# Regenerate sample outputs
python scripts\generate_sample_outputs.py
```

---

## Delivery artifacts

| File | Purpose |
|------|---------|
| `README.md` | Setup, usage, troubleshooting |
| `FINAL_DELIVERY_SUMMARY.md` | This document |
| `LIVE_VALIDATION_REPORT.md` | Live pipeline validation (ingestion + API) |
| `ANSWER_QUALITY_REPORT.md` | RAG hardening before/after |
| `VALIDATION_REPORT.md` | Offline structural validation |
| `sample_outputs.md` | Live Gemini answer examples |
| `.env.example` | Environment template (no secrets) |
| `requirements.txt` | Python dependencies |

---

## Security

- `.env` is listed in `.gitignore` — API key must not be committed.
- No API key values found in project source, logs, or output markdown files.
- Store `GEMINI_API_KEY` only in local `.env`.

---

## Final validation (2026-07-07)

| Check | Result |
|-------|--------|
| `pytest -v` | **64 passed** |
| Citation consistency (`top_k=1`) | **Passed** — see `ANSWER_QUALITY_REPORT.md` |
| Source balance check | **Passed** — `source_balance_results.json` |
| Chunk count | **841** |
| Vector index | **Loaded** (`index.faiss` + `metadata.json`) |
| `sample_outputs.md` | **Regenerated** (2026-07-07 11:14 UTC) |
| API `/health` | `{"status":"ok","vector_store_loaded":true,"chunks_count":841}` |
| Uvicorn | Running at `http://127.0.0.1:8000` with `--reload` |
