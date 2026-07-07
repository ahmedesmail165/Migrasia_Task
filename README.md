# PoBot Expansion – AI Assistant for Migrant Support

PoBot is a **Retrieval-Augmented Generation (RAG)** chatbot built with **FastAPI** and **Google Gemini**. It answers questions about Hong Kong labour regulations, migrant worker protections, foreign domestic helpers (FDHs), employment agencies, wages, leave, recruitment rules, immigration-related employment rules, and work injury compensation.

The system ingests official PDF documents from `data/`, chunks and embeds them with Gemini, stores vectors in **FAISS**, and generates grounded answers with source citations using **hybrid retrieval** (semantic search + lexical keyword boosting + query expansion).

## Features

- Automatic document ingestion from `data/` (PDF, TXT, MD, HTML)
- Gemini embeddings and chat via `google-genai`
- **Hybrid retrieval:** query expansion, FAISS vector search, lexical metadata scan, and re-ranking
- **Answerability gate** to prevent false “insufficient information” when evidence exists
- Automatic **embedding model fallback** with rate-limit handling
- FAISS vector store with confidence-based fallback
- FastAPI REST API (`/health`, `/ingest`, `/chat`, `/sources`)
- CLI for ingestion and interactive chat
- Source citations with file name and page numbers (evidence-prioritized, not raw retrieval rank)
- Multilingual answers (English, Arabic, Tagalog per prompt rules)

## Project structure

```
.
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # pydantic-settings configuration
│   ├── schemas.py           # API request/response models
│   ├── cli.py               # Command-line interface
│   └── services/
│       ├── document_loader.py
│       ├── chunker.py
│       ├── embeddings.py    # Gemini embeddings + model fallback
│       ├── vector_store.py  # FAISS index
│       ├── retrieval.py     # Hybrid retrieval + query expansion
│       ├── rag_service.py   # RAG pipeline
│       └── ingestion.py
├── data/                    # Source documents (PDFs)
├── processed/chunks.jsonl   # Generated chunks
├── vector_store/            # FAISS index + metadata
├── scripts/
│   ├── ingest.py
│   ├── validate_e2e.py
│   ├── generate_sample_outputs.py
│   ├── source_balance_check.py
│   └── test_top_k_one.py
└── tests/                   # pytest tests (64 tests)
```

## Requirements

- Python 3.11+
- A valid [Gemini API key](https://aistudio.google.com/apikey)

## Setup

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
python scripts/ingest.py
uvicorn app.main:app --reload
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set GEMINI_API_KEY (save the file!)
python scripts\ingest.py
uvicorn app.main:app --reload
```

API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Environment variables

Copy `.env.example` to `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
DATA_DIR=data
PROCESSED_DIR=processed
VECTOR_STORE_DIR=vector_store
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
TOP_K=8
MIN_RETRIEVAL_K=8
MAX_RETRIEVAL_K=20
MAX_DISPLAY_SOURCES=10
MIN_CONFIDENCE_SCORE=0.35
```

Optional tuning:

```env
EMBEDDING_DIMENSIONALITY=768
EMBEDDING_BATCH_SIZE=16
EMBEDDING_REQUEST_DELAY=1.5
```

**Never commit `.env`** — it is listed in `.gitignore`.

### Embedding model fallback

The default embedding model is `gemini-embedding-2`. If that model is unavailable or rate-limited, PoBot automatically tries these fallbacks in order:

1. `GEMINI_EMBEDDING_MODEL` from `.env` (default: `gemini-embedding-2`)
2. `gemini-embedding-001`
3. `text-embedding-004`

Once a fallback succeeds, it is reused for all subsequent embeddings in the same run. To skip the rate-limited primary model:

```env
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

After changing the embedding model, always re-run ingestion:

```powershell
python scripts\ingest.py
```

## Hybrid retrieval

PoBot does **not** rely on FAISS semantic search alone. The retrieval pipeline (`app/services/retrieval.py`) includes:

1. **Query expansion** — For legal yes/no questions (e.g. passport retention), the original question is expanded into keyword variants before embedding.
2. **Multi-query FAISS search** — Results from all variants are merged by `chunk_id`.
3. **Lexical scan** — A metadata keyword pass boosts chunks containing terms like `passport`, `consent`, `personal identification documents`, `withhold`, etc.
4. **Hybrid re-ranking** — Final score combines vector similarity (50%), lexical match (35%), and source-title relevance (15%).
5. **Answerability check** — Before returning a fallback, the system checks whether retrieved chunks contain direct legal support.

### `top_k` behavior

The `top_k` field on `/chat` controls **how many source citations are returned**, not how many chunks are retrieved for answering.

| Setting | Default | Purpose |
|---------|---------|---------|
| `DEFAULT_TOP_K` | 8 | Default citation count when `top_k` is omitted |
| `MIN_RETRIEVAL_K` | 8 | Minimum internal retrieval depth (always used for context + answerability) |
| `MAX_RETRIEVAL_K` | 20 | Upper cap on internal retrieval depth |
| `MAX_DISPLAY_SOURCES` | 10 | Maximum citations returned even if `top_k` is higher |

Example: `{"question": "...", "top_k": 1}` returns **one** source citation, but the system still retrieves at least **8** chunks internally so the LLM sees enough evidence (e.g. passport rules in `FDHguideEnglish.pdf`).

Legacy env name `TOP_K` is still accepted as an alias for `DEFAULT_TOP_K`.

## Ingest documents

Place official documents in `data/`, then run:

```powershell
python scripts\ingest.py
```

Or via CLI:

```powershell
python -m app.cli ingest
```

Or via API:

```powershell
curl -X POST http://127.0.0.1:8000/ingest
```

First ingestion of ~841 chunks may take **20–30 minutes** due to Gemini embedding rate limits.

## Run FastAPI

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### API examples

**Health check**

```powershell
curl http://127.0.0.1:8000/health
```

Expected: `{"status":"ok","vector_store_loaded":true,"chunks_count":841}`

**List sources**

```powershell
curl http://127.0.0.1:8000/sources
```

**Chat (PowerShell-safe — use Python for apostrophes in JSON)**

```powershell
python -c "import urllib.request,json; b=json.dumps({\"question\": \"Can an employer keep a helper's passport?\", \"top_k\": 10}).encode(); r=urllib.request.Request('http://127.0.0.1:8000/chat',data=b,headers={'Content-Type':'application/json'},method='POST'); print(urllib.request.urlopen(r).read().decode())"
```

**Chat (bash / Git Bash)**

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Can an employer keep a helper'\''s passport?", "top_k": 10}'
```

## CLI usage

```powershell
# Ingest documents
python -m app.cli ingest

# Single question
python -m app.cli ask "What are the rules for recruitment agencies?"

# Interactive chat (type exit, quit, or q to leave)
python -m app.cli chat
```

## Sample questions

Try these after ingestion (full live outputs in `sample_outputs.md`):

1. What are the rights of foreign domestic helpers in Hong Kong?
2. What are the rules for recruitment agencies?
3. Can an employer keep a helper's passport?
4. When should wages be paid to a domestic helper?
5. What happens if a helper is injured at work?
6. Can a foreign domestic helper work part-time for another employer?
7. What should an employment agency charge a job seeker?
8. هل يمكن لصاحب العمل الاحتفاظ بجواز سفر العاملة؟ (Arabic)
9. Pwede bang kunin ng employer ang passport ng helper? (Tagalog)

Regenerate live sample outputs:

```powershell
python scripts\generate_sample_outputs.py
```

## Run tests

```powershell
pytest -v
```

Unit tests use mocks and do not call Gemini. Passport regression tests use the live vector index and `.env` API key when available.

## Validation

| Script / report | Purpose |
|-----------------|---------|
| `python scripts/validate_e2e.py` | End-to-end structural check |
| `VALIDATION_REPORT.md` | Offline validation |
| `LIVE_VALIDATION_REPORT.md` | Live ingestion + API validation |
| `python scripts/source_balance_check.py` | Retrieval source-balance sanity check |
| `python scripts/test_top_k_one.py` | Passport `top_k=1` smoke test (in-process + optional API) |
| `ANSWER_QUALITY_REPORT.md` | Passport retrieval hardening + citation consistency |
| `FINAL_DELIVERY_SUMMARY.md` | Delivery handoff summary |

## Troubleshooting

### Invalid Gemini API key (`API_KEY_INVALID`)

**Symptoms:** Ingestion or chat fails with `400 INVALID_ARGUMENT` / `API_KEY_INVALID`.

**Fix:**
1. Create a new key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Set `GEMINI_API_KEY=...` in `.env` and **save the file to disk**.
3. Restart uvicorn and re-run ingestion.

### Gemini 429 rate limits (`RESOURCE_EXHAUSTED`)

**Symptoms:** Ingestion logs show `429` on `gemini-embedding-2` or `gemini-embedding-001`; very slow progress.

**Fix:**
1. Set `GEMINI_EMBEDDING_MODEL=gemini-embedding-001` in `.env` to skip the primary model.
2. Increase `EMBEDDING_REQUEST_DELAY=2.0` (default 1.5) to throttle requests.
3. Wait and retry — the embedding service auto-falls back and retries with backoff.
4. Re-run `python scripts\ingest.py` after rate limits clear.

### Missing vector index

**Symptoms:** `/health` shows `vector_store_loaded: false` or chat returns fallback immediately.

**Fix:**
1. Run `python scripts\ingest.py` to build `vector_store/index.faiss` and `metadata.json`.
2. Confirm files exist in `vector_store/`.
3. Restart uvicorn so the app reloads the index.

### PowerShell `curl` JSON issues

**Symptoms:** `curl` with `-d "{\"question\": \"...helper's passport...\"}"` fails with parser errors on Windows PowerShell (apostrophes break quoting).

**Fix:** Use Python instead:

```powershell
python -c "import urllib.request,json; b=json.dumps({\"question\": \"Can an employer keep a helper's passport?\", \"top_k\": 10}).encode(); r=urllib.request.Request('http://127.0.0.1:8000/chat',data=b,headers={'Content-Type':'application/json'},method='POST'); print(urllib.request.urlopen(r).read().decode())"
```

Or use `Invoke-RestMethod` with a here-string, or call the API from [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Stale uvicorn server (old code still running)

**Symptoms:** Code changes (retrieval, citations, `top_k` guardrail) do not appear in `/chat` responses; duplicate processes on port 8000.

**Fix:**
1. Stop all uvicorn instances on port 8000 (close terminals or `Get-Process` / Task Manager).
2. Start a single fresh server: `uvicorn app.main:app --reload`.
3. Re-test with `python scripts\test_top_k_one.py` or the Python one-liner above.

`--reload` only reloads the process you started; a second stale server will still serve outdated behavior.

## Known limitations

- Answers depend on the quality and coverage of documents in `data/`.
- PDF extraction may miss tables, scanned images, or complex layouts.
- Multilingual support follows prompt rules; Tagalog coverage is limited compared to English.
- Hybrid retrieval uses multiple embeddings per question — `/chat` is slower than single-query RAG.
- This is **informational support, not legal advice**. Users should confirm details with the Labour Department or Immigration Department.
- First ingestion can take 20–30+ minutes for large corpora due to embedding rate limits.

## Future improvements

- Incremental ingestion and index updates
- OCR for scanned PDFs
- Conversation memory and follow-up questions
- Admin UI for document management
- Caching for frequent queries
- Evaluation harness with labelled Q&A pairs

## License

Internal project for migrant support tooling. Ensure compliance with document copyright and API usage terms.
