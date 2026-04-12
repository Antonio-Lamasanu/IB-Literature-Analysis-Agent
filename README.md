# IB AI Tutor

Local-first literary analysis and IB English A exam preparation tool. Upload a PDF, chat with it using a local GGUF model (Learn mode), or practice timed essays graded automatically by the same model (Exam mode). No cloud APIs, no data leaving your machine.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite 7 |
| Backend | FastAPI, uvicorn |
| LLM inference | llama-cpp-python (local GGUF) |
| PDF extraction | PyMuPDF, pdfplumber, pypdf |
| OCR fallback | pytesseract + Pillow |
| NLP / chunking | spaCy 3.7 |
| Embeddings | sentence-transformers |
| Retrieval | BM25 + cosine similarity |
| Storage | SQLite (docs/sessions) + flat JSONL/NumPy (chunks/history) |

---

## Project Structure

```
demo/
├── backend/
│   ├── main.py               # FastAPI app, all REST endpoints (2531 LOC)
│   ├── llm_service.py        # llama-cpp-python wrapper
│   ├── chunking.py           # Semantic PDF chunking (spaCy-based)
│   ├── retrieval.py          # BM25 + embedding retrieval
│   ├── embeddings.py         # sentence-transformers wrapper
│   ├── pdf_extract.py        # Native extraction + OCR fallback
│   ├── exam_service.py       # Essay grading logic + rubrics
│   ├── exam_questions.py     # Hardcoded IB exam passages/questions
│   ├── chat_history.py       # Conversation persistence
│   ├── exam_history.py       # Exam attempt tracking
│   ├── document_registry.py  # Document metadata registry
│   ├── sessions.py           # Named session CRUD
│   ├── database.py           # SQLite setup + connection factory
│   ├── quality.py            # Chunk quality / duplicate detection
│   ├── system_info.py        # Hardware detection + model-tier recommendation
│   ├── prompt_router.py      # Auto base_knowledge vs RAG selection
│   ├── corpus_lookup.py      # Known-works API lookup + caching
│   ├── benchmark.py          # Multi-model benchmarking harness
│   ├── benchmark_config.json # Benchmark run configuration
│   ├── requirements.txt
│   ├── .env.example
│   ├── outputs/              # Runtime-generated (gitignored)
│   │   ├── app.db            # SQLite database
│   │   ├── documents/        # Preprocessed document files
│   │   │   ├── *.txt         # Extracted full text
│   │   │   ├── *.chunks.jsonl
│   │   │   ├── *.chunks.meta.json
│   │   │   └── *.embeddings.npy
│   │   ├── chat_history/     # Per-document chat logs
│   │   ├── exam_history/     # Exam attempt records
│   │   └── benchmarks/       # Benchmark reports
│   └── uploads/              # Temp PDF uploads (gitignored)
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Main UI (Learn + Exam modes)
│   │   └── main.jsx          # React entry point
│   ├── index.html
│   └── package.json
└── useful/                   # Sample IB exam PDFs + study materials
```

---

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set LLM_MODEL_PATH to your GGUF file
uvicorn main:app --reload
```

Backend runs on `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173` (proxies API calls to the backend).

---

## Environment Variables

Key variables in `backend/.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_ENABLED` | `true` | Enable/disable LLM inference |
| `LLM_MODEL_PATH` | — | Absolute path to your `.gguf` model file |
| `LLM_N_THREADS` | `8` | CPU threads for inference |
| `LLM_MAX_TOKENS` | `640` | Max tokens per response |
| `LLM_TEMPERATURE` | `0.4` | Sampling temperature |
| `LLM_N_CTX` | `4096` | Context window size |
| `CHUNK_TARGET_TOKENS` | `500` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `100` | Overlap between chunks |
| `TESSERACT_CMD` | `tesseract` | Path to Tesseract binary |
| `NATIVE_CHAR_THRESHOLD` | `200` | Min chars to skip OCR |
| `MAX_UPLOAD_MB` | `500` | Max PDF upload size |
| `DB_PATH` | `outputs/app.db` | SQLite database file path |

See `.env.example` for the full list.

---

## Modes

### Learn Mode

1. Upload a PDF — text is extracted (native or OCR) and chunked
2. Chunks are embedded with sentence-transformers
3. Ask questions — relevant chunks are retrieved (BM25 + semantic similarity) and passed to the local LLM
4. Sessions are named and persist across page reloads

Three prompt formats are available:
- `base_knowledge` — LLM answers from model knowledge only (no context)
- `rag` — Retrieved chunks formatted with metadata
- `rag_raw` — Retrieved chunks as plain text

### Exam Mode

Practice IB English A Literature essays with automated grading.

| Paper | Total | Criteria |
|---|---|---|
| Paper 1 | 20 pts | A: Understanding, B: Analysis, C: Focus, D: Language (5 pts each) |
| Paper 2 | 40 pts | A: Knowledge, B: Analysis, C: Organization, D: Language (10 pts each) |

The LLM grades your essay per criterion and provides written feedback for each.

---

## Benchmarking

Test multiple GGUF models head-to-head on the Learn mode pipeline:

```bash
cd backend
python benchmark.py
```

Edit `benchmark_config.json` to configure:
- `models_folder` — folder containing your `.gguf` files
- `models` — list of model filenames to test
- `prompt_formats` — which prompt modes to evaluate
- `n_ctx_values`, `n_threads`, `max_tokens`, `temperature`
- `questions` — the 5-question conversation sequence (Q1–Q3 cold start, Q4–Q5 followup)

Results are written to `backend/outputs/benchmarks/`.

---

## Data Storage

| Storage | Contents |
|---|---|
| `outputs/app.db` | SQLite — documents table + sessions/turns tables |
| `outputs/documents/{id}.txt` | Extracted full text |
| `outputs/documents/{stem}_{id}.chunks.jsonl` | Text chunks (one JSON object per line) |
| `outputs/documents/{stem}_{id}.chunks.meta.json` | Chunk metadata (schema version, counts) |
| `outputs/documents/{stem}_{id}.chunks.embeddings.npy` | NumPy embedding matrix |
| `outputs/chat_history/{doc_id}_*.json` | Chat session history |
| `outputs/exam_history/*.json` | Exam attempt records |
