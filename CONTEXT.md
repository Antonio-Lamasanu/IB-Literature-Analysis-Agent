> MAINTENANCE: At the end of any session where a new convention, architectural decision, or major feature was completed, suggest updates to this file and ask for approval before writing them.

# PROJECT CONTEXT

## Identity
IB AI Tutor — local-first literary analysis & IB English A exam prep tool. No cloud APIs. All LLM inference runs locally via llama-cpp-python (GGUF models).

## Stack
| Layer | Tech |
|-------|------|
| Frontend | React 19.1.1, Vite 7.1.2 → `http://localhost:5173` |
| Backend | FastAPI + uvicorn → `http://localhost:8000` |
| LLM | llama-cpp-python (GGUF, local) |
| PDF | PyMuPDF, pdfplumber, pypdf + Tesseract OCR fallback |
| Chunking | spaCy 3.7 (NER, POS) |
| Embeddings | sentence-transformers `multi-qa-MiniLM-L6-cos-v1` → 384-dim L2-normalized |
| Retrieval | BM25 (K1=1.5, B=0.75) + cosine similarity hybrid |
| Storage | Flat JSON/JSONL + NumPy .npy — NO database |

## Directory Structure
```
demo/
├── backend/
│   ├── main.py                 # FastAPI app + all endpoints (2068 LOC)
│   ├── llm_service.py          # LLM wrapper + prompt building (1043 LOC)
│   ├── retrieval.py            # BM25 + semantic search (1578 LOC)
│   ├── chunking.py             # Semantic chunking w/ spaCy (906 LOC)
│   ├── exam_service.py         # Essay grading + rubrics (472 LOC)
│   ├── pdf_extract.py          # Text extraction + OCR (569 LOC)
│   ├── document_registry.py    # Doc metadata persistence (218 LOC)
│   ├── chat_history.py         # Conversation persistence (271 LOC)
│   ├── exam_history.py         # Exam attempt tracking (233 LOC)
│   ├── embeddings.py           # SentenceTransformer wrapper (66 LOC)
│   ├── exam_questions.py       # Question DB loader (59 LOC)
│   ├── benchmark.py            # Multi-model benchmarking (1156 LOC)
│   ├── exam_benchmark.py       # Exam mode benchmarking (649 LOC)
│   ├── evaluate_benchmarks.py  # Results analysis (946 LOC)
│   ├── .env / .env.example     # Runtime config
│   ├── outputs/                # GITIGNORED — all runtime data
│   │   ├── documents/          # {id}.txt, {id}.chunks.jsonl, {id}.chunks.embeddings.npy
│   │   ├── chat_history/       # {doc_id}_{turn_id}.json
│   │   ├── exam_history/       # {attempt_id}.json
│   │   ├── documents.index.json # DocumentRegistry persistence
│   │   └── benchmarks/         # benchmark_{timestamp}.json
│   └── uploads/                # Temp PDF uploads
├── frontend/
│   └── src/
│       ├── App.jsx             # Main UI — Learn + Exam modes (~1200 LOC)
│       └── main.jsx            # React entry point (9 LOC)
└── useful/
    ├── paper1_sub.json         # IB Paper 1 passages + guiding questions
    ├── paper2_sub.json         # IB Paper 2 questions
    ├── paper1_solutions.json   # Example solutions
    ├── paper2_solutions.json
    ├── learn_questions.json
    └── PDFs/                   # Official IB exam PDFs
```

## API Endpoints (backend/main.py)
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Health check |
| GET | `/api/status` | LLM availability + config |
| POST | `/api/process-pdf` | Upload PDF → extract + chunk + embed |
| GET | `/api/documents` | List documents |
| GET | `/api/documents/{id}` | Doc metadata |
| GET | `/api/documents/{id}/download` | Extracted text |
| GET | `/api/documents/{id}/chunks/download` | Chunks JSONL |
| POST | `/api/documents/{id}/generate-chunks` | Regen chunks |
| POST | `/api/chat` | Single-turn chat (buffered) |
| POST | `/api/chat/stream` | Streaming chat (SSE) |
| GET | `/api/exam/paper1` | Get random Paper 1 passage + question |
| GET | `/api/exam/paper2/questions` | All Paper 2 questions |
| POST | `/api/exam/submit-answer` | Grade essay → scores |
| POST | `/api/exam/criterion-feedback/stream` | Stream per-criterion feedback (SSE) |

## Key Functions

### pdf_extract.py
- `extract_text_with_threshold(pdf_path, char_threshold=200)` → ExtractionResult — tries native, falls back to OCR
- `extract_text_with_ocr(pdf_path, lang='eng')` → ExtractionResult

### chunking.py
- `build_chunks(units: list[Unit], params: ChunkParams)` → list[Chunk]
- `parse_units_from_marked_text()` → list[Unit]
- `CHUNK_SCHEMA_VERSION = "simple-novel-v1"`
- `ChunkParams(target_tokens, overlap_tokens, min_tokens, max_tokens, include_raw)`

### retrieval.py
- `retrieve_relevant_excerpts(doc_id, query, max_excerpts, retrieve_candidates)` → list[RetrievedExcerpt]
- `build_chat_context_result_with_history(...)` → ContextResult
- Modes: `chunks_only` | `history_first` | `combined`
- `RetrievedExcerpt`: excerpt_id, heading, page_start, page_end, text, score, token_estimate

### embeddings.py
- `encode_texts(texts: list[str])` → np.ndarray(N, 384) — L2-normalized
- `embeddings_path_for_chunks(chunks_path)` → Path (foo.chunks.jsonl → foo.chunks.embeddings.npy)

### llm_service.py
- `get_llm_service()` → LLMService singleton
- `LLMService.complete(prompt, temperature, max_tokens)` → LLMInferenceResult
- Token estimate: words × 1.33
- Prompt modes: `base_knowledge` | `rag` | `rag_raw`

### exam_service.py
- `grade_answer(paper_type, question, student_answer, context)` → GradingResult
- Criteria A/B/C/D — Paper 1: 5pts each (20 total); Paper 2: 10pts each (40 total)
- `stream_criterion_feedback(...)` → Generator[str] (SSE)
- Grading context query: `"language tone imagery literary techniques psychological state"`

### document_registry.py
- `DocumentRegistry` — thread-safe in-memory index, persists to `outputs/documents.index.json`

### chat_history.py
- `ChatHistoryTurn`: turn_id, document_id, created_at, user_query, assistant_answer, retrieval_mode_used, retrieved_chunk_refs, answer_reused_from_history
- `load_chat_history(doc_id)` → list[ChatHistoryTurn]

### exam_history.py
- `create_exam_attempt()` → ExamAttempt; persists to `outputs/exam_history/{attempt_id}.json`
- ExamAttempt fields: attempt_id, document_id, paper_type, question, student_answer, score_a/b/c/d, feedback_a/b/c/d, total_score, max_score, overall_comments

### exam_questions.py
- `load_paper1_passages()`, `get_random_paper1_passage()`, `load_paper2_questions()`

## Data Flow

### Learn Mode
PDF upload → extract text → chunk → embed chunks → [user query] → BM25+semantic retrieve → LLM → stream response → persist turn

### Exam Mode
Select paper → fetch question → write essay → submit → retrieve context from selected docs → grade each criterion via LLM → stream feedback → persist attempt

## Key Data Schemas

### Chunk (simple-novel-v1)
```json
{
  "content": {"text": "...", "raw": "..."},
  "position": {"chapter": 5, "paragraph": 12, "section": "Chapter 5: ..."},
  "metadata": {"character_mentions": ["Alice"], "is_dialogue": false, "is_description": true}
}
```

### ExamAttempt
```json
{"attempt_id": "uuid", "paper_type": "paper1|paper2", "score_a": 4, "score_b": 3, "score_c": 5, "score_d": 4, "total_score": 16, "max_score": 20}
```

### ChatHistoryTurn
```json
{"turn_id": "uuid", "retrieval_mode_used": "chunks_only", "retrieved_chunk_refs": [{"excerpt_id": "...", "heading": "...", "page_start": 5}]}
```

## Env Vars (backend/.env)
```
# LLM
LLM_ENABLED, LLM_MODEL_PATH, LLM_N_CTX, LLM_N_THREADS, LLM_MAX_TOKENS, LLM_TEMPERATURE

# Chunking
CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_MIN_TOKENS, CHUNK_MAX_TOKENS

# PDF
NATIVE_CHAR_THRESHOLD, TESSERACT_CMD, TESSERACT_LANG

# Chat
CHAT_HISTORY_ENABLED, CHAT_RETRIEVAL_MODE, CHAT_MAX_HISTORY_MESSAGES

# Exam
EXAM_CONTEXT_TOKEN_BUDGET

# CORS
CORS_ALLOW_ORIGINS
```

## Design Constraints
- No database — all storage is flat JSON/JSONL + NumPy arrays
- Local-only LLM inference — no cloud API calls
- Thread-safe registry/history via locks
- SSE streaming for chat and exam feedback
- spaCy, SentenceTransformer, LLM all lazy-loaded
- `outputs/` is gitignored — runtime-generated, portable via zip
- CORS restricted to localhost:5173 by default

## Run Commands
```bash
# Backend
cd backend && uvicorn main:app --reload

# Frontend
cd frontend && npm run dev
```
