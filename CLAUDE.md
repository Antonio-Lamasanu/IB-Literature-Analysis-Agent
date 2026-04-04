# CLAUDE.md

See CONTEXT.md for full project overview, stack, API table, key functions, schemas, and env vars.

---

## main.py section map (2068 lines)

| Lines | Content |
|-------|---------|
| 1–75 | Imports |
| 77–225 | Env helpers (`parse_int_env`, `parse_bool_env`, `parse_float_env`, `parse_csv_env`), app logging config, `_wipe_chat_history_dir`, `parse_cors_allow_origins` |
| 226–260 | FastAPI app init, CORS middleware, storage dirs |
| 261–263 | `GET /api/health` |
| 265–368 | Pydantic models: `DocumentMetadataResponse`, `GenerateChunksResponse`, `ChatMessage`, `ChatRequest`, `ChatDebug*`, `ChatResponse` |
| 369–560 | Helpers: `_dedup_documents`, `_build_base_knowledge_prompt` (L381), URL builders, `to_document_metadata_response`, `get_latest_user_question`, chunk/history ref builders, `save_upload_file`, `remove_file`, `compute_file_fingerprint`, `validate_pdf_upload`, `sanitize_stem` |
| 565–690 | Chunk pipeline helpers: `build_chunk_params`, `build_chunk_outputs`, `generate_and_persist_chunks_for_record` |
| 691–883 | PDF processing: `process_pdf` (L691), `process_pdf_endpoint POST /api/process-pdf` (L724) |
| 884–942 | Document endpoints: `get_document_or_404`, `GET /api/documents`, `GET /api/documents/{id}`, `GET /api/documents/{id}/download`, `GET /api/documents/{id}/chunks/download` |
| 943–1224 | `POST /api/chat` — buffered chat endpoint (retrieval, prompt build, LLM call, history persist) |
| 1225–1558 | `POST /api/chat/stream` — SSE streaming chat; `_sse_event` helper at L1225 |
| 1559–1676 | `POST /api/documents/{id}/generate-chunks`, `POST /api/process-pdf-chunks` |
| 1677–1749 | Exam Pydantic models: `DocumentSummary`, `Paper1ExamResponse`, `Paper2QuestionsResponse`, `SubmitAnswerRequest`, `ExamCriterionResult`, `ExamDebugInfo`, `SubmitAnswerResponse`, `FeedbackStreamRequest` |
| 1751–1944 | Exam endpoints: `_require_chunks` (L1751), `GET /api/status` (L1768), `GET /api/documents` (L1778 list), `GET /api/exam/paper1`, `GET /api/exam/paper2/questions`, `POST /api/exam/submit-answer` |
| 1943–2068 | `POST /api/exam/criterion-feedback/stream` — SSE per-criterion feedback |
