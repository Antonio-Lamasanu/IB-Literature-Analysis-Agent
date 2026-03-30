import asyncio
import json as _json
import os
import re
import time
import uuid
import hashlib
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from chat_history import (
    append_chat_history_turn,
    create_chat_history_turn,
    embed_and_save_history,
    load_chat_history,
)
import exam_service
from exam_service import (
    CRITERIA_BY_PAPER,
    GradingResult,
    estimate_feedback_prompt_tokens,
    grade_answer,
    retrieve_multi_doc_context,
    stream_criterion_feedback,
)
from exam_questions import (
    PAPER1_PASSAGE,
    PAPER1_QUESTION,
    get_random_paper1_passage,
    load_paper2_questions,
)
from exam_history import append_exam_attempt, create_exam_attempt
from chunking import (
    CHUNK_SCHEMA_VERSION,
    ChunkParams,
    build_chunks,
    build_doc_id,
    filter_chunks_for_embedding,
    parse_units_from_marked_text,
    write_chunks_jsonl,
    write_meta_json,
)
from pdf_extract import (
    ExtractionResult,
    configure_tesseract,
    extract_text_with_ocr,
    extract_text_with_threshold,
)
from document_registry import DocumentRegistry, DocumentRecord
from llm_service import (
    LLMDisabledError,
    LLMInferenceResult,
    LLMNotConfiguredError,
    LLMServiceError,
    get_llm_service,
)
from retrieval import (
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    DEFAULT_MAX_EXCERPTS,
    DEFAULT_RETRIEVE_CANDIDATES,
    build_chat_context_result_with_history,
)

load_dotenv()
logger = logging.getLogger(__name__)


def parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on", "y"}:
        return True
    if value in {"0", "false", "no", "off", "n"}:
        return False
    return default


def parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


def parse_csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return default.copy()

    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return default.copy()
    return values


def configure_app_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    if parse_bool_env("PDF_EXTRACT_DEBUG", False):
        logging.getLogger("pdf_extract").setLevel(logging.DEBUG)
        logger.info("Enabled DEBUG logging for pdf_extract via PDF_EXTRACT_DEBUG=1")


BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
DOCUMENTS_DIR = OUTPUTS_DIR / "documents"
DEFAULT_CHAT_HISTORY_STORAGE_DIR = OUTPUTS_DIR / "chat_history"
DEFAULT_EXAM_HISTORY_STORAGE_DIR = OUTPUTS_DIR / "exam_history"
DOCUMENTS_INDEX_PATH = OUTPUTS_DIR / "documents.index.json"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
document_registry = DocumentRegistry(DOCUMENTS_INDEX_PATH)
llm_service = get_llm_service()

ALLOWED_CONTENT_TYPES = {"application/pdf"}


NATIVE_CHAR_THRESHOLD = parse_int_env("NATIVE_CHAR_THRESHOLD", 200)
MAX_UPLOAD_MB = parse_int_env("MAX_UPLOAD_MB", 500)
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng")
CHUNK_TARGET_TOKENS = parse_int_env("CHUNK_TARGET_TOKENS", 500)
CHUNK_OVERLAP_TOKENS = parse_int_env("CHUNK_OVERLAP_TOKENS", 100)
CHUNK_MIN_TOKENS = parse_int_env("CHUNK_MIN_TOKENS", 200)
CHUNK_MAX_TOKENS = parse_int_env("CHUNK_MAX_TOKENS", 800)
CHUNK_INCLUDE_RAW = parse_bool_env("CHUNK_INCLUDE_RAW", False)
CHUNK_FILTER_FOR_EMBEDDING = parse_bool_env("CHUNK_FILTER_FOR_EMBEDDING", False)
CHUNK_EMBED_MIN_TOKENS = parse_int_env("CHUNK_EMBED_MIN_TOKENS", 120)
CHUNK_EMBED_EXCLUDE_SECTION_TYPES = [
    section_type.lower()
    for section_type in parse_csv_env("CHUNK_EMBED_EXCLUDE_SECTION_TYPES", ["toc", "front_matter"])
]
CHUNK_DROP_NOISE_PARAS = parse_bool_env("CHUNK_DROP_NOISE_PARAS", True)
KEEP_OUTPUT_FILES = parse_bool_env("KEEP_OUTPUT_FILES", False)
CHAT_MAX_HISTORY_MESSAGES = parse_int_env("CHAT_MAX_HISTORY_MESSAGES", 10)
CHAT_RETRIEVE_CANDIDATES = parse_int_env("CHAT_RETRIEVE_CANDIDATES", DEFAULT_RETRIEVE_CANDIDATES)
CHAT_CONTEXT_MAX_EXCERPTS = parse_int_env("CHAT_CONTEXT_MAX_EXCERPTS", DEFAULT_MAX_EXCERPTS)
CHAT_CONTEXT_TOKEN_BUDGET = parse_int_env("CHAT_CONTEXT_TOKEN_BUDGET", DEFAULT_CONTEXT_TOKEN_BUDGET)
EXAM_CONTEXT_TOKEN_BUDGET = parse_int_env("EXAM_CONTEXT_TOKEN_BUDGET", 450)
CHAT_HISTORY_ENABLED = parse_bool_env("CHAT_HISTORY_ENABLED", True)
CHAT_RETRIEVAL_MODE = (os.getenv("CHAT_RETRIEVAL_MODE", "chunks_only").strip().lower() or "chunks_only")
if CHAT_RETRIEVAL_MODE not in {"combined", "history_first", "chunks_only"}:
    CHAT_RETRIEVAL_MODE = "chunks_only"
CHAT_HISTORY_REUSE_THRESHOLD = parse_float_env("CHAT_HISTORY_REUSE_THRESHOLD", 2.75)
CHAT_HISTORY_MAX_CANDIDATES = parse_int_env("CHAT_HISTORY_MAX_CANDIDATES", DEFAULT_RETRIEVE_CANDIDATES)
CHAT_HISTORY_MAX_EXCERPTS = parse_int_env("CHAT_HISTORY_MAX_EXCERPTS", 2)
_chat_history_storage_dir_env = (os.getenv("CHAT_HISTORY_STORAGE_DIR") or "").strip()
CHAT_HISTORY_STORAGE_DIR = Path(
    _chat_history_storage_dir_env or str(DEFAULT_CHAT_HISTORY_STORAGE_DIR)
).expanduser()
_exam_history_storage_dir_env = (os.getenv("EXAM_HISTORY_STORAGE_DIR") or "").strip()
EXAM_HISTORY_STORAGE_DIR = Path(
    _exam_history_storage_dir_env or str(DEFAULT_EXAM_HISTORY_STORAGE_DIR)
).expanduser()
REUSE_PROCESSED_PDFS = parse_bool_env("REUSE_PROCESSED_PDFS", False)

exam_service.set_grading_token_budget(EXAM_CONTEXT_TOKEN_BUDGET)

WIPE_CHAT_HISTORY_ON_START = parse_bool_env("WIPE_CHAT_HISTORY_ON_START", False)

configure_app_logging()


def _wipe_chat_history_dir(storage_dir: Path) -> None:
    """Delete all chat history JSON and history embedding files on startup."""
    if not storage_dir.exists():
        return
    removed = 0
    for pattern in ("*.json", "*.history.embeddings.npy"):
        for path in storage_dir.glob(pattern):
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                logging.getLogger(__name__).warning("Could not delete %s: %s", path, exc)
    logging.getLogger(__name__).info("WIPE_CHAT_HISTORY_ON_START: removed %d file(s) from %s", removed, storage_dir)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]


def parse_cors_allow_origins(origins_env: str | None) -> list[str]:
    if not origins_env:
        return DEFAULT_CORS_ORIGINS.copy()

    parsed = [origin.strip() for origin in origins_env.split(",") if origin.strip()]
    if not parsed:
        return DEFAULT_CORS_ORIGINS.copy()

    # Deduplicate while preserving declaration order.
    return list(dict.fromkeys(parsed))


CORS_ALLOW_ORIGINS = parse_cors_allow_origins(os.getenv("CORS_ALLOW_ORIGINS"))
# This app does not use cookie/session auth, so credentials can stay disabled.
CORS_ALLOW_CREDENTIALS = False

app = FastAPI(title="PDF Text Extractor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
        "X-Processing-Mode",
        "X-Pages",
        "X-Text-Chars",
        "X-Document-Id",
        "X-Document-Filename",
        "X-Chat-Available",
        "X-Chunks-Available",
        "X-Chunks-Count",
        "X-Chunks-Download-Url",
        "X-Chunk-Schema-Version",
        "X-Chunk-Target-Tokens",
        "X-Chunk-Overlap-Tokens",
    ],
)

configure_tesseract(os.getenv("TESSERACT_CMD"))

if WIPE_CHAT_HISTORY_ON_START:
    _wipe_chat_history_dir(CHAT_HISTORY_STORAGE_DIR)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


class DocumentMetadataResponse(BaseModel):
    document_id: str
    filename: str
    processing_mode: str
    pages: int
    text_chars: int
    chat_available: bool
    download_url: str
    chunks_available: bool = False
    chunks_count: int | None = None
    chunks_download_url: str | None = None
    chunk_meta_path: str | None = None
    chunk_schema_version: str | None = None


class GenerateChunksResponse(BaseModel):
    document_id: str
    filename: str
    chunks_available: bool
    chunks_count: int
    chunk_schema_version: str
    chunks_path: str | None = None
    chunk_meta_path: str | None = None
    chunks_download_url: str | None = None
    chunk_target_tokens: int
    chunk_overlap_tokens: int


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8_000)


class ChatRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=128)
    messages: list[ChatMessage] = Field(..., min_length=1)
    prompt_format: Literal["rag", "rag_raw", "base_knowledge"] = "rag"


class ChatDebugTimingResponse(BaseModel):
    retrieval_seconds: float
    prompt_build_seconds: float
    inference_seconds: float
    total_seconds: float


class ChatDebugExcerptResponse(BaseModel):
    excerpt_id: str
    score: float
    heading: str
    page_start: int
    page_end: int
    text: str


class ChatDebugHistoryCandidateResponse(BaseModel):
    turn_id: str
    created_at: str
    score: float
    user_query: str
    assistant_answer: str
    answer_reused_from_history: bool = False
    reused_from_turn_id: str | None = None


class ChatDebugCandidateResponse(BaseModel):
    source: str
    source_id: str
    score: float
    text: str
    heading: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    turn_id: str | None = None
    created_at: str | None = None
    user_query: str | None = None
    assistant_answer: str | None = None


class ChatDebugResponse(BaseModel):
    final_prompt: str
    total_seconds: float
    retrieval_mode: str
    retrieved_excerpts: list[ChatDebugExcerptResponse]
    timing: ChatDebugTimingResponse
    chunk_retrieval_mode: str | None = None
    history_checked: bool = False
    history_reuse_hit: bool = False
    history_top_score: float | None = None
    history_threshold: float | None = None
    reused_turn_id: str | None = None
    retrieved_history_candidates: list[ChatDebugHistoryCandidateResponse] = Field(default_factory=list)
    retrieved_chunk_candidates: list[ChatDebugExcerptResponse] = Field(default_factory=list)
    raw_retrieved_excerpts: list[ChatDebugExcerptResponse] = Field(default_factory=list)
    final_candidate_mix: list[ChatDebugCandidateResponse] = Field(default_factory=list)
    history_persisted: bool = True
    history_persist_error: str | None = None


class ChatResponse(BaseModel):
    reply: str
    debug: ChatDebugResponse | None = None


def _dedup_documents(records: list) -> list:
    """Return the most-recent copy of each distinct work (by source_fingerprint or filename)."""
    seen: set[str] = set()
    out = []
    for r in records:  # already sorted newest-first
        key = r.source_fingerprint or r.filename
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _build_base_knowledge_prompt(
    title: str,
    author: str,
    session_messages: list[dict[str, str]],
    question: str,
) -> str:
    """Build a prompt that relies on model training knowledge (no document context)."""
    history_lines = []
    for msg in session_messages:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{speaker}: {msg['content']}")
    conversation_history = "\n".join(history_lines).strip() or "(none)"
    return (
        f'SYSTEM INSTRUCTIONS:\n'
        f'- You are answering questions about "{title}" by {author}.\n'
        f'- Answer from your general knowledge of this literary work.\n'
        f'- Be analytical and specific.\n\n'
        f'RECENT CONVERSATION:\n{conversation_history}\n\n'
        f'LATEST USER QUESTION:\nUser: {question}\n'
        f'Assistant:'
    )


def build_download_url(document_id: str) -> str:
    return f"/api/documents/{document_id}/download"


def build_chunks_download_url(document_id: str) -> str:
    return f"/api/documents/{document_id}/chunks/download"


def build_document_headers(record: DocumentRecord) -> dict[str, str]:
    headers = {
        "X-Processing-Mode": record.processing_mode,
        "X-Pages": str(record.pages),
        "X-Text-Chars": str(record.text_chars),
        "X-Document-Id": record.document_id,
        "X-Document-Filename": record.filename,
        "X-Chat-Available": "1" if llm_service.is_chat_available() else "0",
        "X-Chunks-Available": "1" if record.chunks_available else "0",
        "X-Chunks-Count": str(record.chunks_count or 0),
        "X-Chunk-Schema-Version": record.chunk_schema_version or "",
    }
    if record.chunks_available and record.chunks_path:
        headers["X-Chunks-Download-Url"] = build_chunks_download_url(record.document_id)
    return headers


def to_document_metadata_response(record: DocumentRecord) -> DocumentMetadataResponse:
    return DocumentMetadataResponse(
        document_id=record.document_id,
        filename=record.filename,
        processing_mode=record.processing_mode,
        pages=record.pages,
        text_chars=record.text_chars,
        chat_available=llm_service.is_chat_available(),
        download_url=build_download_url(record.document_id),
        chunks_available=record.chunks_available,
        chunks_count=record.chunks_count,
        chunks_download_url=(
            build_chunks_download_url(record.document_id)
            if record.chunks_available and record.chunks_path
            else None
        ),
        chunk_meta_path=record.chunk_meta_path,
        chunk_schema_version=record.chunk_schema_version,
    )


def get_latest_user_question(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = str(message.get("content", "")).strip()
            if content:
                return content
    return ""


def build_chunk_history_refs(excerpts: list[Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for excerpt in excerpts:
        refs.append(
            {
                "excerpt_id": excerpt.excerpt_id,
                "heading": excerpt.heading,
                "page_start": excerpt.page_start,
                "page_end": excerpt.page_end,
                "score": excerpt.score,
            }
        )
    return refs


def build_history_candidate_refs(candidates: list[Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for candidate in candidates:
        refs.append(
            {
                "turn_id": candidate.turn_id,
                "created_at": candidate.created_at,
                "score": candidate.score,
                "user_query": candidate.user_query,
                "assistant_answer": candidate.assistant_answer,
            }
        )
    return refs


async def save_upload_file(upload_file: UploadFile, destination: Path, max_upload_mb: int) -> None:
    max_size_bytes = max_upload_mb * 1024 * 1024
    total_bytes = 0

    try:
        with destination.open("wb") as out_file:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_size_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File too large. Maximum allowed size is {max_upload_mb}MB.",
                    )
                out_file.write(chunk)
    except HTTPException:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {exc}",
        ) from exc
    finally:
        await upload_file.close()


def remove_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        # Cleanup failure should not break the response path.
        pass


def compute_file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_pdf_upload(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded.")

    ext = Path(file.filename).suffix.lower()
    content_type = (file.content_type or "").split(";")[0].strip().lower()

    if ext != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file extension. Please upload a .pdf file.",
        )

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content type. Expected application/pdf.",
        )


def sanitize_stem(filename_stem: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", filename_stem).strip("_")
    return normalized or "document"


def build_chunk_params() -> ChunkParams:
    min_tokens = max(1, CHUNK_MIN_TOKENS)
    max_tokens = max(min_tokens, CHUNK_MAX_TOKENS)
    target_tokens = min(max(CHUNK_TARGET_TOKENS, min_tokens), max_tokens)
    overlap_tokens = max(0, min(CHUNK_OVERLAP_TOKENS, max_tokens - 1))
    return ChunkParams(
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        include_raw=CHUNK_INCLUDE_RAW,
    )


def build_chunk_outputs(
    *,
    document_id: str,
    final_text: str,
    original_filename: str,
    extraction: ExtractionResult,
) -> tuple[Path, Path, Path | None, int, ChunkParams, str]:
    filename_stem = Path(original_filename).stem
    safe_stem = sanitize_stem(filename_stem)
    doc_id = build_doc_id(filename_stem, final_text)
    params = build_chunk_params()

    units = parse_units_from_marked_text(
        final_text,
        doc_id=doc_id,
        drop_noise_paras=CHUNK_DROP_NOISE_PARAS,
    )
    chunks = build_chunks(units, params)
    chapter_count = len(
        {
            chunk.position.get("chapter")
            for chunk in chunks
            if chunk.position.get("chapter") is not None
        }
    )

    chunks_path = DOCUMENTS_DIR / f"{safe_stem}_{document_id}.chunks.jsonl"
    meta_path = DOCUMENTS_DIR / f"{safe_stem}_{document_id}.chunks.meta.json"
    filtered_chunks_path: Path | None = None

    write_chunks_jsonl(chunks_path, chunks, include_raw=params.include_raw)

    try:
        from embeddings import encode_texts, embeddings_path_for_chunks
        import numpy as _np
        _emb_texts = [
            str(c.content.get("text") or "").strip()
            for c in chunks
            if c.content.get("text")
        ]
        if _emb_texts:
            _emb = encode_texts(_emb_texts)
            if _emb.shape[0] == len(_emb_texts):
                _np.save(str(embeddings_path_for_chunks(chunks_path)), _emb)
                logger.info("Saved %d embeddings to %s", _emb.shape[0], chunks_path)
            else:
                logger.warning(
                    "Embedding count mismatch (%d vs %d chunks); skipping save.",
                    _emb.shape[0], len(_emb_texts),
                )
    except Exception as _exc:
        logger.warning("Embedding generation failed (non-fatal): %s", _exc)

    if CHUNK_FILTER_FOR_EMBEDDING:
        filtered_chunks_path = DOCUMENTS_DIR / f"{safe_stem}_{document_id}.chunks.filtered.jsonl"
        filtered_chunks = filter_chunks_for_embedding(
            chunks,
            min_tokens=CHUNK_EMBED_MIN_TOKENS,
            exclude_section_types=CHUNK_EMBED_EXCLUDE_SECTION_TYPES,
        )
        write_chunks_jsonl(filtered_chunks_path, filtered_chunks, include_raw=params.include_raw)

    write_meta_json(
        meta_path,
        doc_id=doc_id,
        original_filename=original_filename,
        pages=extraction.pages_count,
        processing_mode=extraction.mode,
        total_chars=len(final_text),
        total_chunks=len(chunks),
        params=params,
        chapter_count=chapter_count,
    )

    return chunks_path, meta_path, filtered_chunks_path, len(chunks), params, doc_id


def generate_and_persist_chunks_for_record(
    record: DocumentRecord,
    final_text: str,
) -> tuple[DocumentRecord, ChunkParams]:
    extraction = ExtractionResult(
        text=final_text,
        pages_count=record.pages,
        chars_count=record.text_chars,
        mode=record.processing_mode,
    )
    chunks_path, meta_path, _filtered_chunks_path, chunks_count, chunk_params, _chunk_doc_id = (
        build_chunk_outputs(
            document_id=record.document_id,
            final_text=final_text,
            original_filename=record.filename,
            extraction=extraction,
        )
    )

    updated_record = document_registry.update_chunks(
        record.document_id,
        chunks_available=chunks_count > 0,
        chunks_path=chunks_path,
        chunk_meta_path=meta_path,
        chunks_count=chunks_count,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
    )
    if updated_record is None:
        raise RuntimeError(f"Document not found for document_id={record.document_id}.")

    logger.info("Persisting generated chunk output file: %s", chunks_path)
    logger.info("Persisting generated chunk meta file: %s", meta_path)
    return updated_record, chunk_params


def process_pdf(upload_path: Path, output_path: Path) -> tuple[ExtractionResult, str]:
    extraction = extract_text_with_threshold(
        upload_path,
        char_threshold=NATIVE_CHAR_THRESHOLD,
    )
    final_text = extraction.text.strip()

    
    if extraction.mode == "ocr":
        final_text = extract_text_with_ocr(upload_path, lang=TESSERACT_LANG).strip()
        extraction = ExtractionResult(
            text=final_text,
            pages_count=extraction.pages_count,
            chars_count=len(final_text),
            mode="ocr",
        )
        
    if not final_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Could not extract text from this PDF. "
                "If this is an image-based PDF, verify Tesseract is installed and configured."
            ),
        )

    output_path.write_text(final_text, encoding="utf-8")
    return extraction, final_text


# Temporary debug compatibility alias. Remove after all clients use /api/process-pdf.
@app.post("/upload-pdf", include_in_schema=False)
@app.post("/api/process-pdf")
async def process_pdf_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> FileResponse:
    validate_pdf_upload(file)

    request_id = str(uuid.uuid4())
    upload_path = UPLOADS_DIR / f"{request_id}.pdf"
    output_path = DOCUMENTS_DIR / f"{request_id}.txt"

    await save_upload_file(file, upload_path, MAX_UPLOAD_MB)
    try:
        source_fingerprint = compute_file_fingerprint(upload_path)
    except OSError as exc:
        remove_file(upload_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fingerprint uploaded PDF: {exc}",
        ) from exc

    if REUSE_PROCESSED_PDFS:
        existing_record = document_registry.find_reusable_by_source_fingerprint(source_fingerprint)
        if existing_record:
            existing_text_path = Path(existing_record.text_path)
            if (
                not existing_record.chunks_available
                or not existing_record.chunks_path
                or not Path(existing_record.chunks_path).exists()
            ):
                try:
                    existing_text = existing_text_path.read_text(encoding="utf-8").strip()
                    if not existing_text:
                        raise HTTPException(
                            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Reused document text is empty; cannot generate chunks.",
                        )
                    existing_record, _chunk_params = await run_in_threadpool(
                        generate_and_persist_chunks_for_record,
                        existing_record,
                        existing_text,
                    )
                except HTTPException:
                    remove_file(upload_path)
                    raise
                except OSError as exc:
                    remove_file(upload_path)
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to prepare reused document chunks: {exc}",
                    ) from exc
                except Exception as exc:
                    remove_file(upload_path)
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to generate chunks for reused document: {exc}",
                    ) from exc

            logger.info(
                "Reusing processed PDF output for fingerprint %s from document_id=%s",
                source_fingerprint,
                existing_record.document_id,
            )
            if existing_record.title is None and llm_service.is_chat_available():
                try:
                    existing_text = existing_text_path.read_text(encoding="utf-8")
                    title, author = await run_in_threadpool(
                        llm_service.extract_title_and_author, existing_text[:1500]
                    )
                    existing_record = document_registry.update_title_author(
                        existing_record.document_id, title=title, author=author
                    ) or existing_record
                    logger.info(
                        "Extracted title=%r author=%r for reused document_id=%s",
                        title,
                        author,
                        existing_record.document_id,
                    )
                except Exception:
                    logger.warning(
                        "Title/author extraction failed for reused document_id=%s",
                        existing_record.document_id,
                    )
            background_tasks.add_task(remove_file, upload_path)
            download_name = f"{Path(existing_record.filename).stem}.txt"
            return FileResponse(
                path=existing_text_path,
                media_type="text/plain; charset=utf-8",
                filename=download_name,
                background=background_tasks,
                headers=build_document_headers(existing_record),
            )

        logger.info(
            "No reusable processed output found for fingerprint %s; processing upload normally",
            source_fingerprint,
        )

    try:
        extraction, final_text = await run_in_threadpool(process_pdf, upload_path, output_path)
    except HTTPException:
        remove_file(upload_path)
        remove_file(output_path)
        raise
    except Exception as exc:
        remove_file(upload_path)
        remove_file(output_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {exc}",
        ) from exc

    document_id = request_id
    record = document_registry.register(
        document_id=document_id,
        text_path=output_path,
        filename=file.filename,
        processing_mode=extraction.mode,
        pages=extraction.pages_count,
        text_chars=len(final_text),
        source_fingerprint=source_fingerprint,
    )

    try:
        record, _chunk_params = await run_in_threadpool(
            generate_and_persist_chunks_for_record,
            record,
            final_text,
        )
    except Exception as exc:
        remove_file(upload_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate chunks: {exc}",
        ) from exc

    if llm_service.is_chat_available():
        try:
            title, author = await run_in_threadpool(
                llm_service.extract_title_and_author, final_text[:1500]
            )
            document_registry.update_title_author(document_id, title=title, author=author)
            logger.info("Extracted title=%r author=%r for document_id=%s", title, author, document_id)
        except Exception:
            logger.warning("Title/author extraction failed for document_id=%s", document_id)

    background_tasks.add_task(remove_file, upload_path)
    # Keep /api/process-pdf outputs for reuse (for example as aiModel.py context input).
    logger.info("Persisting extracted text output file: %s", output_path)

    download_name = f"{Path(file.filename).stem}.txt"

    return FileResponse(
        path=output_path,
        media_type="text/plain; charset=utf-8",
        filename=download_name,
        background=background_tasks,
        headers=build_document_headers(record),
    )


def get_document_or_404(document_id: str) -> DocumentRecord:
    record = document_registry.get(document_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found for document_id={document_id}.",
        )
    return record


@app.get("/api/documents/{document_id}", response_model=DocumentMetadataResponse)
def get_document_metadata(document_id: str) -> DocumentMetadataResponse:
    record = get_document_or_404(document_id)
    return to_document_metadata_response(record)


@app.get("/api/documents/{document_id}/download")
def download_document_text(document_id: str) -> FileResponse:
    record = get_document_or_404(document_id)
    text_path = Path(record.text_path)
    if not text_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document text file is missing.",
        )

    download_name = f"{Path(record.filename).stem}.txt"
    return FileResponse(
        path=text_path,
        media_type="text/plain; charset=utf-8",
        filename=download_name,
    )


@app.get("/api/documents/{document_id}/chunks/download")
def download_document_chunks(document_id: str) -> FileResponse:
    record = get_document_or_404(document_id)
    if not record.chunks_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document chunks are not available.",
        )

    chunks_path = Path(record.chunks_path)
    if not chunks_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document chunk file is missing.",
        )

    download_name = f"{Path(record.filename).stem}.chunks.jsonl"
    return FileResponse(
        path=chunks_path,
        media_type="application/x-ndjson; charset=utf-8",
        filename=download_name,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    request_started = time.perf_counter()
    record = get_document_or_404(payload.document_id)
    document_text = ""
    text_path = Path(record.text_path)
    persisted_chunks_path: Path | None = None
    if record.chunks_available and record.chunks_path:
        candidate_chunks_path = Path(record.chunks_path)
        if candidate_chunks_path.exists():
            persisted_chunks_path = candidate_chunks_path

    if persisted_chunks_path is None and text_path.exists():
        try:
            document_text = text_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read document text: {exc}",
            ) from exc

    messages = [{"role": message.role, "content": message.content} for message in payload.messages]
    latest_user_question = get_latest_user_question(messages)
    effective_retrieval_mode = CHAT_RETRIEVAL_MODE if CHAT_HISTORY_ENABLED else "chunks_only"

    if not document_text and persisted_chunks_path is None and effective_retrieval_mode == "chunks_only":
        if not text_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document text file is missing.",
            )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document text is empty and no persisted chunks are available; cannot chat.",
        )

    # --- base_knowledge: skip retrieval entirely ---
    if payload.prompt_format == "base_knowledge":
        retrieval_seconds = 0.0
        reply_text = ""
        final_prompt = ""
        prompt_build_seconds = 0.0
        inference_seconds = 0.0
        title = record.title or "Unknown"
        author = record.author or "Unknown"
        history_messages = messages[:-1]  # everything before the latest question
        prompt_build_t0 = time.perf_counter()
        base_prompt = _build_base_knowledge_prompt(title, author, history_messages, latest_user_question)
        prompt_build_seconds = time.perf_counter() - prompt_build_t0
        try:
            inference_result: LLMInferenceResult = await run_in_threadpool(
                llm_service.generate_raw_reply,
                base_prompt,
            )
        except LLMDisabledError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except LLMNotConfiguredError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except LLMServiceError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Chat inference failed: {exc}",
            ) from exc
        reply_text = inference_result.reply
        final_prompt = inference_result.final_prompt
        inference_seconds = inference_result.inference_seconds

        # Persist history even for base_knowledge turns
        history_persisted = True
        history_persist_error: str | None = None
        if CHAT_HISTORY_ENABLED and latest_user_question:
            turn = create_chat_history_turn(
                document_id=record.document_id,
                user_query=latest_user_question,
                assistant_answer=reply_text,
                retrieval_mode_used="base_knowledge",
            )
            try:
                await run_in_threadpool(
                    append_chat_history_turn,
                    record.document_id,
                    turn,
                    CHAT_HISTORY_STORAGE_DIR,
                )
                background_tasks.add_task(embed_and_save_history, record.document_id, CHAT_HISTORY_STORAGE_DIR)
            except OSError as exc:
                history_persisted = False
                history_persist_error = str(exc)

        total_seconds = time.perf_counter() - request_started
        return ChatResponse(
            reply=reply_text,
            debug=ChatDebugResponse(
                final_prompt=final_prompt,
                total_seconds=total_seconds,
                retrieval_mode="base_knowledge",
                retrieved_excerpts=[],
                timing=ChatDebugTimingResponse(
                    retrieval_seconds=0.0,
                    prompt_build_seconds=prompt_build_seconds,
                    inference_seconds=inference_seconds,
                    total_seconds=total_seconds,
                ),
                chunk_retrieval_mode=None,
                history_checked=False,
                history_reuse_hit=False,
                history_top_score=None,
                history_threshold=None,
                reused_turn_id=None,
                retrieved_history_candidates=[],
                retrieved_chunk_candidates=[],
                final_candidate_mix=[],
                history_persisted=history_persisted,
                history_persist_error=history_persist_error,
            ),
        )

    estimated_context_budget = llm_service.estimate_context_token_budget(
        messages=messages,
        max_history_messages=max(1, CHAT_MAX_HISTORY_MESSAGES),
    )
    context_token_budget = max(
        256,
        min(estimated_context_budget, CHAT_CONTEXT_TOKEN_BUDGET),
    )
    retrieval_started = time.perf_counter()
    history_turns = []
    if CHAT_HISTORY_ENABLED:
        history_turns = await run_in_threadpool(
            load_chat_history,
            record.document_id,
            CHAT_HISTORY_STORAGE_DIR,
        )
    try:
        context_result = build_chat_context_result_with_history(
            document_text,
            latest_user_question,
            history_turns=history_turns,
            chat_retrieval_mode=effective_retrieval_mode,
            history_reuse_threshold=CHAT_HISTORY_REUSE_THRESHOLD,
            history_max_candidates=max(1, CHAT_HISTORY_MAX_CANDIDATES),
            history_max_excerpts=max(0, CHAT_HISTORY_MAX_EXCERPTS),
            max_excerpts=max(1, CHAT_CONTEXT_MAX_EXCERPTS),
            retrieve_candidates=max(1, CHAT_RETRIEVE_CANDIDATES),
            context_token_budget=context_token_budget,
            document_id=record.document_id,
            persisted_chunks_path=persisted_chunks_path,
            history_storage_dir=CHAT_HISTORY_STORAGE_DIR if CHAT_HISTORY_ENABLED else None,
            apply_sub_chunking=payload.prompt_format != "rag_raw",
        )
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load persisted chunks: {exc}",
        ) from exc
    retrieval_seconds = time.perf_counter() - retrieval_started

    if not context_result.history_reuse_hit and not context_result.context.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No document or chat-history context is available for this chat request.",
        )

    reply_text = ""
    final_prompt = ""
    prompt_build_seconds = 0.0
    inference_seconds = 0.0
    if context_result.history_reuse_hit and context_result.reused_answer is not None:
        reply_text = "[Answer retrieved from chat history]\n\n" + context_result.reused_answer
    else:
        try:
            inference_result = await run_in_threadpool(
                llm_service.generate_reply_with_debug,
                document_text=context_result.context,
                messages=messages,
                max_history_messages=max(1, CHAT_MAX_HISTORY_MESSAGES),
                title=record.title or "",
                author=record.author or "",
            )
        except LLMDisabledError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except LLMNotConfiguredError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except LLMServiceError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Chat inference failed: {exc}",
            ) from exc

        reply_text = inference_result.reply
        final_prompt = inference_result.final_prompt
        prompt_build_seconds = inference_result.prompt_build_seconds
        inference_seconds = inference_result.inference_seconds

    history_persisted = True
    history_persist_error: str | None = None
    if CHAT_HISTORY_ENABLED and latest_user_question:
        turn = create_chat_history_turn(
            document_id=record.document_id,
            user_query=latest_user_question,
            assistant_answer=reply_text,
            retrieval_mode_used=context_result.retrieval_mode,
            retrieved_chunk_refs=build_chunk_history_refs(context_result.retrieved_excerpts),
            retrieved_history_refs=build_history_candidate_refs(context_result.retrieved_history),
            answer_reused_from_history=context_result.history_reuse_hit,
            reused_from_turn_id=context_result.reused_turn_id,
            history_reuse_score=(
                context_result.history_top_score if context_result.history_reuse_hit else None
            ),
            document_source_fingerprint=record.source_fingerprint,
            chunk_corpus_path=record.chunks_path,
            chunk_meta_path=record.chunk_meta_path,
        )
        try:
            await run_in_threadpool(
                append_chat_history_turn,
                record.document_id,
                turn,
                CHAT_HISTORY_STORAGE_DIR,
            )
            background_tasks.add_task(
                embed_and_save_history,
                record.document_id,
                CHAT_HISTORY_STORAGE_DIR,
            )
        except OSError as exc:
            history_persisted = False
            history_persist_error = str(exc)
            logger.warning(
                "Failed to persist chat history for document_id=%s: %s",
                record.document_id,
                exc,
            )

    total_seconds = time.perf_counter() - request_started
    return ChatResponse(
        reply=reply_text,
        debug=ChatDebugResponse(
            final_prompt=final_prompt,
            total_seconds=total_seconds,
            retrieval_mode=context_result.retrieval_mode,
            retrieved_excerpts=[
                ChatDebugExcerptResponse(**excerpt.to_debug_dict())
                for excerpt in context_result.retrieved_excerpts
            ],
            timing=ChatDebugTimingResponse(
                retrieval_seconds=retrieval_seconds,
                prompt_build_seconds=prompt_build_seconds,
                inference_seconds=inference_seconds,
                total_seconds=total_seconds,
            ),
            chunk_retrieval_mode=context_result.chunk_retrieval_mode,
            history_checked=context_result.history_checked,
            history_reuse_hit=context_result.history_reuse_hit,
            history_top_score=context_result.history_top_score,
            history_threshold=context_result.history_threshold,
            reused_turn_id=context_result.reused_turn_id,
            retrieved_history_candidates=[
                ChatDebugHistoryCandidateResponse(**candidate.to_debug_dict())
                for candidate in context_result.retrieved_history
            ],
            retrieved_chunk_candidates=[
                ChatDebugExcerptResponse(**excerpt.to_debug_dict())
                for excerpt in context_result.retrieved_chunk_candidates
            ],
            raw_retrieved_excerpts=[
                ChatDebugExcerptResponse(**excerpt.to_debug_dict())
                for excerpt in context_result.raw_retrieved_excerpts
            ],
            final_candidate_mix=[
                ChatDebugCandidateResponse(**candidate.to_debug_dict())
                for candidate in context_result.final_candidate_mix
            ],
            history_persisted=history_persisted,
            history_persist_error=history_persist_error,
        ),
    )


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream_endpoint(payload: ChatRequest) -> StreamingResponse:
    """SSE streaming chat endpoint.

    Emits Server-Sent Events:
      {"type":"meta",  ...retrieval info...}
      {"type":"token", "text": "..."}   (one per token)
      {"type":"done",  "debug": {...}}
    """
    request_started = time.perf_counter()
    record = get_document_or_404(payload.document_id)
    document_text = ""
    text_path = Path(record.text_path)
    persisted_chunks_path: Path | None = None
    if record.chunks_available and record.chunks_path:
        candidate = Path(record.chunks_path)
        if candidate.exists():
            persisted_chunks_path = candidate

    if persisted_chunks_path is None and text_path.exists():
        try:
            document_text = text_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read document text: {exc}",
            ) from exc

    messages = [{"role": m.role, "content": m.content} for m in payload.messages]
    latest_user_question = get_latest_user_question(messages)
    effective_retrieval_mode = CHAT_RETRIEVAL_MODE if CHAT_HISTORY_ENABLED else "chunks_only"

    if not document_text and persisted_chunks_path is None and payload.prompt_format != "base_knowledge":
        if not text_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document text file is missing.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document text is empty and no persisted chunks are available; cannot chat.",
        )

    # --- base_knowledge: skip retrieval, stream directly ---
    if payload.prompt_format == "base_knowledge":
        _bk_title = record.title or "Unknown"
        _bk_author = record.author or "Unknown"
        _bk_history_messages = messages[:-1]
        _bk_question = latest_user_question
        _bk_request_started = request_started
        _bk_record = record

        async def _bk_event_generator() -> AsyncGenerator[str, None]:
            yield _sse_event({"type": "meta", "retrieval_seconds": 0.0, "retrieval_mode": "base_knowledge", "retrieved_excerpts": []})

            base_prompt = _build_base_knowledge_prompt(_bk_title, _bk_author, _bk_history_messages, _bk_question)

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

            def _produce_bk() -> None:
                try:
                    gen = llm_service.generate_raw_reply_stream(base_prompt)
                    try:
                        while True:
                            token = next(gen)
                            loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                    except StopIteration as exc:
                        loop.call_soon_threadsafe(queue.put_nowait, ("done", exc.value))
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

            loop.run_in_executor(None, _produce_bk)

            reply_text = ""
            stream_result = None
            while True:
                kind, data = await queue.get()
                if kind == "token":
                    reply_text += data
                    yield _sse_event({"type": "token", "text": data})
                elif kind == "done":
                    stream_result = data
                    break
                elif kind == "error":
                    exc = data
                    if isinstance(exc, (LLMDisabledError, LLMNotConfiguredError, LLMServiceError)):
                        yield _sse_event({"type": "error", "detail": str(exc)})
                    else:
                        yield _sse_event({"type": "error", "detail": f"Chat inference failed: {exc}"})
                    return

            inference_seconds = stream_result.inference_seconds if stream_result else 0.0
            if stream_result:
                reply_text = stream_result.reply
            total_seconds = time.perf_counter() - _bk_request_started

            if CHAT_HISTORY_ENABLED and _bk_question and reply_text:
                turn = create_chat_history_turn(
                    document_id=_bk_record.document_id,
                    user_query=_bk_question,
                    assistant_answer=reply_text,
                    retrieval_mode_used="base_knowledge",
                )
                try:
                    await run_in_threadpool(
                        append_chat_history_turn,
                        _bk_record.document_id,
                        turn,
                        CHAT_HISTORY_STORAGE_DIR,
                    )
                    asyncio.get_running_loop().run_in_executor(
                        None, embed_and_save_history, _bk_record.document_id, str(CHAT_HISTORY_STORAGE_DIR)
                    )
                except OSError:
                    pass

            yield _sse_event({
                "type": "done",
                "debug": {
                    "final_prompt": stream_result.final_prompt if stream_result else base_prompt,
                    "total_seconds": round(total_seconds, 3),
                    "retrieval_mode": "base_knowledge",
                    "timing": {
                        "retrieval_seconds": 0.0,
                        "prompt_build_seconds": 0.0,
                        "inference_seconds": round(inference_seconds, 3),
                        "total_seconds": round(total_seconds, 3),
                    },
                    "history_checked": False,
                    "history_reuse_hit": False,
                    "history_top_score": None,
                    "reused_turn_id": None,
                    "history_persisted": True,
                    "history_persist_error": None,
                    "retrieved_excerpts": [],
                },
            })

        return StreamingResponse(
            _bk_event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    estimated_context_budget = llm_service.estimate_context_token_budget(
        messages=messages,
        max_history_messages=max(1, CHAT_MAX_HISTORY_MESSAGES),
    )
    context_token_budget = max(256, min(estimated_context_budget, CHAT_CONTEXT_TOKEN_BUDGET))

    retrieval_started = time.perf_counter()
    history_turns = []
    if CHAT_HISTORY_ENABLED:
        history_turns = await run_in_threadpool(
            load_chat_history,
            record.document_id,
            CHAT_HISTORY_STORAGE_DIR,
        )
    try:
        context_result = build_chat_context_result_with_history(
            document_text,
            latest_user_question,
            history_turns=history_turns,
            chat_retrieval_mode=effective_retrieval_mode,
            history_reuse_threshold=CHAT_HISTORY_REUSE_THRESHOLD,
            history_max_candidates=max(1, CHAT_HISTORY_MAX_CANDIDATES),
            history_max_excerpts=max(0, CHAT_HISTORY_MAX_EXCERPTS),
            max_excerpts=max(1, CHAT_CONTEXT_MAX_EXCERPTS),
            retrieve_candidates=max(1, CHAT_RETRIEVE_CANDIDATES),
            context_token_budget=context_token_budget,
            document_id=record.document_id,
            persisted_chunks_path=persisted_chunks_path,
            history_storage_dir=CHAT_HISTORY_STORAGE_DIR if CHAT_HISTORY_ENABLED else None,
            apply_sub_chunking=payload.prompt_format != "rag_raw",
        )
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load persisted chunks: {exc}",
        ) from exc
    retrieval_seconds = time.perf_counter() - retrieval_started

    if not context_result.history_reuse_hit and not context_result.context.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No document or chat-history context available for this request.",
        )

    # Capture mutable state for the generator closure
    _record = record
    _context_result = context_result
    _messages = messages
    _latest_user_question = latest_user_question
    _retrieval_seconds = retrieval_seconds
    _request_started = request_started

    async def _event_generator() -> AsyncGenerator[str, None]:
        # 1. Meta event with retrieval info
        yield _sse_event({
            "type": "meta",
            "retrieval_seconds": round(_retrieval_seconds, 3),
            "retrieval_mode": _context_result.retrieval_mode,
            "retrieved_excerpts": [e.to_debug_dict() for e in _context_result.retrieved_excerpts],
        })

        reply_text = ""
        final_prompt = ""
        prompt_build_seconds = 0.0
        inference_seconds = 0.0

        if _context_result.history_reuse_hit and _context_result.reused_answer is not None:
            reply_text = "[Answer retrieved from chat history]\n\n" + _context_result.reused_answer
            yield _sse_event({"type": "token", "text": reply_text})
        else:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

            def _produce() -> None:
                try:
                    gen = llm_service.generate_reply_stream(
                        document_text=_context_result.context,
                        messages=_messages,
                        max_history_messages=max(1, CHAT_MAX_HISTORY_MESSAGES),
                        title=_record.title or "",
                        author=_record.author or "",
                    )
                    try:
                        while True:
                            token = next(gen)
                            loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                    except StopIteration as exc:
                        loop.call_soon_threadsafe(queue.put_nowait, ("done", exc.value))
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

            loop.run_in_executor(None, _produce)

            stream_result = None
            while True:
                kind, data = await queue.get()
                if kind == "token":
                    reply_text += data
                    yield _sse_event({"type": "token", "text": data})
                elif kind == "done":
                    stream_result = data
                    break
                elif kind == "error":
                    exc = data
                    if isinstance(exc, (LLMDisabledError, LLMNotConfiguredError, LLMServiceError)):
                        yield _sse_event({"type": "error", "detail": str(exc)})
                    else:
                        yield _sse_event({"type": "error", "detail": f"Chat inference failed: {exc}"})
                    return

            if stream_result is not None:
                reply_text = stream_result.reply
                final_prompt = stream_result.final_prompt
                prompt_build_seconds = stream_result.prompt_build_seconds
                inference_seconds = stream_result.inference_seconds

        total_seconds = time.perf_counter() - _request_started

        # Persist history turn (synchronous, runs before done event)
        history_persisted = True
        history_persist_error: str | None = None
        if CHAT_HISTORY_ENABLED and _latest_user_question and reply_text:
            turn = create_chat_history_turn(
                document_id=_record.document_id,
                user_query=_latest_user_question,
                assistant_answer=reply_text,
                retrieval_mode_used=_context_result.retrieval_mode,
                retrieved_chunk_refs=build_chunk_history_refs(_context_result.retrieved_excerpts),
                retrieved_history_refs=build_history_candidate_refs(_context_result.retrieved_history),
                answer_reused_from_history=_context_result.history_reuse_hit,
                reused_from_turn_id=_context_result.reused_turn_id,
                history_reuse_score=(
                    _context_result.history_top_score if _context_result.history_reuse_hit else None
                ),
                document_source_fingerprint=_record.source_fingerprint,
                chunk_corpus_path=_record.chunks_path,
                chunk_meta_path=_record.chunk_meta_path,
            )
            try:
                await run_in_threadpool(
                    append_chat_history_turn,
                    _record.document_id,
                    turn,
                    CHAT_HISTORY_STORAGE_DIR,
                )
                # Embed history in background after stream completes
                asyncio.get_running_loop().run_in_executor(
                    None,
                    embed_and_save_history,
                    _record.document_id,
                    str(CHAT_HISTORY_STORAGE_DIR),
                )
            except OSError as exc:
                history_persisted = False
                history_persist_error = str(exc)

        # 3. Done event with full debug
        yield _sse_event({
            "type": "done",
            "debug": {
                "final_prompt": final_prompt,
                "total_seconds": round(total_seconds, 3),
                "retrieval_mode": _context_result.retrieval_mode,
                "timing": {
                    "retrieval_seconds": round(_retrieval_seconds, 3),
                    "prompt_build_seconds": round(prompt_build_seconds, 3),
                    "inference_seconds": round(inference_seconds, 3),
                    "total_seconds": round(total_seconds, 3),
                },
                "history_checked": _context_result.history_checked,
                "history_reuse_hit": _context_result.history_reuse_hit,
                "history_top_score": _context_result.history_top_score,
                "reused_turn_id": _context_result.reused_turn_id,
                "history_persisted": history_persisted,
                "history_persist_error": history_persist_error,
                "retrieved_excerpts": [e.to_debug_dict() for e in _context_result.retrieved_excerpts],
                "raw_retrieved_excerpts": [e.to_debug_dict() for e in _context_result.raw_retrieved_excerpts],
            },
        })

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/documents/{document_id}/generate-chunks", response_model=GenerateChunksResponse)
async def generate_document_chunks_endpoint(document_id: str) -> GenerateChunksResponse:
    record = get_document_or_404(document_id)
    text_path = Path(record.text_path)
    if not text_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document text file is missing.",
        )

    try:
        final_text = text_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read document text: {exc}",
        ) from exc

    if not final_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document text is empty; cannot generate chunks.",
        )

    try:
        updated_record, chunk_params = await run_in_threadpool(
            generate_and_persist_chunks_for_record,
            record,
            final_text,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate chunks: {exc}",
        ) from exc

    return GenerateChunksResponse(
        document_id=updated_record.document_id,
        filename=updated_record.filename,
        chunks_available=updated_record.chunks_available,
        chunks_count=updated_record.chunks_count or 0,
        chunk_schema_version=updated_record.chunk_schema_version or CHUNK_SCHEMA_VERSION,
        chunks_path=updated_record.chunks_path,
        chunk_meta_path=updated_record.chunk_meta_path,
        chunks_download_url=(
            build_chunks_download_url(updated_record.document_id)
            if updated_record.chunks_available and updated_record.chunks_path
            else None
        ),
        chunk_target_tokens=chunk_params.target_tokens,
        chunk_overlap_tokens=chunk_params.overlap_tokens,
    )


@app.post("/api/process-pdf-chunks", include_in_schema=False)
async def process_pdf_chunks_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> FileResponse:
    validate_pdf_upload(file)

    request_id = str(uuid.uuid4())
    upload_path = UPLOADS_DIR / f"{request_id}.pdf"
    output_path = DOCUMENTS_DIR / f"{request_id}.txt"

    await save_upload_file(file, upload_path, MAX_UPLOAD_MB)
    try:
        source_fingerprint = compute_file_fingerprint(upload_path)
        extraction, final_text = await run_in_threadpool(process_pdf, upload_path, output_path)
    except HTTPException:
        remove_file(upload_path)
        remove_file(output_path)
        raise
    except Exception as exc:
        remove_file(upload_path)
        remove_file(output_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF chunks: {exc}",
        ) from exc

    record = document_registry.register(
        document_id=request_id,
        text_path=output_path,
        filename=file.filename,
        processing_mode=extraction.mode,
        pages=extraction.pages_count,
        text_chars=len(final_text),
        source_fingerprint=source_fingerprint,
    )

    try:
        chunks_result = await generate_document_chunks_endpoint(record.document_id)
    except HTTPException:
        remove_file(upload_path)
        raise

    background_tasks.add_task(remove_file, upload_path)
    if not chunks_result.chunks_path:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chunk generation completed without a persisted chunk file path.",
        )
    download_name = f"{Path(file.filename).stem}.chunks.jsonl"

    return FileResponse(
        path=Path(chunks_result.chunks_path),
        media_type="application/x-ndjson; charset=utf-8",
        filename=download_name,
        background=background_tasks,
        headers={
            **build_document_headers(get_document_or_404(record.document_id)),
            "X-Chunks-Count": str(chunks_result.chunks_count),
            "X-Chunk-Target-Tokens": str(chunks_result.chunk_target_tokens),
            "X-Chunk-Overlap-Tokens": str(chunks_result.chunk_overlap_tokens),
        },
    )


# --------------------------------------------------------------------------- #
# Exam Mode — Pydantic models                                                  #
# --------------------------------------------------------------------------- #

class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    chunks_available: bool
    chunks_count: int | None
    created_at: str
    title: str | None
    author: str | None


class Paper1ExamResponse(BaseModel):
    passage: str
    question: str


class Paper2QuestionsResponse(BaseModel):
    questions: list[dict]


class SubmitAnswerRequest(BaseModel):
    paper_type: Literal["paper1", "paper2"] = "paper1"
    question: str = Field(..., min_length=1, max_length=2_000)
    student_answer: str = Field(..., min_length=1, max_length=8_000)
    document_ids: list[str] = Field(default_factory=list)
    context_mode: Literal["chunks", "titles_only"] = "chunks"
    passage_text: str | None = Field(None, max_length=10_000)


class ExamCriterionResult(BaseModel):
    criterion: str
    label: str
    score: int
    max_score: int
    feedback: str


class ExamDebugInfo(BaseModel):
    prompt: str
    raw_output: str
    inference_seconds: float
    prompt_tokens: int


class SubmitAnswerResponse(BaseModel):
    total_score: int
    max_score: int
    paper_type: str
    context_mode: str
    criteria: list[ExamCriterionResult]
    overall_comments: str
    inference_seconds: float
    debug: ExamDebugInfo


class FeedbackStreamRequest(BaseModel):
    paper_type: Literal["paper1", "paper2"]
    criterion: Literal["A", "B", "C", "D"]
    score: int
    max_score: int
    student_answer: str = Field(..., min_length=1, max_length=8_000)
    question: str = Field(..., min_length=1, max_length=2_000)
    passage_text: str | None = Field(None, max_length=10_000)
    document_ids: list[str] = Field(default_factory=list)
    context_mode: Literal["chunks", "titles_only"] = "chunks"


# --------------------------------------------------------------------------- #
# Exam Mode — endpoints                                                        #
# --------------------------------------------------------------------------- #

def _require_chunks(record: Any) -> Path:
    """Return the validated chunks Path, or raise HTTP 422/404."""
    if not record.chunks_available or not record.chunks_path:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document chunks are not available. Generate chunks first.",
        )
    chunks_path = Path(record.chunks_path)
    if not chunks_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunk file is missing on disk.",
        )
    return chunks_path


@app.get("/api/status")
async def get_status() -> dict:
    """Return LLM availability and basic service status."""
    try:
        chat_available = get_llm_service().is_chat_available()
    except Exception:
        chat_available = False
    return {"chat_available": chat_available}


@app.get("/api/documents", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    """List all registered documents, most recently created first (one per unique work)."""
    records = _dedup_documents(document_registry.list_all())
    return [
        DocumentSummary(
            document_id=r.document_id,
            filename=r.filename,
            chunks_available=r.chunks_available,
            chunks_count=r.chunks_count,
            created_at=r.created_at,
            title=r.title,
            author=r.author,
        )
        for r in records
    ]


@app.get("/api/exam/paper1", response_model=Paper1ExamResponse)
async def exam_get_paper1() -> Paper1ExamResponse:
    """Return a randomly selected Paper 1 unseen passage and guiding question."""
    entry = get_random_paper1_passage()
    return Paper1ExamResponse(passage=entry["passage"], question=entry["guiding_question"])


@app.get("/api/exam/paper2/questions", response_model=Paper2QuestionsResponse)
async def exam_get_paper2_questions() -> Paper2QuestionsResponse:
    """Return the Paper 2 question bank (loaded from paper2_sub.json)."""
    return Paper2QuestionsResponse(questions=load_paper2_questions())


@app.post("/api/exam/submit-answer", response_model=SubmitAnswerResponse)
async def exam_submit_answer(payload: SubmitAnswerRequest) -> SubmitAnswerResponse:
    # ---- Paper 1: no document validation needed ----
    if payload.paper_type == "paper1":
        try:
            result: GradingResult = await run_in_threadpool(
                grade_answer,
                paper_type="paper1",
                question=payload.question,
                student_answer=payload.student_answer,
                passage_text=payload.passage_text or PAPER1_PASSAGE,
            )
        except LLMDisabledError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except LLMNotConfiguredError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except LLMServiceError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Grading failed: {exc}",
            ) from exc

        history_doc_id = "paper1"
        history_doc_ids: list[str] = []
        history_chunks_path: str | None = None

    # ---- Paper 2: validate exactly 2 documents ----
    else:
        if len(payload.document_ids) != 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Paper 2 requires exactly 2 document_ids.",
            )

        records = [get_document_or_404(doc_id) for doc_id in payload.document_ids]

        # Chunks are only required for "chunks" context mode
        chunks_paths: list[Path] = []
        if payload.context_mode == "chunks":
            for rec in records:
                chunks_paths.append(_require_chunks(rec))

        doc_titles: list[str] = [
            f"{rec.title or rec.filename} by {rec.author or 'Unknown'}"
            for rec in records
        ]

        try:
            result = await run_in_threadpool(
                grade_answer,
                paper_type="paper2",
                question=payload.question,
                student_answer=payload.student_answer,
                chunks_paths=[str(p) for p in chunks_paths] or None,
                context_mode=payload.context_mode,
                doc_titles=doc_titles,
            )
        except LLMDisabledError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except LLMNotConfiguredError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except LLMServiceError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Grading failed: {exc}",
            ) from exc

        sorted_ids = sorted(payload.document_ids)
        history_doc_id = f"{sorted_ids[0]}_{sorted_ids[1][:8]}"
        history_doc_ids = payload.document_ids
        history_chunks_path = str(chunks_paths[0]) if chunks_paths else None

    # ---- Persist attempt ----
    score_by_criterion = {c.criterion: c for c in result.criteria}
    attempt = create_exam_attempt(
        document_id=history_doc_id,
        paper_type=payload.paper_type,
        question=payload.question,
        student_answer=payload.student_answer,
        chunks_path=history_chunks_path,
        score_a=score_by_criterion["A"].score if "A" in score_by_criterion else None,
        score_b=score_by_criterion["B"].score if "B" in score_by_criterion else None,
        score_c=score_by_criterion["C"].score if "C" in score_by_criterion else None,
        score_d=score_by_criterion["D"].score if "D" in score_by_criterion else None,
        total_score=result.total_score,
        max_score=result.max_score,
        feedback_a=score_by_criterion["A"].feedback if "A" in score_by_criterion else None,
        feedback_b=score_by_criterion["B"].feedback if "B" in score_by_criterion else None,
        feedback_c=score_by_criterion["C"].feedback if "C" in score_by_criterion else None,
        feedback_d=score_by_criterion["D"].feedback if "D" in score_by_criterion else None,
        overall_comments=result.overall_comments,
        grading_raw_output=result.raw_output,
        document_ids=history_doc_ids,
        context_mode=payload.context_mode,
    )
    try:
        await run_in_threadpool(
            append_exam_attempt,
            history_doc_id,
            attempt,
            EXAM_HISTORY_STORAGE_DIR,
        )
    except OSError as exc:
        logger.warning("Failed to persist exam attempt for document_id=%s: %s", history_doc_id, exc)

    return SubmitAnswerResponse(
        total_score=result.total_score,
        max_score=result.max_score,
        paper_type=result.paper_type,
        context_mode=result.context_mode,
        criteria=[
            ExamCriterionResult(
                criterion=c.criterion,
                label=c.label,
                score=c.score,
                max_score=c.max_score,
                feedback=c.feedback,
            )
            for c in result.criteria
        ],
        overall_comments=result.overall_comments,
        inference_seconds=result.inference_seconds,
        debug=ExamDebugInfo(
            prompt=result.prompt,
            raw_output=result.raw_output,
            inference_seconds=result.inference_seconds,
            prompt_tokens=get_llm_service().count_tokens(result.prompt),
        ),
    )


# Module-level set tracking active feedback stream keys.
# The frontend calls criteria sequentially so concurrent requests from the same
# session should not happen, but this guard handles accidental double-sends.
_active_feedback_streams: set[str] = set()


@app.post("/api/exam/criterion-feedback/stream")
async def exam_criterion_feedback_stream(
    payload: FeedbackStreamRequest,
) -> StreamingResponse:
    """SSE streaming endpoint for per-criterion detailed coaching feedback.

    Emits token / done / error events using the same pattern as /api/chat/stream.
    Returns HTTP 409 if a stream for the same student answer is already active.
    """
    stream_key = payload.student_answer[:32]
    if stream_key in _active_feedback_streams:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A feedback stream for this response is already in progress.",
        )

    # Resolve criterion label
    criteria_defs = CRITERIA_BY_PAPER.get(payload.paper_type, [])
    criterion_info = next(
        ((lbl, mx) for c, lbl, mx in criteria_defs if c == payload.criterion),
        None,
    )
    if criterion_info is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown criterion {payload.criterion!r} for {payload.paper_type}.",
        )
    criterion_label, _ = criterion_info

    # Build context_text for Paper 2 chunks mode
    context_text: str | None = None
    if payload.paper_type == "paper2" and payload.context_mode == "chunks":
        if len(payload.document_ids) != 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Paper 2 chunks mode requires exactly 2 document_ids.",
            )
        records = [get_document_or_404(doc_id) for doc_id in payload.document_ids]
        chunks_paths = [_require_chunks(rec) for rec in records]
        context_text = await run_in_threadpool(
            retrieve_multi_doc_context,
            [str(p) for p in chunks_paths],
            "paper2",
        )

    svc = get_llm_service()
    n_ctx = svc.get_context_window_size()
    prompt_tokens = await run_in_threadpool(
        estimate_feedback_prompt_tokens,
        paper_type=payload.paper_type,
        criterion=payload.criterion,
        criterion_label=criterion_label,
        score=payload.score,
        max_score=payload.max_score,
        student_answer=payload.student_answer,
        passage_text=payload.passage_text,
        context_text=context_text,
        question=payload.question,
    )
    remaining_budget = max(0, n_ctx - prompt_tokens)

    async def _event_generator() -> AsyncGenerator[str, None]:
        _active_feedback_streams.add(stream_key)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _produce() -> None:
            try:
                gen = stream_criterion_feedback(
                    paper_type=payload.paper_type,
                    criterion=payload.criterion,
                    criterion_label=criterion_label,
                    score=payload.score,
                    max_score=payload.max_score,
                    student_answer=payload.student_answer,
                    passage_text=payload.passage_text,
                    context_text=context_text,
                    question=payload.question,
                )
                try:
                    while True:
                        token = next(gen)
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                except StopIteration as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, ("done", exc.value))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        try:
            loop.run_in_executor(None, _produce)
            while True:
                kind, data = await queue.get()
                if kind == "token":
                    yield _sse_event({"type": "token", "text": data})
                elif kind == "done":
                    inf_result = data
                    yield _sse_event({
                        "type": "done",
                        "debug": {
                            "criterion": payload.criterion,
                            "prompt": inf_result.final_prompt,
                            "inference_seconds": round(inf_result.inference_seconds, 3),
                            "prompt_tokens": prompt_tokens,
                            "remaining_budget": remaining_budget,
                            "n_ctx": n_ctx,
                        },
                    })
                    return
                elif kind == "error":
                    exc = data
                    yield _sse_event({"type": "error", "detail": str(exc)})
                    return
        finally:
            _active_feedback_streams.discard(stream_key)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
