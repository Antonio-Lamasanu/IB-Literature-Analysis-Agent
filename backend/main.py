import os
import re
import time
import uuid
import hashlib
import logging
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from chat_history import (
    append_chat_history_turn,
    create_chat_history_turn,
    load_chat_history,
)
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
DEFAULT_CHAT_HISTORY_STORAGE_DIR = OUTPUTS_DIR / "chat_history"
DOCUMENTS_INDEX_PATH = OUTPUTS_DIR / "documents.index.json"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
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
REUSE_PROCESSED_PDFS = parse_bool_env("REUSE_PROCESSED_PDFS", False)

configure_app_logging()

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
    final_candidate_mix: list[ChatDebugCandidateResponse] = Field(default_factory=list)
    history_persisted: bool = True
    history_persist_error: str | None = None


class ChatResponse(BaseModel):
    reply: str
    debug: ChatDebugResponse | None = None


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

    chunks_path = OUTPUTS_DIR / f"{safe_stem}_{document_id}.chunks.jsonl"
    meta_path = OUTPUTS_DIR / f"{safe_stem}_{document_id}.chunks.meta.json"
    filtered_chunks_path: Path | None = None

    write_chunks_jsonl(chunks_path, chunks, include_raw=params.include_raw)
    if CHUNK_FILTER_FOR_EMBEDDING:
        filtered_chunks_path = OUTPUTS_DIR / f"{safe_stem}_{document_id}.chunks.filtered.jsonl"
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
    output_path = OUTPUTS_DIR / f"{request_id}.txt"

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
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
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
        reply_text = context_result.reused_answer
    else:
        try:
            inference_result = await run_in_threadpool(
                llm_service.generate_reply_with_debug,
                document_text=context_result.context,
                messages=messages,
                max_history_messages=max(1, CHAT_MAX_HISTORY_MESSAGES),
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
            final_candidate_mix=[
                ChatDebugCandidateResponse(**candidate.to_debug_dict())
                for candidate in context_result.final_candidate_mix
            ],
            history_persisted=history_persisted,
            history_persist_error=history_persist_error,
        ),
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
    output_path = OUTPUTS_DIR / f"{request_id}.txt"

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
