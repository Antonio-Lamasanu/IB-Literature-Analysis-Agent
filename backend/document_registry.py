from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from database import get_connection, get_db_path, init_db

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    document_id: str
    text_path: str
    filename: str
    processing_mode: str
    pages: int
    text_chars: int
    created_at: str
    source_fingerprint: str | None = None
    chunks_available: bool = False
    chunks_path: str | None = None
    chunk_meta_path: str | None = None
    chunks_count: int | None = None
    chunk_schema_version: str | None = None
    title: str | None = None
    author: str | None = None
    known_work_confidence: float | None = None
    known_work_source: str | None = None
    known_work_confidence_pending: bool = False
    quality_score: float | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> "DocumentRecord" | None:
        try:
            return cls(
                document_id=str(payload["document_id"]),
                text_path=str(payload["text_path"]),
                filename=str(payload["filename"]),
                processing_mode=str(payload["processing_mode"]),
                pages=int(payload["pages"]),
                text_chars=int(payload["text_chars"]),
                created_at=str(payload["created_at"]),
                source_fingerprint=(
                    str(payload["source_fingerprint"])
                    if payload.get("source_fingerprint") is not None
                    else None
                ),
                chunks_available=bool(payload.get("chunks_available", False)),
                chunks_path=(
                    str(payload["chunks_path"])
                    if payload.get("chunks_path") is not None
                    else None
                ),
                chunk_meta_path=(
                    str(payload["chunk_meta_path"])
                    if payload.get("chunk_meta_path") is not None
                    else None
                ),
                chunks_count=(
                    int(payload["chunks_count"])
                    if payload.get("chunks_count") is not None
                    else None
                ),
                chunk_schema_version=(
                    str(payload["chunk_schema_version"])
                    if payload.get("chunk_schema_version") is not None
                    else None
                ),
                title=(
                    str(payload["title"])
                    if payload.get("title") is not None
                    else None
                ),
                author=(
                    str(payload["author"])
                    if payload.get("author") is not None
                    else None
                ),
                known_work_confidence=(
                    float(payload["known_work_confidence"])
                    if payload.get("known_work_confidence") is not None
                    else None
                ),
                known_work_source=(
                    str(payload["known_work_source"])
                    if payload.get("known_work_source") is not None
                    else None
                ),
                known_work_confidence_pending=bool(payload.get("known_work_confidence_pending", False)),
                quality_score=(
                    float(payload["quality_score"])
                    if payload.get("quality_score") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _record_from_row(row: object) -> DocumentRecord | None:
    if row is None:
        return None
    try:
        return DocumentRecord(
            document_id=row["document_id"],
            text_path=row["text_path"],
            filename=row["filename"],
            processing_mode=row["processing_mode"],
            pages=row["pages"],
            text_chars=row["text_chars"],
            created_at=row["created_at"],
            source_fingerprint=row["source_fingerprint"],
            chunks_available=bool(row["chunks_available"]),
            chunks_path=row["chunks_path"],
            chunk_meta_path=row["chunk_meta_path"],
            chunks_count=row["chunks_count"],
            chunk_schema_version=row["chunk_schema_version"],
            title=row["title"],
            author=row["author"],
            known_work_confidence=row["known_work_confidence"],
            known_work_source=row["known_work_source"],
            known_work_confidence_pending=bool(row["known_work_confidence_pending"]),
            quality_score=row["quality_score"],
        )
    except Exception:
        return None


class DocumentRegistry:
    def __init__(self, index_path: str | Path):
        self._index_path = Path(index_path)
        self._db_path = get_db_path()
        init_db(self._db_path)
        self._records: dict[str, DocumentRecord] = {}
        self._lock = threading.Lock()
        self._load()
        self._migrate_json(self._index_path)

    # ------------------------------------------------------------------
    # Public API (signatures unchanged)
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        document_id: str,
        text_path: str | Path,
        filename: str,
        processing_mode: str,
        pages: int,
        text_chars: int,
        source_fingerprint: str | None = None,
    ) -> DocumentRecord:
        record = DocumentRecord(
            document_id=document_id,
            text_path=str(Path(text_path).resolve()),
            filename=filename,
            processing_mode=processing_mode,
            pages=pages,
            text_chars=text_chars,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_fingerprint=source_fingerprint,
        )
        with self._lock:
            self._records[document_id] = record
            self._db_insert(record)
        return record

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._records.get(document_id)

    def update_chunks(
        self,
        document_id: str,
        *,
        chunks_available: bool,
        chunks_path: str | Path | None,
        chunk_meta_path: str | Path | None,
        chunks_count: int | None,
        chunk_schema_version: str | None,
    ) -> DocumentRecord | None:
        with self._lock:
            record = self._records.get(document_id)
            if record is None:
                return None
            record.chunks_available = bool(chunks_available)
            record.chunks_path = str(Path(chunks_path).resolve()) if chunks_path else None
            record.chunk_meta_path = str(Path(chunk_meta_path).resolve()) if chunk_meta_path else None
            record.chunks_count = int(chunks_count) if chunks_count is not None else None
            record.chunk_schema_version = chunk_schema_version
            with get_connection(self._db_path) as conn:
                conn.execute(
                    """UPDATE documents SET chunks_available=?, chunks_path=?,
                       chunk_meta_path=?, chunks_count=?, chunk_schema_version=?
                       WHERE document_id=?""",
                    (
                        int(record.chunks_available),
                        record.chunks_path,
                        record.chunk_meta_path,
                        record.chunks_count,
                        record.chunk_schema_version,
                        document_id,
                    ),
                )
            return record

    def update_title_author(
        self,
        document_id: str,
        *,
        title: str | None,
        author: str | None,
    ) -> DocumentRecord | None:
        with self._lock:
            record = self._records.get(document_id)
            if record is None:
                return None
            record.title = title
            record.author = author
            with get_connection(self._db_path) as conn:
                conn.execute(
                    "UPDATE documents SET title=?, author=? WHERE document_id=?",
                    (title, author, document_id),
                )
            return record

    def update_known_work_confidence(
        self,
        document_id: str,
        *,
        confidence: float,
        source: str,
    ) -> DocumentRecord | None:
        with self._lock:
            record = self._records.get(document_id)
            if record is None:
                return None
            record.known_work_confidence = confidence
            record.known_work_source = source
            with get_connection(self._db_path) as conn:
                conn.execute(
                    "UPDATE documents SET known_work_confidence=?, known_work_source=? WHERE document_id=?",
                    (confidence, source, document_id),
                )
            return record

    def update_corpus_pending(self, document_id: str, *, pending: bool) -> DocumentRecord | None:
        with self._lock:
            record = self._records.get(document_id)
            if record is None:
                return None
            record.known_work_confidence_pending = pending
            with get_connection(self._db_path) as conn:
                conn.execute(
                    "UPDATE documents SET known_work_confidence_pending=? WHERE document_id=?",
                    (int(pending), document_id),
                )
            return record

    def update_quality_score(self, document_id: str, *, quality_score: float) -> DocumentRecord | None:
        with self._lock:
            record = self._records.get(document_id)
            if record is None:
                return None
            record.quality_score = quality_score
            with get_connection(self._db_path) as conn:
                conn.execute(
                    "UPDATE documents SET quality_score=? WHERE document_id=?",
                    (quality_score, document_id),
                )
            return record

    def replace_files(
        self,
        document_id: str,
        *,
        text_path: str | Path,
        chunks_path: str | Path,
        chunk_meta_path: str | Path,
        chunks_count: int,
        chunk_schema_version: str,
        processing_mode: str,
        text_chars: int,
        pages: int,
        quality_score: float | None = None,
    ) -> DocumentRecord | None:
        """Replace all file paths and metadata for an existing record (quality upgrade)."""
        with self._lock:
            record = self._records.get(document_id)
            if record is None:
                return None
            record.text_path = str(Path(text_path).resolve())
            record.chunks_path = str(Path(chunks_path).resolve())
            record.chunk_meta_path = str(Path(chunk_meta_path).resolve())
            record.chunks_count = chunks_count
            record.chunk_schema_version = chunk_schema_version
            record.chunks_available = chunks_count > 0
            record.processing_mode = processing_mode
            record.text_chars = text_chars
            record.pages = pages
            if quality_score is not None:
                record.quality_score = quality_score
            with get_connection(self._db_path) as conn:
                conn.execute(
                    """UPDATE documents SET text_path=?, chunks_path=?, chunk_meta_path=?,
                       chunks_count=?, chunk_schema_version=?, chunks_available=?,
                       processing_mode=?, text_chars=?, pages=?, quality_score=?
                       WHERE document_id=?""",
                    (
                        record.text_path,
                        record.chunks_path,
                        record.chunk_meta_path,
                        record.chunks_count,
                        record.chunk_schema_version,
                        int(record.chunks_available),
                        record.processing_mode,
                        record.text_chars,
                        record.pages,
                        record.quality_score,
                        document_id,
                    ),
                )
            return record

    def find_by_source_fingerprint(self, fingerprint: str) -> DocumentRecord | None:
        with self._lock:
            for record in self._records.values():
                if record.source_fingerprint == fingerprint:
                    return record
        return None

    def find_by_title_author(
        self,
        title: str | None,
        author: str | None,
        exclude_id: str | None = None,
    ) -> DocumentRecord | None:
        """Return the best existing record for the same title+author, or None.

        If both the incoming author and the stored author are empty/None, matches
        on title alone — handles cases where LLM fails to extract the author on
        re-upload but extracted it correctly the first time (or vice versa).
        """
        if not title and not author:
            return None
        norm_title = (title or "").strip().lower()
        norm_author = (author or "").strip().lower()
        with self._lock:
            candidates = [
                r for r in self._records.values()
                if (
                    r.document_id != exclude_id
                    and (r.title or "").strip().lower() == norm_title
                    and (
                        (r.author or "").strip().lower() == norm_author
                        # title-only fallback: both sides have no author
                        or (not norm_author and not (r.author or "").strip())
                    )
                    and Path(r.text_path).exists()
                )
            ]
        if not candidates:
            return None
        candidates.sort(key=lambda r: r.created_at, reverse=True)
        return candidates[0]

    def list_all(self) -> list[DocumentRecord]:
        """Return registered documents whose text file still exists, most recently created first."""
        with self._lock:
            records = [r for r in self._records.values() if Path(r.text_path).exists()]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def remove(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            record = self._records.pop(document_id, None)
            if record is not None:
                with get_connection(self._db_path) as conn:
                    conn.execute("DELETE FROM documents WHERE document_id=?", (document_id,))
            return record

    def find_reusable_by_source_fingerprint(self, fingerprint: str) -> DocumentRecord | None:
        with self._lock:
            matches = [
                record for record in self._records.values() if record.source_fingerprint == fingerprint
            ]

        if not matches:
            return None

        matches.sort(key=lambda record: record.created_at, reverse=True)

        for record in matches:
            if Path(record.text_path).exists():
                return record

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with get_connection(self._db_path) as conn:
            rows = conn.execute("SELECT * FROM documents").fetchall()
        loaded: dict[str, DocumentRecord] = {}
        for row in rows:
            rec = _record_from_row(row)
            if rec:
                loaded[rec.document_id] = rec
        self._records = loaded

    def _db_insert(self, record: DocumentRecord) -> None:
        with get_connection(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO documents
                   (document_id, filename, text_path, processing_mode, pages, text_chars,
                    created_at, source_fingerprint, chunks_available, chunks_path,
                    chunk_meta_path, chunks_count, chunk_schema_version, title, author,
                    known_work_confidence, known_work_source, known_work_confidence_pending,
                    quality_score)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.document_id,
                    record.filename,
                    record.text_path,
                    record.processing_mode,
                    record.pages,
                    record.text_chars,
                    record.created_at,
                    record.source_fingerprint,
                    int(record.chunks_available),
                    record.chunks_path,
                    record.chunk_meta_path,
                    record.chunks_count,
                    record.chunk_schema_version,
                    record.title,
                    record.author,
                    record.known_work_confidence,
                    record.known_work_source,
                    int(record.known_work_confidence_pending),
                    record.quality_score,
                ),
            )

    def _migrate_json(self, index_path: Path) -> None:
        if not index_path.exists():
            return
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            documents = payload.get("documents", [])
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("document_registry: skipping JSON migration (%s)", exc)
            return

        migrated = 0
        with get_connection(self._db_path) as conn:
            for item in documents:
                if not isinstance(item, dict):
                    continue
                rec = DocumentRecord.from_dict(item)
                if rec is None:
                    continue
                # INSERT OR IGNORE: never overwrite DB rows that already exist
                conn.execute(
                    """INSERT OR IGNORE INTO documents
                       (document_id, filename, text_path, processing_mode, pages, text_chars,
                        created_at, source_fingerprint, chunks_available, chunks_path,
                        chunk_meta_path, chunks_count, chunk_schema_version, title, author,
                        known_work_confidence, known_work_source, known_work_confidence_pending,
                        quality_score)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        rec.document_id, rec.filename, rec.text_path, rec.processing_mode,
                        rec.pages, rec.text_chars, rec.created_at, rec.source_fingerprint,
                        int(rec.chunks_available), rec.chunks_path, rec.chunk_meta_path,
                        rec.chunks_count, rec.chunk_schema_version, rec.title, rec.author,
                        rec.known_work_confidence, rec.known_work_source,
                        int(rec.known_work_confidence_pending), rec.quality_score,
                    ),
                )
                migrated += 1

        if migrated:
            logger.info("document_registry: migrated %d records from %s", migrated, index_path)
            # Reload in-memory cache to include migrated records
            self._load()

        migrated_path = index_path.with_suffix(".json.migrated")
        try:
            index_path.rename(migrated_path)
            logger.info("document_registry: renamed %s → %s", index_path.name, migrated_path.name)
        except OSError as exc:
            logger.warning("document_registry: could not rename old index file: %s", exc)
