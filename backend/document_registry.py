from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


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
            )
        except (KeyError, TypeError, ValueError):
            return None


class DocumentRegistry:
    def __init__(self, index_path: str | Path):
        self._index_path = Path(index_path)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, DocumentRecord] = {}
        self._lock = threading.Lock()
        self._load()

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
            self._persist_unlocked()

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
            self._persist_unlocked()
            return record

    def find_by_source_fingerprint(self, fingerprint: str) -> DocumentRecord | None:
        with self._lock:
            for record in self._records.values():
                if record.source_fingerprint == fingerprint:
                    return record
        return None

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

    def _load(self) -> None:
        if not self._index_path.exists():
            return

        try:
            payload = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        documents = payload.get("documents")
        if not isinstance(documents, list):
            return

        loaded_records: dict[str, DocumentRecord] = {}
        for item in documents:
            if not isinstance(item, dict):
                continue
            parsed = DocumentRecord.from_dict(item)
            if parsed:
                loaded_records[parsed.document_id] = parsed

        self._records = loaded_records

    def _persist_unlocked(self) -> None:
        items = [asdict(record) for record in self._records.values()]
        payload = {"documents": items}
        temp_path = self._index_path.with_suffix(".tmp")

        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self._index_path)
