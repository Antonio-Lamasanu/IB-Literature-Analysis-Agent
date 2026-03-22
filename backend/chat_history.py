from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
CHAT_HISTORY_SCHEMA_VERSION = "chat-history-v1"
_persist_lock = threading.Lock()


def _sanitize_document_id(document_id: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9._-]+", "_", (document_id or "").strip())
    return compact or "document"


@dataclass
class ChatHistoryTurn:
    turn_id: str
    document_id: str
    created_at: str
    user_query: str
    assistant_answer: str
    retrieval_mode_used: str | None = None
    retrieved_chunk_refs: list[dict[str, Any]] = field(default_factory=list)
    retrieved_history_refs: list[dict[str, Any]] = field(default_factory=list)
    answer_reused_from_history: bool = False
    reused_from_turn_id: str | None = None
    history_reuse_score: float | None = None
    document_source_fingerprint: str | None = None
    chunk_corpus_path: str | None = None
    chunk_meta_path: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChatHistoryTurn" | None:
        try:
            turn_id = str(payload["turn_id"]).strip()
            document_id = str(payload["document_id"]).strip()
            created_at = str(payload["created_at"]).strip()
            user_query = str(payload["user_query"]).strip()
            assistant_answer = str(payload.get("assistant_answer") or "")
        except (KeyError, TypeError, ValueError):
            return None

        if not turn_id or not document_id or not created_at or not user_query:
            return None

        return cls(
            turn_id=turn_id,
            document_id=document_id,
            created_at=created_at,
            user_query=user_query,
            assistant_answer=assistant_answer,
            retrieval_mode_used=(
                str(payload["retrieval_mode_used"])
                if payload.get("retrieval_mode_used") is not None
                else None
            ),
            retrieved_chunk_refs=_coerce_ref_list(payload.get("retrieved_chunk_refs")),
            retrieved_history_refs=_coerce_ref_list(payload.get("retrieved_history_refs")),
            answer_reused_from_history=bool(payload.get("answer_reused_from_history", False)),
            reused_from_turn_id=(
                str(payload["reused_from_turn_id"])
                if payload.get("reused_from_turn_id") is not None
                else None
            ),
            history_reuse_score=_coerce_optional_float(payload.get("history_reuse_score")),
            document_source_fingerprint=(
                str(payload["document_source_fingerprint"])
                if payload.get("document_source_fingerprint") is not None
                else None
            ),
            chunk_corpus_path=(
                str(payload["chunk_corpus_path"])
                if payload.get("chunk_corpus_path") is not None
                else None
            ),
            chunk_meta_path=(
                str(payload["chunk_meta_path"])
                if payload.get("chunk_meta_path") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_ref_list(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    refs: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            refs.append(dict(item))
    return refs


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_history_path(document_id: str, storage_dir: str | Path) -> Path:
    base_dir = Path(storage_dir)
    return base_dir / f"{_sanitize_document_id(document_id)}.json"


def _load_chat_history_from_path(history_path: Path, document_id: str) -> list[ChatHistoryTurn]:
    if not history_path.exists():
        return []

    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load chat history from %s: %s", history_path, exc)
        return []

    if isinstance(payload, dict):
        turns_payload = payload.get("turns")
    elif isinstance(payload, list):
        turns_payload = payload
    else:
        logger.warning("Ignoring malformed chat history payload in %s", history_path)
        return []

    if not isinstance(turns_payload, list):
        logger.warning("Ignoring chat history without turns list in %s", history_path)
        return []

    turns: list[ChatHistoryTurn] = []
    for item in turns_payload:
        if not isinstance(item, dict):
            continue
        parsed = ChatHistoryTurn.from_dict(item)
        if parsed and parsed.document_id == document_id:
            turns.append(parsed)
    return turns


def load_chat_history(document_id: str, storage_dir: str | Path) -> list[ChatHistoryTurn]:
    history_path = get_history_path(document_id, storage_dir)
    return _load_chat_history_from_path(history_path, document_id)


def create_chat_history_turn(
    *,
    document_id: str,
    user_query: str,
    assistant_answer: str,
    retrieval_mode_used: str | None = None,
    retrieved_chunk_refs: list[dict[str, Any]] | None = None,
    retrieved_history_refs: list[dict[str, Any]] | None = None,
    answer_reused_from_history: bool = False,
    reused_from_turn_id: str | None = None,
    history_reuse_score: float | None = None,
    document_source_fingerprint: str | None = None,
    chunk_corpus_path: str | None = None,
    chunk_meta_path: str | None = None,
) -> ChatHistoryTurn:
    return ChatHistoryTurn(
        turn_id=str(uuid.uuid4()),
        document_id=document_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        user_query=user_query.strip(),
        assistant_answer=(assistant_answer or "").strip(),
        retrieval_mode_used=retrieval_mode_used,
        retrieved_chunk_refs=[dict(item) for item in (retrieved_chunk_refs or []) if isinstance(item, dict)],
        retrieved_history_refs=[
            dict(item) for item in (retrieved_history_refs or []) if isinstance(item, dict)
        ],
        answer_reused_from_history=bool(answer_reused_from_history),
        reused_from_turn_id=reused_from_turn_id,
        history_reuse_score=history_reuse_score,
        document_source_fingerprint=document_source_fingerprint,
        chunk_corpus_path=chunk_corpus_path,
        chunk_meta_path=chunk_meta_path,
    )


def persist_chat_history(
    document_id: str,
    turns: list[ChatHistoryTurn],
    storage_dir: str | Path,
) -> Path:
    history_path = get_history_path(document_id, storage_dir)
    with _persist_lock:
        _write_chat_history_unlocked(history_path, document_id, turns)
    return history_path


def _write_chat_history_unlocked(
    history_path: Path,
    document_id: str,
    turns: list[ChatHistoryTurn],
) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHAT_HISTORY_SCHEMA_VERSION,
        "document_id": document_id,
        "turns": [turn.to_dict() for turn in turns],
    }
    temp_path = history_path.with_suffix(f"{history_path.suffix}.tmp")

    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(history_path)


def append_chat_history_turn(
    document_id: str,
    turn: ChatHistoryTurn,
    storage_dir: str | Path,
) -> Path:
    history_path = get_history_path(document_id, storage_dir)
    with _persist_lock:
        turns = _load_chat_history_from_path(history_path, document_id)
        turns.append(turn)
        _write_chat_history_unlocked(history_path, document_id, turns)
    return history_path


def history_embeddings_path(document_id: str, storage_dir: str | Path) -> Path:
    """Path to the per-document history embeddings file."""
    base_dir = Path(storage_dir)
    return base_dir / f"{_sanitize_document_id(document_id)}.history.embeddings.npy"


def embed_and_save_history(document_id: str, storage_dir: str | Path) -> None:
    """Rebuild and persist embeddings for all history turns (called as a background task).

    Loads all current turns, encodes query+answer for each, and saves to
    {doc_id}.history.embeddings.npy — a full rebuild keeps the index aligned with the
    turns array without needing to track row counts.
    """
    try:
        from embeddings import encode_texts
        import numpy as np
    except ImportError:
        return

    turns = load_chat_history(document_id, storage_dir)
    if not turns:
        return

    texts = [f"{t.user_query} {t.assistant_answer}" for t in turns]
    try:
        embeddings = encode_texts(texts)
    except Exception as exc:
        logger.warning("embed_and_save_history: encoding failed for doc %s: %s", document_id, exc)
        return

    emb_path = history_embeddings_path(document_id, storage_dir)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.save(str(emb_path), embeddings)
    except Exception as exc:
        logger.warning("embed_and_save_history: save failed for doc %s: %s", document_id, exc)
