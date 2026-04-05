from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from database import get_connection, get_db_path, init_db

logger = logging.getLogger(__name__)
CHAT_HISTORY_SCHEMA_VERSION = "chat-history-v1"


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
    session_id: str | None = None

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
            session_id=(
                str(payload["session_id"])
                if payload.get("session_id") is not None
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


def _migrate_json_files(storage_dir: str | Path) -> None:
    """One-time migration: import existing *.json history files into the DB."""
    base_dir = Path(storage_dir)
    if not base_dir.exists():
        return
    db_path = get_db_path()
    for json_file in sorted(base_dir.glob("*.json")):
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if isinstance(payload, dict):
            turns_payload = payload.get("turns", [])
            doc_id = payload.get("document_id", json_file.stem)
        elif isinstance(payload, list):
            turns_payload = payload
            doc_id = json_file.stem
        else:
            continue

        if not isinstance(turns_payload, list):
            continue

        migrated = 0
        with get_connection(db_path) as conn:
            for item in turns_payload:
                if not isinstance(item, dict):
                    continue
                turn = ChatHistoryTurn.from_dict(item)
                if turn is None or turn.document_id != doc_id:
                    continue
                conn.execute(
                    """INSERT OR IGNORE INTO chat_turns
                       (turn_id, document_id, created_at, user_query, assistant_answer,
                        retrieval_mode_used, retrieved_chunk_refs, retrieved_history_refs,
                        answer_reused_from_history, reused_from_turn_id, history_reuse_score,
                        document_source_fingerprint, chunk_corpus_path, chunk_meta_path, session_id)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        turn.turn_id, turn.document_id, turn.created_at,
                        turn.user_query, turn.assistant_answer, turn.retrieval_mode_used,
                        json.dumps(turn.retrieved_chunk_refs, ensure_ascii=False),
                        json.dumps(turn.retrieved_history_refs, ensure_ascii=False),
                        int(turn.answer_reused_from_history), turn.reused_from_turn_id,
                        turn.history_reuse_score, turn.document_source_fingerprint,
                        turn.chunk_corpus_path, turn.chunk_meta_path, turn.session_id,
                    ),
                )
                migrated += 1

        if migrated:
            logger.info("chat_history: migrated %d turns from %s", migrated, json_file.name)
        try:
            json_file.rename(json_file.with_suffix(".json.migrated"))
        except OSError as exc:
            logger.warning("chat_history: could not rename %s: %s", json_file.name, exc)


_migrated_dirs: set[str] = set()


def _ensure_migrated(storage_dir: str | Path) -> None:
    key = str(Path(storage_dir).resolve())
    if key not in _migrated_dirs:
        _migrated_dirs.add(key)
        _migrate_json_files(storage_dir)


def load_chat_history(
    document_id: str,
    storage_dir: str | Path,
    *,
    session_id: str | None = None,
) -> list[ChatHistoryTurn]:
    _ensure_migrated(storage_dir)
    db_path = get_db_path()
    with get_connection(db_path) as conn:
        if session_id is not None:
            rows = conn.execute(
                "SELECT * FROM chat_turns WHERE document_id=? AND session_id=? ORDER BY created_at ASC",
                (document_id, session_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM chat_turns WHERE document_id=? ORDER BY created_at ASC",
                (document_id,),
            ).fetchall()
    turns: list[ChatHistoryTurn] = []
    for row in rows:
        try:
            row_dict = dict(row)
            turn = ChatHistoryTurn(
                turn_id=row_dict["turn_id"],
                document_id=row_dict["document_id"],
                created_at=row_dict["created_at"],
                user_query=row_dict["user_query"],
                assistant_answer=row_dict["assistant_answer"],
                retrieval_mode_used=row_dict["retrieval_mode_used"],
                retrieved_chunk_refs=json.loads(row_dict["retrieved_chunk_refs"] or "[]"),
                retrieved_history_refs=json.loads(row_dict["retrieved_history_refs"] or "[]"),
                answer_reused_from_history=bool(row_dict["answer_reused_from_history"]),
                reused_from_turn_id=row_dict["reused_from_turn_id"],
                history_reuse_score=row_dict["history_reuse_score"],
                document_source_fingerprint=row_dict["document_source_fingerprint"],
                chunk_corpus_path=row_dict["chunk_corpus_path"],
                chunk_meta_path=row_dict["chunk_meta_path"],
                session_id=row_dict.get("session_id"),
            )
            turns.append(turn)
        except Exception as exc:
            logger.warning("chat_history: skipping malformed row: %s", exc)
    return turns


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
    session_id: str | None = None,
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
        session_id=session_id,
    )


def persist_chat_history(
    document_id: str,
    turns: list[ChatHistoryTurn],
    storage_dir: str | Path,
) -> None:
    _ensure_migrated(storage_dir)
    db_path = get_db_path()
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM chat_turns WHERE document_id=?", (document_id,))
        for turn in turns:
            _insert_turn(conn, turn)


def append_chat_history_turn(
    document_id: str,
    turn: ChatHistoryTurn,
    storage_dir: str | Path,
) -> None:
    _ensure_migrated(storage_dir)
    db_path = get_db_path()
    with get_connection(db_path) as conn:
        _insert_turn(conn, turn)


def _insert_turn(conn: object, turn: ChatHistoryTurn) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO chat_turns
           (turn_id, document_id, created_at, user_query, assistant_answer,
            retrieval_mode_used, retrieved_chunk_refs, retrieved_history_refs,
            answer_reused_from_history, reused_from_turn_id, history_reuse_score,
            document_source_fingerprint, chunk_corpus_path, chunk_meta_path, session_id)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            turn.turn_id, turn.document_id, turn.created_at,
            turn.user_query, turn.assistant_answer, turn.retrieval_mode_used,
            json.dumps(turn.retrieved_chunk_refs, ensure_ascii=False),
            json.dumps(turn.retrieved_history_refs, ensure_ascii=False),
            int(turn.answer_reused_from_history), turn.reused_from_turn_id,
            turn.history_reuse_score, turn.document_source_fingerprint,
            turn.chunk_corpus_path, turn.chunk_meta_path, turn.session_id,
        ),
    )


def history_embeddings_path(document_id: str, storage_dir: str | Path) -> Path:
    """Path to the per-document history embeddings file (stays on disk)."""
    base_dir = Path(storage_dir)
    return base_dir / f"{_sanitize_document_id(document_id)}.history.embeddings.npy"


def embed_and_save_history(document_id: str, storage_dir: str | Path) -> None:
    """Rebuild and persist embeddings for all history turns (background task)."""
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
