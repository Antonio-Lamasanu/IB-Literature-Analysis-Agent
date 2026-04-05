"""
sessions.py — CRUD operations for named chat sessions.

Sessions namespace chat turns per user and document/mode. Each session has a
human-readable name that can be renamed by the user.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from database import get_connection, get_db_path

_MAX_NAME_LENGTH = 100


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session(
    *,
    user_id: str,
    document_id: str,
    mode: str,
    name: str = "New Session",
) -> dict[str, Any]:
    """Create a new session and return its dict (including turn_count=0)."""
    if mode not in {"learn", "exam"}:
        raise ValueError(f"Invalid mode: {mode!r}")
    session_id = str(uuid.uuid4())
    now = _now_iso()
    name = name.strip()[:_MAX_NAME_LENGTH] or "New Session"
    with get_connection(get_db_path()) as conn:
        conn.execute(
            """INSERT INTO chat_sessions
               (session_id, user_id, document_id, mode, name, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (session_id, user_id, document_id, mode, name, now, now),
        )
    return {
        "session_id": session_id,
        "user_id": user_id,
        "document_id": document_id,
        "mode": mode,
        "name": name,
        "created_at": now,
        "updated_at": now,
        "turn_count": 0,
    }


def list_sessions(
    *,
    user_id: str,
    document_id: str,
    mode: str,
) -> list[dict[str, Any]]:
    """Return sessions for a user+document+mode, newest first, with turn counts."""
    with get_connection(get_db_path()) as conn:
        rows = conn.execute(
            """SELECT s.session_id, s.user_id, s.document_id, s.mode, s.name,
                      s.created_at, s.updated_at, s.metadata,
                      COUNT(t.turn_id) AS turn_count
               FROM chat_sessions s
               LEFT JOIN chat_turns t ON t.session_id = s.session_id
               WHERE s.user_id=? AND s.document_id=? AND s.mode=?
               GROUP BY s.session_id
               ORDER BY s.updated_at DESC""",
            (user_id, document_id, mode),
        ).fetchall()
    return [dict(row) for row in rows]


def get_session(session_id: str) -> dict[str, Any] | None:
    """Return a single session dict (without turn_count) or None."""
    with get_connection(get_db_path()) as conn:
        row = conn.execute(
            "SELECT * FROM chat_sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def rename_session(session_id: str, name: str) -> None:
    """Update the session name (max 100 chars)."""
    name = name.strip()[:_MAX_NAME_LENGTH] or "New Session"
    with get_connection(get_db_path()) as conn:
        conn.execute(
            "UPDATE chat_sessions SET name=?, updated_at=? WHERE session_id=?",
            (name, _now_iso(), session_id),
        )


def touch_session(session_id: str) -> None:
    """Update updated_at to now (called after appending a turn)."""
    with get_connection(get_db_path()) as conn:
        conn.execute(
            "UPDATE chat_sessions SET updated_at=? WHERE session_id=?",
            (_now_iso(), session_id),
        )


def delete_session(session_id: str) -> None:
    """Delete a session and all its chat turns."""
    with get_connection(get_db_path()) as conn:
        conn.execute("DELETE FROM chat_turns WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE session_id=?", (session_id,))


def count_session_turns(session_id: str) -> int:
    """Return the number of turns stored for this session."""
    with get_connection(get_db_path()) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM chat_turns WHERE session_id=?",
            (session_id,),
        ).fetchone()
    return int(row[0]) if row else 0


def update_session_metadata(session_id: str, metadata: dict) -> None:
    """Persist a JSON metadata blob for a session (used by exam mode)."""
    import json as _json
    with get_connection(get_db_path()) as conn:
        conn.execute(
            "UPDATE chat_sessions SET metadata=?, updated_at=? WHERE session_id=?",
            (_json.dumps(metadata), _now_iso(), session_id),
        )


def get_session_turns(session_id: str) -> list[dict[str, Any]]:
    """Return a lightweight list of turns (user_query, assistant_answer, created_at)."""
    with get_connection(get_db_path()) as conn:
        rows = conn.execute(
            """SELECT user_query, assistant_answer, created_at
               FROM chat_turns WHERE session_id=? ORDER BY created_at ASC""",
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]
