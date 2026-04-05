"""
database.py — SQLite setup and connection factory.

Single DB file at outputs/app.db (overridable via DB_PATH env var).
All tables are created here; each module connects via get_connection().
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

_DEFAULT_DB_DIR = Path(__file__).parent / "outputs"

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id                   TEXT PRIMARY KEY,
    filename                      TEXT NOT NULL,
    text_path                     TEXT NOT NULL,
    processing_mode               TEXT NOT NULL,
    pages                         INTEGER NOT NULL,
    text_chars                    INTEGER NOT NULL,
    created_at                    TEXT NOT NULL,
    source_fingerprint            TEXT,
    chunks_available              INTEGER NOT NULL DEFAULT 0,
    chunks_path                   TEXT,
    chunk_meta_path               TEXT,
    chunks_count                  INTEGER,
    chunk_schema_version          TEXT,
    title                         TEXT,
    author                        TEXT,
    known_work_confidence         REAL,
    known_work_source             TEXT,
    known_work_confidence_pending INTEGER NOT NULL DEFAULT 0,
    quality_score                 REAL
);
CREATE INDEX IF NOT EXISTS idx_doc_fingerprint ON documents(source_fingerprint);
CREATE INDEX IF NOT EXISTS idx_doc_title_author ON documents(title, author);

CREATE TABLE IF NOT EXISTS chat_turns (
    turn_id                      TEXT PRIMARY KEY,
    document_id                  TEXT NOT NULL,
    created_at                   TEXT NOT NULL,
    user_query                   TEXT NOT NULL,
    assistant_answer             TEXT NOT NULL,
    retrieval_mode_used          TEXT,
    retrieved_chunk_refs         TEXT NOT NULL DEFAULT '[]',
    retrieved_history_refs       TEXT NOT NULL DEFAULT '[]',
    answer_reused_from_history   INTEGER NOT NULL DEFAULT 0,
    reused_from_turn_id          TEXT,
    history_reuse_score          REAL,
    document_source_fingerprint  TEXT,
    chunk_corpus_path            TEXT,
    chunk_meta_path              TEXT
);
CREATE INDEX IF NOT EXISTS idx_chat_turns_doc ON chat_turns(document_id, created_at);

CREATE TABLE IF NOT EXISTS exam_attempts (
    attempt_id         TEXT PRIMARY KEY,
    document_id        TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    paper_type         TEXT NOT NULL,
    question           TEXT NOT NULL,
    student_answer     TEXT NOT NULL,
    score_a            INTEGER,
    score_b            INTEGER,
    score_c            INTEGER,
    score_d            INTEGER,
    total_score        INTEGER,
    max_score          INTEGER,
    feedback_a         TEXT,
    feedback_b         TEXT,
    feedback_c         TEXT,
    feedback_d         TEXT,
    overall_comments   TEXT,
    grading_raw_output TEXT,
    chunks_path        TEXT,
    document_ids       TEXT NOT NULL DEFAULT '[]',
    context_mode       TEXT NOT NULL DEFAULT 'chunks'
);
CREATE INDEX IF NOT EXISTS idx_exam_attempts_doc ON exam_attempts(document_id, created_at);

CREATE TABLE IF NOT EXISTS corpus_cache (
    cache_key  TEXT PRIMARY KEY,
    confidence REAL NOT NULL,
    source     TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
"""


def get_db_path() -> Path:
    custom = os.environ.get("DB_PATH", "").strip()
    if custom:
        return Path(custom)
    return _DEFAULT_DB_DIR / "app.db"


@contextmanager
def get_connection(db_path: str | Path | None = None):
    """Context manager that yields a connection, commits on success, rolls back and closes on exit."""
    path = db_path if db_path is not None else get_db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path | None = None) -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    path = db_path if db_path is not None else get_db_path()
    with get_connection(path) as conn:
        conn.executescript(_DDL)
