from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from database import get_connection, get_db_path

logger = logging.getLogger(__name__)
EXAM_HISTORY_SCHEMA_VERSION = "exam-history-v1"


def _sanitize_document_id(document_id: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9._-]+", "_", (document_id or "").strip())
    return compact or "document"


@dataclass
class ExamAttempt:
    attempt_id: str
    document_id: str
    created_at: str
    paper_type: str
    question: str
    student_answer: str
    score_a: int | None = None
    score_b: int | None = None
    score_c: int | None = None
    score_d: int | None = None
    total_score: int | None = None
    max_score: int | None = None
    feedback_a: str | None = None
    feedback_b: str | None = None
    feedback_c: str | None = None
    feedback_d: str | None = None
    overall_comments: str | None = None
    grading_raw_output: str | None = None
    chunks_path: str | None = None
    document_ids: list[str] = field(default_factory=list)
    context_mode: str = "chunks"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExamAttempt | None":
        try:
            attempt_id = str(payload["attempt_id"]).strip()
            document_id = str(payload["document_id"]).strip()
            created_at = str(payload["created_at"]).strip()
            paper_type = str(payload["paper_type"]).strip()
            question = str(payload["question"]).strip()
            student_answer = str(payload.get("student_answer") or "")
        except (KeyError, TypeError, ValueError):
            return None

        if not attempt_id or not document_id or not created_at or not paper_type or not question:
            return None

        def _opt_int(val: Any) -> int | None:
            if val is None:
                return None
            try:
                return int(val)
            except (TypeError, ValueError):
                return None

        def _opt_str(val: Any) -> str | None:
            if val is None:
                return None
            return str(val)

        return cls(
            attempt_id=attempt_id,
            document_id=document_id,
            created_at=created_at,
            paper_type=paper_type,
            question=question,
            student_answer=student_answer,
            score_a=_opt_int(payload.get("score_a")),
            score_b=_opt_int(payload.get("score_b")),
            score_c=_opt_int(payload.get("score_c")),
            score_d=_opt_int(payload.get("score_d")),
            total_score=_opt_int(payload.get("total_score")),
            max_score=_opt_int(payload.get("max_score")),
            feedback_a=_opt_str(payload.get("feedback_a")),
            feedback_b=_opt_str(payload.get("feedback_b")),
            feedback_c=_opt_str(payload.get("feedback_c")),
            feedback_d=_opt_str(payload.get("feedback_d")),
            overall_comments=_opt_str(payload.get("overall_comments")),
            grading_raw_output=_opt_str(payload.get("grading_raw_output")),
            chunks_path=_opt_str(payload.get("chunks_path")),
            document_ids=[str(d) for d in payload.get("document_ids") or []],
            context_mode=str(payload.get("context_mode") or "chunks"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _migrate_json_files(storage_dir: str | Path) -> None:
    """One-time migration: import existing *.json exam history files into the DB."""
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
            attempts_payload = payload.get("attempts", [])
            doc_id = payload.get("document_id", json_file.stem)
        elif isinstance(payload, list):
            attempts_payload = payload
            doc_id = json_file.stem
        else:
            continue

        if not isinstance(attempts_payload, list):
            continue

        migrated = 0
        with get_connection(db_path) as conn:
            for item in attempts_payload:
                if not isinstance(item, dict):
                    continue
                attempt = ExamAttempt.from_dict(item)
                if attempt is None or attempt.document_id != doc_id:
                    continue
                conn.execute(
                    """INSERT OR IGNORE INTO exam_attempts
                       (attempt_id, document_id, created_at, paper_type, question,
                        student_answer, score_a, score_b, score_c, score_d,
                        total_score, max_score, feedback_a, feedback_b, feedback_c,
                        feedback_d, overall_comments, grading_raw_output, chunks_path,
                        document_ids, context_mode)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        attempt.attempt_id, attempt.document_id, attempt.created_at,
                        attempt.paper_type, attempt.question, attempt.student_answer,
                        attempt.score_a, attempt.score_b, attempt.score_c, attempt.score_d,
                        attempt.total_score, attempt.max_score,
                        attempt.feedback_a, attempt.feedback_b, attempt.feedback_c,
                        attempt.feedback_d, attempt.overall_comments,
                        attempt.grading_raw_output, attempt.chunks_path,
                        json.dumps(attempt.document_ids, ensure_ascii=False),
                        attempt.context_mode,
                    ),
                )
                migrated += 1

        if migrated:
            logger.info("exam_history: migrated %d attempts from %s", migrated, json_file.name)
        try:
            json_file.rename(json_file.with_suffix(".json.migrated"))
        except OSError as exc:
            logger.warning("exam_history: could not rename %s: %s", json_file.name, exc)


_migrated_dirs: set[str] = set()


def _ensure_migrated(storage_dir: str | Path) -> None:
    key = str(Path(storage_dir).resolve())
    if key not in _migrated_dirs:
        _migrated_dirs.add(key)
        _migrate_json_files(storage_dir)


def load_exam_history(document_id: str, storage_dir: str | Path) -> list[ExamAttempt]:
    _ensure_migrated(storage_dir)
    db_path = get_db_path()
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM exam_attempts WHERE document_id=? ORDER BY created_at ASC",
            (document_id,),
        ).fetchall()
    attempts: list[ExamAttempt] = []
    for row in rows:
        try:
            attempt = ExamAttempt(
                attempt_id=row["attempt_id"],
                document_id=row["document_id"],
                created_at=row["created_at"],
                paper_type=row["paper_type"],
                question=row["question"],
                student_answer=row["student_answer"],
                score_a=row["score_a"],
                score_b=row["score_b"],
                score_c=row["score_c"],
                score_d=row["score_d"],
                total_score=row["total_score"],
                max_score=row["max_score"],
                feedback_a=row["feedback_a"],
                feedback_b=row["feedback_b"],
                feedback_c=row["feedback_c"],
                feedback_d=row["feedback_d"],
                overall_comments=row["overall_comments"],
                grading_raw_output=row["grading_raw_output"],
                chunks_path=row["chunks_path"],
                document_ids=json.loads(row["document_ids"] or "[]"),
                context_mode=row["context_mode"] or "chunks",
            )
            attempts.append(attempt)
        except Exception as exc:
            logger.warning("exam_history: skipping malformed row: %s", exc)
    return attempts


def create_exam_attempt(
    *,
    document_id: str,
    paper_type: str,
    question: str,
    student_answer: str,
    chunks_path: str | None = None,
    score_a: int | None = None,
    score_b: int | None = None,
    score_c: int | None = None,
    score_d: int | None = None,
    total_score: int | None = None,
    max_score: int | None = None,
    feedback_a: str | None = None,
    feedback_b: str | None = None,
    feedback_c: str | None = None,
    feedback_d: str | None = None,
    overall_comments: str | None = None,
    grading_raw_output: str | None = None,
    document_ids: list[str] | None = None,
    context_mode: str = "chunks",
) -> ExamAttempt:
    return ExamAttempt(
        attempt_id=str(uuid.uuid4()),
        document_id=document_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        paper_type=paper_type,
        question=question.strip(),
        student_answer=(student_answer or "").strip(),
        score_a=score_a,
        score_b=score_b,
        score_c=score_c,
        score_d=score_d,
        total_score=total_score,
        max_score=max_score,
        feedback_a=feedback_a,
        feedback_b=feedback_b,
        feedback_c=feedback_c,
        feedback_d=feedback_d,
        overall_comments=overall_comments,
        grading_raw_output=grading_raw_output,
        chunks_path=chunks_path,
        document_ids=list(document_ids or []),
        context_mode=context_mode,
    )


def persist_exam_history(
    document_id: str,
    attempts: list[ExamAttempt],
    storage_dir: str | Path,
) -> None:
    _ensure_migrated(storage_dir)
    db_path = get_db_path()
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM exam_attempts WHERE document_id=?", (document_id,))
        for attempt in attempts:
            _insert_attempt(conn, attempt)


def append_exam_attempt(
    document_id: str,
    attempt: ExamAttempt,
    storage_dir: str | Path,
) -> None:
    _ensure_migrated(storage_dir)
    db_path = get_db_path()
    with get_connection(db_path) as conn:
        _insert_attempt(conn, attempt)


def _insert_attempt(conn: object, attempt: ExamAttempt) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO exam_attempts
           (attempt_id, document_id, created_at, paper_type, question,
            student_answer, score_a, score_b, score_c, score_d,
            total_score, max_score, feedback_a, feedback_b, feedback_c,
            feedback_d, overall_comments, grading_raw_output, chunks_path,
            document_ids, context_mode)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            attempt.attempt_id, attempt.document_id, attempt.created_at,
            attempt.paper_type, attempt.question, attempt.student_answer,
            attempt.score_a, attempt.score_b, attempt.score_c, attempt.score_d,
            attempt.total_score, attempt.max_score,
            attempt.feedback_a, attempt.feedback_b, attempt.feedback_c,
            attempt.feedback_d, attempt.overall_comments,
            attempt.grading_raw_output, attempt.chunks_path,
            json.dumps(attempt.document_ids, ensure_ascii=False),
            attempt.context_mode,
        ),
    )
