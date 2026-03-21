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
EXAM_HISTORY_SCHEMA_VERSION = "exam-history-v1"
_persist_lock = threading.Lock()


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


def get_exam_history_path(document_id: str, storage_dir: str | Path) -> Path:
    base_dir = Path(storage_dir)
    return base_dir / f"{_sanitize_document_id(document_id)}.json"


def _load_exam_history_from_path(history_path: Path, document_id: str) -> list[ExamAttempt]:
    if not history_path.exists():
        return []

    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load exam history from %s: %s", history_path, exc)
        return []

    if isinstance(payload, dict):
        attempts_payload = payload.get("attempts")
    elif isinstance(payload, list):
        attempts_payload = payload
    else:
        logger.warning("Ignoring malformed exam history payload in %s", history_path)
        return []

    if not isinstance(attempts_payload, list):
        logger.warning("Ignoring exam history without attempts list in %s", history_path)
        return []

    attempts: list[ExamAttempt] = []
    for item in attempts_payload:
        if not isinstance(item, dict):
            continue
        parsed = ExamAttempt.from_dict(item)
        if parsed and parsed.document_id == document_id:
            attempts.append(parsed)
    return attempts


def load_exam_history(document_id: str, storage_dir: str | Path) -> list[ExamAttempt]:
    history_path = get_exam_history_path(document_id, storage_dir)
    return _load_exam_history_from_path(history_path, document_id)


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


def _write_exam_history_unlocked(
    history_path: Path,
    document_id: str,
    attempts: list[ExamAttempt],
) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": EXAM_HISTORY_SCHEMA_VERSION,
        "document_id": document_id,
        "attempts": [attempt.to_dict() for attempt in attempts],
    }
    temp_path = history_path.with_suffix(f"{history_path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(history_path)


def persist_exam_history(
    document_id: str,
    attempts: list[ExamAttempt],
    storage_dir: str | Path,
) -> Path:
    history_path = get_exam_history_path(document_id, storage_dir)
    with _persist_lock:
        _write_exam_history_unlocked(history_path, document_id, attempts)
    return history_path


def append_exam_attempt(
    document_id: str,
    attempt: ExamAttempt,
    storage_dir: str | Path,
) -> Path:
    history_path = get_exam_history_path(document_id, storage_dir)
    with _persist_lock:
        attempts = _load_exam_history_from_path(history_path, document_id)
        attempts.append(attempt)
        _write_exam_history_unlocked(history_path, document_id, attempts)
    return history_path
