from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from llm_service import (
    LLMDisabledError,
    LLMNotConfiguredError,
    LLMServiceError,
    get_llm_service,
)
from retrieval import format_excerpt_context, retrieve_relevant_excerpts


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

EXAM_CONTEXT_MAX_EXCERPTS = 5
EXAM_CONTEXT_TOKEN_BUDGET = 900
EXAM_RETRIEVE_CANDIDATES = 12

# BM25 query strings differ by paper type to surface the most useful excerpts.
QUESTION_GEN_QUERY: dict[str, str] = {
    "paper1": "passage language tone imagery literary techniques close reading",
    "paper2": "themes motifs narrative voice symbolism comparative analysis",
}

# Per-paper criteria: (letter, label, max_score)
CRITERIA_BY_PAPER: dict[str, list[tuple[str, str, int]]] = {
    "paper1": [
        ("A", "Understanding and Interpretation", 5),
        ("B", "Analysis and Evaluation", 5),
        ("C", "Focus and Organization", 5),
        ("D", "Language", 5),
    ],
    "paper2": [
        ("A", "Knowledge, Understanding and Interpretation", 10),
        ("B", "Analysis and Evaluation", 10),
        ("C", "Focus, Organisation and Development", 10),
        ("D", "Language", 10),
    ],
}

MAX_SCORE_BY_PAPER: dict[str, int] = {"paper1": 20, "paper2": 40}

_PAPER_LABEL: dict[str, str] = {
    "paper1": "Paper 1",
    "paper2": "Paper 2",
}

_CRITERIA_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "paper1": {
        "A": "depth of interpretation of the passage's meaning, themes, and context",
        "B": "identification and analysis of literary features and their effects",
        "C": "coherent structure and clear argument throughout the response",
        "D": "accuracy, effectiveness, and appropriateness of language used",
    },
    "paper2": {
        "A": "knowledge and understanding of the work; quality of interpretation",
        "B": "identification and analysis of literary features and their effects across the work",
        "C": "coherent essay structure, development of ideas, and sustained argument",
        "D": "accuracy, effectiveness, and appropriateness of language used",
    },
}

_QUESTION_GEN_USER_MSG: dict[str, str] = {
    "paper1": (
        "Based on the passage excerpts in the document context above, write ONE IB Literature\n"
        "Paper 1 exam question. The question must ask the student to write a close analysis of\n"
        "the passage as a whole: its themes, tone, language, and literary techniques.\n"
        "Output a single line beginning with \"Question:\" followed by the question text only.\n"
        "Question:"
    ),
    "paper2": (
        "Based on the work represented in the document context above, write ONE IB Literature\n"
        "Paper 2 essay question. The question must ask the student to explore a theme, motif,\n"
        "or literary technique across the work as a whole using textual evidence.\n"
        "Output a single line beginning with \"Question:\" followed by the question text only.\n"
        "Question:"
    ),
}

_SCORE_RE = re.compile(r"Score:\s*(\d+)/\d+")
_FEEDBACK_RE = re.compile(r"Feedback:\s*(.+?)(?=\n\[|\nScore:|\Z)", re.DOTALL)
_OVERALL_RE = re.compile(r"\[Overall\].*?Comments:\s*(.+?)(?=\n\[|\Z)", re.DOTALL)
_CRITERION_BLOCK_RE = re.compile(
    r"\[Criterion\s+([A-D])\](.*?)(?=\[Criterion\s+[A-D]\]|\[Overall\]|\Z)",
    re.DOTALL,
)


# --------------------------------------------------------------------------- #
# Dataclasses                                                                  #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ExamQuestionResult:
    question: str
    paper_type: str
    inference_seconds: float


@dataclass(frozen=True)
class CriterionScore:
    criterion: str    # "A"–"D"
    label: str
    score: int
    max_score: int    # 5 for paper1, 10 for paper2
    feedback: str


@dataclass(frozen=True)
class GradingResult:
    criteria: list[CriterionScore]
    overall_comments: str
    total_score: int
    max_score: int    # 20 for paper1, 40 for paper2
    paper_type: str
    raw_output: str
    inference_seconds: float


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _validate_paper_type(paper_type: str) -> None:
    if paper_type not in CRITERIA_BY_PAPER:
        raise ValueError(f"Invalid paper_type {paper_type!r}. Must be 'paper1' or 'paper2'.")


def retrieve_exam_context(chunks_path: str | Path, paper_type: str) -> str:
    """Retrieve and format BM25 excerpts appropriate for the given paper type."""
    _validate_paper_type(paper_type)
    query = QUESTION_GEN_QUERY[paper_type]
    excerpts, _mode = retrieve_relevant_excerpts(
        "",  # document_text unused when persisted_chunks_path is provided
        query,
        persisted_chunks_path=Path(chunks_path),
        max_excerpts=EXAM_CONTEXT_MAX_EXCERPTS,
        context_token_budget=EXAM_CONTEXT_TOKEN_BUDGET,
        retrieve_candidates=EXAM_RETRIEVE_CANDIDATES,
    )
    return format_excerpt_context(excerpts)


def _build_grading_user_msg(question: str, student_answer: str, paper_type: str) -> str:
    criteria = CRITERIA_BY_PAPER[paper_type]
    descriptions = _CRITERIA_DESCRIPTIONS[paper_type]
    paper_label = _PAPER_LABEL[paper_type]

    criteria_lines = "\n".join(
        f"  {letter} \u2013 {label} ({max_score}): {descriptions[letter]}"
        for letter, label, max_score in criteria
    )

    # Build the expected output template with correct max scores
    output_blocks = "\n\n".join(
        f"[Criterion {letter}]\nScore: N/{max_score}\nFeedback: <1\u20133 sentences>"
        for letter, _label, max_score in criteria
    )

    first_letter = criteria[0][0]
    first_max = criteria[0][2]

    return (
        f"You are an IB Literature examiner. Grade the student's response strictly using the\n"
        f"four criteria below. Output each block in this exact format:\n\n"
        f"{output_blocks}\n\n"
        f"[Overall]\n"
        f"Comments: <2\u20134 sentences>\n\n"
        f"IB CRITERIA ({paper_label}):\n"
        f"{criteria_lines}\n\n"
        f"EXAM QUESTION:\n{question}\n\n"
        f"STUDENT ANSWER:\n{student_answer}\n\n"
        f"[Criterion {first_letter}]\n"
        f"Score: N/{first_max}\n"
        f"Feedback:"
    )


def _parse_grading_output(raw: str, paper_type: str, inference_seconds: float) -> GradingResult:
    criteria_defs = CRITERIA_BY_PAPER[paper_type]
    criterion_map: dict[str, tuple[str, int]] = {
        letter: (label, max_score) for letter, label, max_score in criteria_defs
    }

    parsed_criteria: list[CriterionScore] = []
    for match in _CRITERION_BLOCK_RE.finditer(raw):
        letter = match.group(1).upper()
        block_text = match.group(2)

        if letter not in criterion_map:
            continue

        label, max_score = criterion_map[letter]

        score_match = _SCORE_RE.search(block_text)
        score = int(score_match.group(1)) if score_match else 0
        score = max(0, min(score, max_score))

        feedback_match = _FEEDBACK_RE.search(block_text)
        feedback = feedback_match.group(1).strip() if feedback_match else "(could not parse)"

        parsed_criteria.append(
            CriterionScore(
                criterion=letter,
                label=label,
                score=score,
                max_score=max_score,
                feedback=feedback,
            )
        )

    # Ensure all four criteria are present, inserting defaults for any missing
    present = {c.criterion for c in parsed_criteria}
    for letter, label, max_score in criteria_defs:
        if letter not in present:
            parsed_criteria.append(
                CriterionScore(
                    criterion=letter,
                    label=label,
                    score=0,
                    max_score=max_score,
                    feedback="(could not parse)",
                )
            )

    # Restore canonical A→D order
    order = {letter: idx for idx, (letter, _, _) in enumerate(criteria_defs)}
    parsed_criteria.sort(key=lambda c: order.get(c.criterion, 99))

    overall_match = _OVERALL_RE.search(raw)
    overall_comments = overall_match.group(1).strip() if overall_match else "(could not parse)"

    total_score = sum(c.score for c in parsed_criteria)

    return GradingResult(
        criteria=parsed_criteria,
        overall_comments=overall_comments,
        total_score=total_score,
        max_score=MAX_SCORE_BY_PAPER[paper_type],
        paper_type=paper_type,
        raw_output=raw,
        inference_seconds=inference_seconds,
    )


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def generate_question(chunks_path: str | Path, paper_type: str) -> ExamQuestionResult:
    """Generate an IB exam question from the document's BM25 excerpts."""
    _validate_paper_type(paper_type)
    context_text = retrieve_exam_context(chunks_path, paper_type)
    user_msg = _QUESTION_GEN_USER_MSG[paper_type]

    t0 = time.perf_counter()
    result = get_llm_service().generate_reply_with_debug(
        document_text=context_text,
        messages=[{"role": "user", "content": user_msg}],
        max_history_messages=0,
    )
    inference_seconds = time.perf_counter() - t0

    # Strip any "Question:" primer the model may have echoed
    question = result.reply.strip()
    if question.lower().startswith("question:"):
        question = question[len("question:"):].strip()

    return ExamQuestionResult(
        question=question,
        paper_type=paper_type,
        inference_seconds=inference_seconds,
    )


def grade_answer(
    chunks_path: str | Path,
    *,
    paper_type: str,
    question: str,
    student_answer: str,
) -> GradingResult:
    """Grade a student answer against IB criteria for the given paper type."""
    _validate_paper_type(paper_type)
    context_text = retrieve_exam_context(chunks_path, paper_type)
    grading_msg = _build_grading_user_msg(question, student_answer, paper_type)

    t0 = time.perf_counter()
    result = get_llm_service().generate_reply_with_debug(
        document_text=context_text,
        messages=[{"role": "user", "content": grading_msg}],
        max_history_messages=0,
    )
    inference_seconds = time.perf_counter() - t0

    return _parse_grading_output(result.reply, paper_type, inference_seconds)
