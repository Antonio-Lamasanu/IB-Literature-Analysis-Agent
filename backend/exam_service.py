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

_GRADING_MAX_EXCERPTS = 3
_GRADING_TOKEN_BUDGET = 450
_GRADING_RETRIEVE_CANDIDATES = 12

# BM25 query strings used when retrieving context for grading
GRADING_QUERY: dict[str, str] = {
    "paper1": "language tone imagery literary techniques psychological state",
    "paper2": "themes motifs narrative voice symbolism setting memory appearance",
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
    context_mode: str  # "chunks" | "titles_only"
    raw_output: str
    inference_seconds: float
    prompt: str


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _validate_paper_type(paper_type: str) -> None:
    if paper_type not in CRITERIA_BY_PAPER:
        raise ValueError(f"Invalid paper_type {paper_type!r}. Must be 'paper1' or 'paper2'.")


def retrieve_multi_doc_context(chunks_paths: list[str | Path], paper_type: str) -> str:
    """Retrieve and format BM25 excerpts from one or two documents.

    Each document's excerpts are labelled with a '--- Work N ---' header so the
    grading prompt clearly distinguishes the two works.
    """
    _validate_paper_type(paper_type)
    query = GRADING_QUERY[paper_type]
    parts: list[str] = []

    for idx, chunks_path in enumerate(chunks_paths, start=1):
        excerpts, _mode = retrieve_relevant_excerpts(
            "",  # document_text unused when persisted_chunks_path is given
            query,
            persisted_chunks_path=Path(chunks_path),
            max_excerpts=_GRADING_MAX_EXCERPTS,
            context_token_budget=_GRADING_TOKEN_BUDGET,
            retrieve_candidates=_GRADING_RETRIEVE_CANDIDATES,
        )
        label = f"--- Work {idx} ---"
        parts.append(f"{label}\n{format_excerpt_context(excerpts)}")

    return "\n\n".join(parts)


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

    return (
        f"You are an IB Literature examiner. Grade the student's response strictly using the\n"
        f"four criteria below. Output each block in this exact format:\n\n"
        f"{output_blocks}\n\n"
        f"[Overall]\n"
        f"Comments: <2\u20134 sentences>\n\n"
        f"IB CRITERIA ({paper_label}):\n"
        f"{criteria_lines}\n\n"
        f"EXAM QUESTION:\n{question}\n\n"
        f"STUDENT ANSWER:\n{student_answer}"
    )


def _parse_grading_output(
    raw: str,
    paper_type: str,
    context_mode: str,
    inference_seconds: float,
    prompt: str,
) -> GradingResult:
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
        context_mode=context_mode,
        raw_output=raw,
        inference_seconds=inference_seconds,
        prompt=prompt,
    )


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def grade_answer(
    *,
    paper_type: str,
    question: str,
    student_answer: str,
    passage_text: str | None = None,
    chunks_paths: list[str | Path] | None = None,
    context_mode: str = "chunks",
    doc_titles: list[str] | None = None,
) -> GradingResult:
    """Grade a student answer against IB criteria for the given paper type.

    Paper 1: pass ``passage_text`` — the hardcoded unseen passage is used directly.
    Paper 2, chunks mode: pass ``chunks_paths`` — BM25 retrieval from each work.
    Paper 2, titles_only mode: pass ``doc_titles`` — only work titles/authors as context.
    """
    _validate_paper_type(paper_type)

    if paper_type == "paper1":
        context_text = passage_text or ""
    elif context_mode == "titles_only":
        lines = [f"Work {i + 1}: {t}" for i, t in enumerate(doc_titles or [])]
        context_text = "\n".join(lines) if lines else "(no work information provided)"
    else:
        context_text = retrieve_multi_doc_context(chunks_paths or [], paper_type)

    grading_msg = _build_grading_user_msg(question, student_answer, paper_type)

    t0 = time.perf_counter()
    result = get_llm_service().generate_reply_with_debug(
        document_text=context_text,
        messages=[{"role": "user", "content": grading_msg}],
        max_history_messages=0,
    )
    inference_seconds = time.perf_counter() - t0

    return _parse_grading_output(result.reply, paper_type, context_mode, inference_seconds, grading_msg)
