from __future__ import annotations

import re
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

from llm_service import (
    LLMDisabledError,
    LLMInferenceResult,
    LLMNotConfiguredError,
    LLMServiceError,
    get_llm_service,
)
from prompt_router import route_prompt_mode
from retrieval import format_excerpt_context, retrieve_relevant_excerpts


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

_GRADING_MAX_EXCERPTS = 3
_grading_token_budget: int = 450
_GRADING_RETRIEVE_CANDIDATES = 12


def set_grading_token_budget(n: int) -> None:
    global _grading_token_budget
    _grading_token_budget = max(64, int(n))

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

# TODO: IB PDF cover shows 30 marks but the 4-criteria breakdown here sums to 40.
# Verify correct per-criterion weights (SL vs HL) before changing.
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

_SCORE_RE = re.compile(r"Score:\s*(\d+)")
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
    context_mode: str           # user's requested mode
    context_mode_effective: str # actual mode used (may differ if router overrides)
    raw_output: str
    inference_seconds: float
    prompt: str
    retrieval_modes: list[str] = field(default_factory=list)
    routing_reason: str = ""
    top_semantic_score: float | None = None
    excerpts_per_doc: list[list[dict]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _validate_paper_type(paper_type: str) -> None:
    if paper_type not in CRITERIA_BY_PAPER:
        raise ValueError(f"Invalid paper_type {paper_type!r}. Must be 'paper1' or 'paper2'.")


@dataclass
class MultiDocRetrievalResult:
    context_text: str
    excerpts_per_doc: list[list[dict]]  # one list per work; each item is excerpt.to_dict()
    top_semantic_score: float
    retrieval_modes: list[str]


def retrieve_multi_doc_context(
    chunks_paths: list[str | Path],
    paper_type: str,
    student_answer: str = "",
    question: str = "",
) -> MultiDocRetrievalResult:
    """Retrieve and format excerpts from one or two documents.

    Uses the exam question + student answer as the retrieval query so that
    the returned passages are targeted to what the student actually wrote.
    Falls back to the static topic-keyword query when both are empty (e.g.
    benchmark runs that do not supply a live answer).

    Each document's excerpts are labelled with a '--- Work N ---' header so
    the grading prompt clearly distinguishes the two works.
    """
    _validate_paper_type(paper_type)
    answer_snippet = (question + " " + student_answer).strip()
    query = answer_snippet[:2000] if answer_snippet else GRADING_QUERY[paper_type]

    parts: list[str] = []
    all_excerpts_per_doc: list[list[dict]] = []
    top_semantic: float = 0.0
    modes: list[str] = []

    for idx, chunks_path in enumerate(chunks_paths, start=1):
        excerpts, raw_top, retrieval_mode = retrieve_relevant_excerpts(
            "",  # document_text unused when persisted_chunks_path is given
            query,
            persisted_chunks_path=Path(chunks_path),
            max_excerpts=_GRADING_MAX_EXCERPTS,
            context_token_budget=_grading_token_budget,
            retrieve_candidates=_GRADING_RETRIEVE_CANDIDATES,
        )
        if raw_top:
            top_semantic = max(top_semantic, max(e.semantic_score for e in raw_top))
        label = f"--- Work {idx} ---"
        parts.append(f"{label}\n{format_excerpt_context(excerpts)}")
        all_excerpts_per_doc.append([e.to_debug_dict() for e in excerpts])
        modes.append(retrieval_mode)

    return MultiDocRetrievalResult(
        context_text="\n\n".join(parts),
        excerpts_per_doc=all_excerpts_per_doc,
        top_semantic_score=top_semantic,
        retrieval_modes=modes,
    )


_FEEDBACK_COACHING: dict[str, dict[str, str]] = {
    "paper1": {
        "A": "the student's interpretation of the passage's meaning, themes, and literary context",
        "B": "identification and analysis of specific literary techniques and their effects",
        "C": "clarity and coherence of argument structure and essay organisation",
        "D": "use of precise, accurate literary language and register",
    },
    "paper2": {
        "A": "knowledge and understanding of both works, and the quality of interpretation",
        "B": "comparative analysis of literary features across both works",
        "C": "essay structure, development of a sustained comparative argument",
        "D": "use of precise, accurate literary language and formal register",
    },
}


def _build_output_template(paper_type: str) -> str:
    criteria = CRITERIA_BY_PAPER[paper_type]
    blocks = "\n\n".join(
        f"[Criterion {letter}]\nScore: N/{max_score}\nFeedback: <one sentence>"
        for letter, _label, max_score in criteria
    )
    return blocks + "\n\n[Overall]\nComments: <2\u20134 sentences>"


def _build_criteria_lines(paper_type: str) -> str:
    criteria = CRITERIA_BY_PAPER[paper_type]
    descriptions = _CRITERIA_DESCRIPTIONS[paper_type]
    return "\n".join(
        f"  {letter} \u2013 {label} ({max_score}): {descriptions[letter]}"
        for letter, label, max_score in criteria
    )


def _build_paper1_grading_prompt(
    passage: str,
    guiding_question: str,
    student_answer: str,
) -> str:
    """Build a self-contained grading prompt for Paper 1.

    The unseen passage and guiding question are embedded directly so context and
    instructions are physically adjacent in the token stream.
    """
    return (
        "You are an IB Literature examiner grading a Paper 1 guided literary analysis.\n"
        "The following is an IB specimen marking scheme. These are marking notes, not an\n"
        "exhaustive checklist \u2014 alternative formal or technical approaches that demonstrate\n"
        "the same skills should also be rewarded.\n\n"
        "Output each criterion block in this exact format, then the Overall block:\n\n"
        f"{_build_output_template('paper1')}\n\n"
        f"IB CRITERIA (Paper 1):\n{_build_criteria_lines('paper1')}\n\n"
        f"UNSEEN PASSAGE:\n{passage}\n\n"
        f"GUIDING QUESTION:\n{guiding_question}\n\n"
        f"STUDENT ANSWER:\n{student_answer}"
    )


def _build_paper2_grading_prompt(
    question: str,
    student_answer: str,
    context_mode: str,
    context_text: str,
) -> str:
    """Build a self-contained grading prompt for Paper 2.

    Context (titles or excerpts) is embedded inline so it is physically adjacent
    to the grading instructions in the token stream.
    """
    if context_mode == "titles_only":
        context_header = (
            "The student cannot bring copies of the works into the exam.\n"
            "Draw on your own knowledge of the works listed below to evaluate\n"
            "the student\u2019s textual references and interpretations.\n\n"
            f"WORKS:\n{context_text}"
        )
    else:
        context_header = (
            "The student cannot bring copies of the works into the exam.\n"
            "The excerpts below are passages from the studied works. Use them to\n"
            "verify the accuracy of the student\u2019s textual references.\n\n"
            f"EXCERPTS FROM STUDIED WORKS:\n{context_text}"
        )

    return (
        "You are an IB Literature examiner grading a Paper 2 comparative essay.\n"
        "The following is an IB specimen marking scheme. These are marking notes, not an\n"
        "exhaustive checklist \u2014 alternative formal or technical approaches should also be rewarded.\n\n"
        f"{context_header}\n\n"
        "Output each criterion block in this exact format, then the Overall block:\n\n"
        f"{_build_output_template('paper2')}\n\n"
        f"IB CRITERIA (Paper 2):\n{_build_criteria_lines('paper2')}\n\n"
        f"EXAM QUESTION:\n{question}\n\n"
        f"STUDENT ANSWER:\n{student_answer}"
    )


def _parse_grading_output(
    raw: str,
    paper_type: str,
    context_mode: str,
    context_mode_effective: str,
    inference_seconds: float,
    prompt: str,
    retrieval_result: MultiDocRetrievalResult | None = None,
    routing_reason: str = "",
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
        context_mode_effective=context_mode_effective,
        raw_output=raw,
        inference_seconds=inference_seconds,
        prompt=prompt,
        retrieval_modes=retrieval_result.retrieval_modes if retrieval_result else [],
        routing_reason=routing_reason,
        top_semantic_score=retrieval_result.top_semantic_score if retrieval_result else None,
        excerpts_per_doc=retrieval_result.excerpts_per_doc if retrieval_result else [],
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
    Paper 2, chunks mode: pass ``chunks_paths`` — retrieval from each work using the
      student answer + question as the query. The System 2 router then decides whether
      the retrieved excerpts are relevant enough to include; if not, falls back to
      titles_only (base knowledge).
    Paper 2, titles_only mode: pass ``doc_titles`` — only work titles/authors as context.
    """
    _validate_paper_type(paper_type)

    retrieval_result: MultiDocRetrievalResult | None = None
    routing_reason: str = ""
    context_mode_effective: str = context_mode

    if paper_type == "paper1":
        prompt = _build_paper1_grading_prompt(
            passage=passage_text or "",
            guiding_question=question,
            student_answer=student_answer,
        )
    else:
        if context_mode == "titles_only":
            lines = [f"Work {i + 1}: {t}" for i, t in enumerate(doc_titles or [])]
            context_text = "\n".join(lines) if lines else "(no work information provided)"
            routing_reason = "user_selected_titles_only"
        else:
            retrieval_result = retrieve_multi_doc_context(
                chunks_paths or [],
                paper_type,
                student_answer=student_answer,
                question=question,
            )
            chosen_mode, routing_reason = route_prompt_mode(
                known_work_confidence=None,
                top_semantic_score=retrieval_result.top_semantic_score,
                has_conversation_history=False,
            )
            if chosen_mode == "base_knowledge":
                # Router determined retrieval signal is too weak; fall back to titles
                lines = [f"Work {i + 1}: {t}" for i, t in enumerate(doc_titles or [])]
                context_text = "\n".join(lines) if lines else "(no work information provided)"
                context_mode_effective = "titles_only"
            else:
                context_text = retrieval_result.context_text
                context_mode_effective = "chunks"

        prompt = _build_paper2_grading_prompt(question, student_answer, context_mode_effective, context_text)

    t0 = time.perf_counter()
    result = get_llm_service().generate_raw_reply(prompt)
    inference_seconds = time.perf_counter() - t0

    return _parse_grading_output(
        result.reply,
        paper_type,
        context_mode,
        context_mode_effective,
        inference_seconds,
        prompt,
        retrieval_result=retrieval_result,
        routing_reason=routing_reason,
    )


# --------------------------------------------------------------------------- #
# Per-criterion streaming feedback                                             #
# --------------------------------------------------------------------------- #

def _build_criterion_feedback_prompt(
    *,
    paper_type: str,
    criterion: str,
    criterion_label: str,
    score: int,
    max_score: int,
    student_answer: str,
    passage_text: str | None,
    context_text: str | None,
    question: str | None,
) -> str:
    coaching = _FEEDBACK_COACHING[paper_type][criterion]
    lines = [
        "You are an IB Literature coach providing detailed feedback on ONE criterion only.",
        "",
        f"CRITERION {criterion} \u2014 {criterion_label} ({score}/{max_score})",
        f"Focus area: {coaching}",
        "",
        "TASK:",
        "Write 3-5 sentences of specific, actionable coaching feedback. You must:",
        '1. Quote at least one short phrase from the student\'s answer (use "quotation marks")',
        "2. Explain what the student did well or poorly on this specific criterion",
        "3. Give one concrete suggestion for improvement",
        "Do NOT re-score. Do NOT comment on other criteria.",
    ]

    if paper_type == "paper1" and passage_text:
        lines += ["", "UNSEEN PASSAGE:", passage_text]
    elif paper_type == "paper2" and context_text:
        lines += ["", "RELEVANT EXCERPTS FROM STUDIED WORKS:", context_text]
    elif paper_type == "paper2":
        lines += [
            "",
            "NOTE: No source text is available. The student did not bring texts into the exam.",
            "Ground your feedback entirely in the student's own references and quotations within their answer.",
            "If the student has not quoted or referenced specific passages, note this as a weakness.",
        ]

    if question:
        lines += ["", "EXAM QUESTION:", question]

    lines += ["", "STUDENT ANSWER:", student_answer, "", f"FEEDBACK FOR CRITERION {criterion}:"]
    return "\n".join(lines)


def estimate_feedback_prompt_tokens(
    *,
    paper_type: str,
    criterion: str,
    criterion_label: str,
    score: int,
    max_score: int,
    student_answer: str,
    passage_text: str | None = None,
    context_text: str | None = None,
    question: str | None = None,
) -> int:
    """Return token count for the criterion feedback prompt.

    Uses the model's tokenizer for an exact count when a local model is loaded;
    falls back to words × 1.33 otherwise.
    """
    prompt = _build_criterion_feedback_prompt(
        paper_type=paper_type,
        criterion=criterion,
        criterion_label=criterion_label,
        score=score,
        max_score=max_score,
        student_answer=student_answer,
        passage_text=passage_text,
        context_text=context_text,
        question=question,
    )
    return get_llm_service().count_tokens(prompt)


def stream_criterion_feedback(
    *,
    paper_type: str,
    criterion: str,
    criterion_label: str,
    score: int,
    max_score: int,
    student_answer: str,
    passage_text: str | None = None,
    context_text: str | None = None,
    question: str | None = None,
) -> Generator[str, None, LLMInferenceResult]:
    """Stream detailed coaching feedback for a single IB criterion.

    Yields tokens one by one. Raises StopIteration with an LLMInferenceResult
    as .value when complete. Uses generate_feedback_stream (no stop tokens).
    """
    _validate_paper_type(paper_type)
    prompt = _build_criterion_feedback_prompt(
        paper_type=paper_type,
        criterion=criterion,
        criterion_label=criterion_label,
        score=score,
        max_score=max_score,
        student_answer=student_answer,
        passage_text=passage_text,
        context_text=context_text,
        question=question,
    )
    return (yield from get_llm_service().generate_feedback_stream(prompt))
