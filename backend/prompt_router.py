"""
prompt_router.py — System 2: automatic prompt mode selection.

Selects 'base_knowledge' or 'rag' before each inference call based on
context-need scoring (query semantics vs. pre-scored question dataset),
corpus confidence, and conversation history.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level embedding cache (populated at startup via preload_context_need_embeddings)
# ---------------------------------------------------------------------------

_embed_lock = threading.Lock()
_dataset_embeddings: Any = None   # numpy ndarray (N, 384) or None
_dataset_scores: list[float] = []
_dataset_questions: list[str] = []
_dataset_categories: list[str] = []


def preload_context_need_embeddings() -> None:
    """Load all scored questions from the DB and pre-compute their embeddings.

    Called once at app startup. Thread-safe — multiple calls are safe but only
    the first actually runs. Subsequent calls return immediately.
    """
    global _dataset_embeddings, _dataset_scores, _dataset_questions, _dataset_categories

    with _embed_lock:
        if _dataset_embeddings is not None:
            return  # already loaded

        try:
            import numpy as np
            from database import get_connection, get_db_path
            from embeddings import encode_texts

            with get_connection(get_db_path()) as conn:
                rows = conn.execute(
                    "SELECT question, score, category FROM query_context_scores ORDER BY id"
                ).fetchall()

            if not rows:
                logger.warning("prompt_router: query_context_scores table is empty — context_need scoring disabled")
                return

            questions = [r["question"] for r in rows]
            scores = [float(r["score"]) for r in rows]
            categories = [r["category"] for r in rows]

            embeddings = encode_texts(questions)  # (N, 384) L2-normalised
            _dataset_questions = questions
            _dataset_scores = scores
            _dataset_categories = categories
            _dataset_embeddings = embeddings
            logger.info("prompt_router: preloaded %d context-need embeddings", len(questions))

        except Exception as exc:
            logger.warning("prompt_router: could not preload context-need embeddings: %s", exc)


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _confidence_threshold() -> float:
    try:
        return float(os.environ.get("ROUTER_CONFIDENCE_THRESHOLD", "0.65"))
    except (TypeError, ValueError):
        return 0.65


def _rag_semantic_threshold() -> float:
    """Fallback cosine similarity threshold used when context_need scoring is unavailable."""
    try:
        return float(os.environ.get("ROUTER_RAG_SEMANTIC_THRESHOLD", "0.25"))
    except (TypeError, ValueError):
        return 0.25


def _context_need_threshold() -> float:
    try:
        return float(os.environ.get("ROUTER_CONTEXT_NEED_THRESHOLD", "0.45"))
    except (TypeError, ValueError):
        return 0.45


def _context_need_top_k() -> int:
    try:
        return max(1, int(os.environ.get("ROUTER_CONTEXT_NEED_TOP_K", "5")))
    except (TypeError, ValueError):
        return 5


def _default_mode() -> str:
    mode = os.environ.get("ROUTER_DEFAULT_MODE", "rag")
    return mode if mode in ("rag", "base_knowledge") else "rag"


# ---------------------------------------------------------------------------
# Context-need scoring
# ---------------------------------------------------------------------------

def score_context_need(query: str) -> tuple[float, list[dict]]:
    """Score how much document context this query requires (0.0–1.0).

    Finds the top-K nearest questions in the pre-loaded dataset (by cosine
    similarity) and returns their similarity-weighted average score.

    Returns:
        (weighted_score, top_k_matches)
        where top_k_matches is a list of dicts:
        {"question": str, "category": str, "similarity": float, "score": float}

    Returns (0.5, []) if embeddings are not loaded yet.
    """
    if _dataset_embeddings is None:
        return 0.5, []

    try:
        from embeddings import encode_query

        q_vec = encode_query(query)  # (1, 384) L2-normalised
        if q_vec.shape[0] != 1:
            return 0.5, []

        sims = (_dataset_embeddings @ q_vec[0]).tolist()  # (N,)

        k = _context_need_top_k()
        top_indices = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]

        top_matches = [
            {
                "question": _dataset_questions[i],
                "category": _dataset_categories[i],
                "similarity": round(float(sims[i]), 4),
                "score": _dataset_scores[i],
            }
            for i in top_indices
        ]

        # Weighted average — clamp negatives to 0 so irrelevant matches don't distort the score
        total_weight = sum(max(m["similarity"], 0.0) for m in top_matches)
        if total_weight <= 0:
            weighted = float(sum(m["score"] for m in top_matches) / len(top_matches))
        else:
            weighted = sum(
                m["score"] * max(m["similarity"], 0.0) for m in top_matches
            ) / total_weight

        return round(float(weighted), 4), top_matches

    except Exception as exc:
        logger.warning("prompt_router: score_context_need failed: %s", exc)
        return 0.5, []


# ---------------------------------------------------------------------------
# Public router
# ---------------------------------------------------------------------------

def route_prompt_mode(
    known_work_confidence: float | None,
    top_semantic_score: float | None,
    has_conversation_history: bool,
    document_id: str = "",
    rag_semantic_threshold: float | None = None,
    query: str | None = None,
) -> tuple[str, str, dict]:
    """Return (mode, reason, routing_debug). Mode is 'base_knowledge' or 'rag'.

    Priority order when query is provided:
    1. context_need_score >= ROUTER_CONTEXT_NEED_THRESHOLD → rag
    2. known_work_confidence >= ROUTER_CONFIDENCE_THRESHOLD → base_knowledge
    3. Ongoing conversation → rag (context continuity)
    4. Default (ROUTER_DEFAULT_MODE)

    When query is None (e.g. benchmark sweep), falls back to top_semantic_score threshold.

    routing_debug contains all intermediate values for the frontend debug panel.
    """
    confidence = known_work_confidence or 0.0
    sem_score = top_semantic_score or 0.0
    threshold = rag_semantic_threshold if rag_semantic_threshold is not None else _rag_semantic_threshold()
    need_threshold = _context_need_threshold()

    context_need_score: float | None = None
    top_k_matches: list[dict] = []

    if query is not None:
        context_need_score, top_k_matches = score_context_need(query)

    # Routing decision
    if context_need_score is not None:
        if context_need_score >= need_threshold:
            mode, reason = "rag", "context_need_high"
        elif confidence >= _confidence_threshold():
            mode, reason = "base_knowledge", "known_work_confidence_high"
        elif has_conversation_history:
            mode, reason = "rag", "has_conversation_history"
        else:
            mode, reason = _default_mode(), "default"
    else:
        # Legacy path: no query provided, fall back to semantic score
        if sem_score >= threshold:
            mode, reason = "rag", "top_semantic_score_high"
        elif confidence >= _confidence_threshold():
            mode, reason = "base_knowledge", "known_work_confidence_high"
        elif has_conversation_history:
            mode, reason = "rag", "has_conversation_history"
        else:
            mode, reason = _default_mode(), "default"

    routing_debug: dict = {
        "mode": mode,
        "reason": reason,
        "context_need_score": context_need_score,
        "context_need_threshold": need_threshold,
        "known_work_confidence": round(confidence, 4),
        "confidence_threshold": _confidence_threshold(),
        "top_semantic_score": round(sem_score, 4),
        "semantic_threshold": threshold,
        "has_conversation_history": has_conversation_history,
        "top_k_matches": top_k_matches,
    }

    logger.info(
        "prompt_router: mode=%s reason=%s doc=%s confidence=%.2f context_need=%s top_semantic=%.3f has_history=%s",
        mode,
        reason,
        document_id,
        confidence,
        f"{context_need_score:.3f}" if context_need_score is not None else "n/a",
        sem_score,
        has_conversation_history,
    )
    return mode, reason, routing_debug
