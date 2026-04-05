"""
prompt_router.py — System 2: automatic prompt mode selection.

Selects 'base_knowledge' or 'rag' before each inference call based on corpus
confidence, retrieval signal strength, and conversation history.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _confidence_threshold() -> float:
    try:
        return float(os.environ.get("ROUTER_CONFIDENCE_THRESHOLD", "0.65"))
    except (TypeError, ValueError):
        return 0.65


def _rag_semantic_threshold() -> float:
    """Cosine similarity threshold for RAG selection.

    Uses raw cosine similarity (not normalized hybrid score) to avoid false
    positives from min-max normalization inflating all chunk scores.
    Default 0.25 requires genuine semantic overlap between query and chunk.
    """
    try:
        return float(os.environ.get("ROUTER_RAG_SEMANTIC_THRESHOLD", "0.25"))
    except (TypeError, ValueError):
        return 0.25


def _default_mode() -> str:
    mode = os.environ.get("ROUTER_DEFAULT_MODE", "rag")
    return mode if mode in ("rag", "base_knowledge") else "rag"


def route_prompt_mode(
    known_work_confidence: float | None,
    top_semantic_score: float | None,
    has_conversation_history: bool,
    document_id: str = "",
    rag_semantic_threshold: float | None = None,
) -> tuple[str, str]:
    """Return (mode, reason). Mode is 'base_knowledge' or 'rag'.

    Priority order:
    1. Strong semantic retrieval signal overrides confidence → rag
    2. High corpus confidence → base_knowledge
    3. Ongoing conversation → rag (prefer context continuity)
    4. Default (env var ROUTER_DEFAULT_MODE)

    top_semantic_score is the raw cosine similarity of the top chunk (not the
    normalized hybrid score), so it genuinely reflects query-chunk relevance.

    rag_semantic_threshold overrides ROUTER_RAG_SEMANTIC_THRESHOLD env var when
    provided (used by the benchmark to sweep multiple threshold values).
    """
    confidence = known_work_confidence or 0.0
    sem_score = top_semantic_score or 0.0
    threshold = rag_semantic_threshold if rag_semantic_threshold is not None else _rag_semantic_threshold()

    if sem_score >= threshold:
        mode, reason = "rag", "top_semantic_score_high"
    elif confidence >= _confidence_threshold():
        mode, reason = "base_knowledge", "known_work_confidence_high"
    elif has_conversation_history:
        mode, reason = "rag", "has_conversation_history"
    else:
        mode, reason = _default_mode(), "default"

    logger.info(
        "prompt_router: mode=%s reason=%s doc=%s confidence=%.2f top_semantic=%.3f has_history=%s",
        mode, reason, document_id, confidence, sem_score, has_conversation_history,
    )
    return mode, reason
