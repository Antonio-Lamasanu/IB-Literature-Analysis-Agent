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


def _rag_score_threshold() -> float:
    try:
        return float(os.environ.get("ROUTER_RAG_SCORE_THRESHOLD", "0.30"))
    except (TypeError, ValueError):
        return 0.30


def _default_mode() -> str:
    mode = os.environ.get("ROUTER_DEFAULT_MODE", "rag")
    return mode if mode in ("rag", "base_knowledge") else "rag"


def route_prompt_mode(
    known_work_confidence: float | None,
    top_chunk_score: float | None,
    has_conversation_history: bool,
    document_id: str = "",
) -> tuple[str, str]:
    """Return (mode, reason). Mode is 'base_knowledge' or 'rag'.

    Priority order:
    1. Strong retrieval signal overrides confidence → rag
    2. High corpus confidence → base_knowledge
    3. Ongoing conversation → rag (prefer context continuity)
    4. Default (env var ROUTER_DEFAULT_MODE)
    """
    confidence = known_work_confidence or 0.0
    chunk_score = top_chunk_score or 0.0

    if chunk_score >= _rag_score_threshold():
        mode, reason = "rag", "top_chunk_score_high"
    elif confidence >= _confidence_threshold():
        mode, reason = "base_knowledge", "known_work_confidence_high"
    elif has_conversation_history:
        mode, reason = "rag", "has_conversation_history"
    else:
        mode, reason = _default_mode(), "default"

    logger.info(
        "prompt_router: mode=%s reason=%s doc=%s confidence=%.2f top_chunk=%.3f has_history=%s",
        mode, reason, document_id, confidence, chunk_score, has_conversation_history,
    )
    return mode, reason
