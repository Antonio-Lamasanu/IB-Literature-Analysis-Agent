from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
_model = None  # None = not tried; False = failed/unavailable
_model_lock = threading.Lock()


def _get_model():
    """Lazy-load SentenceTransformer; returns None if unavailable."""
    global _model
    if _model is not None:
        return _model if _model is not False else None
    with _model_lock:
        if _model is not None:
            return _model if _model is not False else None
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(_MODEL_NAME)
            logger.info("SentenceTransformer '%s' loaded.", _MODEL_NAME)
        except Exception as exc:
            logger.warning(
                "sentence-transformers unavailable (%s); semantic search disabled.", exc
            )
            _model = False
    return _model if _model is not False else None


def encode_texts(texts: list[str]) -> np.ndarray:
    """
    Encode texts to L2-normalised float32 embeddings of shape (N, 384).

    Because embeddings are L2-normalised, dot product == cosine similarity.
    Returns an empty (0, 384) array if the model is unavailable or texts is empty.
    """
    EMPTY = np.empty((0, 384), dtype=np.float32)
    if not texts:
        return EMPTY
    model = _get_model()
    if model is None:
        return EMPTY
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def embeddings_path_for_chunks(chunks_path: str | Path) -> Path:
    """
    Return the path where embeddings are stored for a given chunks file.

    Example: foo.chunks.jsonl  →  foo.chunks.embeddings.npy
    """
    p = Path(chunks_path)
    return p.parent / f"{p.stem}.embeddings.npy"
