from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "google/embeddinggemma-300m"
EMBEDDING_DIM = 768
_QUERY_PROMPT = "task: search result | query: "
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
    Encode document passages to L2-normalised float32 embeddings of shape (N, 768).

    No task prompt is applied — use encode_query() for retrieval queries.
    Because embeddings are L2-normalised, dot product == cosine similarity.
    Returns an empty (0, 768) array if the model is unavailable or texts is empty.
    """
    EMPTY = np.empty((0, 768), dtype=np.float32)
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


def encode_query(text: str) -> np.ndarray:
    """
    Encode a single retrieval query with the required task prompt prefix.

    EmbeddingGemma requires task prompts on queries but not on documents.
    Returns shape (1, 768), L2-normalised. Returns empty (0, 768) if model unavailable.
    """
    EMPTY = np.empty((0, 768), dtype=np.float32)
    if not text:
        return EMPTY
    model = _get_model()
    if model is None:
        return EMPTY
    return model.encode(
        [_QUERY_PROMPT + text],
        batch_size=1,
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
