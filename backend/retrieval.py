from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from chat_history import ChatHistoryTurn
from chunking import ChunkParams, build_chunks, estimate_tokens, get_spacy_nlp, parse_units_from_marked_text


TOKEN_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "s",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}
CHAT_EXCERPT_PARAMS = ChunkParams(
    target_tokens=120,
    overlap_tokens=20,
    min_tokens=40,
    max_tokens=180,
    include_raw=False,
)
DEFAULT_MAX_EXCERPTS = 4
DEFAULT_RETRIEVE_CANDIDATES = 12
DEFAULT_CONTEXT_TOKEN_BUDGET = 900
DEFAULT_FALLBACK_CHARS = 2_500
BM25_K1 = 1.5
BM25_B = 0.75
MAX_CACHE_ITEMS = 24

_DIALOGUE_QUERY_TERMS = {"dialogue", "conversation", "speak", "speaks", "speech", "talk", "talks", "quote", "quoted", "said"}
_DESCRIPTION_QUERY_TERMS = {"describe", "description", "look", "looks", "scene", "setting", "appearance"}
_NARRATION_QUERY_TERMS = {"happen", "happens", "happened", "story", "plot", "narrate", "narration"}

_chunk_record_cache: OrderedDict[tuple[str, str, int, int], list[dict[str, Any]]] = OrderedDict()
_retrieval_corpus_cache: OrderedDict[tuple[str, str, int, int], "RetrievalCorpus"] = OrderedDict()
_history_emb_cache: OrderedDict[tuple, "Any"] = OrderedDict()  # cache key → np.ndarray | None
_cache_lock = threading.Lock()


@dataclass(frozen=True)
class RetrievedExcerpt:
    excerpt_id: str
    heading: str
    page_start: int
    page_end: int
    text: str
    score: float
    token_estimate: int
    semantic_score: float = 0.0  # raw cosine similarity; used as routing signal (unaffected by normalization/boost)

    def to_debug_dict(self) -> dict:
        return {
            "excerpt_id": self.excerpt_id,
            "score": self.score,
            "semantic_score": self.semantic_score,
            "heading": self.heading or "UNKNOWN",
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
        }


@dataclass(frozen=True)
class ChatContextResult:
    context: str
    retrieval_mode: str
    retrieved_excerpts: list[RetrievedExcerpt]
    retrieved_history: list["RetrievedHistoryTurn"] = field(default_factory=list)
    retrieved_chunk_candidates: list[RetrievedExcerpt] = field(default_factory=list)
    raw_retrieved_excerpts: list[RetrievedExcerpt] = field(default_factory=list)
    final_candidate_mix: list["RetrievedCandidate"] = field(default_factory=list)
    history_checked: bool = False
    history_reuse_hit: bool = False
    history_top_score: float | None = None
    history_threshold: float | None = None
    reused_turn_id: str | None = None
    reused_answer: str | None = None
    chunk_retrieval_mode: str | None = None


@dataclass(frozen=True)
class RetrievedHistoryTurn:
    turn_id: str
    document_id: str
    created_at: str
    user_query: str
    assistant_answer: str
    score: float
    token_estimate: int
    answer_reused_from_history: bool
    reused_from_turn_id: str | None = None

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "created_at": self.created_at,
            "score": self.score,
            "user_query": self.user_query,
            "assistant_answer": self.assistant_answer,
            "answer_reused_from_history": self.answer_reused_from_history,
            "reused_from_turn_id": self.reused_from_turn_id,
        }


@dataclass(frozen=True)
class RetrievedCandidate:
    source: str
    source_id: str
    score: float
    token_estimate: int
    text: str
    heading: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    turn_id: str | None = None
    created_at: str | None = None
    user_query: str | None = None
    assistant_answer: str | None = None

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "score": self.score,
            "heading": self.heading,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "turn_id": self.turn_id,
            "created_at": self.created_at,
            "user_query": self.user_query,
            "assistant_answer": self.assistant_answer,
            "text": self.text,
        }


@dataclass(frozen=True)
class ExcerptFeature:
    excerpt: RetrievedExcerpt
    term_freq: Counter[str]
    heading_terms: frozenset[str]
    character_terms: frozenset[str]
    location_terms: frozenset[str]
    organization_terms: frozenset[str]
    event_terms: frozenset[str]
    token_set: frozenset[str]
    document_length: int
    unit_type: str
    has_dialogue: bool
    dialogue_ratio: float


@dataclass(frozen=True)
class HistoryFeature:
    turn: RetrievedHistoryTurn
    combined_term_freq: Counter[str]
    query_term_freq: Counter[str]
    answer_term_freq: Counter[str]
    token_set: frozenset[str]
    document_length: int
    query_length: int
    answer_length: int


@dataclass(frozen=True)
class RetrievalCorpus:
    features: tuple[ExcerptFeature, ...]
    doc_freq: Counter[str]
    avg_doc_length: float
    source_mode: str
    chunk_embeddings: "Any" = None  # np.ndarray shape (N, 384), row i = features[i]; None = BM25-only


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_query_for_match(q: str) -> str:
    """Normalize a query for exact-match comparison: lowercase, strip punctuation, collapse whitespace."""
    q = (q or "").strip().lower()
    q = re.sub(r"[^\w\s]", "", q)
    return re.sub(r"\s+", " ", q).strip()


def _tokenize(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_RE.findall((text or "").lower()):
        if len(token) < 3:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _remember_cache_value(cache: OrderedDict, key: tuple[str, str, int, int], value: Any) -> Any:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > MAX_CACHE_ITEMS:
        cache.popitem(last=False)
    return value


def _build_chunk_cache_key(
    path: str | Path,
    *,
    document_id: str | None = None,
) -> tuple[str, str, int, int] | None:
    chunk_path = Path(path)
    if not chunk_path.exists():
        return None

    resolved = chunk_path.resolve()
    stat = resolved.stat()

    emb_mtime = 0
    try:
        from embeddings import embeddings_path_for_chunks
        emb_path = embeddings_path_for_chunks(resolved)
        if emb_path.exists():
            emb_mtime = emb_path.stat().st_mtime_ns
    except Exception:
        pass

    return (
        document_id or resolved.name,
        str(resolved),
        stat.st_mtime_ns ^ emb_mtime,
        stat.st_size,
    )


def _build_history_emb_cache_key(
    history_json_path: Path,
    emb_path: Path,
    document_id: str,
) -> tuple[str, str, int, int, int]:
    json_mtime = history_json_path.stat().st_mtime_ns if history_json_path.exists() else 0
    json_size = history_json_path.stat().st_size if history_json_path.exists() else 0
    emb_mtime = emb_path.stat().st_mtime_ns if emb_path.exists() else 0
    return (document_id, str(history_json_path.resolve()), json_mtime, emb_mtime, json_size)


def load_history_embeddings(
    document_id: str,
    storage_dir: str | Path,
) -> "Any":
    """Load history embeddings from disk with mtime-based cache invalidation.

    Returns an np.ndarray of shape (N, 384) aligned with the history turns in the JSON,
    or None if the file doesn't exist or is unreadable.
    """
    from chat_history import history_embeddings_path, get_history_path
    storage_dir = Path(storage_dir)
    history_json_path = get_history_path(document_id, storage_dir)
    emb_path = history_embeddings_path(document_id, storage_dir)

    if not emb_path.exists():
        return None

    cache_key = _build_history_emb_cache_key(history_json_path, emb_path, document_id)
    with _cache_lock:
        cached = _history_emb_cache.get(cache_key)
        if cached is not None:
            _history_emb_cache.move_to_end(cache_key)
            return cached

    try:
        import numpy as np
        arr = np.load(str(emb_path)).astype(np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        from embeddings import EMBEDDING_DIM
        if arr.shape[1] != EMBEDDING_DIM:
            logger.warning(
                "Stale history embeddings (dim=%d, expected=%d) — skipping history search",
                arr.shape[1], EMBEDDING_DIM,
            )
            return None
    except Exception as exc:
        logger.warning("Failed to load history embeddings from %s: %s", emb_path, exc)
        return None

    with _cache_lock:
        return _remember_cache_value(_history_emb_cache, cache_key, arr)


def load_persisted_chunks(
    path: str | Path,
    *,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    cache_key = _build_chunk_cache_key(path, document_id=document_id)
    if cache_key is None:
        return []

    with _cache_lock:
        cached = _chunk_record_cache.get(cache_key)
        if cached is not None:
            _chunk_record_cache.move_to_end(cache_key)
            return cached

    chunk_path = Path(path)
    records: list[dict[str, Any]] = []
    with chunk_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)

    with _cache_lock:
        return _remember_cache_value(_chunk_record_cache, cache_key, records)


def _chapter_heading_from_value(value: Any) -> str:
    if value is None:
        return "Chapter unknown"
    if isinstance(value, (int, float)) and int(value) == value:
        return f"Chapter {int(value)}"
    compact = _normalize_space(str(value))
    return compact or "Chapter unknown"


def _build_chunk_excerpt_feature(index: int, chunk: dict[str, Any]) -> ExcerptFeature | None:
    position = chunk.get("position") or {}
    content = chunk.get("content") or {}
    metadata = chunk.get("metadata") or {}
    text = str(content.get("text") or "").strip()
    if not text:
        return None

    page_start = int(position.get("page_start") or 0)
    page_end = int(position.get("page_end") or page_start or 0)
    heading = _chapter_heading_from_value(position.get("chapter"))
    token_estimate = int(content.get("token_estimate") or estimate_tokens(text))
    excerpt = RetrievedExcerpt(
        excerpt_id=str(chunk.get("unit_id") or f"chunk-{index}"),
        heading=heading,
        page_start=page_start,
        page_end=page_end,
        text=text,
        score=0.0,
        token_estimate=max(1, token_estimate),
    )

    term_freq = Counter(_tokenize(text))
    document_length = max(1, sum(term_freq.values()))
    heading_terms = frozenset(_tokenize(heading))
    character_terms = frozenset(
        token
        for item in metadata.get("character_mentions") or []
        for token in _tokenize(str(item))
    )
    location_terms = frozenset(
        token
        for item in metadata.get("location_mentions") or []
        for token in _tokenize(str(item))
    )
    organization_terms = frozenset(
        token
        for item in metadata.get("organization_mentions") or []
        for token in _tokenize(str(item))
    )
    event_terms = frozenset(
        token
        for item in metadata.get("event_mentions") or []
        for token in _tokenize(str(item))
    )

    return ExcerptFeature(
        excerpt=excerpt,
        term_freq=term_freq,
        heading_terms=heading_terms,
        character_terms=character_terms,
        location_terms=location_terms,
        organization_terms=organization_terms,
        event_terms=event_terms,
        token_set=frozenset(term_freq),
        document_length=document_length,
        unit_type=str(metadata.get("unit_type") or "unknown").strip().lower() or "unknown",
        has_dialogue=bool(metadata.get("has_dialogue", False)),
        dialogue_ratio=float(metadata.get("dialogue_ratio") or 0.0),
    )


def _build_retrieval_corpus_from_chunks(chunk_records: list[dict[str, Any]]) -> RetrievalCorpus:
    features: list[ExcerptFeature] = []
    doc_freq: Counter[str] = Counter()
    total_length = 0

    for index, chunk in enumerate(chunk_records, start=1):
        feature = _build_chunk_excerpt_feature(index, chunk)
        if feature is None:
            continue
        features.append(feature)
        total_length += feature.document_length
        doc_freq.update(feature.token_set)

    avg_doc_length = total_length / len(features) if features else 1.0
    return RetrievalCorpus(
        features=tuple(features),
        doc_freq=doc_freq,
        avg_doc_length=max(1.0, avg_doc_length),
        source_mode="persisted_chunks",
    )


def _load_chunk_embeddings(chunks_path: str | Path) -> "Any":
    """Load .embeddings.npy next to chunks_path; return None if absent or unreadable."""
    try:
        import numpy as np
        from embeddings import embeddings_path_for_chunks
        emb_path = embeddings_path_for_chunks(chunks_path)
        if not emb_path.exists():
            return None
        arr = np.load(str(emb_path)).astype(np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        from embeddings import EMBEDDING_DIM
        if arr.shape[1] != EMBEDDING_DIM:
            logger.warning(
                "Stale chunk embeddings (dim=%d, expected=%d) at %s — falling back to BM25 only",
                arr.shape[1], EMBEDDING_DIM, emb_path,
            )
            return None
        return arr
    except Exception as exc:
        logger.warning("Failed to load embeddings from %s: %s", chunks_path, exc)
        return None


def get_persisted_chunk_corpus(
    path: str | Path,
    *,
    document_id: str | None = None,
) -> RetrievalCorpus | None:
    cache_key = _build_chunk_cache_key(path, document_id=document_id)
    if cache_key is None:
        return None

    with _cache_lock:
        cached = _retrieval_corpus_cache.get(cache_key)
        if cached is not None:
            _retrieval_corpus_cache.move_to_end(cache_key)
            return cached

    records = load_persisted_chunks(path, document_id=document_id)
    corpus = _build_retrieval_corpus_from_chunks(records)

    chunk_embeddings = _load_chunk_embeddings(path)
    if chunk_embeddings is not None:
        if chunk_embeddings.shape[0] == len(corpus.features):
            corpus = RetrievalCorpus(
                features=corpus.features,
                doc_freq=corpus.doc_freq,
                avg_doc_length=corpus.avg_doc_length,
                source_mode=corpus.source_mode,
                chunk_embeddings=chunk_embeddings,
            )
        else:
            logger.warning(
                "EMBEDDING SYNC MISMATCH for %s: embeddings have %d rows but corpus has "
                "%d chunks. Re-upload or run scripts/regenerate_embeddings.py to fix. "
                "Using BM25-only.",
                path, chunk_embeddings.shape[0], len(corpus.features),
            )

    with _cache_lock:
        return _remember_cache_value(_retrieval_corpus_cache, cache_key, corpus)


def _build_candidate_excerpts_from_text(document_text: str) -> list[RetrievedExcerpt]:
    units = parse_units_from_marked_text(
        document_text,
        doc_id="chat_document",
        drop_noise_paras=True,
    )
    chunks = build_chunks(units, CHAT_EXCERPT_PARAMS)

    excerpts: list[RetrievedExcerpt] = []
    for chunk in chunks:
        text = str(chunk.content.get("text") or "").strip()
        if not text:
            continue
        position = chunk.position or {}
        heading = _chapter_heading_from_value(position.get("chapter"))
        if heading == "Chapter unknown" and chunk.chapter_label:
            heading = chunk.chapter_label
        excerpts.append(
            RetrievedExcerpt(
                excerpt_id=chunk.unit_id,
                heading=heading,
                page_start=int(position.get("page_start") or 0),
                page_end=int(position.get("page_end") or 0),
                text=text,
                score=0.0,
                token_estimate=max(1, int(chunk.content.get("token_estimate") or estimate_tokens(text))),
            )
        )
    return excerpts


def _legacy_score_excerpt(excerpt: RetrievedExcerpt, query: str, query_terms: Counter[str]) -> float:
    if not query_terms:
        return 0.0

    excerpt_terms = Counter(_tokenize(excerpt.text))
    heading_terms = set(_tokenize(excerpt.heading))
    overlap = sum(min(count, excerpt_terms.get(term, 0)) for term, count in query_terms.items())
    if overlap <= 0:
        overlap = len(set(query_terms) & set(excerpt_terms))

    heading_bonus = 0.3 * sum(1 for term in query_terms if term in heading_terms)
    exact_match_bonus = 0.0
    normalized_query = _normalize_space(query).lower()
    if len(normalized_query) >= 8 and normalized_query in _normalize_space(excerpt.text).lower():
        exact_match_bonus = 0.75

    return float(overlap) + heading_bonus + exact_match_bonus


def _legacy_retrieve_relevant_excerpts(
    document_text: str,
    latest_user_question: str,
    *,
    max_excerpts: int,
) -> tuple[list[RetrievedExcerpt], str]:
    candidates = _build_candidate_excerpts_from_text(document_text)
    query_terms = Counter(_tokenize(latest_user_question))
    if not candidates or not query_terms:
        return [], "legacy_text_fallback"

    scored: list[RetrievedExcerpt] = []
    for candidate in candidates:
        score = _legacy_score_excerpt(candidate, latest_user_question, query_terms)
        if score <= 0:
            continue
        scored.append(
            RetrievedExcerpt(
                excerpt_id=candidate.excerpt_id,
                heading=candidate.heading,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
                text=candidate.text,
                score=score,
                token_estimate=candidate.token_estimate,
            )
        )

    scored.sort(key=lambda item: (-item.score, item.page_start, item.excerpt_id))
    return scored[: max(1, max_excerpts)], "legacy_text_lexical_fallback"


def _bm25_idf(total_docs: int, doc_freq: int) -> float:
    numerator = total_docs - doc_freq + 0.5
    denominator = doc_freq + 0.5
    return math.log(1.0 + (numerator / max(0.5, denominator)))


def _score_bm25_feature(
    feature: ExcerptFeature,
    query_terms: Counter[str],
    *,
    doc_freq: Counter[str],
    total_docs: int,
    avg_doc_length: float,
) -> float:
    return _score_bm25_term_freq(
        feature.term_freq,
        feature.document_length,
        query_terms,
        doc_freq=doc_freq,
        total_docs=total_docs,
        avg_doc_length=avg_doc_length,
    )


def _score_bm25_term_freq(
    term_freq: Counter[str],
    document_length: int,
    query_terms: Counter[str],
    *,
    doc_freq: Counter[str],
    total_docs: int,
    avg_doc_length: float,
) -> float:
    if not query_terms:
        return 0.0

    score = 0.0
    doc_length = max(1, document_length)
    for term, query_count in query_terms.items():
        tf = term_freq.get(term, 0)
        if tf <= 0:
            continue
        idf = _bm25_idf(total_docs, doc_freq.get(term, 0))
        numerator = tf * (BM25_K1 + 1.0)
        denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_length / max(1.0, avg_doc_length)))
        score += idf * (numerator / denominator) * query_count
    return score


def _metadata_boost(feature: ExcerptFeature, query: str, query_terms: Counter[str]) -> float:
    if not query_terms:
        return 0.0

    query_term_set = set(query_terms)
    normalized_query = _normalize_space(query).lower()
    boost = 0.0

    heading_matches = len(query_term_set & feature.heading_terms)
    if heading_matches:
        boost += min(0.45, 0.12 * heading_matches)

    character_matches = len(query_term_set & feature.character_terms)
    if character_matches:
        boost += min(0.35, 0.18 + 0.05 * character_matches)

    location_matches = len(query_term_set & feature.location_terms)
    if location_matches:
        boost += min(0.35, 0.15 + 0.05 * location_matches)

    event_matches = len(query_term_set & feature.event_terms)
    if event_matches:
        boost += min(0.25, 0.10 + 0.05 * event_matches)

    organization_matches = len(query_term_set & feature.organization_terms)
    if organization_matches:
        boost += min(0.15, 0.07 + 0.03 * organization_matches)

    if query_term_set & _DIALOGUE_QUERY_TERMS and feature.has_dialogue:
        boost += 0.18

    if query_term_set & _DESCRIPTION_QUERY_TERMS and feature.unit_type == "description":
        boost += 0.12
    elif query_term_set & _NARRATION_QUERY_TERMS and feature.unit_type == "narration":
        boost += 0.12
    elif query_term_set & _DIALOGUE_QUERY_TERMS and feature.unit_type in {"dialogue", "mixed"}:
        boost += 0.12

    excerpt_text = _normalize_space(feature.excerpt.text).lower()
    if len(normalized_query) >= 8 and normalized_query in excerpt_text:
        boost += 0.35

    return min(1.1, boost)


def _looks_near_duplicate(candidate: RetrievedExcerpt, selected: list[RetrievedExcerpt]) -> bool:
    candidate_terms = set(_tokenize(candidate.text))
    if not candidate_terms:
        return False

    for existing in selected:
        if candidate.excerpt_id == existing.excerpt_id:
            return True

        existing_terms = set(_tokenize(existing.text))
        if not existing_terms:
            continue
        overlap = len(candidate_terms & existing_terms) / max(1, len(candidate_terms | existing_terms))
        if (
            candidate.page_start == existing.page_start
            and candidate.page_end == existing.page_end
            and candidate.heading == existing.heading
            and overlap >= 0.45
        ):
            return True
        if overlap >= 0.82:
            return True

    return False


BM25_WEIGHT = 0.55
SEMANTIC_WEIGHT = 0.35
_HYBRID_EPS = 1e-9


def _rank_persisted_excerpt_features(
    corpus: RetrievalCorpus,
    latest_user_question: str,
    *,
    retrieve_candidates: int,
) -> list[RetrievedExcerpt]:
    query_terms = Counter(_tokenize(latest_user_question))
    if not corpus.features or not query_terms:
        return []

    total_docs = len(corpus.features)

    # BM25 pass — compute raw scores for all features up front
    bm25_scores = [
        _score_bm25_feature(
            feature,
            query_terms,
            doc_freq=corpus.doc_freq,
            total_docs=total_docs,
            avg_doc_length=corpus.avg_doc_length,
        )
        for feature in corpus.features
    ]

    # Semantic pass — only when embeddings are available
    cosine_scores: list[float] | None = None
    if corpus.chunk_embeddings is not None:
        try:
            import numpy as np
            from embeddings import encode_query
            q_vec = encode_query(latest_user_question)
            if q_vec.shape[0] == 1:
                cosine_scores = np.clip(
                    corpus.chunk_embeddings @ q_vec[0], -1.0, 1.0
                ).tolist()
        except Exception as exc:
            logger.warning("Semantic scoring failed (non-fatal): %s", exc)

    # Min-max normalise BM25 across the full candidate set so it maps to [0, 1]
    # before combining with cosine similarity (already in [-1, 1]).
    min_bm25 = min(bm25_scores)
    max_bm25 = max(bm25_scores)
    bm25_range = max_bm25 - min_bm25 + _HYBRID_EPS

    ranked: list[RetrievedExcerpt] = []
    for i, feature in enumerate(corpus.features):
        bm25_raw = bm25_scores[i]
        cos = cosine_scores[i] if cosine_scores is not None else 0.0  # raw cosine; preserved for routing signal

        if cosine_scores is not None:
            norm_bm25 = (bm25_raw - min_bm25) / bm25_range
            total_score = BM25_WEIGHT * norm_bm25 + SEMANTIC_WEIGHT * cos
            # Skip only if both signals are non-positive (semantically irrelevant)
            if total_score <= 0 and bm25_raw <= 0:
                continue
        else:
            # BM25-only: preserve original raw-score behaviour exactly
            if bm25_raw <= 0:
                continue
            total_score = bm25_raw

        total_score += _metadata_boost(feature, latest_user_question, query_terms)
        ranked.append(
            RetrievedExcerpt(
                excerpt_id=feature.excerpt.excerpt_id,
                heading=feature.excerpt.heading,
                page_start=feature.excerpt.page_start,
                page_end=feature.excerpt.page_end,
                text=feature.excerpt.text,
                score=round(total_score, 6),
                token_estimate=feature.excerpt.token_estimate,
                semantic_score=round(float(cos), 6),
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.page_start, item.excerpt_id))
    if retrieve_candidates <= 0:
        return ranked
    return ranked[:retrieve_candidates]


def _budget_excerpt_selection(
    excerpts: list[RetrievedExcerpt],
    *,
    max_excerpts: int,
    context_token_budget: int,
) -> list[RetrievedExcerpt]:
    if not excerpts:
        return []

    selected: list[RetrievedExcerpt] = []
    remaining_budget = max(1, context_token_budget)
    per_excerpt_overhead = 24

    for excerpt in excerpts:
        if len(selected) >= max(1, max_excerpts):
            break
        if _looks_near_duplicate(excerpt, selected):
            continue

        estimated_cost = max(1, excerpt.token_estimate) + per_excerpt_overhead
        if selected and estimated_cost > remaining_budget:
            continue
        if not selected and estimated_cost > remaining_budget:
            selected.append(excerpt)
            break

        selected.append(excerpt)
        remaining_budget -= estimated_cost
        if remaining_budget <= 0:
            break

    if selected:
        return selected
    return excerpts[:1]


def _build_history_feature(turn: ChatHistoryTurn) -> HistoryFeature | None:
    user_query = _normalize_space(turn.user_query)
    assistant_answer = _normalize_space(turn.assistant_answer)
    combined_text = "\n".join(part for part in (user_query, assistant_answer) if part).strip()
    if not combined_text:
        return None

    query_term_freq = Counter(_tokenize(user_query))
    answer_term_freq = Counter(_tokenize(assistant_answer))
    combined_term_freq = query_term_freq + answer_term_freq
    document_length = max(1, sum(combined_term_freq.values()))
    token_estimate = max(1, estimate_tokens(combined_text))

    return HistoryFeature(
        turn=RetrievedHistoryTurn(
            turn_id=turn.turn_id,
            document_id=turn.document_id,
            created_at=turn.created_at,
            user_query=user_query,
            assistant_answer=assistant_answer,
            score=0.0,
            token_estimate=token_estimate,
            answer_reused_from_history=turn.answer_reused_from_history,
            reused_from_turn_id=turn.reused_from_turn_id,
        ),
        combined_term_freq=combined_term_freq,
        query_term_freq=query_term_freq,
        answer_term_freq=answer_term_freq,
        token_set=frozenset(combined_term_freq),
        document_length=document_length,
        query_length=max(1, sum(query_term_freq.values())),
        answer_length=max(1, sum(answer_term_freq.values())),
    )


def _build_history_corpus(history_turns: list[ChatHistoryTurn]) -> tuple[tuple[HistoryFeature, ...], Counter[str], float]:
    features: list[HistoryFeature] = []
    doc_freq: Counter[str] = Counter()
    total_length = 0

    for turn in history_turns:
        feature = _build_history_feature(turn)
        if feature is None:
            continue
        features.append(feature)
        total_length += feature.document_length
        doc_freq.update(feature.token_set)

    avg_doc_length = total_length / len(features) if features else 1.0
    return tuple(features), doc_freq, max(1.0, avg_doc_length)


def _history_metadata_boost(feature: HistoryFeature, query: str, query_terms: Counter[str]) -> float:
    if not query_terms:
        return 0.0

    normalized_query = _normalize_space(query).lower()
    query_tokens = set(query_terms)
    query_matches = len(query_tokens & set(feature.query_term_freq))
    answer_matches = len(query_tokens & set(feature.answer_term_freq))
    boost = 0.0

    if query_matches:
        boost += min(0.45, 0.12 * query_matches)
    if answer_matches:
        boost += min(0.35, 0.09 * answer_matches)

    normalized_user_query = feature.turn.user_query.lower()
    normalized_answer = feature.turn.assistant_answer.lower()
    if len(normalized_query) >= 8 and normalized_query in normalized_user_query:
        boost += 0.65
    if len(normalized_query) >= 8 and normalized_query in normalized_answer:
        boost += 0.4
    if feature.turn.answer_reused_from_history:
        boost += 0.05

    return min(1.1, boost)


def retrieve_relevant_history_turns(
    history_turns: list[ChatHistoryTurn],
    latest_user_question: str,
    *,
    max_candidates: int = DEFAULT_RETRIEVE_CANDIDATES,
    history_embeddings: "Any" = None,
) -> list[RetrievedHistoryTurn]:
    query_terms = Counter(_tokenize(latest_user_question))
    if not history_turns or not query_terms:
        return []

    features, doc_freq, avg_doc_length = _build_history_corpus(history_turns)
    if not features:
        return []

    # Semantic scores — available when history_embeddings aligns with features length
    cosine_scores: list[float] | None = None
    if history_embeddings is not None:
        try:
            import numpy as np
            from embeddings import encode_query
            if history_embeddings.shape[0] == len(features):
                q_vec = encode_query(latest_user_question)
                if q_vec.shape[0] == 1:
                    cosine_scores = np.clip(history_embeddings @ q_vec[0], -1.0, 1.0).tolist()
        except Exception as exc:
            logger.debug("History semantic scoring failed: %s", exc)

    BM25_W = 0.6 if cosine_scores else 1.0
    SEM_W = 0.4

    total_docs = len(features)
    ranked: list[RetrievedHistoryTurn] = []
    for i, feature in enumerate(features):
        query_score = _score_bm25_term_freq(
            feature.query_term_freq,
            feature.query_length,
            query_terms,
            doc_freq=doc_freq,
            total_docs=total_docs,
            avg_doc_length=avg_doc_length,
        )
        answer_score = _score_bm25_term_freq(
            feature.answer_term_freq,
            feature.answer_length,
            query_terms,
            doc_freq=doc_freq,
            total_docs=total_docs,
            avg_doc_length=avg_doc_length,
        )
        bm25_raw = (1.15 * query_score) + (0.9 * answer_score)
        if cosine_scores is not None:
            cos = cosine_scores[i]
            total_score = BM25_W * bm25_raw + SEM_W * cos
            if total_score <= 0 and bm25_raw <= 0:
                continue
        else:
            total_score = bm25_raw
            if total_score <= 0:
                continue
        total_score += _history_metadata_boost(feature, latest_user_question, query_terms)
        ranked.append(
            RetrievedHistoryTurn(
                turn_id=feature.turn.turn_id,
                document_id=feature.turn.document_id,
                created_at=feature.turn.created_at,
                user_query=feature.turn.user_query,
                assistant_answer=feature.turn.assistant_answer,
                score=round(total_score, 6),
                token_estimate=feature.turn.token_estimate,
                answer_reused_from_history=feature.turn.answer_reused_from_history,
                reused_from_turn_id=feature.turn.reused_from_turn_id,
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.created_at, item.turn_id))
    if max_candidates <= 0:
        return ranked
    return ranked[:max_candidates]


def format_history_context(history_turns: list[RetrievedHistoryTurn]) -> str:
    blocks: list[str] = []
    for index, turn in enumerate(history_turns, start=1):
        blocks.append(
            f"[History Match {index} | Turn {turn.turn_id} | {turn.created_at}]\n"
            f"Previous user question: {turn.user_query}\n"
            f"Previous assistant answer: {turn.assistant_answer}"
        )
    return "\n\n".join(blocks).strip()


def _build_chunk_candidate(excerpt: RetrievedExcerpt) -> RetrievedCandidate:
    return RetrievedCandidate(
        source="chunk",
        source_id=excerpt.excerpt_id,
        score=excerpt.score,
        token_estimate=excerpt.token_estimate,
        text=excerpt.text,
        heading=excerpt.heading,
        page_start=excerpt.page_start,
        page_end=excerpt.page_end,
    )


def _build_history_candidate(turn: RetrievedHistoryTurn) -> RetrievedCandidate:
    return RetrievedCandidate(
        source="history",
        source_id=turn.turn_id,
        score=turn.score,
        token_estimate=turn.token_estimate,
        text=(
            f"Previous user question: {turn.user_query}\n"
            f"Previous assistant answer: {turn.assistant_answer}"
        ).strip(),
        turn_id=turn.turn_id,
        created_at=turn.created_at,
        user_query=turn.user_query,
        assistant_answer=turn.assistant_answer,
    )


def _looks_near_duplicate_candidate(candidate: RetrievedCandidate, selected: list[RetrievedCandidate]) -> bool:
    candidate_terms = set(_tokenize(candidate.text))
    if not candidate_terms:
        return False

    for existing in selected:
        if candidate.source == existing.source and candidate.source_id == existing.source_id:
            return True

        existing_terms = set(_tokenize(existing.text))
        if not existing_terms:
            continue
        overlap = len(candidate_terms & existing_terms) / max(1, len(candidate_terms | existing_terms))
        if overlap >= 0.82:
            return True

    return False


def _budget_candidate_selection(
    candidates: list[RetrievedCandidate],
    *,
    max_chunk_excerpts: int,
    max_history_excerpts: int,
    context_token_budget: int,
) -> list[RetrievedCandidate]:
    if not candidates:
        return []

    selected: list[RetrievedCandidate] = []
    remaining_budget = max(1, context_token_budget)
    per_candidate_overhead = 28
    chunk_count = 0
    history_count = 0

    for candidate in candidates:
        if candidate.source == "chunk" and chunk_count >= max(1, max_chunk_excerpts):
            continue
        if candidate.source == "history" and history_count >= max(0, max_history_excerpts):
            continue
        if _looks_near_duplicate_candidate(candidate, selected):
            continue

        estimated_cost = max(1, candidate.token_estimate) + per_candidate_overhead
        if selected and estimated_cost > remaining_budget:
            continue
        if not selected and estimated_cost > remaining_budget:
            selected.append(candidate)
            break

        selected.append(candidate)
        remaining_budget -= estimated_cost
        if candidate.source == "chunk":
            chunk_count += 1
        else:
            history_count += 1
        if remaining_budget <= 0:
            break

    if selected:
        return selected
    return candidates[:1]


def format_candidate_context(candidates: list[RetrievedCandidate]) -> str:
    blocks: list[str] = []
    chunk_index = 0
    history_index = 0

    for candidate in candidates:
        if candidate.source == "chunk":
            chunk_index += 1
            page_start = candidate.page_start or 0
            page_end = candidate.page_end or page_start
            page_label = (
                f"Page {page_start}"
                if page_start == page_end
                else f"Pages {page_start}-{page_end}"
            )
            blocks.append(
                f"[Excerpt {chunk_index} | Source {candidate.source_id} | {candidate.heading or 'UNKNOWN'} | {page_label}]\n"
                f"{candidate.text}"
            )
            continue

        history_index += 1
        blocks.append(
            f"[History Match {history_index} | Turn {candidate.turn_id} | {candidate.created_at}]\n"
            f"Previous user question: {candidate.user_query or ''}\n"
            f"Previous assistant answer: {candidate.assistant_answer or ''}"
        )

    return "\n\n".join(blocks).strip()


def retrieve_ranked_chunk_excerpts(
    document_text: str,
    latest_user_question: str,
    *,
    retrieve_candidates: int = DEFAULT_RETRIEVE_CANDIDATES,
    document_id: str | None = None,
    persisted_chunks_path: str | Path | None = None,
) -> tuple[list[RetrievedExcerpt], str]:
    if persisted_chunks_path:
        corpus = get_persisted_chunk_corpus(persisted_chunks_path, document_id=document_id)
        if corpus and corpus.features:
            ranked = _rank_persisted_excerpt_features(
                corpus,
                latest_user_question,
                retrieve_candidates=retrieve_candidates,
            )
            if ranked:
                mode = "persisted_chunk_hybrid" if corpus.chunk_embeddings is not None else "persisted_chunk_bm25"
                return ranked, mode
            fallback = [feature.excerpt for feature in corpus.features[: max(1, min(4, retrieve_candidates))]]
            return fallback, "fallback_persisted_chunks"

    return _legacy_retrieve_relevant_excerpts(
        document_text,
        latest_user_question,
        max_excerpts=max(1, retrieve_candidates),
    )


def _sub_chunk_excerpt(
    excerpt: RetrievedExcerpt,
    query_tokens: frozenset[str],
    nlp: Any,
) -> RetrievedExcerpt:
    """Split an excerpt into sentences and return a ±2-sentence window around the best match."""
    doc = nlp(excerpt.text)
    if doc.has_annotation("SENT_START"):
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    else:
        # Fallback: naive split on sentence-ending punctuation
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", excerpt.text) if s.strip()]

    if len(sentences) <= 1:
        return excerpt

    best_idx = 0
    best_score = -1
    for i, sent in enumerate(sentences):
        score = len(query_tokens & frozenset(_tokenize(sent)))
        if score > best_score:
            best_score = score
            best_idx = i
    # If no query terms matched any sentence (best_score == 0), best_idx stays at 0,
    # returning the opening sentences — acceptable fallback.

    window_start = max(0, best_idx - 1)
    window_end = min(len(sentences), best_idx + 2)
    window_text = " ".join(sentences[window_start:window_end])

    return RetrievedExcerpt(
        excerpt_id=excerpt.excerpt_id,
        heading=excerpt.heading,
        page_start=excerpt.page_start,
        page_end=excerpt.page_end,
        text=window_text,
        score=excerpt.score,
        token_estimate=max(1, estimate_tokens(window_text)),
    )


def _apply_sub_chunking(
    excerpts: list[RetrievedExcerpt],
    query: str,
    nlp: Any,
) -> list[RetrievedExcerpt]:
    """Apply sentence-window sub-chunking to the given excerpts. Returns unchanged if nlp is None."""
    if nlp is None:
        return excerpts
    query_tokens = frozenset(_tokenize(query))
    return [_sub_chunk_excerpt(e, query_tokens, nlp) for e in excerpts]


def retrieve_relevant_excerpts(
    document_text: str,
    latest_user_question: str,
    *,
    max_excerpts: int = DEFAULT_MAX_EXCERPTS,
    retrieve_candidates: int = DEFAULT_RETRIEVE_CANDIDATES,
    context_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    document_id: str | None = None,
    persisted_chunks_path: str | Path | None = None,
    apply_sub_chunking: bool = True,
) -> tuple[list[RetrievedExcerpt], list[RetrievedExcerpt], str]:
    """Returns (final_excerpts, raw_top_N_before_subchunking, retrieval_mode)."""
    ranked_excerpts, retrieval_mode = retrieve_ranked_chunk_excerpts(
        document_text,
        latest_user_question,
        retrieve_candidates=retrieve_candidates,
        document_id=document_id,
        persisted_chunks_path=persisted_chunks_path,
    )
    # Slice to the top-N finalists before sub-chunking so spaCy only processes ≤max_excerpts chunks
    raw_top = ranked_excerpts[:max(1, max_excerpts)]
    processed = _apply_sub_chunking(raw_top, latest_user_question, get_spacy_nlp()) if apply_sub_chunking else raw_top
    return (
        _budget_excerpt_selection(
            processed,
            max_excerpts=max_excerpts,
            context_token_budget=context_token_budget,
        ),
        raw_top,
        retrieval_mode,
    )


def format_excerpt_context(excerpts: list[RetrievedExcerpt]) -> str:
    blocks: list[str] = []
    for index, excerpt in enumerate(excerpts, start=1):
        page_label = (
            f"Page {excerpt.page_start}"
            if excerpt.page_start == excerpt.page_end
            else f"Pages {excerpt.page_start}-{excerpt.page_end}"
        )
        heading = excerpt.heading or "Chapter unknown"
        blocks.append(
            f"[Excerpt {index} | Source {excerpt.excerpt_id} | {heading} | {page_label}]\n"
            f"{excerpt.text}"
        )
    return "\n\n".join(blocks).strip()


def build_chat_context(
    document_text: str,
    latest_user_question: str,
    *,
    max_excerpts: int = DEFAULT_MAX_EXCERPTS,
    retrieve_candidates: int = DEFAULT_RETRIEVE_CANDIDATES,
    context_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    fallback_chars: int = DEFAULT_FALLBACK_CHARS,
    document_id: str | None = None,
    persisted_chunks_path: str | Path | None = None,
) -> str:
    return build_chat_context_result(
        document_text,
        latest_user_question,
        max_excerpts=max_excerpts,
        retrieve_candidates=retrieve_candidates,
        context_token_budget=context_token_budget,
        fallback_chars=fallback_chars,
        document_id=document_id,
        persisted_chunks_path=persisted_chunks_path,
    ).context


def build_chat_context_result(
    document_text: str,
    latest_user_question: str,
    *,
    max_excerpts: int = DEFAULT_MAX_EXCERPTS,
    retrieve_candidates: int = DEFAULT_RETRIEVE_CANDIDATES,
    context_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    fallback_chars: int = DEFAULT_FALLBACK_CHARS,
    document_id: str | None = None,
    persisted_chunks_path: str | Path | None = None,
    apply_sub_chunking: bool = True,
) -> ChatContextResult:
    excerpts, raw_excerpts, retrieval_mode = retrieve_relevant_excerpts(
        document_text,
        latest_user_question,
        max_excerpts=max_excerpts,
        retrieve_candidates=retrieve_candidates,
        context_token_budget=context_token_budget,
        document_id=document_id,
        persisted_chunks_path=persisted_chunks_path,
        apply_sub_chunking=apply_sub_chunking,
    )
    if excerpts:
        return ChatContextResult(
            context=format_excerpt_context(excerpts),
            retrieval_mode=retrieval_mode,
            retrieved_excerpts=excerpts,
            raw_retrieved_excerpts=raw_excerpts,
        )

    if persisted_chunks_path:
        corpus = get_persisted_chunk_corpus(persisted_chunks_path, document_id=document_id)
        if corpus and corpus.features:
            fallback_excerpts = _budget_excerpt_selection(
                [feature.excerpt for feature in corpus.features[: max(1, min(3, max_excerpts))]],
                max_excerpts=max_excerpts,
                context_token_budget=context_token_budget,
            )
            return ChatContextResult(
                context=format_excerpt_context(fallback_excerpts),
                retrieval_mode="fallback_persisted_chunks",
                retrieved_excerpts=fallback_excerpts,
            )

    fallback_excerpts = _build_candidate_excerpts_from_text(document_text)[: max(1, min(2, max_excerpts))]
    if fallback_excerpts:
        fallback_context = format_excerpt_context(fallback_excerpts)
        context = fallback_context[:fallback_chars].rstrip() if len(fallback_context) > fallback_chars else fallback_context
        return ChatContextResult(
            context=context,
            retrieval_mode="legacy_text_leading_excerpts",
            retrieved_excerpts=fallback_excerpts,
        )

    return ChatContextResult(
        context=(document_text or "").strip()[:fallback_chars].rstrip(),
        retrieval_mode="fallback_leading_text",
        retrieved_excerpts=[],
    )


def build_chat_context_result_with_history(
    document_text: str,
    latest_user_question: str,
    *,
    history_turns: list[ChatHistoryTurn] | None = None,
    chat_retrieval_mode: str = "chunks_only",
    history_reuse_threshold: float = 0.0,
    history_max_candidates: int = DEFAULT_RETRIEVE_CANDIDATES,
    history_max_excerpts: int = 2,
    max_excerpts: int = DEFAULT_MAX_EXCERPTS,
    retrieve_candidates: int = DEFAULT_RETRIEVE_CANDIDATES,
    context_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    fallback_chars: int = DEFAULT_FALLBACK_CHARS,
    document_id: str | None = None,
    persisted_chunks_path: str | Path | None = None,
    history_storage_dir: str | Path | None = None,
    apply_sub_chunking: bool = True,
    skip_history_reuse: bool = False,
) -> ChatContextResult:
    normalized_mode = (chat_retrieval_mode or "chunks_only").strip().lower()
    if normalized_mode not in {"combined", "history_first", "chunks_only"}:
        normalized_mode = "chunks_only"

    available_history_turns = history_turns or []
    if normalized_mode == "chunks_only":
        base_result = build_chat_context_result(
            document_text,
            latest_user_question,
            max_excerpts=max_excerpts,
            retrieve_candidates=retrieve_candidates,
            context_token_budget=context_token_budget,
            fallback_chars=fallback_chars,
            document_id=document_id,
            persisted_chunks_path=persisted_chunks_path,
            apply_sub_chunking=apply_sub_chunking,
        )
        return ChatContextResult(
            context=base_result.context,
            retrieval_mode=base_result.retrieval_mode,
            retrieved_excerpts=base_result.retrieved_excerpts,
            retrieved_chunk_candidates=base_result.raw_retrieved_excerpts,
            raw_retrieved_excerpts=base_result.raw_retrieved_excerpts,
            final_candidate_mix=[_build_chunk_candidate(item) for item in base_result.retrieved_excerpts],
            chunk_retrieval_mode=base_result.retrieval_mode,
        )

    # Fast path for history_first: exact match on normalized user query — bypasses BM25 entirely.
    # BM25 IDF goes negative with tiny corpora, making threshold-based reuse unreliable.
    # skip_history_reuse=True when there are prior assistant turns in the current session
    # (follow-up questions must not accidentally reuse old session answers).
    if normalized_mode == "history_first" and available_history_turns and not skip_history_reuse:
        new_q_norm = _normalize_query_for_match(latest_user_question)
        for turn in available_history_turns:
            if (
                _normalize_query_for_match(turn.user_query) == new_q_norm
                and turn.assistant_answer.strip()
            ):
                matched_turn = RetrievedHistoryTurn(
                    turn_id=turn.turn_id,
                    document_id=turn.document_id,
                    created_at=turn.created_at,
                    user_query=turn.user_query,
                    assistant_answer=turn.assistant_answer,
                    score=999.0,
                    token_estimate=estimate_tokens(turn.assistant_answer),
                    answer_reused_from_history=True,
                    reused_from_turn_id=turn.reused_from_turn_id,
                )
                return ChatContextResult(
                    context="",
                    retrieval_mode="history_exact_match",
                    retrieved_excerpts=[],
                    retrieved_history=[matched_turn],
                    history_checked=True,
                    history_reuse_hit=True,
                    history_top_score=999.0,
                    history_threshold=history_reuse_threshold,
                    reused_turn_id=turn.turn_id,
                    reused_answer=turn.assistant_answer,
                )

    _history_embeddings = None
    if history_storage_dir is not None and document_id is not None:
        try:
            _history_embeddings = load_history_embeddings(document_id, history_storage_dir)
        except Exception:
            pass

    history_candidates = retrieve_relevant_history_turns(
        available_history_turns,
        latest_user_question,
        max_candidates=max(1, history_max_candidates),
        history_embeddings=_history_embeddings,
    )
    history_top_score = history_candidates[0].score if history_candidates else None

    if normalized_mode == "history_first":
        top_history = history_candidates[0] if history_candidates else None
        if (
            not skip_history_reuse
            and top_history is not None
            and top_history.score >= history_reuse_threshold
            and top_history.assistant_answer.strip()
        ):
            reused_candidate = _build_history_candidate(top_history)
            return ChatContextResult(
                context="",
                retrieval_mode="history_first_reuse",
                retrieved_excerpts=[],
                retrieved_history=history_candidates[: max(1, history_max_candidates)],
                retrieved_chunk_candidates=[],
                final_candidate_mix=[reused_candidate],
                history_checked=True,
                history_reuse_hit=True,
                history_top_score=top_history.score,
                history_threshold=history_reuse_threshold,
                reused_turn_id=top_history.turn_id,
                reused_answer=top_history.assistant_answer,
            )

        chunk_candidates, chunk_retrieval_mode = retrieve_ranked_chunk_excerpts(
            document_text,
            latest_user_question,
            retrieve_candidates=retrieve_candidates,
            document_id=document_id,
            persisted_chunks_path=persisted_chunks_path,
        )
        selected_chunks = _budget_excerpt_selection(
            chunk_candidates,
            max_excerpts=max_excerpts,
            context_token_budget=context_token_budget,
        )
        if selected_chunks:
            return ChatContextResult(
                context=format_excerpt_context(selected_chunks),
                retrieval_mode="history_first_chunk_fallback",
                retrieved_excerpts=selected_chunks,
                retrieved_history=history_candidates[: max(1, history_max_candidates)],
                retrieved_chunk_candidates=chunk_candidates,
                final_candidate_mix=[_build_chunk_candidate(item) for item in selected_chunks],
                history_checked=True,
                history_reuse_hit=False,
                history_top_score=history_top_score,
                history_threshold=history_reuse_threshold,
                chunk_retrieval_mode=chunk_retrieval_mode,
            )

        base_result = build_chat_context_result(
            document_text,
            latest_user_question,
            max_excerpts=max_excerpts,
            retrieve_candidates=retrieve_candidates,
            context_token_budget=context_token_budget,
            fallback_chars=fallback_chars,
            document_id=document_id,
            persisted_chunks_path=persisted_chunks_path,
            apply_sub_chunking=apply_sub_chunking,
        )
        return ChatContextResult(
            context=base_result.context,
            retrieval_mode="history_first_chunk_fallback",
            retrieved_excerpts=base_result.retrieved_excerpts,
            retrieved_history=history_candidates[: max(1, history_max_candidates)],
            retrieved_chunk_candidates=chunk_candidates,
            raw_retrieved_excerpts=base_result.raw_retrieved_excerpts,
            final_candidate_mix=[_build_chunk_candidate(item) for item in base_result.retrieved_excerpts],
            history_checked=True,
            history_reuse_hit=False,
            history_top_score=history_top_score,
            history_threshold=history_reuse_threshold,
            chunk_retrieval_mode=base_result.retrieval_mode,
        )

    chunk_candidates, chunk_retrieval_mode = retrieve_ranked_chunk_excerpts(
        document_text,
        latest_user_question,
        retrieve_candidates=retrieve_candidates,
        document_id=document_id,
        persisted_chunks_path=persisted_chunks_path,
    )
    unified_candidates = [_build_chunk_candidate(item) for item in chunk_candidates]
    unified_candidates.extend(_build_history_candidate(item) for item in history_candidates)
    unified_candidates.sort(
        key=lambda item: (
            -item.score,
            0 if item.source == "history" else 1,
            item.source_id,
        )
    )

    selected_candidates = _budget_candidate_selection(
        unified_candidates,
        max_chunk_excerpts=max_excerpts,
        max_history_excerpts=max(0, history_max_excerpts),
        context_token_budget=context_token_budget,
    )
    if selected_candidates:
        selected_chunks = [
            item
            for item in chunk_candidates
            if any(
                candidate.source == "chunk" and candidate.source_id == item.excerpt_id
                for candidate in selected_candidates
            )
        ]
        return ChatContextResult(
            context=format_candidate_context(selected_candidates),
            retrieval_mode="combined",
            retrieved_excerpts=selected_chunks,
            retrieved_history=history_candidates[: max(1, history_max_candidates)],
            retrieved_chunk_candidates=chunk_candidates,
            final_candidate_mix=selected_candidates,
            history_checked=True,
            history_reuse_hit=False,
            history_top_score=history_top_score,
            history_threshold=history_reuse_threshold,
            chunk_retrieval_mode=chunk_retrieval_mode,
        )

    base_result = build_chat_context_result(
        document_text,
        latest_user_question,
        max_excerpts=max_excerpts,
        retrieve_candidates=retrieve_candidates,
        context_token_budget=context_token_budget,
        fallback_chars=fallback_chars,
        document_id=document_id,
        persisted_chunks_path=persisted_chunks_path,
        apply_sub_chunking=apply_sub_chunking,
    )
    return ChatContextResult(
        context=base_result.context,
        retrieval_mode="combined",
        retrieved_excerpts=base_result.retrieved_excerpts,
        retrieved_history=history_candidates[: max(1, history_max_candidates)],
        retrieved_chunk_candidates=chunk_candidates,
        raw_retrieved_excerpts=base_result.raw_retrieved_excerpts,
        final_candidate_mix=[_build_chunk_candidate(item) for item in base_result.retrieved_excerpts],
        history_checked=True,
        history_reuse_hit=False,
        history_top_score=history_top_score,
        history_threshold=history_reuse_threshold,
        chunk_retrieval_mode=base_result.retrieval_mode,
    )
