"""
corpus_lookup.py — System 1 corpus confidence lookup.

Pipeline: Option B (static JSON) → Option A (Open Library + Google Books API).
Returns a (confidence: float, source: str) tuple.
All API results are cached to known_works_api_cache.json.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from database import get_connection, get_db_path

logger = logging.getLogger(__name__)

_STATIC_JSON_PATH = Path(__file__).parent / "known_works.json"
_LEGACY_CACHE_PATH = Path(__file__).parent / "known_works_api_cache.json"
_CACHE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# One-time migration from known_works_api_cache.json → DB corpus_cache table
# ---------------------------------------------------------------------------

def _migrate_json_cache_once() -> None:
    if not _LEGACY_CACHE_PATH.exists():
        return
    # Ensure tables exist — this module is imported before DocumentRegistry calls init_db()
    init_db()
    try:
        data = json.loads(_LEGACY_CACHE_PATH.read_text(encoding="utf-8"))
        entries = data.get("cache", {})
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("corpus_lookup: skipping cache migration (%s)", exc)
        return

    db_path = get_db_path()
    migrated = 0
    with get_connection(db_path) as conn:
        for key, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO corpus_cache (cache_key, confidence, source, fetched_at) VALUES (?,?,?,?)",
                    (
                        str(key),
                        float(entry.get("confidence", 0.0)),
                        str(entry.get("source", "unknown")),
                        str(entry.get("fetched_at", "")),
                    ),
                )
                migrated += 1
            except Exception:
                continue

    if migrated:
        logger.info("corpus_lookup: migrated %d cache entries from %s", migrated, _LEGACY_CACHE_PATH.name)
    try:
        _LEGACY_CACHE_PATH.rename(_LEGACY_CACHE_PATH.with_suffix(".json.migrated"))
    except OSError as exc:
        logger.warning("corpus_lookup: could not rename cache file: %s", exc)


_migrate_json_cache_once()

_ARTICLES_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Lowercase, strip leading articles, collapse whitespace, strip punctuation."""
    s = s.lower().strip()
    s = _ARTICLES_RE.sub("", s)
    s = _PUNCT_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Option B — static lookup
# ---------------------------------------------------------------------------

def _build_static_index() -> dict[str, list[dict[str, Any]]]:
    """Load known_works.json once and build a normalized title index."""
    try:
        data = json.loads(_STATIC_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("corpus_lookup: could not load %s", _STATIC_JSON_PATH)
        return {}

    index: dict[str, list[dict[str, Any]]] = {}
    for work in data.get("works", []):
        titles = [work.get("title", "")] + work.get("title_variants", [])
        for t in titles:
            key = _normalize(t)
            if key:
                index.setdefault(key, []).append(work)
    return index


_STATIC_INDEX: dict[str, list[dict[str, Any]]] = _build_static_index()


def _static_lookup(title: str | None, author: str | None) -> float:
    """Return confidence from static list, or 0.0 if not found."""
    if not title:
        return 0.0

    norm_title = _normalize(title)
    candidates = _STATIC_INDEX.get(norm_title, [])
    if not candidates:
        return 0.0

    norm_author = _normalize(author) if author else ""

    for work in candidates:
        if not norm_author:
            # No author provided — accept first title match
            return float(work.get("confidence", 0.0))

        all_author_forms = [work.get("author", "")] + work.get("author_variants", [])
        norm_forms = [_normalize(a) for a in all_author_forms if a]
        if norm_author in norm_forms:
            return float(work.get("confidence", 0.0))

    # Title matched but no author match — still return first candidate's confidence
    # (author may have been extracted in a different form)
    return float(candidates[0].get("confidence", 0.0))


# ---------------------------------------------------------------------------
# Option A — Open Library + Google Books API lookup with cache
# ---------------------------------------------------------------------------

def _cache_ttl_days() -> int:
    try:
        return int(os.environ.get("CORPUS_CACHE_TTL_DAYS", "30"))
    except (TypeError, ValueError):
        return 30


def _get_cached(key: str) -> dict[str, Any] | None:
    try:
        with get_connection(get_db_path()) as conn:
            row = conn.execute(
                "SELECT confidence, source, fetched_at FROM corpus_cache WHERE cache_key=?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {"confidence": row["confidence"], "source": row["source"], "fetched_at": row["fetched_at"]}
    except Exception as exc:
        logger.warning("corpus_lookup: DB read failed: %s", exc)
        return None


def _set_cached(key: str, confidence: float, source: str, fetched_at: str) -> None:
    try:
        with get_connection(get_db_path()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO corpus_cache (cache_key, confidence, source, fetched_at) VALUES (?,?,?,?)",
                (key, confidence, source, fetched_at),
            )
    except Exception as exc:
        logger.warning("corpus_lookup: DB write failed: %s", exc)


def _cache_key(title: str | None, author: str | None) -> str:
    return f"{_normalize(title or '')}:{_normalize(author or '')}"


def _is_cache_fresh(entry: dict[str, Any]) -> bool:
    fetched_at = entry.get("fetched_at", "")
    if not fetched_at:
        return False
    try:
        fetched = datetime.fromisoformat(fetched_at)
        age_days = (datetime.now(timezone.utc) - fetched).days
        return age_days < _cache_ttl_days()
    except (ValueError, TypeError):
        return False


def _score_from_open_library(result: dict[str, Any]) -> float:
    """Derive a confidence score from a single Open Library search result."""
    has_fulltext = bool(result.get("has_fulltext", False))
    if has_fulltext:
        # Strong standalone signal: Gutenberg presence ≈ definitely in training data
        return 0.82

    year = result.get("first_publish_year")
    edition_count = result.get("edition_count", 0)

    if year and year < 1928:
        base = 0.45
    elif year and year < 2000:
        base = 0.30
    else:
        base = 0.10

    if edition_count >= 50:
        base += 0.12
    elif edition_count >= 20:
        base += 0.06

    return min(base, 0.88)


def _api_lookup(title: str | None, author: str | None) -> tuple[float, str]:
    """Query Open Library (+ Google Books supplement). Returns (confidence, source)."""
    # Lazily import requests to avoid import-time failure if not installed
    try:
        import requests
    except ImportError:
        logger.warning("corpus_lookup: 'requests' not installed; skipping API lookup")
        return 0.0, "unknown"

    key = _cache_key(title, author)

    with _CACHE_LOCK:
        entry = _get_cached(key)
        if entry and _is_cache_fresh(entry):
            return float(entry["confidence"]), str(entry["source"])

    score = 0.0
    source = "unknown"

    # --- Open Library ---
    try:
        params: dict[str, Any] = {"limit": 1}
        if title:
            params["title"] = title
        if author:
            params["author"] = author

        resp = requests.get(
            "https://openlibrary.org/search.json",
            params=params,
            timeout=4,
            headers={"User-Agent": "IB-AI-Tutor/1.0 (corpus-lookup)"},
        )
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("docs", [])
        if docs:
            score = _score_from_open_library(docs[0])
            source = "open_library"
    except Exception as exc:
        logger.warning("corpus_lookup: Open Library request failed: %s", exc)

    # --- Google Books supplement (only when OL score is weak) ---
    if score < 0.40:
        try:
            query_parts = []
            if title:
                query_parts.append(f"intitle:{title}")
            if author:
                query_parts.append(f"inauthor:{author}")

            gb_resp = requests.get(
                "https://www.googleapis.com/books/v1/volumes",
                params={"q": " ".join(query_parts), "maxResults": 1},
                timeout=4,
                headers={"User-Agent": "IB-AI-Tutor/1.0 (corpus-lookup)"},
            )
            gb_resp.raise_for_status()
            gb_data = gb_resp.json()
            total_items = int(gb_data.get("totalItems", 0))
            if total_items > 500:
                score += 0.12
                source = "google_books"
        except Exception as exc:
            logger.warning("corpus_lookup: Google Books request failed: %s", exc)

    score = round(min(max(score, 0.0), 0.92), 4)

    with _CACHE_LOCK:
        _set_cached(key, score, source, datetime.now(timezone.utc).isoformat())

    return score, source


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _option_c_threshold() -> float:
    try:
        return float(os.environ.get("CORPUS_OPTION_C_THRESHOLD", "0.4"))
    except (TypeError, ValueError):
        return 0.4


def lookup_confidence(title: str | None, author: str | None) -> tuple[float, str, bool]:
    """Return (confidence 0.0–1.0, source, option_c_pending) for a literary work.

    option_c_pending=True when score is below CORPUS_OPTION_C_THRESHOLD and LLM fallback needed.
    source values: 'static_list' | 'open_library' | 'google_books' | 'unknown'
    confidence == 0.0 means not found or lookup failed.
    """
    # Short-circuit: nothing useful to look up
    if not (title or author):
        return 0.0, "unknown", False

    # Option B: static list (instant, zero I/O)
    score = _static_lookup(title, author)
    if score > 0.0:
        rounded = round(score, 4)
        threshold = _option_c_threshold()
        if rounded < threshold:
            logger.warning(
                "corpus_lookup: OPTION C PENDING — score %.3f below threshold %.3f for title=%r author=%r",
                rounded, threshold, title, author,
            )
            return rounded, "static_list", True
        return rounded, "static_list", False

    # Option A: API lookup with cache
    score, source = _api_lookup(title, author)
    threshold = _option_c_threshold()
    if score < threshold:
        logger.warning(
            "corpus_lookup: OPTION C PENDING — score %.3f below threshold %.3f for title=%r author=%r",
            score, threshold, title, author,
        )
        return score, source, True
    return score, source, False


def fire_option_c(title: str | None, author: str | None) -> tuple[float, str]:
    """Manually trigger LLM-API fallback for a pending document. Dev use only.

    Env vars required: CORPUS_LLM_API_URL, CORPUS_LLM_API_KEY, CORPUS_LLM_MODEL
    Returns (confidence, "llm_api").
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx is required for Option C. Install it with: pip install httpx")

    api_url = os.environ.get("CORPUS_LLM_API_URL", "").rstrip("/")
    api_key = os.environ.get("CORPUS_LLM_API_KEY", "")
    model = os.environ.get("CORPUS_LLM_MODEL", "")

    key = _cache_key(title, author)
    with _CACHE_LOCK:
        entry = _get_cached(key)
    if entry and entry.get("source") == "llm_api" and _is_cache_fresh(entry):
        return float(entry["confidence"]), "llm_api"

    # NOTE: LLM self-reports on training data are heuristic; treat as weak signal only.
    prompt = (
        f"Is '{title}' by '{author}' well-represented in the training data of a 2024 "
        "open-source LLM such as Llama-3 or Gemma-2? Respond with only a single decimal "
        "number between 0.0 and 1.0 representing probability."
    )
    response = httpx.post(
        f"{api_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 10},
        timeout=30.0,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"].strip()
    score = round(max(0.0, min(0.92, float(raw))), 4)
    with _CACHE_LOCK:
        # Always overwrites Option A cache entries; llm_api result is authoritative for this key.
        _set_cached(key, score, "llm_api", datetime.now(timezone.utc).isoformat())
    return score, "llm_api"
