"""
quality.py — Chunk quality comparison for duplicate document detection.

Two-tier approach:
  1. noise_check()  — fast regex heuristic, no LLM cost
  2. llm_score()    — LLM judgment, called only when noise check is inconclusive
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_service import LLMService

logger = logging.getLogger(__name__)

# Characters that reliably indicate OCR garbage
_GARBAGE_RE = re.compile(
    r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # replacement char + control chars
    r"|\?{3,}"                                    # runs of ??? (unknown glyphs)
    r"|[^\x00-\x7F]{3,}(?=[^\x00-\x7F]{3,})"    # long runs of non-ASCII (not proper words)
)

# Lines that are just noise (very short, punctuation-only, page numbers)
_FRAGMENT_LINE_RE = re.compile(r"^\s*[\d\W]{0,6}\s*$")


def _noise_density(text: str) -> float:
    """Return fraction of characters that are garbage/noise (0.0 = clean, 1.0 = all noise)."""
    if not text:
        return 0.0
    garbage_chars = sum(len(m.group()) for m in _GARBAGE_RE.finditer(text))
    lines = text.splitlines()
    fragment_chars = sum(len(line) for line in lines if _FRAGMENT_LINE_RE.match(line))
    noise = garbage_chars + fragment_chars * 0.5
    return min(1.0, noise / max(1, len(text)))


def noise_check(
    chunks_a: list[str],
    chunks_b: list[str],
) -> str | None:
    """Quick pre-filter before calling the LLM.

    Returns:
        "a"   — chunks_a is clearly cleaner (>3× less noise density)
        "b"   — chunks_b is clearly cleaner
        None  — inconclusive, LLM comparison needed
    """
    if not chunks_a and not chunks_b:
        return None
    density_a = sum(_noise_density(c) for c in chunks_a) / max(1, len(chunks_a))
    density_b = sum(_noise_density(c) for c in chunks_b) / max(1, len(chunks_b))

    # If both are very clean, heuristic can't distinguish — defer to LLM
    if density_a < 0.005 and density_b < 0.005:
        return None

    if density_a == 0.0 and density_b == 0.0:
        return None

    ratio = (density_b + 1e-9) / (density_a + 1e-9)
    if ratio > 3.0:
        return "a"  # b has 3× more noise → a wins
    if ratio < (1.0 / 3.0):
        return "b"  # a has 3× more noise → b wins
    return None


_SCORE_JSON_RE = re.compile(r'\{[^{}]*"score_a"\s*:\s*(\d+)[^{}]*"score_b"\s*:\s*(\d+)[^{}]*\}', re.DOTALL)
_SCORE_JSON_ALT_RE = re.compile(r'\{[^{}]*"score_b"\s*:\s*(\d+)[^{}]*"score_a"\s*:\s*(\d+)[^{}]*\}', re.DOTALL)


def _parse_scores(raw: str) -> tuple[float, float] | None:
    """Parse {"score_a": N, "score_b": N} from LLM output (tolerant of surrounding text)."""
    m = _SCORE_JSON_RE.search(raw)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = _SCORE_JSON_ALT_RE.search(raw)
    if m:
        # alt order: score_b comes first in regex groups
        return float(m.group(2)), float(m.group(1))
    # Fallback: look for bare numbers
    nums = re.findall(r"\b([1-9]|10)\b", raw)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None


def llm_score(
    chunks_a: list[str],
    chunks_b: list[str],
    llm_service: "LLMService",
) -> tuple[float, float]:
    """Ask the LLM to rate the quality of two chunk sets (1–10 each).

    Returns (score_a, score_b). Falls back to (5.0, 5.0) on parse failure.
    """
    sep = "\n---\n"
    extract_a = sep.join(c.strip() for c in chunks_a if c.strip())
    extract_b = sep.join(c.strip() for c in chunks_b if c.strip())

    prompt = (
        "You are a text quality evaluator. Score each extract from 1 to 10.\n"
        "Criteria: readability, absence of OCR noise or garbled text, complete sentences,\n"
        "no fragmented words or random symbols.\n"
        'Reply ONLY with valid JSON: {"score_a": <integer>, "score_b": <integer>}\n\n'
        f"Extract A:\n{extract_a}\n\n"
        f"Extract B:\n{extract_b}"
    )

    try:
        result = llm_service.generate_raw_reply(prompt)
        scores = _parse_scores(result.text)
        if scores is not None:
            logger.info("quality.llm_score: score_a=%.1f score_b=%.1f", scores[0], scores[1])
            return scores
        logger.warning("quality.llm_score: could not parse scores from: %r", result.text[:200])
    except Exception as exc:
        logger.warning("quality.llm_score: inference failed: %s", exc)

    return 5.0, 5.0  # tie → keep existing


def get_text_samples(text: str, n: int = 10, tokens_per_sample: int = 400) -> list[str]:
    """Split text into rough word-count-based samples for quality comparison.

    Fast (no NLP), good enough to detect OCR garbage vs. clean text.
    Returns up to `n` blocks of approximately `tokens_per_sample` tokens each.
    """
    words = text.split()
    words_per_block = max(1, int(tokens_per_sample / 1.33))
    samples: list[str] = []
    for i in range(0, len(words), words_per_block):
        block = " ".join(words[i : i + words_per_block]).strip()
        if block:
            samples.append(block)
        if len(samples) >= n:
            break
    return samples


def load_comparison_chunks(
    chunks_path: str | Path,
    *,
    sample_size: int = 10,
    compare_slice: tuple[int, int] = (6, 10),
) -> list[str]:
    """Load chunk texts from a .chunks.jsonl file for quality comparison.

    Reads the first `sample_size` chunks, then returns the texts at
    indices compare_slice[0]:compare_slice[1] (skipping front matter).
    If the file has fewer than sample_size chunks, returns the last
    min(compare_slice[1]-compare_slice[0], n) chunks.
    """
    path = Path(chunks_path)
    if not path.exists():
        return []
    chunks: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("content", {}).get("text", "")
                    if text:
                        chunks.append(text)
                except (json.JSONDecodeError, AttributeError):
                    continue
                if len(chunks) >= sample_size:
                    break
    except OSError as exc:
        logger.warning("quality: could not read %s: %s", path, exc)
        return []

    if not chunks:
        return []

    start, end = compare_slice
    if len(chunks) >= end:
        return chunks[start:end]
    # Fewer chunks than expected: return last min(4, n) chunks
    take = min(end - start, len(chunks))
    return chunks[-take:]
