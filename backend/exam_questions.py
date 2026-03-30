"""Static question database for Exam Mode.

Paper 1: passages loaded from useful/paper1_sub.json at runtime (with legacy
         hardcoded fallbacks kept for backwards compatibility).
Paper 2: three questions; one is randomly selected per session.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

_PAPER1_JSON = Path(__file__).resolve().parents[1] / "useful" / "paper1_sub.json"
_PAPER2_JSON = Path(__file__).resolve().parents[1] / "useful" / "paper2_sub.json"

PAPER1_PASSAGE = (
    "The streetlight flickered again.\n"
    "Mara paused beneath it, her shadow breaking apart and stitching itself back together across "
    "the pavement. It was late\u2014too late for anyone to notice her standing there\u2014but she "
    "lingered, watching the light struggle against the dark.\n"
    "The city had changed. Or maybe she had. The bakery on the corner, once warm with the smell "
    "of bread at dawn, now sat behind a metal shutter layered with graffiti. Words clawed over "
    "each other in angry colors, none of them legible anymore.\n"
    "She remembered when things made sense. When mornings came with certainty, and nights were "
    "only temporary interruptions. Now, the hours blurred. Days collapsed into one another, "
    "indistinguishable except for the slow accumulation of silence.\n"
    "A car passed, its headlights slicing through her thoughts. For a moment, everything was "
    "illuminated\u2014sharp, undeniable. Then it was gone, and the darkness returned, thicker "
    "than before.\n"
    "Mara stepped forward, out from under the streetlight. The flickering stopped behind her. "
    "Or maybe she had just stopped looking."
)

PAPER1_QUESTION = (
    "Analyze how the writer uses stylistic and structural features to convey "
    "Mara\u2019s psychological state."
)

def load_paper1_passages() -> list[dict]:
    """Return all entries from useful/paper1_sub.json.

    Each entry has keys: title, author, year, passage, guiding_question.
    """
    with _PAPER1_JSON.open(encoding="utf-8") as f:
        return json.load(f)


def get_random_paper1_passage() -> dict:
    """Return a randomly selected passage entry from paper1_sub.json."""
    return random.choice(load_paper1_passages())


def load_paper2_questions() -> list[dict]:
    """Return all entries from useful/paper2_sub.json.

    Each entry has keys: id, text.
    """
    with _PAPER2_JSON.open(encoding="utf-8") as f:
        return json.load(f)
