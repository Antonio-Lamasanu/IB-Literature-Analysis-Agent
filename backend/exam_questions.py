"""Static question database for Exam Mode.

Paper 1: passages stored in the DB (paper1_passages table), seeded from
         useful/paper1_sub.json on first run. Legacy hardcoded constants kept
         for backwards compatibility.
Paper 2: questions stored in the DB (paper2_questions table), seeded from
         useful/paper2_sub.json on first run.
"""
from __future__ import annotations

import random

from database import get_connection

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
    """Return all entries from the paper1_passages DB table.

    Each dict has keys: id, title, author, year, passage, guiding_question.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, author, year, passage, guiding_question FROM paper1_passages ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def get_random_paper1_passage() -> dict:
    """Return a randomly selected passage entry from the DB."""
    passages = load_paper1_passages()
    if not passages:
        return {"title": "", "author": "", "year": None, "passage": PAPER1_PASSAGE, "guiding_question": PAPER1_QUESTION}
    return random.choice(passages)


def load_paper2_questions() -> list[dict]:
    """Return all entries from the paper2_questions DB table.

    Each dict has keys: id, text.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, text FROM paper2_questions ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]
