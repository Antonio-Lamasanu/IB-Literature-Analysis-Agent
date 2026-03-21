"""Static question database for Exam Mode.

Paper 1: one unseen passage + one fixed question.
Paper 2: three questions; one is randomly selected per session.
"""
from __future__ import annotations

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

PAPER2_QUESTIONS: list[dict] = [
    {
        "id": 1,
        "text": (
            "\u201cLiterary works often explore the tension between appearance and reality.\u201d\n"
            "To what extent is this true of two works you have studied?"
        ),
    },
    {
        "id": 2,
        "text": (
            "How do two works you have studied present the impact of memory "
            "on individuals or societies?"
        ),
    },
    {
        "id": 3,
        "text": (
            "Writers use setting not only as a backdrop, but as a means of shaping meaning.\n"
            "Discuss how this is achieved in two works you have studied."
        ),
    },
]
