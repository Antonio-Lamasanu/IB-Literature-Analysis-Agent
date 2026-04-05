from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database import init_db
from exam_history import (
    ExamAttempt,
    append_exam_attempt,
    create_exam_attempt,
    load_exam_history,
    persist_exam_history,
)
import exam_history as _exam_history_mod


class ExamHistoryBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._root = Path(self._tmpdir.name)
        self._db_path = self._root / "test.db"
        init_db(self._db_path)
        self._patcher = mock.patch("exam_history.get_db_path", return_value=self._db_path)
        self._patcher.start()
        _exam_history_mod._migrated_dirs.clear()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmpdir.cleanup()

    @property
    def _storage_dir(self) -> Path:
        d = self._root / "exam_history"
        d.mkdir(exist_ok=True)
        return d


class AppendAndLoadTests(ExamHistoryBase):
    def test_round_trip_minimal_fields(self) -> None:
        attempt = create_exam_attempt(
            document_id="doc-1",
            paper_type="paper1",
            question="What is the theme of Animal Farm?",
            student_answer="It is about revolution.",
        )
        append_exam_attempt("doc-1", attempt, self._storage_dir)

        loaded = load_exam_history("doc-1", self._storage_dir)
        self.assertEqual(1, len(loaded))
        self.assertEqual("What is the theme of Animal Farm?", loaded[0].question)
        self.assertEqual("It is about revolution.", loaded[0].student_answer)
        self.assertEqual("paper1", loaded[0].paper_type)

    def test_round_trip_all_score_and_feedback_fields(self) -> None:
        attempt = create_exam_attempt(
            document_id="doc-2",
            paper_type="paper2",
            question="Discuss the role of Boxer.",
            student_answer="Boxer is loyal.",
            score_a=3,
            score_b=2,
            score_c=4,
            score_d=1,
            total_score=10,
            max_score=12,
            feedback_a="Good point about loyalty.",
            feedback_b="Could expand on work ethic.",
            feedback_c="Strong analysis.",
            feedback_d="Needs more evidence.",
            overall_comments="Solid attempt.",
            grading_raw_output='{"raw": true}',
            document_ids=["doc-2", "doc-3"],
            context_mode="full_text",
        )
        append_exam_attempt("doc-2", attempt, self._storage_dir)

        loaded = load_exam_history("doc-2", self._storage_dir)
        self.assertEqual(1, len(loaded))
        a = loaded[0]
        self.assertEqual(3, a.score_a)
        self.assertEqual(10, a.total_score)
        self.assertEqual("Good point about loyalty.", a.feedback_a)
        self.assertEqual("Solid attempt.", a.overall_comments)
        self.assertEqual(["doc-2", "doc-3"], a.document_ids)
        self.assertEqual("full_text", a.context_mode)

    def test_multiple_attempts_ordered_by_created_at(self) -> None:
        for i in range(3):
            a = create_exam_attempt(
                document_id="doc-order",
                paper_type="paper1",
                question=f"Q{i}",
                student_answer=f"A{i}",
            )
            append_exam_attempt("doc-order", a, self._storage_dir)

        loaded = load_exam_history("doc-order", self._storage_dir)
        self.assertEqual(3, len(loaded))
        # created_at is ISO; they should come back in ascending order
        timestamps = [a.created_at for a in loaded]
        self.assertEqual(sorted(timestamps), timestamps)

    def test_documents_do_not_bleed_into_each_other(self) -> None:
        for doc_id in ["doc-a", "doc-b"]:
            a = create_exam_attempt(
                document_id=doc_id,
                paper_type="paper1",
                question="Who is Napoleon?",
                student_answer="A pig.",
            )
            append_exam_attempt(doc_id, a, self._storage_dir)

        self.assertEqual(1, len(load_exam_history("doc-a", self._storage_dir)))
        self.assertEqual(1, len(load_exam_history("doc-b", self._storage_dir)))


class PersistExamHistoryTests(ExamHistoryBase):
    def test_persist_replaces_all_attempts_for_document(self) -> None:
        # Insert two original attempts
        for i in range(2):
            a = create_exam_attempt(
                document_id="doc-p",
                paper_type="paper1",
                question=f"Old Q{i}",
                student_answer="old",
            )
            append_exam_attempt("doc-p", a, self._storage_dir)

        # Persist a single replacement
        new_attempt = create_exam_attempt(
            document_id="doc-p",
            paper_type="paper2",
            question="New Q",
            student_answer="new",
        )
        persist_exam_history("doc-p", [new_attempt], self._storage_dir)

        loaded = load_exam_history("doc-p", self._storage_dir)
        self.assertEqual(1, len(loaded))
        self.assertEqual("New Q", loaded[0].question)

    def test_persist_with_empty_list_clears_history(self) -> None:
        a = create_exam_attempt(
            document_id="doc-clear",
            paper_type="paper1",
            question="Q",
            student_answer="A",
        )
        append_exam_attempt("doc-clear", a, self._storage_dir)
        persist_exam_history("doc-clear", [], self._storage_dir)

        self.assertEqual([], load_exam_history("doc-clear", self._storage_dir))

    def test_persist_does_not_affect_other_documents(self) -> None:
        for doc_id in ["doc-x", "doc-y"]:
            a = create_exam_attempt(
                document_id=doc_id,
                paper_type="paper1",
                question="Q",
                student_answer="A",
            )
            append_exam_attempt(doc_id, a, self._storage_dir)

        persist_exam_history("doc-x", [], self._storage_dir)

        self.assertEqual([], load_exam_history("doc-x", self._storage_dir))
        self.assertEqual(1, len(load_exam_history("doc-y", self._storage_dir)))


class CreateExamAttemptTests(ExamHistoryBase):
    def test_creates_attempt_with_unique_ids(self) -> None:
        a1 = create_exam_attempt(
            document_id="doc-1", paper_type="paper1", question="Q", student_answer="A"
        )
        a2 = create_exam_attempt(
            document_id="doc-1", paper_type="paper1", question="Q", student_answer="A"
        )
        self.assertNotEqual(a1.attempt_id, a2.attempt_id)

    def test_strips_whitespace_from_question_and_answer(self) -> None:
        a = create_exam_attempt(
            document_id="doc-1",
            paper_type="paper1",
            question="  Q?  ",
            student_answer="  A.  ",
        )
        self.assertEqual("Q?", a.question)
        self.assertEqual("A.", a.student_answer)

    def test_default_context_mode_is_chunks(self) -> None:
        a = create_exam_attempt(
            document_id="doc-1", paper_type="paper1", question="Q", student_answer="A"
        )
        self.assertEqual("chunks", a.context_mode)


if __name__ == "__main__":
    unittest.main()
