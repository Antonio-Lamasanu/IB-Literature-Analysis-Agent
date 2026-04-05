from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database import get_connection, get_db_path, init_db


class GetDbPathTests(unittest.TestCase):
    def test_returns_default_path_when_env_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DB_PATH", None)
            path = get_db_path()
        self.assertIn("app.db", str(path))
        self.assertIn("outputs", str(path))

    def test_returns_custom_path_when_env_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = str(Path(tmpdir) / "custom.db")
            with mock.patch.dict(os.environ, {"DB_PATH": custom}):
                path = get_db_path()
            self.assertEqual(Path(custom), path)

    def test_strips_whitespace_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = str(Path(tmpdir) / "ws.db")
            with mock.patch.dict(os.environ, {"DB_PATH": f"  {custom}  "}):
                path = get_db_path()
            self.assertEqual(Path(custom), path)


class InitDbTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_creates_all_four_tables(self) -> None:
        init_db(self._db_path)
        conn = sqlite3.connect(str(self._db_path))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        self.assertIn("documents", tables)
        self.assertIn("chat_turns", tables)
        self.assertIn("exam_attempts", tables)
        self.assertIn("corpus_cache", tables)

    def test_idempotent_safe_to_call_twice(self) -> None:
        init_db(self._db_path)
        # should not raise
        init_db(self._db_path)

    def test_creates_parent_directory_if_missing(self) -> None:
        nested_path = Path(self._tmpdir.name) / "sub" / "dir" / "app.db"
        init_db(nested_path)
        self.assertTrue(nested_path.exists())


class GetConnectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        init_db(self._db_path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_commits_on_success(self) -> None:
        with get_connection(self._db_path) as conn:
            conn.execute(
                "INSERT INTO corpus_cache (cache_key, confidence, source, fetched_at) "
                "VALUES ('k1', 0.9, 'test', '2024-01-01T00:00:00')"
            )
        # Read in a new connection to confirm commit
        with get_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT confidence FROM corpus_cache WHERE cache_key='k1'"
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertAlmostEqual(0.9, row["confidence"])

    def test_rolls_back_on_exception(self) -> None:
        try:
            with get_connection(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO corpus_cache (cache_key, confidence, source, fetched_at) "
                    "VALUES ('k2', 0.5, 'test', '2024-01-01T00:00:00')"
                )
                raise RuntimeError("simulated failure")
        except RuntimeError:
            pass

        with get_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT confidence FROM corpus_cache WHERE cache_key='k2'"
            ).fetchone()
        self.assertIsNone(row)

    def test_re_raises_exception_after_rollback(self) -> None:
        with self.assertRaises(ValueError):
            with get_connection(self._db_path) as conn:
                raise ValueError("must propagate")

    def test_row_factory_returns_sqlite_row(self) -> None:
        with get_connection(self._db_path) as conn:
            conn.execute(
                "INSERT INTO corpus_cache (cache_key, confidence, source, fetched_at) "
                "VALUES ('k3', 0.7, 'src', '2024-01-01T00:00:00')"
            )
        with get_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT cache_key, confidence FROM corpus_cache WHERE cache_key='k3'"
            ).fetchone()
        # sqlite3.Row supports column-name access
        self.assertEqual("k3", row["cache_key"])

    def test_wal_mode_is_applied(self) -> None:
        with get_connection(self._db_path) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual("wal", mode)

    def test_foreign_keys_are_enabled(self) -> None:
        with get_connection(self._db_path) as conn:
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        self.assertEqual(1, fk)


if __name__ == "__main__":
    unittest.main()
