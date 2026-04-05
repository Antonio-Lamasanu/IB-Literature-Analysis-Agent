from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database import init_db
from document_registry import DocumentRegistry


class DocumentRegistryBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._root = Path(self._tmpdir.name)
        self._db_path = self._root / "test.db"
        init_db(self._db_path)
        self._patcher = mock.patch("document_registry.get_db_path", return_value=self._db_path)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmpdir.cleanup()

    def _make_registry(self) -> DocumentRegistry:
        return DocumentRegistry(self._root / "documents.index.json")

    def _make_text_file(self, name: str = "doc.txt") -> Path:
        p = self._root / name
        p.write_text("sample", encoding="utf-8")
        return p


class RegisterAndGetTests(DocumentRegistryBase):
    def test_register_persists_and_get_returns_record(self) -> None:
        reg = self._make_registry()
        text = self._make_text_file()
        record = reg.register(
            document_id="doc-1",
            text_path=text,
            filename="doc.pdf",
            processing_mode="native",
            pages=5,
            text_chars=1000,
            source_fingerprint="fp-1",
        )
        self.assertEqual("doc-1", record.document_id)
        self.assertEqual("doc.pdf", record.filename)
        self.assertEqual(5, record.pages)
        self.assertEqual("fp-1", record.source_fingerprint)

        fetched = reg.get("doc-1")
        self.assertIsNotNone(fetched)
        self.assertEqual("doc-1", fetched.document_id)

    def test_get_returns_none_for_unknown_id(self) -> None:
        reg = self._make_registry()
        self.assertIsNone(reg.get("nonexistent"))

    def test_persists_across_instances(self) -> None:
        """Data written by one registry must be reloaded by a second registry on the same DB."""
        text = self._make_text_file()
        reg1 = self._make_registry()
        reg1.register(
            document_id="doc-persist",
            text_path=text,
            filename="persist.pdf",
            processing_mode="native",
            pages=2,
            text_chars=500,
        )

        reg2 = self._make_registry()
        fetched = reg2.get("doc-persist")
        self.assertIsNotNone(fetched)
        self.assertEqual("persist.pdf", fetched.filename)


class UpdateTests(DocumentRegistryBase):
    def setUp(self) -> None:
        super().setUp()
        self._reg = self._make_registry()
        self._text = self._make_text_file()
        self._reg.register(
            document_id="doc-1",
            text_path=self._text,
            filename="doc.pdf",
            processing_mode="native",
            pages=3,
            text_chars=300,
        )

    def test_update_chunks(self) -> None:
        chunks = self._root / "doc.chunks.jsonl"
        chunks.touch()
        meta = self._root / "doc.meta.json"
        meta.touch()

        rec = self._reg.update_chunks(
            "doc-1",
            chunks_available=True,
            chunks_path=chunks,
            chunk_meta_path=meta,
            chunks_count=10,
            chunk_schema_version="v2",
        )
        self.assertTrue(rec.chunks_available)
        self.assertEqual(10, rec.chunks_count)
        self.assertEqual("v2", rec.chunk_schema_version)

        # Verify DB persistence
        reloaded = self._make_registry().get("doc-1")
        self.assertTrue(reloaded.chunks_available)
        self.assertEqual(10, reloaded.chunks_count)

    def test_update_title_author(self) -> None:
        rec = self._reg.update_title_author("doc-1", title="Animal Farm", author="Orwell")
        self.assertEqual("Animal Farm", rec.title)
        self.assertEqual("Orwell", rec.author)

        reloaded = self._make_registry().get("doc-1")
        self.assertEqual("Animal Farm", reloaded.title)

    def test_update_known_work_confidence(self) -> None:
        rec = self._reg.update_known_work_confidence("doc-1", confidence=0.95, source="openlibrary")
        self.assertAlmostEqual(0.95, rec.known_work_confidence)
        self.assertEqual("openlibrary", rec.known_work_source)

        reloaded = self._make_registry().get("doc-1")
        self.assertAlmostEqual(0.95, reloaded.known_work_confidence)

    def test_update_corpus_pending(self) -> None:
        rec = self._reg.update_corpus_pending("doc-1", pending=True)
        self.assertTrue(rec.known_work_confidence_pending)

        reloaded = self._make_registry().get("doc-1")
        self.assertTrue(reloaded.known_work_confidence_pending)

    def test_update_quality_score(self) -> None:
        rec = self._reg.update_quality_score("doc-1", quality_score=0.87)
        self.assertAlmostEqual(0.87, rec.quality_score)

        reloaded = self._make_registry().get("doc-1")
        self.assertAlmostEqual(0.87, reloaded.quality_score)

    def test_update_returns_none_for_unknown_id(self) -> None:
        self.assertIsNone(self._reg.update_title_author("ghost", title="X", author="Y"))

    def test_replace_files(self) -> None:
        new_text = self._make_text_file("new.txt")
        new_chunks = self._root / "new.chunks.jsonl"
        new_chunks.touch()
        new_meta = self._root / "new.meta.json"
        new_meta.touch()

        rec = self._reg.replace_files(
            "doc-1",
            text_path=new_text,
            chunks_path=new_chunks,
            chunk_meta_path=new_meta,
            chunks_count=20,
            chunk_schema_version="v3",
            processing_mode="ocr",
            text_chars=800,
            pages=6,
            quality_score=0.92,
        )
        self.assertEqual("ocr", rec.processing_mode)
        self.assertEqual(20, rec.chunks_count)
        self.assertAlmostEqual(0.92, rec.quality_score)

        reloaded = self._make_registry().get("doc-1")
        self.assertEqual("v3", reloaded.chunk_schema_version)


class FindTests(DocumentRegistryBase):
    def setUp(self) -> None:
        super().setUp()
        self._reg = self._make_registry()
        self._text = self._make_text_file()
        self._reg.register(
            document_id="doc-1",
            text_path=self._text,
            filename="doc.pdf",
            processing_mode="native",
            pages=1,
            text_chars=100,
            source_fingerprint="fp-abc",
        )
        self._reg.update_title_author("doc-1", title="Animal Farm", author="Orwell")

    def test_find_by_source_fingerprint_returns_match(self) -> None:
        found = self._reg.find_by_source_fingerprint("fp-abc")
        self.assertIsNotNone(found)
        self.assertEqual("doc-1", found.document_id)

    def test_find_by_source_fingerprint_returns_none_for_miss(self) -> None:
        self.assertIsNone(self._reg.find_by_source_fingerprint("fp-unknown"))

    def test_find_by_title_author_returns_match(self) -> None:
        found = self._reg.find_by_title_author("Animal Farm", "Orwell")
        self.assertIsNotNone(found)
        self.assertEqual("doc-1", found.document_id)

    def test_find_by_title_author_is_case_insensitive(self) -> None:
        found = self._reg.find_by_title_author("ANIMAL FARM", "ORWELL")
        self.assertIsNotNone(found)

    def test_find_by_title_author_excludes_self(self) -> None:
        found = self._reg.find_by_title_author("Animal Farm", "Orwell", exclude_id="doc-1")
        self.assertIsNone(found)

    def test_find_by_title_author_returns_none_for_empty_query(self) -> None:
        self.assertIsNone(self._reg.find_by_title_author(None, None))


class ListAndRemoveTests(DocumentRegistryBase):
    def test_list_all_returns_only_existing_text_files(self) -> None:
        reg = self._make_registry()
        existing = self._make_text_file("exists.txt")
        missing = self._root / "gone.txt"
        missing.write_text("x")
        reg.register(
            document_id="doc-exists",
            text_path=existing,
            filename="exists.pdf",
            processing_mode="native",
            pages=1,
            text_chars=10,
        )
        reg.register(
            document_id="doc-missing",
            text_path=missing,
            filename="gone.pdf",
            processing_mode="native",
            pages=1,
            text_chars=10,
        )
        missing.unlink()  # delete the file

        listed = reg.list_all()
        ids = {r.document_id for r in listed}
        self.assertIn("doc-exists", ids)
        self.assertNotIn("doc-missing", ids)

    def test_remove_deletes_from_cache_and_db(self) -> None:
        reg = self._make_registry()
        text = self._make_text_file()
        reg.register(
            document_id="doc-del",
            text_path=text,
            filename="del.pdf",
            processing_mode="native",
            pages=1,
            text_chars=10,
        )
        removed = reg.remove("doc-del")
        self.assertIsNotNone(removed)
        self.assertIsNone(reg.get("doc-del"))

        # Confirm DB deletion
        reloaded = self._make_registry().get("doc-del")
        self.assertIsNone(reloaded)

    def test_remove_returns_none_for_unknown_id(self) -> None:
        reg = self._make_registry()
        self.assertIsNone(reg.remove("ghost"))


class JsonMigrationTests(DocumentRegistryBase):
    def test_migrates_json_index_on_init(self) -> None:
        text = self._make_text_file("migrated.txt")
        index_path = self._root / "documents.index.json"
        payload = {
            "documents": [
                {
                    "document_id": "doc-migrated",
                    "text_path": str(text),
                    "filename": "migrated.pdf",
                    "processing_mode": "native",
                    "pages": 2,
                    "text_chars": 200,
                    "created_at": "2024-01-01T00:00:00+00:00",
                }
            ]
        }
        index_path.write_text(json.dumps(payload), encoding="utf-8")

        reg = DocumentRegistry(index_path)
        record = reg.get("doc-migrated")
        self.assertIsNotNone(record)
        self.assertEqual("migrated.pdf", record.filename)

    def test_json_index_renamed_after_migration(self) -> None:
        index_path = self._root / "documents.index.json"
        index_path.write_text(json.dumps({"documents": []}), encoding="utf-8")
        DocumentRegistry(index_path)
        self.assertFalse(index_path.exists())
        self.assertTrue((self._root / "documents.index.json.migrated").exists())


if __name__ == "__main__":
    unittest.main()
