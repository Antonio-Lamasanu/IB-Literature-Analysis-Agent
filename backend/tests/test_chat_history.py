from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from chat_history import append_chat_history_turn, create_chat_history_turn, load_chat_history
from document_registry import DocumentRegistry
from main import app
import main
from retrieval import build_chat_context_result, build_chat_context_result_with_history, retrieve_relevant_history_turns


def _write_chunks_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _build_chunk_record(*, unit_id: str, text: str, chapter: int = 1, page: int = 1) -> dict:
    return {
        "unit_id": unit_id,
        "doc_id": "doc1",
        "position": {
            "chapter": chapter,
            "index_in_chapter": 1,
            "absolute_percent": 0.1,
            "page_start": page,
            "page_end": page,
        },
        "content": {
            "text": text,
            "word_count": len(text.split()),
            "token_estimate": max(1, int(len(text.split()) * 1.33)),
        },
        "metadata": {
            "unit_type": "mixed",
            "character_mentions": ["Squealer"],
            "has_dialogue": False,
            "dialogue_ratio": 0.0,
        },
    }


class ChatHistoryPersistenceTests(unittest.TestCase):
    def test_append_and_load_round_trip_creates_history_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir) / "chat_history"
            turn = create_chat_history_turn(
                document_id="doc-1",
                user_query="Who is Boxer?",
                assistant_answer="Boxer is the hardworking cart-horse.",
                retrieval_mode_used="chunks_only",
            )

            history_path = append_chat_history_turn("doc-1", turn, storage_dir)

            self.assertTrue(history_path.exists())
            loaded_turns = load_chat_history("doc-1", storage_dir)
            self.assertEqual(1, len(loaded_turns))
            self.assertEqual("Who is Boxer?", loaded_turns[0].user_query)
            self.assertEqual("Boxer is the hardworking cart-horse.", loaded_turns[0].assistant_answer)


class HistoryRetrievalTests(unittest.TestCase):
    def test_history_retrieval_searches_query_and_answer_fields(self) -> None:
        turn_from_answer = create_chat_history_turn(
            document_id="doc-1",
            user_query="What happened earlier?",
            assistant_answer="Old Major gives the speech that inspires the rebellion.",
        )
        turn_from_query = create_chat_history_turn(
            document_id="doc-1",
            user_query="Who leads the rebellion speech?",
            assistant_answer="It is an important moment in the barn.",
        )

        ranked = retrieve_relevant_history_turns(
            [turn_from_answer, turn_from_query],
            "Who gives the rebellion speech?",
            max_candidates=5,
        )

        self.assertGreaterEqual(len(ranked), 2)
        ranked_turn_ids = {item.turn_id for item in ranked[:2]}
        self.assertIn(turn_from_answer.turn_id, ranked_turn_ids)
        self.assertIn(turn_from_query.turn_id, ranked_turn_ids)


class ChatRetrievalModeTests(unittest.TestCase):
    def test_combined_mode_mixes_history_and_chunk_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks_path = Path(tmpdir) / "doc1.chunks.jsonl"
            _write_chunks_jsonl(
                chunks_path,
                [
                    _build_chunk_record(
                        unit_id="chunk-1",
                        text="Squealer explains that the pigs need the milk and apples for their health.",
                    )
                ],
            )
            history_turn = create_chat_history_turn(
                document_id="doc1",
                user_query="Why do the pigs keep the milk?",
                assistant_answer="Squealer says the pigs need the milk because they must stay healthy.",
            )

            result = build_chat_context_result_with_history(
                document_text="",
                latest_user_question="What does Squealer say about the milk?",
                history_turns=[history_turn],
                chat_retrieval_mode="combined",
                history_reuse_threshold=99.0,
                history_max_candidates=3,
                history_max_excerpts=1,
                max_excerpts=1,
                retrieve_candidates=3,
                context_token_budget=1200,
                document_id="doc1",
                persisted_chunks_path=chunks_path,
            )

            self.assertEqual("combined", result.retrieval_mode)
            self.assertEqual({"chunk", "history"}, {item.source for item in result.final_candidate_mix})

    def test_history_first_reuses_answer_when_threshold_is_met(self) -> None:
        history_turn = create_chat_history_turn(
            document_id="doc1",
            user_query="Who is Snowball?",
            assistant_answer="Snowball is one of the pigs who leads the revolution.",
        )

        result = build_chat_context_result_with_history(
            document_text="",
            latest_user_question="Who is Snowball?",
            history_turns=[history_turn],
            chat_retrieval_mode="history_first",
            history_reuse_threshold=0.1,
            history_max_candidates=3,
            history_max_excerpts=1,
            max_excerpts=1,
            retrieve_candidates=3,
            context_token_budget=800,
            document_id="doc1",
            persisted_chunks_path=None,
        )

        self.assertTrue(result.history_reuse_hit)
        self.assertEqual("history_first_reuse", result.retrieval_mode)
        self.assertEqual(history_turn.assistant_answer, result.reused_answer)

    def test_history_first_falls_back_to_chunks_when_threshold_is_not_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks_path = Path(tmpdir) / "doc1.chunks.jsonl"
            _write_chunks_jsonl(
                chunks_path,
                [_build_chunk_record(unit_id="chunk-1", text="Boxer is the strongest horse on the farm.")],
            )
            history_turn = create_chat_history_turn(
                document_id="doc1",
                user_query="What color is the barn?",
                assistant_answer="The book does not focus on that detail.",
            )

            result = build_chat_context_result_with_history(
                document_text="",
                latest_user_question="Who is Boxer?",
                history_turns=[history_turn],
                chat_retrieval_mode="history_first",
                history_reuse_threshold=99.0,
                history_max_candidates=3,
                history_max_excerpts=1,
                max_excerpts=1,
                retrieve_candidates=3,
                context_token_budget=800,
                document_id="doc1",
                persisted_chunks_path=chunks_path,
            )

            self.assertFalse(result.history_reuse_hit)
            self.assertEqual("history_first_chunk_fallback", result.retrieval_mode)
            self.assertGreaterEqual(len(result.retrieved_excerpts), 1)

    def test_chunks_only_mode_matches_legacy_context_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks_path = Path(tmpdir) / "doc1.chunks.jsonl"
            _write_chunks_jsonl(
                chunks_path,
                [_build_chunk_record(unit_id="chunk-1", text="Napoleon drives Snowball off the farm.")],
            )

            legacy = build_chat_context_result(
                document_text="",
                latest_user_question="Who drives Snowball away?",
                max_excerpts=1,
                retrieve_candidates=3,
                context_token_budget=800,
                document_id="doc1",
                persisted_chunks_path=chunks_path,
            )
            with_history = build_chat_context_result_with_history(
                document_text="",
                latest_user_question="Who drives Snowball away?",
                history_turns=[
                    create_chat_history_turn(
                        document_id="doc1",
                        user_query="Unrelated question",
                        assistant_answer="Unrelated answer",
                    )
                ],
                chat_retrieval_mode="chunks_only",
                history_reuse_threshold=0.1,
                history_max_candidates=3,
                history_max_excerpts=1,
                max_excerpts=1,
                retrieve_candidates=3,
                context_token_budget=800,
                document_id="doc1",
                persisted_chunks_path=chunks_path,
            )

            self.assertEqual(legacy.context, with_history.context)
            self.assertEqual(legacy.retrieval_mode, with_history.retrieval_mode)
            self.assertEqual(
                [item.excerpt_id for item in legacy.retrieved_excerpts],
                [item.excerpt_id for item in with_history.retrieved_excerpts],
            )


class ChatEndpointReuseTests(unittest.TestCase):
    def test_history_reuse_path_skips_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            text_path = temp_root / "doc1.txt"
            text_path.write_text("", encoding="utf-8")
            storage_dir = temp_root / "chat_history"
            registry = DocumentRegistry(temp_root / "documents.index.json")
            record = registry.register(
                document_id="doc1",
                text_path=text_path,
                filename="doc1.pdf",
                processing_mode="native",
                pages=1,
                text_chars=0,
                source_fingerprint="fingerprint-1",
            )
            turn = create_chat_history_turn(
                document_id="doc1",
                user_query="Who is Snowball?",
                assistant_answer="Snowball is one of the leading pigs on the farm.",
            )
            append_chat_history_turn(record.document_id, turn, storage_dir)

            client = TestClient(app)
            with mock.patch.object(main, "document_registry", registry), mock.patch.object(
                main, "CHAT_HISTORY_ENABLED", True
            ), mock.patch.object(main, "CHAT_RETRIEVAL_MODE", "history_first"), mock.patch.object(
                main, "CHAT_HISTORY_REUSE_THRESHOLD", 0.1
            ), mock.patch.object(
                main, "CHAT_HISTORY_STORAGE_DIR", storage_dir
            ), mock.patch.object(
                main.llm_service,
                "generate_reply_with_debug",
                side_effect=AssertionError("inference should not run"),
            ):
                response = client.post(
                    "/api/chat",
                    json={
                        "document_id": "doc1",
                        "messages": [{"role": "user", "content": "Who is Snowball?"}],
                    },
                )

            self.assertEqual(200, response.status_code)
            payload = response.json()
            self.assertEqual(turn.assistant_answer, payload["reply"])
            self.assertTrue(payload["debug"]["history_reuse_hit"])
            self.assertEqual(0.0, payload["debug"]["timing"]["inference_seconds"])


if __name__ == "__main__":
    unittest.main()
