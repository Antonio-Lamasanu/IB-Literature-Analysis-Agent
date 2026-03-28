"""
Benchmark runner for Exam Mode (Paper 1 and Paper 2 grading).

Loops over:
  Paper 1: passage (from paper1_sub.json) × model × sample_answer
  Paper 2: question × context_mode × model × sample_answer

Each combination calls grade_answer() from exam_service with a fixed sample
answer, recording the criteria scores and grading prompt for analysis.

Usage:
    python exam_benchmark.py
    python exam_benchmark.py --config path/to/exam_benchmark_config.json

Edit exam_benchmark_config.json to change models, questions, or sample answers.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow importing from the backend directory
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import llm_service as _llm_service_module
from document_registry import DocumentRegistry
from exam_service import grade_answer, GradingResult, _build_paper1_grading_prompt
from llm_service import LLMConfig, LLMService, RemoteLLMService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("exam_benchmark")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = _BACKEND_DIR / "exam_benchmark_config.json"
DOCUMENTS_INDEX_PATH = _BACKEND_DIR / "outputs" / "documents.index.json"

# Rough token estimate: 4 chars ≈ 1 token. Warn if Paper 1 prompt exceeds this
# at small context windows.
_TOKEN_WARNING_THRESHOLD = 1600


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        sys.exit(1)
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_models(config: dict[str, Any]) -> list[Path]:
    folder = Path(config["models_folder"])
    if not folder.exists():
        log.error("Models folder not found: %s", folder)
        sys.exit(1)

    explicit: list[str] = config.get("models", [])
    if explicit:
        paths = [folder / name for name in explicit]
        missing = [p for p in paths if not p.exists()]
        if missing:
            log.warning(
                "Skipping %d missing model file(s): %s",
                len(missing), [p.name for p in missing],
            )
        available = [p for p in paths if p.exists()]
        if not available:
            log.error("No model files found from explicit list.")
            sys.exit(1)
        return available

    paths = sorted(folder.glob("*.gguf"))
    if not paths:
        log.error("No .gguf files found in %s", folder)
        sys.exit(1)
    return paths


def resolve_paper2_chunks(config: dict[str, Any]) -> list[str] | None:
    """Return chunks paths for the two Paper 2 documents, or None if IDs are placeholders."""
    doc_ids: list[str] = config.get("paper2_document_ids", [])
    if not doc_ids or any("REPLACE_WITH" in d for d in doc_ids):
        log.warning(
            "paper2_document_ids contains placeholders — Paper 2 'chunks' mode will be skipped. "
            "Update exam_benchmark_config.json with real document IDs from documents.index.json."
        )
        return None

    if len(doc_ids) != 2:
        log.error("paper2_document_ids must contain exactly 2 IDs, got %d.", len(doc_ids))
        sys.exit(1)

    if not DOCUMENTS_INDEX_PATH.exists():
        log.error("Documents index not found: %s", DOCUMENTS_INDEX_PATH)
        sys.exit(1)

    registry = DocumentRegistry(DOCUMENTS_INDEX_PATH)
    chunks_paths: list[str] = []
    for doc_id in doc_ids:
        record = registry.get(doc_id)
        if record is None:
            log.error("Unknown document_id: %s", doc_id)
            sys.exit(1)
        if not record.chunks_path or not Path(record.chunks_path).exists():
            log.error("No chunks for document_id %s. Run generate-chunks first.", doc_id)
            sys.exit(1)
        chunks_paths.append(record.chunks_path)
    return chunks_paths


def resolve_paper2_doc_titles(config: dict[str, Any]) -> list[str]:
    """Return human-readable titles for the two Paper 2 documents."""
    doc_ids: list[str] = config.get("paper2_document_ids", [])
    if not doc_ids or any("REPLACE_WITH" in d for d in doc_ids) or not DOCUMENTS_INDEX_PATH.exists():
        return ["Work 1 (unknown)", "Work 2 (unknown)"]

    registry = DocumentRegistry(DOCUMENTS_INDEX_PATH)
    titles: list[str] = []
    for doc_id in doc_ids:
        record = registry.get(doc_id)
        if record:
            titles.append(f"{record.title or record.filename} by {record.author or 'Unknown'}")
        else:
            titles.append(f"Unknown ({doc_id[:8]})")
    return titles


def resolve_server_models(config: dict[str, Any]) -> list[tuple[str, str]] | None:
    default_url: str = (config.get("server_url") or "").strip()
    entries: list = config.get("server_models") or []
    if not entries:
        return None

    pairs: list[tuple[str, str]] = []
    for entry in entries:
        if isinstance(entry, str):
            if not default_url:
                log.warning("Skipping server model '%s' — no 'server_url' configured.", entry)
                continue
            pairs.append((default_url, entry))
        elif isinstance(entry, dict):
            alias = (entry.get("alias") or "").strip()
            url = (entry.get("url") or default_url or "").strip()
            if not alias or not url:
                log.warning("Skipping incomplete server_models entry: %s", entry)
                continue
            pairs.append((url, alias))
    return pairs or None


def _load_json_from_config_path(config: dict[str, Any], config_path: Path, key: str) -> list:
    path_str: str | None = config.get(key)
    if not path_str:
        log.error("'%s' not set in config.", key)
        sys.exit(1)
    p = (config_path.parent / path_str).resolve()
    if not p.exists():
        log.error("%s not found: %s", key, p)
        sys.exit(1)
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def load_paper1_passages(config: dict[str, Any], config_path: Path) -> list[dict]:
    return _load_json_from_config_path(config, config_path, "paper1_passages_path")


def load_paper1_solutions(config: dict[str, Any], config_path: Path) -> list[dict]:
    return _load_json_from_config_path(config, config_path, "paper1_solutions_path")


def load_paper2_questions(config: dict[str, Any], config_path: Path) -> list[dict]:
    return _load_json_from_config_path(config, config_path, "paper2_questions_path")


def load_paper2_solutions(config: dict[str, Any], config_path: Path) -> list[dict]:
    return _load_json_from_config_path(config, config_path, "paper2_solutions_path")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def open_output_dir(config: dict[str, Any]) -> Path:
    base = _BACKEND_DIR / config.get("output_dir", "outputs/benchmarks")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"exam_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_jsonl_record(jsonl_file: Path, record: dict[str, Any]) -> None:
    with jsonl_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# LLM service injection
# ---------------------------------------------------------------------------

def _inject_service(svc: LLMService | RemoteLLMService) -> None:
    """Override the module-level LLM service used by exam_service.grade_answer."""
    with _llm_service_module._service_instance_lock:
        _llm_service_module._service_instance = svc


def _clear_service() -> None:
    with _llm_service_module._service_instance_lock:
        _llm_service_module._service_instance = None


# ---------------------------------------------------------------------------
# Single grading run
# ---------------------------------------------------------------------------

def run_one_grading(
    *,
    paper_type: str,
    question: str,
    student_answer: str,
    passage_text: str | None = None,
    guiding_question: str | None = None,
    chunks_paths: list[str] | None = None,
    context_mode: str = "chunks",
    doc_titles: list[str] | None = None,
) -> dict[str, Any]:
    """Call grade_answer and return a flat dict of timing + scores."""
    t_start = time.perf_counter()
    try:
        result: GradingResult = grade_answer(
            paper_type=paper_type,
            question=question,
            student_answer=student_answer,
            passage_text=passage_text,
            chunks_paths=chunks_paths,
            context_mode=context_mode,
            doc_titles=doc_titles,
        )
        error = None
    except Exception as exc:
        log.error("  Grading error: %s", exc)
        elapsed = time.perf_counter() - t_start
        return {
            "error": str(exc),
            "score_A": None, "score_B": None, "score_C": None, "score_D": None,
            "feedback_A": None, "feedback_B": None, "feedback_C": None, "feedback_D": None,
            "total_score": None, "max_score": None,
            "overall_comments": None,
            "inference_seconds": round(elapsed, 3),
            "grading_prompt": "",
        }

    scores = {c.criterion: c for c in result.criteria}
    return {
        "error": None,
        "score_A": scores["A"].score if "A" in scores else None,
        "score_B": scores["B"].score if "B" in scores else None,
        "score_C": scores["C"].score if "C" in scores else None,
        "score_D": scores["D"].score if "D" in scores else None,
        "feedback_A": scores["A"].feedback if "A" in scores else None,
        "feedback_B": scores["B"].feedback if "B" in scores else None,
        "feedback_C": scores["C"].feedback if "C" in scores else None,
        "feedback_D": scores["D"].feedback if "D" in scores else None,
        "total_score": result.total_score,
        "max_score": result.max_score,
        "overall_comments": result.overall_comments,
        "inference_seconds": round(result.inference_seconds, 3),
        "grading_prompt": result.prompt,
    }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(config_path: Path) -> None:  # noqa: C901
    config = load_config(config_path)

    raw_ctx = config.get("n_ctx_values", config.get("n_ctx", 4096))
    n_ctx_values: list[int] = [int(v) for v in (raw_ctx if isinstance(raw_ctx, list) else [raw_ctx])]
    n_threads: int = int(config.get("n_threads", 4))
    max_tokens: int = int(config.get("max_tokens", 800))
    temperature: float = float(config.get("temperature", 0.4))
    chat_model_keywords: list[str] = config.get("chat_models", [])

    model_paths = resolve_models(config)
    server_model_pairs = resolve_server_models(config)

    paper1_passages = load_paper1_passages(config, config_path)
    paper1_solutions = load_paper1_solutions(config, config_path)
    paper2_questions = load_paper2_questions(config, config_path)
    paper2_solutions = load_paper2_solutions(config, config_path)
    paper2_context_modes: list[str] = config.get("paper2_context_modes", ["titles_only", "chunks"])

    chunks_paths = resolve_paper2_chunks(config)
    doc_titles = resolve_paper2_doc_titles(config)

    out_dir = open_output_dir(config)
    jsonl_file = out_dir / "results.jsonl"
    benchmark_ts = datetime.now(timezone.utc).isoformat()

    log.info("=" * 60)
    log.info("Exam benchmark started")
    log.info("Paper 1 passages : %d", len(paper1_passages))
    log.info("Paper 2 questions: %d", len(paper2_questions))
    log.info("Context modes    : %s", paper2_context_modes)
    log.info("Models           : %d", len(model_paths))
    if server_model_pairs:
        log.info("Server models    : %d", len(server_model_pairs))
    log.info("Output           : %s", out_dir)
    log.info("=" * 60)

    all_records: list[dict[str, Any]] = []

    # Build a lookup: passage_title -> list of answers
    p1_answers_by_title: dict[str, list[dict]] = {}
    for entry in paper1_solutions:
        title = entry.get("passage_title", "")
        p1_answers_by_title[title] = entry.get("answers", [])

    # Build a lookup: question_id -> list of answers
    p2_answers_by_qid: dict[int, list[dict]] = {}
    for entry in paper2_solutions:
        qid = int(entry.get("question_id", 0))
        p2_answers_by_qid[qid] = entry.get("answers", [])

    def _run_for_model(
        model_name: str,
        n_ctx: int,
        is_server: bool = False,
        server_url: str = "",
    ) -> None:
        if is_server:
            svc: LLMService | RemoteLLMService = RemoteLLMService(
                server_url=server_url,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            use_chat = any(kw.lower() in model_name.lower() for kw in chat_model_keywords)
            matching = [p for p in model_paths if p.name == model_name]
            if not matching:
                log.error("Cannot find model path for %s", model_name)
                return
            cfg = LLMConfig(
                enabled=True,
                model_path=str(matching[0]),
                n_ctx=n_ctx,
                n_threads=n_threads,
                max_tokens=max_tokens,
                temperature=temperature,
                use_chat_api=use_chat,
            )
            svc = LLMService(cfg)

        _inject_service(svc)

        # ---- Paper 1 ----
        for passage_entry in paper1_passages:
            passage_title = passage_entry.get("title", "unknown")
            passage_text = passage_entry.get("passage", "")
            guiding_question = passage_entry.get("guiding_question", "")
            answers = p1_answers_by_title.get(passage_title, [])

            if not answers:
                log.warning("  No sample answers for passage '%s' — skipping.", passage_title)
                continue

            for ans in answers:
                answer_label = ans.get("answer_label", "?")
                student_answer = ans.get("answer", "")
                if not student_answer or "ADD YOUR SAMPLE" in student_answer:
                    log.warning(
                        "  Skipping placeholder answer for passage '%s' label='%s'.",
                        passage_title, answer_label,
                    )
                    continue

                # Token budget warning for small context windows
                estimated_prompt = _build_paper1_grading_prompt(
                    passage=passage_text,
                    guiding_question=guiding_question,
                    student_answer=student_answer,
                )
                estimated_tokens = len(estimated_prompt) // 4
                if n_ctx <= 2048 and estimated_tokens > _TOKEN_WARNING_THRESHOLD:
                    log.warning(
                        "  Paper 1 prompt for '%s' is ~%d tokens but n_ctx=%d — "
                        "output may be truncated.",
                        passage_title, estimated_tokens, n_ctx,
                    )

                log.info(
                    "  P1 | passage='%s' | answer='%s' | ~%d tokens",
                    passage_title, answer_label, estimated_tokens,
                )
                grading = run_one_grading(
                    paper_type="paper1",
                    question=guiding_question,
                    student_answer=student_answer,
                    passage_text=passage_text,
                    context_mode="passage",
                )
                log.info(
                    "    -> %s/%s  (%.1fs)",
                    grading["total_score"], grading["max_score"], grading["inference_seconds"],
                )

                record: dict[str, Any] = {
                    "run_id": str(uuid.uuid4()),
                    "benchmark_timestamp": benchmark_ts,
                    "paper_type": "paper1",
                    "model": model_name,
                    "n_ctx": 0 if is_server else n_ctx,
                    "n_threads": 0 if is_server else n_threads,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "passage_title": passage_title,
                    "guiding_question": guiding_question,
                    "question_id": None,
                    "question_text": guiding_question,
                    "answer_label": answer_label,
                    "student_answer": student_answer,
                    "context_mode": "passage",
                    **grading,
                }
                write_jsonl_record(jsonl_file, record)
                all_records.append(record)

        # ---- Paper 2 ----
        for q in paper2_questions:
            qid = int(q.get("id", 0))
            question_text = q.get("text", "")
            answers = p2_answers_by_qid.get(qid, [])

            if not answers:
                log.warning("  No sample answers for P2 question id=%d — skipping.", qid)
                continue

            for ctx_mode in paper2_context_modes:
                if ctx_mode == "chunks" and chunks_paths is None:
                    log.warning("  Skipping P2 chunks mode — no valid document IDs configured.")
                    continue

                for ans in answers:
                    answer_label = ans.get("answer_label", "?")
                    student_answer = ans.get("answer", "")
                    if not student_answer or "ADD YOUR SAMPLE" in student_answer:
                        log.warning(
                            "  Skipping placeholder answer for P2 q=%d label='%s'.",
                            qid, answer_label,
                        )
                        continue

                    log.info(
                        "  P2 | q=%d | mode=%s | answer='%s'",
                        qid, ctx_mode, answer_label,
                    )
                    grading = run_one_grading(
                        paper_type="paper2",
                        question=question_text,
                        student_answer=student_answer,
                        chunks_paths=chunks_paths if ctx_mode == "chunks" else None,
                        context_mode=ctx_mode,
                        doc_titles=doc_titles if ctx_mode == "titles_only" else None,
                    )
                    log.info(
                        "    -> %s/%s  (%.1fs)",
                        grading["total_score"], grading["max_score"], grading["inference_seconds"],
                    )

                    record = {
                        "run_id": str(uuid.uuid4()),
                        "benchmark_timestamp": benchmark_ts,
                        "paper_type": "paper2",
                        "model": model_name,
                        "n_ctx": 0 if is_server else n_ctx,
                        "n_threads": 0 if is_server else n_threads,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "passage_title": None,
                        "guiding_question": None,
                        "question_id": qid,
                        "question_text": question_text,
                        "answer_label": answer_label,
                        "student_answer": student_answer,
                        "context_mode": ctx_mode,
                        **grading,
                    }
                    write_jsonl_record(jsonl_file, record)
                    all_records.append(record)

        if not is_server:
            log.info("  Unloading model...")
            svc.unload()  # type: ignore[union-attr]
            gc.collect()
        _clear_service()

    # ---- Local GGUF models ----
    for model_path in model_paths:
        model_name = model_path.name
        for n_ctx in n_ctx_values:
            log.info("\n>>> MODEL: %s  n_ctx=%d", model_name, n_ctx)
            _run_for_model(model_name, n_ctx, is_server=False)

    # ---- Server models ----
    if server_model_pairs:
        for server_url, model_alias in server_model_pairs:
            log.info("\n>>> SERVER MODEL: %s  (%s)", model_alias, server_url)
            _run_for_model(model_alias, 0, is_server=True, server_url=server_url)

    _write_report(out_dir / "report.txt", all_records)
    log.info("\nExam benchmark complete.")
    log.info("Results : %s", jsonl_file)
    log.info("Report  : %s", out_dir / "report.txt")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(report_file: Path, records: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("EXAM BENCHMARK REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    models_seen: list[str] = []
    for r in records:
        if r["model"] not in models_seen:
            models_seen.append(r["model"])

    for model_name in models_seen:
        model_records = [r for r in records if r["model"] == model_name]
        lines.append(f"MODEL: {model_name}")
        lines.append("-" * 80)

        p1_recs = [r for r in model_records if r["paper_type"] == "paper1"]
        if p1_recs:
            lines.append("  PAPER 1")
            lines.append(f"  {'Passage':<30} {'Answer':<15} {'A':>4} {'B':>4} {'C':>4} {'D':>4} {'Tot':>5} {'Max':>5} {'Infer(s)':>9}")
            lines.append(f"  {'-'*30} {'-'*15} {'--':>4} {'--':>4} {'--':>4} {'--':>4} {'---':>5} {'---':>5} {'--------':>9}")
            for r in p1_recs:
                title = (r["passage_title"] or "")[:28]
                label = (r["answer_label"] or "")[:13]
                sA = r["score_A"] if r["score_A"] is not None else "-"
                sB = r["score_B"] if r["score_B"] is not None else "-"
                sC = r["score_C"] if r["score_C"] is not None else "-"
                sD = r["score_D"] if r["score_D"] is not None else "-"
                tot = r["total_score"] if r["total_score"] is not None else "-"
                mx = r["max_score"] if r["max_score"] is not None else "-"
                infer = r["inference_seconds"]
                lines.append(f"  {title:<30} {label:<15} {sA!s:>4} {sB!s:>4} {sC!s:>4} {sD!s:>4} {tot!s:>5} {mx!s:>5} {infer:>9.1f}")
            lines.append("")

        p2_recs = [r for r in model_records if r["paper_type"] == "paper2"]
        if p2_recs:
            lines.append("  PAPER 2")
            lines.append(f"  {'Q':>3} {'Mode':<12} {'Answer':<15} {'A':>4} {'B':>4} {'C':>4} {'D':>4} {'Tot':>5} {'Max':>5} {'Infer(s)':>9}")
            lines.append(f"  {'--':>3} {'-'*12} {'-'*15} {'--':>4} {'--':>4} {'--':>4} {'--':>4} {'---':>5} {'---':>5} {'--------':>9}")
            for r in p2_recs:
                qid = r["question_id"] or "-"
                mode = (r["context_mode"] or "")[:10]
                label = (r["answer_label"] or "")[:13]
                sA = r["score_A"] if r["score_A"] is not None else "-"
                sB = r["score_B"] if r["score_B"] is not None else "-"
                sC = r["score_C"] if r["score_C"] is not None else "-"
                sD = r["score_D"] if r["score_D"] is not None else "-"
                tot = r["total_score"] if r["total_score"] is not None else "-"
                mx = r["max_score"] if r["max_score"] is not None else "-"
                infer = r["inference_seconds"]
                lines.append(f"  {qid!s:>3} {mode:<12} {label:<15} {sA!s:>4} {sB!s:>4} {sC!s:>4} {sD!s:>4} {tot!s:>5} {mx!s:>5} {infer:>9.1f}")
            lines.append("")

        lines.append("")

    # Full prompts + feedback section
    lines.append("=" * 80)
    lines.append("GRADING DETAILS")
    lines.append("=" * 80)
    for r in records:
        lines.append(f"\n[{r['paper_type'].upper()}] model={r['model']}")
        if r["paper_type"] == "paper1":
            lines.append(f"Passage : {r['passage_title']}")
            lines.append(f"Question: {r['guiding_question']}")
        else:
            lines.append(f"Q{r['question_id']}: {(r['question_text'] or '')[:120]}")
            lines.append(f"Mode    : {r['context_mode']}")
        lines.append(f"Answer  : {r['answer_label']}")
        lines.append(f"Scores  : A={r['score_A']} B={r['score_B']} C={r['score_C']} D={r['score_D']}  total={r['total_score']}/{r['max_score']}")
        lines.append(f"Overall : {r['overall_comments']}")
        if r.get("error"):
            lines.append(f"ERROR   : {r['error']}")
        lines.append(f"Inference: {r['inference_seconds']}s")
        lines.append("")

    report_file.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exam mode benchmark (Paper 1 and Paper 2 grading)")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to exam_benchmark_config.json (default: backend/exam_benchmark_config.json)",
    )
    args = parser.parse_args()
    run_benchmark(args.config)
