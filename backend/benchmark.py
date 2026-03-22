"""
Benchmark runner for GGUF models — Learn mode only.

Loops over: model x prompt_format x questions
  Q1, Q2  — cold (model reloaded each time)
  Q3      — cold, starts a followup session
  Q4, Q5  — followup to Q3 (same model session, history accumulated)

Usage:
    python benchmark.py
    python benchmark.py --config path/to/benchmark_config.json

Edit benchmark_config.json to change questions, models, or inference settings.
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

from document_registry import DocumentRegistry
from llm_service import LLMConfig, LLMService
from retrieval import build_chat_context_result

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = _BACKEND_DIR / "benchmark_config.json"
DOCUMENTS_INDEX_PATH = _BACKEND_DIR / "outputs" / "documents.index.json"

PROMPT_FORMATS = ["base_knowledge", "rag"]

# Q indices (0-based) that are "cold" — model is reloaded before them
COLD_QUESTION_INDICES = {0, 1, 2}

# Q indices that are followups — they share the session started at Q3 (index 2)
FOLLOWUP_QUESTION_INDICES = {3, 4}


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
            log.error("Model file(s) not found: %s", missing)
            sys.exit(1)
        return paths

    paths = sorted(folder.glob("*.gguf"))
    if not paths:
        log.error("No .gguf files found in %s", folder)
        sys.exit(1)
    return paths


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def open_output_dir(config: dict[str, Any]) -> Path:
    base = _BACKEND_DIR / config.get("output_dir", "outputs/benchmarks")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_jsonl_record(jsonl_file: Path, record: dict[str, Any]) -> None:
    with jsonl_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Prompt building for base_knowledge format
# ---------------------------------------------------------------------------

def build_base_knowledge_prompt(
    title: str,
    author: str,
    session_messages: list[dict[str, str]],
    question: str,
) -> str:
    history_lines = []
    for msg in session_messages:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{speaker}: {msg['content']}")
    conversation_history = "\n".join(history_lines).strip() or "(none)"

    return (
        f'SYSTEM INSTRUCTIONS:\n'
        f'- You are answering questions about "{title}" by {author}.\n'
        f'- Answer from your general knowledge of this literary work.\n'
        f'- Be analytical and specific.\n\n'
        f'RECENT CONVERSATION:\n{conversation_history}\n\n'
        f'LATEST USER QUESTION:\nUser: {question}\n'
        f'Assistant:'
    )


# ---------------------------------------------------------------------------
# Single question runner
# ---------------------------------------------------------------------------

def run_one_question(
    *,
    llm_service: LLMService,
    prompt_format: str,
    question: str,
    session_messages: list[dict[str, str]],
    title: str,
    author: str,
    chunks_path: str | None,
    document_text: str,
    max_history_messages: int = 10,
) -> dict[str, Any]:
    """Run one question and return timing + answer dict."""
    retrieval_time_s = 0.0
    prompt_build_time_s = 0.0

    if prompt_format == "rag":
        t0 = time.perf_counter()
        ctx_result = build_chat_context_result(
            document_text,
            question,
            persisted_chunks_path=chunks_path,
            document_id=None,
        )
        retrieval_time_s = time.perf_counter() - t0
        doc_context = ctx_result.context

        # Build message list for the LLM call (include session history + new question)
        messages_for_llm = list(session_messages) + [{"role": "user", "content": question}]

        t1 = time.perf_counter()
        result = llm_service.generate_reply_with_debug(
            document_text=doc_context,
            messages=messages_for_llm,
            max_history_messages=max_history_messages,
        )
        prompt_build_time_s = result.prompt_build_seconds
        inference_time_s = result.inference_seconds
        answer = result.reply
        final_prompt = result.final_prompt

    else:  # base_knowledge
        messages_for_prompt = list(session_messages)
        t1 = time.perf_counter()
        prompt = build_base_knowledge_prompt(title, author, messages_for_prompt, question)
        prompt_build_time_s = time.perf_counter() - t1

        result = llm_service.generate_raw_reply(prompt)
        inference_time_s = result.inference_seconds
        answer = result.reply
        final_prompt = result.final_prompt

    total_time_s = retrieval_time_s + prompt_build_time_s + inference_time_s

    return {
        "answer": answer,
        "final_prompt": final_prompt,
        "retrieval_time_s": round(retrieval_time_s, 3),
        "prompt_build_time_s": round(prompt_build_time_s, 3),
        "inference_time_s": round(inference_time_s, 3),
        "total_time_s": round(total_time_s, 3),
    }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(config_path: Path) -> None:
    config = load_config(config_path)

    document_id: str = config.get("document_id", "")
    if not document_id or document_id == "REPLACE_WITH_DOCUMENT_ID":
        log.error("Set 'document_id' in %s before running.", config_path)
        sys.exit(1)

    questions: list[str] = config.get("questions", [])
    if len(questions) < 5:
        log.error("Expected 5 questions in config, got %d.", len(questions))
        sys.exit(1)

    raw_ctx = config.get("n_ctx_values", config.get("n_ctx", 2048))
    n_ctx_values: list[int] = [int(v) for v in (raw_ctx if isinstance(raw_ctx, list) else [raw_ctx])]
    n_threads: int = int(config.get("n_threads", 4))
    max_tokens: int = int(config.get("max_tokens", 512))
    temperature: float = float(config.get("temperature", 0.2))

    # Load document record
    registry = DocumentRegistry(DOCUMENTS_INDEX_PATH)
    record = registry.get(document_id)
    if record is None:
        log.error("Document not found: %s", document_id)
        sys.exit(1)

    title = record.title or "Unknown"
    author = record.author or "Unknown"
    chunks_path = record.chunks_path

    if not chunks_path or not Path(chunks_path).exists():
        log.error(
            "Document has no chunks. Run /api/documents/{id}/generate-chunks first.\n"
            "chunks_path=%s",
            chunks_path,
        )
        sys.exit(1)

    # Load document text (used as fallback by retrieval if chunks miss)
    document_text = ""
    if record.text_path and Path(record.text_path).exists():
        document_text = Path(record.text_path).read_text(encoding="utf-8")

    model_paths = resolve_models(config)
    out_dir = open_output_dir(config)
    jsonl_file = out_dir / "results.jsonl"
    report_file = out_dir / "report.txt"
    benchmark_ts = datetime.now(timezone.utc).isoformat()

    log.info("=" * 60)
    log.info("Benchmark started")
    log.info("Document : %s  (%s)", record.filename, document_id)
    log.info("Title    : %s", title)
    log.info("Author   : %s", author)
    log.info("Models   : %d", len(model_paths))
    log.info("Formats  : %s", PROMPT_FORMATS)
    log.info("Questions: %d", len(questions))
    log.info("n_ctx_values=%s  n_threads=%d  max_tokens=%d  temperature=%s",
             n_ctx_values, n_threads, max_tokens, temperature)
    log.info("Output   : %s", out_dir)
    log.info("=" * 60)

    all_records: list[dict[str, Any]] = []

    for model_path in model_paths:
        model_name = model_path.name
        log.info("\n>>> MODEL: %s", model_name)

        for n_ctx in n_ctx_values:
            log.info("  n_ctx: %d", n_ctx)

            for prompt_format in PROMPT_FORMATS:
                log.info("    Format: %s", prompt_format)

                llm_cfg = LLMConfig(
                    enabled=True,
                    model_path=str(model_path),
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # One service instance per (model, n_ctx, format) — avoids cross-state
                llm_service = LLMService(llm_cfg)

                # Session state for Q3-Q5 (accumulated across followup questions)
                followup_session: list[dict[str, str]] = []

                for q_idx, question in enumerate(questions):
                    is_cold = q_idx in COLD_QUESTION_INDICES
                    is_followup = q_idx in FOLLOWUP_QUESTION_INDICES
                    q_type = "followup" if is_followup else "cold"

                    if is_cold and q_idx > 0:
                        log.info("      Unloading model for cold reset...")
                        llm_service.unload()
                        gc.collect()

                    log.info(
                        "      Q%d [%s] %s ...",
                        q_idx + 1,
                        q_type,
                        question[:60] + ("..." if len(question) > 60 else ""),
                    )

                    # Session messages to pass for context
                    session_messages = followup_session if is_followup else []

                    t_start = time.perf_counter()
                    try:
                        result = run_one_question(
                            llm_service=llm_service,
                            prompt_format=prompt_format,
                            question=question,
                            session_messages=session_messages,
                            title=title,
                            author=author,
                            chunks_path=chunks_path,
                            document_text=document_text,
                        )
                        error = None
                    except Exception as exc:
                        log.error("      ERROR on Q%d: %s", q_idx + 1, exc)
                        result = {
                            "answer": "",
                            "final_prompt": "",
                            "retrieval_time_s": 0.0,
                            "prompt_build_time_s": 0.0,
                            "inference_time_s": round(time.perf_counter() - t_start, 3),
                            "total_time_s": round(time.perf_counter() - t_start, 3),
                        }
                        error = str(exc)

                    log.info(
                        "      -> retrieval=%.1fs  inference=%.1fs  total=%.1fs",
                        result["retrieval_time_s"],
                        result["inference_time_s"],
                        result["total_time_s"],
                    )

                    # Accumulate session for followup questions (Q3 starts at index 2)
                    if q_idx == 2:
                        # Q3 starts the followup session
                        followup_session = [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": result["answer"]},
                        ]
                    elif is_followup and result["answer"]:
                        followup_session.append({"role": "user", "content": question})
                        followup_session.append({"role": "assistant", "content": result["answer"]})

                    record_entry: dict[str, Any] = {
                        "run_id": str(uuid.uuid4()),
                        "benchmark_timestamp": benchmark_ts,
                        "model": model_name,
                        "prompt_format": prompt_format,
                        "question_index": q_idx + 1,
                        "question_type": q_type,
                        "question": question,
                        "answer": result["answer"],
                        "final_prompt": result["final_prompt"],
                        "n_ctx": n_ctx,
                        "n_threads": n_threads,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "retrieval_time_s": result["retrieval_time_s"],
                        "prompt_build_time_s": result["prompt_build_time_s"],
                        "inference_time_s": result["inference_time_s"],
                        "total_time_s": result["total_time_s"],
                    }
                    if error:
                        record_entry["error"] = error

                    write_jsonl_record(jsonl_file, record_entry)
                    all_records.append(record_entry)

                # Unload after the last question of each format
                log.info("    Unloading model after format=%s", prompt_format)
                llm_service.unload()
                gc.collect()

    # Write text report
    _write_report(report_file, all_records, questions, title, author, document_id)
    log.info("\nBenchmark complete.")
    log.info("Results : %s", jsonl_file)
    log.info("Report  : %s", report_file)


def _write_report(
    report_file: Path,
    records: list[dict[str, Any]],
    questions: list[str],
    title: str,
    author: str,
    document_id: str,
) -> None:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("BENCHMARK REPORT")
    lines.append(f"Document : {title} by {author}")
    lines.append(f"Doc ID   : {document_id}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # Group by model, then n_ctx
    models_seen: list[str] = []
    for r in records:
        if r["model"] not in models_seen:
            models_seen.append(r["model"])

    n_ctx_values_seen: list[int] = []
    for r in records:
        if r["n_ctx"] not in n_ctx_values_seen:
            n_ctx_values_seen.append(r["n_ctx"])

    for model_name in models_seen:
        lines.append(f"MODEL: {model_name}")
        lines.append("-" * 80)

        for n_ctx in n_ctx_values_seen:
            lines.append(f"  n_ctx={n_ctx}")

            for fmt in PROMPT_FORMATS:
                cell_records = [
                    r for r in records
                    if r["model"] == model_name and r["n_ctx"] == n_ctx and r["prompt_format"] == fmt
                ]
                if not cell_records:
                    continue

                lines.append(f"    Prompt format: {fmt}")
                lines.append(f"    {'Q#':<4} {'Type':<10} {'Retr(s)':<9} {'Infer(s)':<10} {'Total(s)':<10} Answer preview")
                lines.append(f"    {'-'*4} {'-'*10} {'-'*9} {'-'*10} {'-'*10} {'-'*40}")

                for r in sorted(cell_records, key=lambda x: x["question_index"]):
                    q_num = r["question_index"]
                    q_type = r["question_type"]
                    retr = r["retrieval_time_s"]
                    infer = r["inference_time_s"]
                    total = r["total_time_s"]
                    preview = (r["answer"] or "[ERROR]")[:60].replace("\n", " ")
                    if len(r["answer"]) > 60:
                        preview += "..."
                    lines.append(f"    {q_num:<4} {q_type:<10} {retr:<9.1f} {infer:<10.1f} {total:<10.1f} {preview}")

                lines.append("")

            # Per-format timing summary for this n_ctx
            lines.append(f"    Timing summary for n_ctx={n_ctx}:")
            lines.append(f"    {'Format':<18} {'Metric':<12} {'Q1':>6} {'Q2':>6} {'Q3':>6} {'Q4':>6} {'Q5':>6} {'Total':>8}")
            lines.append(f"    {'-'*18} {'-'*12} {'---':>6} {'---':>6} {'---':>6} {'---':>6} {'---':>6} {'-----':>8}")
            for fmt in PROMPT_FORMATS:
                fmt_ctx_records = sorted(
                    [r for r in records if r["model"] == model_name and r["n_ctx"] == n_ctx and r["prompt_format"] == fmt],
                    key=lambda x: x["question_index"],
                )
                if not fmt_ctx_records:
                    continue
                retr_times = [r["retrieval_time_s"] for r in fmt_ctx_records]
                infer_times = [r["inference_time_s"] for r in fmt_ctx_records]
                total_retr = sum(retr_times)
                total_infer = sum(infer_times)
                retr_cols = "  ".join(f"{t:>6.1f}" for t in retr_times)
                infer_cols = "  ".join(f"{t:>6.1f}" for t in infer_times)
                lines.append(f"    {fmt:<18} {'retrieval':<12} {retr_cols}  {total_retr:>8.1f}")
                lines.append(f"    {'':<18} {'inference':<12} {infer_cols}  {total_infer:>8.1f}")
            lines.append("")

        lines.append("")

    # Full answers section
    lines.append("=" * 80)
    lines.append("FULL ANSWERS")
    lines.append("=" * 80)
    lines.append("")

    for model_name in models_seen:
        for n_ctx in n_ctx_values_seen:
            for fmt in PROMPT_FORMATS:
                cell_records = sorted(
                    [
                        r for r in records
                        if r["model"] == model_name and r["n_ctx"] == n_ctx and r["prompt_format"] == fmt
                    ],
                    key=lambda x: x["question_index"],
                )
                if not cell_records:
                    continue

                lines.append(f"--- {model_name} | n_ctx={n_ctx} | {fmt} ---")
                for r in cell_records:
                    q_num = r["question_index"]
                    q_text = questions[q_num - 1] if q_num <= len(questions) else r["question"]
                    lines.append(f"\nQ{q_num} [{r['question_type']}]: {q_text}")
                    lines.append(f"Retrieval: {r['retrieval_time_s']:.1f}s  Inference: {r['inference_time_s']:.1f}s")
                    lines.append(f"Prompt sent to model:\n{r.get('final_prompt', '(not recorded)')}")
                    lines.append(f"Answer:\n{r['answer'] or '[ERROR: ' + r.get('error', 'unknown') + ']'}")
                    lines.append("")
                lines.append("")

    report_file.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GGUF model benchmark — Learn mode")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to benchmark_config.json (default: backend/benchmark_config.json)",
    )
    args = parser.parse_args()
    run_benchmark(args.config)
