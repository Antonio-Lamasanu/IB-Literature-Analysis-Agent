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
import random
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
from llm_service import LLMConfig, LLMService, RemoteLLMService
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

PROMPT_FORMATS = ["base_knowledge", "rag", "rag_raw"]

# Q indices (0-based) that are "cold" — model is reloaded before them
COLD_QUESTION_INDICES = {0, 1, 2, 5}

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


def resolve_server_models(config: dict[str, Any]) -> list[tuple[str, str]] | None:
    """Return (server_url, model_alias) pairs if server config is present, else None.

    Each entry in ``server_models`` can be either:
      - a plain string  — uses the top-level ``server_url``
      - an object       — {"alias": "...", "url": "http://..."}  (url overrides server_url)
    """
    default_url: str = (config.get("server_url") or "").strip()
    entries: list = config.get("server_models") or []
    if not entries:
        if default_url:
            log.warning("'server_url' set but 'server_models' is empty — no server models to run.")
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
            if not alias:
                log.warning("Skipping server model entry with missing 'alias': %s", entry)
                continue
            if not url:
                log.warning("Skipping server model '%s' — no url or server_url configured.", alias)
                continue
            pairs.append((url, alias))
        else:
            log.warning("Skipping unrecognised server_models entry: %s", entry)

    return pairs or None


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


def resolve_documents(config: dict[str, Any]) -> list[tuple[str, Any]]:
    """Return list of (document_id, record) for all valid documents in config."""
    registry = DocumentRegistry(DOCUMENTS_INDEX_PATH)

    ids: list[str] = config.get("document_ids") or []
    if not ids:
        single = config.get("document_id", "")
        if single and single != "REPLACE_WITH_DOCUMENT_ID":
            ids = [single]

    if not ids:
        log.error("Set 'document_ids' (list) or 'document_id' in config before running.")
        sys.exit(1)

    result = []
    for doc_id in ids:
        record = registry.get(doc_id)
        if record is None:
            log.warning("Skipping unknown document_id: %s", doc_id)
            continue
        if not record.chunks_path or not Path(record.chunks_path).exists():
            log.warning("Skipping document %s — no chunks. Run generate-chunks first.", doc_id)
            continue
        result.append((doc_id, record))

    if not result:
        log.error("No valid documents available to benchmark.")
        sys.exit(1)
    return result


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

    if prompt_format in ("rag", "rag_raw"):
        t0 = time.perf_counter()
        ctx_result = build_chat_context_result(
            document_text,
            question,
            persisted_chunks_path=chunks_path,
            document_id=None,
            apply_sub_chunking=prompt_format == "rag",
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
            title=title,
            author=author,
        )
        prompt_build_time_s = result.prompt_build_seconds
        inference_time_s = result.inference_seconds
        answer = result.reply
        final_prompt = result.final_prompt

    else:  # base_knowledge
        messages_for_prompt = list(session_messages)
        result = llm_service.generate_base_knowledge_reply_with_debug(
            title=title,
            author=author,
            session_messages=messages_for_prompt,
            question=question,
        )
        prompt_build_time_s = result.prompt_build_seconds
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

def run_benchmark(config_path: Path, formats: list[str] | None = None) -> None:
    config = load_config(config_path)

    questions: list[str] = config.get("questions", [])
    if len(questions) < 5:
        log.error("Expected 5 questions in config, got %d.", len(questions))
        sys.exit(1)

    # Q6: randomly selected from learn_questions.json
    learn_q_path_str: str | None = config.get("learn_questions_path")
    if learn_q_path_str:
        learn_q_path = (config_path.parent / learn_q_path_str).resolve()
        if learn_q_path.exists():
            with learn_q_path.open(encoding="utf-8") as f:
                learn_questions = json.load(f)
            q6 = random.choice(learn_questions)["question"]
            questions = list(questions) + [q6]
            log.info("Q6 (random from learn_questions.json): %s", q6[:80] + ("..." if len(q6) > 80 else ""))
        else:
            log.warning("learn_questions_path not found: %s — skipping Q6", learn_q_path)
    else:
        log.warning("No 'learn_questions_path' in config — skipping Q6")

    raw_ctx = config.get("n_ctx_values", config.get("n_ctx", 2048))
    n_ctx_values: list[int] = [int(v) for v in (raw_ctx if isinstance(raw_ctx, list) else [raw_ctx])]
    n_threads: int = int(config.get("n_threads", 4))
    max_tokens: int = int(config.get("max_tokens", 512))
    temperature: float = float(config.get("temperature", 0.2))

    # Resolve which formats to run: CLI --formats > config "prompt_formats" > all
    config_formats = config.get("prompt_formats")
    if formats:
        active_formats = [f for f in formats if f in PROMPT_FORMATS]
    elif config_formats:
        active_formats = [f for f in config_formats if f in PROMPT_FORMATS]
    else:
        active_formats = list(PROMPT_FORMATS)
    if not active_formats:
        log.error("No valid formats selected. Choices: %s", PROMPT_FORMATS)
        sys.exit(1)

    documents = resolve_documents(config)
    model_paths = resolve_models(config)
    chat_model_keywords: list[str] = config.get("chat_models", [])
    server_model_pairs = resolve_server_models(config)
    server_max_tokens: int = int(config.get("server_max_tokens", max_tokens))
    server_temperature: float = float(config.get("server_temperature", 0.6))
    server_interactive: bool = bool(config.get("server_interactive", False))
    out_dir = open_output_dir(config)
    jsonl_file = out_dir / "results.jsonl"
    report_file = out_dir / "report.txt"
    benchmark_ts = datetime.now(timezone.utc).isoformat()

    log.info("=" * 60)
    log.info("Benchmark started")
    log.info("Documents: %d", len(documents))
    log.info("Models   : %d", len(model_paths))
    if server_model_pairs:
        log.info("Server   : %d model(s) via %s", len(server_model_pairs), server_model_pairs[0][0])
        if server_interactive:
            log.info("           interactive mode ON — will pause before each server model")
    log.info("Formats  : %s", active_formats)
    log.info("Questions: %d", len(questions))
    log.info("n_ctx_values=%s  n_threads=%d  max_tokens=%d  temperature=%s",
             n_ctx_values, n_threads, max_tokens, temperature)
    log.info("Output   : %s", out_dir)
    log.info("=" * 60)

    all_records: list[dict[str, Any]] = []

    for document_id, record in documents:
        title = record.title or "Unknown"
        author = record.author or "Unknown"
        chunks_path = record.chunks_path

        # Load document text (used as fallback by retrieval if chunks miss)
        document_text = ""
        if record.text_path and Path(record.text_path).exists():
            document_text = Path(record.text_path).read_text(encoding="utf-8")

        log.info("\n=== DOCUMENT: %s (%s) ===", record.filename, document_id)
        log.info("    Title : %s", title)
        log.info("    Author: %s", author)

        for model_path in model_paths:
            model_name = model_path.name
            log.info("\n>>> MODEL: %s", model_name)

            use_chat = any(
                kw.lower() in model_name.lower()
                for kw in chat_model_keywords
            )
            if use_chat:
                log.info("  (chat-completion API enabled for this model)")

            for n_ctx in n_ctx_values:
                log.info("  n_ctx: %d", n_ctx)

                for prompt_format in active_formats:
                    log.info("    Format: %s", prompt_format)

                    llm_cfg = LLMConfig(
                        enabled=True,
                        model_path=str(model_path),
                        n_ctx=n_ctx,
                        n_threads=n_threads,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        use_chat_api=use_chat,
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
                            "document_id": document_id,
                            "document_title": title,
                            "document_author": author,
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

        # --- Server models (llama.cpp server / remote OpenAI-compatible API) ---
        if server_model_pairs:
            for srv_idx, (server_url, model_alias) in enumerate(server_model_pairs):
                model_name = model_alias

                if server_interactive:
                    print(
                        f"\n{'=' * 60}\n"
                        f"SERVER MODEL {srv_idx + 1}/{len(server_model_pairs)}\n"
                        f"  Alias : {model_alias}\n"
                        f"  URL   : {server_url}\n"
                        f"\nPlease start (or restart) llama-server with this model,\n"
                        f"then press Enter to continue..."
                    )
                    input()
                    # Verify the server is reachable before proceeding
                    _svc_check = RemoteLLMService(server_url=server_url, model=model_alias)
                    if not _svc_check.is_chat_available():
                        log.error(
                            "Server at %s is not reachable. "
                            "Make sure llama-server is running and try again.",
                            server_url,
                        )
                        sys.exit(1)

                log.info("\n>>> SERVER MODEL: %s  (%s)", model_alias, server_url)

                for prompt_format in active_formats:
                    log.info("    Format: %s", prompt_format)

                    srv_service = RemoteLLMService(
                        server_url=server_url,
                        model=model_alias,
                        max_tokens=server_max_tokens,
                        temperature=server_temperature,
                    )

                    srv_followup_session: list[dict[str, str]] = []

                    for q_idx, question in enumerate(questions):
                        is_cold = q_idx in COLD_QUESTION_INDICES
                        is_followup = q_idx in FOLLOWUP_QUESTION_INDICES
                        q_type = "followup" if is_followup else "cold"

                        if is_cold and q_idx > 0:
                            log.info("      (server model — skipping unload for cold reset)")

                        log.info(
                            "      Q%d [%s] %s ...",
                            q_idx + 1,
                            q_type,
                            question[:60] + ("..." if len(question) > 60 else ""),
                        )

                        session_messages = srv_followup_session if is_followup else []

                        t_start = time.perf_counter()
                        try:
                            result = run_one_question(
                                llm_service=srv_service,
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

                        if q_idx == 2:
                            srv_followup_session = [
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": result["answer"]},
                            ]
                        elif is_followup and result["answer"]:
                            srv_followup_session.append({"role": "user", "content": question})
                            srv_followup_session.append({"role": "assistant", "content": result["answer"]})

                        record_entry: dict[str, Any] = {
                            "run_id": str(uuid.uuid4()),
                            "benchmark_timestamp": benchmark_ts,
                            "document_id": document_id,
                            "document_title": title,
                            "document_author": author,
                            "model": model_name,
                            "prompt_format": prompt_format,
                            "question_index": q_idx + 1,
                            "question_type": q_type,
                            "question": question,
                            "answer": result["answer"],
                            "final_prompt": result["final_prompt"],
                            "n_ctx": 0,
                            "n_threads": 0,
                            "max_tokens": server_max_tokens,
                            "temperature": server_temperature,
                            "retrieval_time_s": result["retrieval_time_s"],
                            "prompt_build_time_s": result["prompt_build_time_s"],
                            "inference_time_s": result["inference_time_s"],
                            "total_time_s": result["total_time_s"],
                        }
                        if error:
                            record_entry["error"] = error

                        write_jsonl_record(jsonl_file, record_entry)
                        all_records.append(record_entry)

                    log.info("    Done with server format=%s", prompt_format)

    # Write text report
    _write_report(report_file, all_records, questions)
    log.info("\nBenchmark complete.")
    log.info("Results : %s", jsonl_file)
    log.info("Report  : %s", report_file)


def _write_report(
    report_file: Path,
    records: list[dict[str, Any]],
    questions: list[str],
) -> None:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("BENCHMARK REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # Collect unique values preserving first-seen order
    docs_seen: list[tuple[str, str, str]] = []  # (document_id, title, author)
    for r in records:
        entry = (r["document_id"], r["document_title"], r["document_author"])
        if entry not in docs_seen:
            docs_seen.append(entry)

    models_seen: list[str] = []
    for r in records:
        if r["model"] not in models_seen:
            models_seen.append(r["model"])

    n_ctx_values_seen: list[int] = []
    for r in records:
        if r["n_ctx"] not in n_ctx_values_seen:
            n_ctx_values_seen.append(r["n_ctx"])

    for doc_id, doc_title, doc_author in docs_seen:
        doc_records = [r for r in records if r["document_id"] == doc_id]
        lines.append("=" * 80)
        lines.append(f"DOCUMENT: {doc_title} by {doc_author}")
        lines.append(f"Doc ID  : {doc_id}")
        lines.append("=" * 80)
        lines.append("")

        for model_name in models_seen:
            lines.append(f"MODEL: {model_name}")
            lines.append("-" * 80)

            for n_ctx in n_ctx_values_seen:
                lines.append(f"  n_ctx={n_ctx}")

                for fmt in PROMPT_FORMATS:
                    cell_records = [
                        r for r in doc_records
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
                lines.append(f"    {'Format':<18} {'Metric':<12} {'Q1':>6} {'Q2':>6} {'Q3':>6} {'Q4':>6} {'Q5':>6} {'Q6':>6} {'Total':>8}")
                lines.append(f"    {'-'*18} {'-'*12} {'---':>6} {'---':>6} {'---':>6} {'---':>6} {'---':>6} {'---':>6} {'-----':>8}")
                for fmt in PROMPT_FORMATS:
                    fmt_ctx_records = sorted(
                        [
                            r for r in doc_records
                            if r["model"] == model_name and r["n_ctx"] == n_ctx and r["prompt_format"] == fmt
                        ],
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

    for doc_id, doc_title, doc_author in docs_seen:
        doc_records = [r for r in records if r["document_id"] == doc_id]
        lines.append(f"{'=' * 40}")
        lines.append(f"DOCUMENT: {doc_title} by {doc_author}")
        lines.append(f"{'=' * 40}")
        lines.append("")

        for model_name in models_seen:
            for n_ctx in n_ctx_values_seen:
                for fmt in PROMPT_FORMATS:
                    cell_records = sorted(
                        [
                            r for r in doc_records
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
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=PROMPT_FORMATS,
        default=None,
        metavar="FORMAT",
        help=f"Prompt formats to run (overrides config). Choices: {' '.join(PROMPT_FORMATS)}",
    )
    args = parser.parse_args()
    run_benchmark(args.config, formats=args.formats)
