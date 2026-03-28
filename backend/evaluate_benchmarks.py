#!/usr/bin/env python3
"""
evaluate_benchmarks.py — LLM-as-judge evaluation for benchmark results.

Uses claude-sonnet-4-6 and gpt-4o to score model answers on:
  - grounding (1-5): textual evidence cited
  - depth    (1-5): literary analysis quality
  - register (1-5): IB-level academic tone
  - coherence (1-5, followup questions only): conversation continuity

Composite = normalize(grounding*0.30 + depth*0.40 + register*0.15) to [1-5].
Coherence is tracked SEPARATELY so cold vs followup questions are on the same scale.

Usage:
  python evaluate_benchmarks.py \\
    --b1 outputs/benchmarks/20260321_224116 \\
    --b2 outputs/benchmarks/20260327_233729 \\
    --debug outputs/benchmarks/20260322_105940 outputs/benchmarks/20260327_223434 outputs/benchmarks/20260328_094518 \\
    --output outputs/evaluations/ \\
    [--dry-run]
"""

import argparse
import json
import os
import hashlib
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert IB English Literature examiner evaluating AI-generated responses to literary analysis questions.

Scoring scale (use integers 1-5 only):
  1 = Names a technique or idea with no analysis
  2 = Mentions technique with minimal or generic commentary
  3 = Identifies technique and explains its effect with some textual reference
  4 = Clear analysis with specific textual evidence and explanation of HOW it works
  5 = Precise textual grounding, explains HOW and WHY, demonstrates sophisticated literary insight

Apply this scale consistently to every dimension you score."""

WEIGHT_GROUNDING = 0.30
WEIGHT_DEPTH     = 0.40
WEIGHT_REGISTER  = 0.15
WEIGHT_SUM = WEIGHT_GROUNDING + WEIGHT_DEPTH + WEIGHT_REGISTER  # 0.85

# B1 and D1 have no document_title field — they were run on Animal Farm
FALLBACK_DOC_TITLE = "Animal Farm"

# ── Lazy API clients ───────────────────────────────────────────────────────────

_anthropic_client = None
_openai_client = None

def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client

def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client

# ── Data loading ───────────────────────────────────────────────────────────────

def load_results(run_dir: str, filter_n_ctx: Optional[int] = None) -> list[dict]:
    path = Path(run_dir) / "results.jsonl"
    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    for r in records:
        # Patch missing document_title
        if not r.get("document_title") or r.get("document_title") == "?":
            r["document_title"] = FALLBACK_DOC_TITLE
        # Synthesise document_id if missing
        if not r.get("document_id"):
            r["document_id"] = hashlib.md5(r["document_title"].encode()).hexdigest()[:8]
        # Tag source run
        r.setdefault("run_dir", str(run_dir))

    if filter_n_ctx is not None:
        records = [r for r in records if r.get("n_ctx") == filter_n_ctx]

    return records


def valid_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("answer") and not r.get("error")]

# ── Grouping and prior Q&A ─────────────────────────────────────────────────────

def group_key(r: dict) -> tuple:
    return (r.get("run_dir", ""), r["model"], r["document_id"], r["prompt_format"])


def build_groups(records: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[group_key(r)].append(r)
    for g in groups.values():
        g.sort(key=lambda x: int(x["question_index"]))
    return groups


def build_prior_qa(group: list[dict], current_index: int) -> str:
    """Return a clean Q-A log (no retrieval context) for all questions before current_index."""
    prior = [
        r for r in group
        if int(r["question_index"]) < current_index
        and r.get("answer")
        and not r.get("error")
    ]
    if not prior:
        return ""
    parts = []
    for r in prior:
        parts.append(f"Q: {r['question']}")
        ans = r["answer"]
        parts.append(f"A: {ans[:400]}{'...' if len(ans) > 400 else ''}")
        parts.append("")
    return "\n".join(parts).strip()

# ── Cache ──────────────────────────────────────────────────────────────────────

def make_cache_key(run_dir: str, r: dict, judge_name: str) -> str:
    parts = [
        Path(run_dir).name,
        r["model"],
        r["document_id"],
        r["prompt_format"],
        str(r["question_index"]),
        judge_name,
    ]
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def load_cache(cache_path: Path) -> dict:
    cache = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        cache[obj["key"]] = obj["value"]
                    except (json.JSONDecodeError, KeyError):
                        continue
    return cache


def append_cache(cache_path: Path, key: str, value: dict):
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

# ── Judge API calls ────────────────────────────────────────────────────────────

def _call_with_retry(fn, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        # Strip markdown code fences
        inner = text.split("```")[1]
        if inner.startswith("json"):
            inner = inner[4:]
        text = inner.strip()
    return json.loads(text)


def judge_claude(user_prompt: str) -> dict:
    def call():
        resp = _get_anthropic().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return _parse_json_response(resp.content[0].text)
    return _call_with_retry(call)


def judge_openai(user_prompt: str) -> dict:
    def call():
        resp = _get_openai().chat.completions.create(
            model="gpt-4o",
            max_tokens=256,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        return json.loads(resp.choices[0].message.content)
    return _call_with_retry(call)


def build_judge_prompt(r: dict, prior_qa: str) -> str:
    lines = [f"Work: {r['document_title']}", ""]
    if prior_qa:
        lines += ["Prior exchange:", prior_qa, ""]
    lines += [
        f"Question: {r['question']}",
        f"Response: {r['answer']}",
        "",
        "Score each dimension with an integer 1-5:",
        "- grounding: Does the response cite specific textual evidence from the work?",
        "- depth: Quality and sophistication of the literary analysis",
        "- register: Appropriate IB-level academic tone and precision of language",
    ]
    if r["question_type"] == "followup":
        lines.append("- coherence: How well does this response build on the prior conversation?")
    lines += ["", 'Return JSON only. Example: {"grounding": 3, "depth": 4, "register": 3, "reasoning": "..."}']
    return "\n".join(lines)


def run_judge(
    r: dict,
    prior_qa: str,
    judge_name: str,
    run_dir: str,
    cache: dict,
    cache_path: Path,
) -> dict:
    key = make_cache_key(run_dir, r, judge_name)
    if key in cache:
        return cache[key]

    prompt = build_judge_prompt(r, prior_qa)
    try:
        raw = judge_claude(prompt) if judge_name == "claude" else judge_openai(prompt)
        # Ensure required keys present
        for dim in ["grounding", "depth", "register"]:
            if dim not in raw or not isinstance(raw.get(dim), (int, float)):
                raw[dim] = None
        # coherence only expected for followup
        if r["question_type"] != "followup":
            raw["coherence"] = None
        elif "coherence" not in raw or not isinstance(raw.get("coherence"), (int, float)):
            raw["coherence"] = None
        result = {"scores": raw, "error": None}
    except Exception as e:
        result = {"scores": {}, "error": str(e)}

    cache[key] = result
    append_cache(cache_path, key, result)
    return result

# ── Score computation ──────────────────────────────────────────────────────────

def composite(scores: dict) -> Optional[float]:
    """
    Weighted composite over grounding/depth/register, normalised to [1-5].
    Coherence is excluded so cold and followup questions share the same scale.
    Returns None if any required dimension is missing.
    """
    vals = {d: scores.get(d) for d in ["grounding", "depth", "register"]}
    if any(v is None for v in vals.values()):
        return None
    weighted = (
        vals["grounding"] * WEIGHT_GROUNDING
        + vals["depth"]    * WEIGHT_DEPTH
        + vals["register"] * WEIGHT_REGISTER
    )
    return weighted / WEIGHT_SUM  # normalise so max is always 5.0


def _avg_dim(dim: str, scores_list: list[dict]) -> Optional[float]:
    vals = [s.get(dim) for s in scores_list if isinstance(s.get(dim), (int, float))]
    return mean(vals) if vals else None

# ── Evaluation runner ─────────────────────────────────────────────────────────

def evaluate_run(
    records: list[dict],
    run_dir: str,
    run_label: str,
    cache: dict,
    cache_path: Path,
    workers: int = 8,
) -> list[dict]:
    """
    Judge all valid records in a run.
    Returns annotated records with averaged judge scores and composite.
    """
    valid = valid_records(records)
    if not valid:
        print(f"  [{run_label}] No valid records to judge.")
        return []

    # Ensure run_dir is tagged
    for r in valid:
        r["run_dir"] = str(run_dir)

    groups = build_groups(valid)

    # Build (task_id, record, prior_qa) triples
    tasks = []
    for group in groups.values():
        for r in group:
            prior_qa = (
                build_prior_qa(group, int(r["question_index"]))
                if r["question_type"] == "followup"
                else ""
            )
            tasks.append((len(tasks), r, prior_qa))  # task_id is list index

    all_calls = [(tid, r, pqa, jname) for (tid, r, pqa) in tasks for jname in ["claude", "openai"]]
    total = len(all_calls)
    done_count = [0]

    def call_judge(args):
        tid, r, prior_qa, judge_name = args
        result = run_judge(r, prior_qa, judge_name, run_dir, cache, cache_path)
        return (tid, judge_name, result)

    judged: dict[int, dict] = {}  # tid -> {claude: result, openai: result}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(call_judge, c): c for c in all_calls}
        for fut in as_completed(futures):
            tid, judge_name, result = fut.result()
            if tid not in judged:
                judged[tid] = {}
            judged[tid][judge_name] = result
            done_count[0] += 1
            if done_count[0] % 20 == 0 or done_count[0] == total:
                print(f"  [{run_label}] {done_count[0]}/{total} judge calls done", flush=True)

    # Annotate records
    annotated = []
    for tid, r, prior_qa in tasks:
        judge_results = judged.get(tid, {})
        claude_scores = judge_results.get("claude", {}).get("scores", {})
        openai_scores = judge_results.get("openai", {}).get("scores", {})

        avg = {d: _avg_dim(d, [claude_scores, openai_scores]) for d in ["grounding", "depth", "register", "coherence"]}
        comp = composite(avg)

        annotated.append({
            **r,
            "judge_claude": claude_scores,
            "judge_openai": openai_scores,
            "judge_avg": avg,
            "composite": comp,
            "coherence": avg.get("coherence"),
            "claude_error": judge_results.get("claude", {}).get("error"),
            "openai_error": judge_results.get("openai", {}).get("error"),
        })

    return annotated

# ── Aggregation: B2 model scores ──────────────────────────────────────────────

def compute_model_scores(b2_annotated: list[dict]) -> dict:
    """Per-model: per-work scores, generalisation, training_reliance_risk, overall."""
    by_model_doc: dict[tuple, list[float]] = defaultdict(list)
    for r in b2_annotated:
        if r["composite"] is not None:
            by_model_doc[(r["model"], r["document_title"])].append(r["composite"])

    models = sorted(set(r["model"] for r in b2_annotated))
    works = ["Animal Farm", "Black Rain", "The Great Gatsby"]
    results = {}

    for model in models:
        per_work = {
            w: (mean(by_model_doc[(model, w)]) if by_model_doc[(model, w)] else None)
            for w in works
        }
        af = per_work.get("Animal Farm")
        br = per_work.get("Black Rain")
        tg = per_work.get("The Great Gatsby")

        gen_vals = [x for x in [br, tg] if x is not None]
        generalisation = mean(gen_vals) if gen_vals else None
        training_reliance_risk = (
            round(af - generalisation, 4)
            if af is not None and generalisation is not None
            else None
        )
        all_vals = [x for x in [af, br, tg] if x is not None]
        overall = mean(all_vals) if all_vals else None

        results[model] = {
            "overall": round(overall, 4) if overall is not None else None,
            "per_work": {w: (round(v, 4) if v is not None else None) for w, v in per_work.items()},
            "generalisation": round(generalisation, 4) if generalisation is not None else None,
            "training_reliance_risk": training_reliance_risk,
        }

    return results


def compute_work_format_matrix(b2_annotated: list[dict]) -> dict:
    """3×3 matrix: work → format → mean composite across all models."""
    by_wf: dict[tuple, list[float]] = defaultdict(list)
    for r in b2_annotated:
        if r["composite"] is not None:
            by_wf[(r["document_title"], r["prompt_format"])].append(r["composite"])

    works = ["Animal Farm", "Black Rain", "The Great Gatsby"]
    formats = ["base_knowledge", "rag", "rag_raw"]
    matrix = {}
    for work in works:
        matrix[work] = {}
        for fmt in formats:
            vals = by_wf.get((work, fmt), [])
            matrix[work][fmt] = round(mean(vals), 4) if vals else None
    return matrix


def compute_format_ranking(b2_annotated: list[dict]) -> dict:
    """Format mean composite separately for Animal Farm and unfamiliar works."""
    familiar: dict[str, list[float]] = defaultdict(list)
    unfamiliar: dict[str, list[float]] = defaultdict(list)
    for r in b2_annotated:
        if r["composite"] is None:
            continue
        target = familiar if r["document_title"] == "Animal Farm" else unfamiliar
        target[r["prompt_format"]].append(r["composite"])

    def rank(d):
        return {fmt: round(mean(v), 4) for fmt, v in d.items() if v}

    return {"animal_farm": rank(familiar), "unfamiliar_works": rank(unfamiliar)}


def compute_coherence_ranking(b1_annotated: list[dict], b2_annotated: list[dict]) -> list[dict]:
    """Multi-turn coherence from B1 Q3-Q5 and B2 Q4-Q6."""
    by_model: dict[str, list[float]] = defaultdict(list)
    for r in b1_annotated:
        if r["coherence"] is not None and int(r["question_index"]) in {3, 4, 5}:
            by_model[r["model"]].append(r["coherence"])
    for r in b2_annotated:
        if r["coherence"] is not None and int(r["question_index"]) in {4, 5, 6}:
            by_model[r["model"]].append(r["coherence"])

    ranked = [
        {"model": m, "coherence_score": round(mean(v), 4), "n": len(v)}
        for m, v in by_model.items()
    ]
    ranked.sort(key=lambda x: x["coherence_score"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    return ranked

# ── Cross-validation ───────────────────────────────────────────────────────────

def cross_validate(b1_annotated: list[dict], b2_annotated: list[dict]) -> list[dict]:
    """
    Compare B1 vs B2 Animal Farm rankings.
    B2 side uses only base_knowledge + rag to match B1's format set.
    """
    b2_af = [
        r for r in b2_annotated
        if r["document_title"] == "Animal Farm"
        and r["prompt_format"] in {"base_knowledge", "rag"}
    ]

    b1_by_model: dict[str, list[float]] = defaultdict(list)
    for r in b1_annotated:
        if r["composite"] is not None:
            b1_by_model[r["model"]].append(r["composite"])

    b2_by_model: dict[str, list[float]] = defaultdict(list)
    for r in b2_af:
        if r["composite"] is not None:
            b2_by_model[r["model"]].append(r["composite"])

    common = set(b1_by_model) & set(b2_by_model)
    if not common:
        return []

    b1_avg = {m: mean(v) for m, v in b1_by_model.items() if m in common}
    b2_avg = {m: mean(v) for m, v in b2_by_model.items() if m in common}

    b1_ranked = sorted(common, key=lambda m: b1_avg[m], reverse=True)
    b2_ranked = sorted(common, key=lambda m: b2_avg[m], reverse=True)
    b1_rank = {m: i + 1 for i, m in enumerate(b1_ranked)}
    b2_rank = {m: i + 1 for i, m in enumerate(b2_ranked)}

    table = [
        {
            "model": m,
            "b1_rank": b1_rank[m],
            "b1_score": round(b1_avg[m], 4),
            "b2_af_rank": b2_rank[m],
            "b2_af_score": round(b2_avg[m], 4),
            "rank_delta": abs(b1_rank[m] - b2_rank[m]),
            "flag": abs(b1_rank[m] - b2_rank[m]) > 3,
        }
        for m in common
    ]
    table.sort(key=lambda x: x["b1_rank"])
    return table

# ── Robustness check ───────────────────────────────────────────────────────────

def robustness_check(
    debug_annotated_list: list[list[dict]],
    b2_model_scores: dict,
    b1_annotated: list[dict],
) -> list[dict]:
    """
    For models overlapping between debug runs and B1/B2, check ordering stability.
    Reports ordering flips; skips score variance if temperature is uniform or missing.
    """
    # Build main rank from B2 overall, fallback to B1
    main_ranks: dict[str, int] = {}
    b2_ordered = sorted(
        [(m, s["overall"]) for m, s in b2_model_scores.items() if s["overall"] is not None],
        key=lambda x: x[1], reverse=True,
    )
    for i, (m, _) in enumerate(b2_ordered):
        main_ranks[m] = i + 1

    b1_by_model: dict[str, list[float]] = defaultdict(list)
    for r in b1_annotated:
        if r["composite"] is not None:
            b1_by_model[r["model"]].append(r["composite"])
    b1_ordered = sorted(
        [(m, mean(v)) for m, v in b1_by_model.items() if m not in main_ranks],
        key=lambda x: x[1], reverse=True,
    )
    offset = len(main_ranks)
    for i, (m, _) in enumerate(b1_ordered):
        main_ranks[m] = offset + i + 1

    findings = []
    for debug_annotated in debug_annotated_list:
        if not debug_annotated:
            continue

        run_name = Path(debug_annotated[0].get("run_dir", "?")).name
        temps = sorted({r.get("temperature") for r in debug_annotated if r.get("temperature") is not None})
        temp_str = ", ".join(str(t) for t in temps) if temps else "unknown"

        debug_by_model: dict[str, list[float]] = defaultdict(list)
        for r in debug_annotated:
            if r["composite"] is not None:
                debug_by_model[r["model"]].append(r["composite"])

        debug_scores = {m: mean(v) for m, v in debug_by_model.items()}
        debug_ranked = sorted(debug_scores, key=lambda m: debug_scores[m], reverse=True)
        debug_rank = {m: i + 1 for i, m in enumerate(debug_ranked)}
        all_debug_models = list(debug_scores.keys())
        overlap = [m for m in all_debug_models if m in main_ranks]

        flips = []
        for i, m1 in enumerate(overlap):
            for m2 in overlap[i + 1:]:
                main_higher = main_ranks[m1] < main_ranks[m2]
                debug_higher = debug_rank[m1] < debug_rank[m2]
                if main_higher != debug_higher:
                    flips.append(f"{m1[:30]} vs {m2[:30]}")

        findings.append({
            "run": run_name,
            "temperature": temp_str,
            "models_in_debug": all_debug_models,
            "overlap_with_main": overlap,
            "ordering_flips": flips,
            "has_flips": bool(flips),
        })

    return findings

# ── Judge agreement ────────────────────────────────────────────────────────────

def judge_agreement(all_annotated: list[dict]) -> dict:
    """Mean absolute difference between Claude and GPT-4o scores per dimension."""
    diffs: dict[str, list[float]] = defaultdict(list)
    for r in all_annotated:
        c = r.get("judge_claude", {})
        o = r.get("judge_openai", {})
        for dim in ["grounding", "depth", "register", "coherence"]:
            cv, ov = c.get(dim), o.get(dim)
            if isinstance(cv, (int, float)) and isinstance(ov, (int, float)):
                diffs[dim].append(abs(cv - ov))

    composite_diffs = []
    for r in all_annotated:
        cc = composite(r.get("judge_claude", {}))
        oc = composite(r.get("judge_openai", {}))
        if cc is not None and oc is not None:
            composite_diffs.append(abs(cc - oc))

    result = {d: round(mean(v), 3) for d, v in diffs.items() if v}
    result["composite"] = round(mean(composite_diffs), 3) if composite_diffs else None
    return result

# ── n_ctx timing analysis (no judging — from raw B1 records) ──────────────────

def n_ctx_timing_analysis(b1_raw: list[dict]) -> dict:
    """Mean inference time per (n_ctx, model) from B1 raw records."""
    by_n_ctx_model: dict[tuple, list[float]] = defaultdict(list)
    for r in b1_raw:
        if r.get("inference_time_s") and not r.get("error"):
            by_n_ctx_model[(str(r["n_ctx"]), r["model"])].append(float(r["inference_time_s"]))

    result: dict[str, dict[str, float]] = defaultdict(dict)
    for (n_ctx, model), times in by_n_ctx_model.items():
        result[n_ctx][model] = round(mean(times), 2)
    return dict(result)

# ── Efficiency ranking ────────────────────────────────────────────────────────

def compute_efficiency_ranking(b2_annotated: list[dict], b2_model_scores: dict) -> list[dict]:
    """Quality ÷ mean inference time per model — higher = better quality per second."""
    by_model_time: dict[str, list[float]] = defaultdict(list)
    for r in b2_annotated:
        t = r.get("total_time_s") or r.get("inference_time_s")
        if t and not r.get("error"):
            by_model_time[r["model"]].append(float(t))

    ranking = []
    for model, s in b2_model_scores.items():
        quality = s.get("overall")
        times = by_model_time.get(model, [])
        mean_time = mean(times) if times else None
        if quality is None or mean_time is None or mean_time == 0:
            continue
        ranking.append({
            "model": model,
            "quality": round(quality, 4),
            "mean_time_s": round(mean_time, 2),
            "efficiency": round(quality / mean_time, 6),
        })

    ranking.sort(key=lambda x: x["efficiency"], reverse=True)
    for i, r in enumerate(ranking):
        r["rank"] = i + 1
    return ranking

# ── Cost estimation (dry-run) ──────────────────────────────────────────────────

def estimate_cost(records: list[dict]) -> dict:
    n = len(records)
    # ~800 avg input tokens, ~100 output tokens per call, ×2 judges
    input_tok = n * 800 * 2
    output_tok = n * 100 * 2
    # claude-sonnet-4-6: $3/1M in, $15/1M out
    claude_cost = input_tok / 2 / 1e6 * 3 + output_tok / 2 / 1e6 * 15
    # gpt-4o: $2.50/1M in, $10/1M out
    gpt_cost = input_tok / 2 / 1e6 * 2.50 + output_tok / 2 / 1e6 * 10
    return {
        "records": n,
        "api_calls": n * 2,
        "estimated_usd": round(claude_cost + gpt_cost, 2),
        "claude_usd": round(claude_cost, 2),
        "gpt_usd": round(gpt_cost, 2),
    }

# ── Report writing ─────────────────────────────────────────────────────────────

def write_report(
    output_dir: Path,
    timestamp: str,
    b2_model_scores: dict,
    work_format_matrix: dict,
    format_ranking: dict,
    coherence_ranking: list[dict],
    cross_val: list[dict],
    robustness: list[dict],
    agreement: dict,
    n_ctx_timing: dict,
    efficiency_ranking: list[dict],
    b2_judged: int,
    b1_judged: int,
) -> Path:
    lines = []

    def h(title):
        lines.append("")
        lines.append("=" * 64)
        lines.append(f"  {title}")
        lines.append("=" * 64)

    lines.append("BENCHMARK EVALUATION REPORT")
    lines.append(f"Generated : {timestamp}")
    lines.append(f"Judges    : claude-sonnet-4-6  +  gpt-4o")
    lines.append(f"B2 records judged : {b2_judged}")
    lines.append(f"B1 records judged : {b1_judged}  (n_ctx=4096 only)")

    comp_mad = agreement.get("composite")
    if comp_mad is not None:
        flag = "  ⚠ WARNING: high disagreement, results may be unreliable" if comp_mad > 1.0 else ""
        lines.append(f"Judge composite MAD : {comp_mad:.3f}{flag}")

    # 1. Overall B2 ranking
    h("1. OVERALL B2 RANKING  (all 3 works, all formats)")
    ranked_overall = sorted(
        [(m, s) for m, s in b2_model_scores.items() if s["overall"] is not None],
        key=lambda x: x[1]["overall"], reverse=True,
    )
    for i, (model, s) in enumerate(ranked_overall, 1):
        lines.append(f"  {i:>3}.  {model[:50]:<50}  {s['overall']:.4f}")

    # 2. Generalisation ranking
    h("2. GENERALISATION RANKING  (Black Rain + Gatsby only)")
    lines.append("       Higher = better on unseen works = safer for production")
    gen_ranked = sorted(
        [(m, s) for m, s in b2_model_scores.items() if s["generalisation"] is not None],
        key=lambda x: x[1]["generalisation"], reverse=True,
    )
    for i, (model, s) in enumerate(gen_ranked, 1):
        lines.append(f"  {i:>3}.  {model[:50]:<50}  {s['generalisation']:.4f}")

    # 3. Training reliance risk
    h("3. TRAINING RELIANCE RISK  (Animal Farm score − generalisation)")
    lines.append("       High positive = model likely using memorisation, not RAG")
    risk_ranked = sorted(
        [(m, s) for m, s in b2_model_scores.items() if s["training_reliance_risk"] is not None],
        key=lambda x: x[1]["training_reliance_risk"], reverse=True,
    )
    for i, (model, s) in enumerate(risk_ranked, 1):
        risk = s["training_reliance_risk"]
        flag = "  ⚠ HIGH RISK" if risk > 0.5 else ""
        lines.append(f"  {i:>3}.  {model[:50]:<50}  {risk:+.4f}{flag}")

    # 4. Work × Format matrix
    h("4. WORK × FORMAT MATRIX  (mean composite per cell)")
    formats = ["base_knowledge", "rag", "rag_raw"]
    lines.append(f"  {'':35} {'base_knowledge':>15} {'rag':>7} {'rag_raw':>9}")
    for work, fmt_scores in work_format_matrix.items():
        def fv(f):
            v = fmt_scores.get(f)
            return f"{v:.4f}" if v is not None else "  N/A "
        lines.append(f"  {work[:35]:<35} {fv('base_knowledge'):>15} {fv('rag'):>7} {fv('rag_raw'):>9}")

    # 5. Format ranking
    h("5. FORMAT RANKING  (mean composite per format)")
    lines.append("  Animal Farm  (familiar / potentially memorised):")
    for fmt, score in sorted(format_ranking.get("animal_farm", {}).items(), key=lambda x: x[1], reverse=True):
        lines.append(f"    {fmt:<22}  {score:.4f}")
    lines.append("  Unfamiliar works  (Black Rain + Gatsby):")
    for fmt, score in sorted(format_ranking.get("unfamiliar_works", {}).items(), key=lambda x: x[1], reverse=True):
        lines.append(f"    {fmt:<22}  {score:.4f}")

    # 6. Multi-turn coherence
    if coherence_ranking:
        h("6. MULTI-TURN COHERENCE RANKING  (B1 Q3-5, B2 Q4-6)")
        for r in coherence_ranking[:20]:
            lines.append(f"  {r['rank']:>3}.  {r['model'][:50]:<50}  {r['coherence_score']:.4f}  (n={r['n']})")

    # 7. Cross-validation
    if cross_val:
        h("7. CROSS-VALIDATION: B1 vs B2 Animal Farm")
        lines.append("  NOTE: B2 scored using base_knowledge + rag only (matching B1 format set)")
        lines.append("  Flag = |rank delta| > 3")
        lines.append(f"  {'Model':<50}  {'B1':>4}  {'B2-AF':>5}  {'Δ':>3}")
        for r in cross_val:
            flag = "  ⚠" if r["flag"] else ""
            lines.append(f"  {r['model'][:50]:<50}  {r['b1_rank']:>4}  {r['b2_af_rank']:>5}  {r['rank_delta']:>3}{flag}")

    # 8. Robustness
    if robustness:
        h("8. ROBUSTNESS CHECK  (debug runs vs B1/B2 ordering)")
        for f in robustness:
            lines.append(f"\n  Run: {f['run']}  (temperature={f['temperature']})")
            lines.append(f"  Models: {', '.join(m[:30] for m in f['models_in_debug'])}")
            overlap_str = ", ".join(m[:30] for m in f["overlap_with_main"]) or "none"
            lines.append(f"  Overlap with B1/B2: {overlap_str}")
            if f["ordering_flips"]:
                lines.append(f"  ⚠ Ordering flips detected:")
                for flip in f["ordering_flips"]:
                    lines.append(f"    - {flip}")
            else:
                lines.append("  ✓ No ordering flips detected")

    # 9. Judge agreement
    h("9. JUDGE AGREEMENT  (Claude vs GPT-4o, mean |diff| per dimension)")
    for dim, val in sorted(agreement.items()):
        if val is None:
            continue
        flag = "  ⚠ HIGH" if val > 1.0 else ""
        lines.append(f"  {dim:<15}  {val:.3f}{flag}")

    # 10. n_ctx timing
    if n_ctx_timing:
        h("10. n_ctx SPEED ANALYSIS  (B1 mean inference time, seconds)")
        for n_ctx in sorted(n_ctx_timing.keys(), key=lambda x: int(x)):
            lines.append(f"\n  n_ctx = {n_ctx}:")
            for model, t in sorted(n_ctx_timing[n_ctx].items(), key=lambda x: x[1]):
                lines.append(f"    {model[:50]:<50}  {t:.2f}s")

    # 11. Efficiency ranking
    if efficiency_ranking:
        h("11. EFFICIENCY RANKING  (quality ÷ mean inference time)")
        lines.append("       Higher = better quality per second of compute")
        lines.append(f"  {'Model':<50}  {'Quality':>7}  {'AvgTime':>8}  {'Eff/s':>8}")
        for r in efficiency_ranking:
            lines.append(
                f"  {r['rank']:>3}.  {r['model'][:50]:<50}"
                f"  {r['quality']:>7.4f}  {r['mean_time_s']:>7.2f}s  {r['efficiency']:>8.6f}"
            )

    report_path = output_dir / f"report_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report:      {report_path}")
    return report_path

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation for benchmark results")
    parser.add_argument("--b1",    required=True,          help="B1 run directory")
    parser.add_argument("--b2",    required=True,          help="B2 run directory")
    parser.add_argument("--debug", nargs="*", default=[],  help="Debug run directories")
    parser.add_argument("--output", default="outputs/evaluations/", help="Output directory")
    parser.add_argument("--dry-run", action="store_true",  help="Print cost estimate and exit")
    parser.add_argument("--workers", type=int, default=8,  help="Concurrent judge threads (default 8)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "judge_cache.jsonl"

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading B1 (all n_ctx values for timing analysis)...")
    b1_raw = load_results(args.b1)
    b1_records_4096 = [r for r in b1_raw if r.get("n_ctx") == 4096]

    print("Loading B2...")
    b2_records = load_results(args.b2)

    debug_records_list = []
    for d in (args.debug or []):
        print(f"Loading debug: {Path(d).name}...")
        debug_records_list.append(load_results(d))

    b1_valid = valid_records(b1_records_4096)
    b2_valid = valid_records(b2_records)
    debug_valid_flat = [r for dr in debug_records_list for r in valid_records(dr)]

    print(f"\nB1 valid (n_ctx=4096): {len(b1_valid)}")
    print(f"B2 valid:              {len(b2_valid)}")
    for i, dr in enumerate(debug_records_list):
        print(f"Debug {i+1} valid:        {len(valid_records(dr))}")

    # ── Dry-run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        all_to_judge = b1_valid + b2_valid + debug_valid_flat
        est = estimate_cost(all_to_judge)
        print("\n=== DRY RUN — cost estimate ===")
        print(f"  Total valid records  : {est['records']}")
        print(f"  API calls (×2 judges): {est['api_calls']}")
        print(f"  Claude sonnet-4-6    : ~${est['claude_usd']}")
        print(f"  GPT-4o               : ~${est['gpt_usd']}")
        print(f"  TOTAL estimate       : ~${est['estimated_usd']}")
        print("\nRemove --dry-run to execute.")
        return

    # ── Load cache ────────────────────────────────────────────────────────────
    print(f"\nLoading cache from {cache_path}...")
    cache = load_cache(cache_path)
    print(f"  {len(cache)} entries cached (will be skipped).")

    # ── Judge ─────────────────────────────────────────────────────────────────
    print("\n[Phase 1] Judging B2 (primary)...")
    b2_annotated = evaluate_run(b2_records, args.b2, "B2", cache, cache_path, args.workers)

    print("\n[Phase 2] Judging B1 (secondary, n_ctx=4096)...")
    b1_annotated = evaluate_run(b1_records_4096, args.b1, "B1", cache, cache_path, args.workers)

    debug_annotated_list = []
    for i, (dr, d_dir) in enumerate(zip(debug_records_list, (args.debug or []))):
        print(f"\n[Debug {i+1}] Judging {Path(d_dir).name}...")
        debug_annotated_list.append(
            evaluate_run(dr, d_dir, f"D{i+1}", cache, cache_path, args.workers)
        )

    # ── Compute ───────────────────────────────────────────────────────────────
    print("\nComputing scores and rankings...")
    b2_model_scores  = compute_model_scores(b2_annotated)
    work_fmt_matrix  = compute_work_format_matrix(b2_annotated)
    format_ranking   = compute_format_ranking(b2_annotated)
    coherence_ranking = compute_coherence_ranking(b1_annotated, b2_annotated)
    cross_val        = cross_validate(b1_annotated, b2_annotated)
    robustness       = robustness_check(debug_annotated_list, b2_model_scores, b1_annotated)
    all_annotated    = b1_annotated + b2_annotated + [r for da in debug_annotated_list for r in da]
    agreement        = judge_agreement(all_annotated)
    n_ctx_timing     = n_ctx_timing_analysis(b1_raw)
    efficiency_ranking = compute_efficiency_ranking(b2_annotated, b2_model_scores)

    # ── Write outputs ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    scores_path = output_dir / f"scores_{timestamp}.json"
    scores_data = {
        "timestamp": timestamp,
        "b2_model_scores": b2_model_scores,
        "work_format_matrix": work_fmt_matrix,
        "format_ranking": format_ranking,
        "coherence_ranking": coherence_ranking,
        "cross_validation": cross_val,
        "robustness": robustness,
        "judge_agreement": agreement,
        "n_ctx_timing": n_ctx_timing,
        "efficiency_ranking": efficiency_ranking,
        "per_answer_scores": all_annotated,
    }
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores_data, f, indent=2, ensure_ascii=False)
    print(f"Scores JSON: {scores_path}")

    write_report(
        output_dir, timestamp,
        b2_model_scores, work_fmt_matrix, format_ranking,
        coherence_ranking, cross_val, robustness, agreement,
        n_ctx_timing, efficiency_ranking, len(b2_annotated), len(b1_annotated),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
