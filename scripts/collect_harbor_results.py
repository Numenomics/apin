#!/usr/bin/env python3
"""Collect Harbor job outputs into the data file behind the site's Agent benchmark page.

Reads Harbor trial directories (`result.json`, `agent/trajectory.json`,
`agent/opencode-v2.txt`, `verifier/reward.json`, `verifier/submission.py`),
extracts exact walltime, token usage (input, cached, cache_write, output, reasoning),
price, turns, tool calls, test runs, solution size & complexity, approach summary,
and all verifier metrics, and writes `flywheel/public/data/benchmark.json`.
The page itself is static: `flywheel/public/benchmark/index.html` reads that file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICING = ROOT / "benchmarks" / "pricing.json"
DEFAULT_OUT_JSON = ROOT / "flywheel" / "public" / "data" / "benchmark.json"
TASKS_DIR = ROOT / "harbor" / "tasks"
DEFAULT_RELEASE = "bench-v0.2.0"  # used for job directories whose name does not start with a release tag

_TEST_CMD_RE = re.compile(
    r"\b(pytest|apin\s+(invariance|autoresearch\s+eval|score|track)|harbor/verify\.py|verify\.py)\b"
)
_TRIVIAL_CONSTS = frozenset({-1, 0, 1, 2, -1.0, 0.0, 1.0, 2.0})


def _parse_iso(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _solution_stats_from_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    loc = sum(1 for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    magic_constants = 0
    ast_nodes = 0
    cyclomatic = 1
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            ast_nodes += 1
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                if node.value not in _TRIVIAL_CONSTS:
                    magic_constants += 1
            elif isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp)):
                cyclomatic += 1
            elif isinstance(node, ast.BoolOp):
                cyclomatic += max(0, len(node.values) - 1)
            elif isinstance(node, ast.comprehension):
                cyclomatic += 1 + len(node.ifs)
    except SyntaxError:
        pass
    return {
        "loc": loc,
        "bytes": len(raw),
        "magic_constants": magic_constants,
        "ast_nodes": ast_nodes,
        "cyclomatic_complexity": cyclomatic,
        "sha256": hashlib.sha256(raw).hexdigest()[:16],
    }


def _extract_from_opencode_jsonl(path: Path) -> dict[str, Any]:
    """Parse raw OpenCode V2 `--format=json` event log for exact tokens, tool calls, test runs, and summary."""
    out: dict[str, Any] = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "api_cost_usd": 0.0,
        "turns": 0,
        "tool_calls": 0,
        "test_runs": 0,
        "approach_summary": "",
        "found": False,
    }
    if not path.is_file():
        return out

    last_texts: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        out["found"] = True
        kind = ev.get("type")
        part = ev.get("part") if isinstance(ev.get("part"), dict) else {}

        if kind == "step_finish":
            out["turns"] += 1
            toks = part.get("tokens") if isinstance(part.get("tokens"), dict) else {}
            out["input_tokens"] += int(toks.get("input") or 0)
            out["output_tokens"] += int(toks.get("output") or 0)
            out["reasoning_tokens"] += int(toks.get("reasoning") or 0)
            cache = toks.get("cache") if isinstance(toks.get("cache"), dict) else {}
            out["cached_tokens"] += int(cache.get("read") or 0)
            out["cache_write_tokens"] += int(cache.get("write") or 0)
            cost = part.get("cost")
            if isinstance(cost, (int, float)) and math.isfinite(cost):
                out["api_cost_usd"] += float(cost)
        elif kind == "tool_use":
            out["tool_calls"] += 1
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            inp = state.get("input") if isinstance(state.get("input"), dict) else {}
            cmd = str(inp.get("command") or inp.get("cmd") or "")
            if cmd and _TEST_CMD_RE.search(cmd):
                out["test_runs"] += 1
        elif kind == "text":
            txt = str(part.get("text") or "").strip()
            if txt:
                last_texts.append(txt)

    if last_texts:
        summary = last_texts[-1]
        out["approach_summary"] = summary[:1600]
    return out


def _extract_from_atif_trajectory(path: Path) -> dict[str, Any]:
    """Parse Harbor ATIF `trajectory.json` for turns, tool calls, test runs, tokens, and final summary."""
    out: dict[str, Any] = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "api_cost_usd": 0.0,
        "turns": 0,
        "tool_calls": 0,
        "test_runs": 0,
        "approach_summary": "",
        "found": False,
    }
    if not path.is_file():
        return out
    try:
        traj = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return out
    out["found"] = True

    fm = traj.get("final_metrics") or {}
    out["input_tokens"] = int(fm.get("total_prompt_tokens") or 0)
    out["cached_tokens"] = int(fm.get("total_cached_tokens") or 0)
    out["output_tokens"] = int(fm.get("total_completion_tokens") or 0)
    if isinstance(fm.get("total_cost_usd"), (int, float)):
        out["api_cost_usd"] = float(fm["total_cost_usd"])

    last_agent_msg = ""
    for step in traj.get("steps") or []:
        if step.get("source") != "agent":
            continue
        out["turns"] += 1
        msg = str(step.get("message") or "").strip()
        if msg:
            last_agent_msg = msg
        for tc in step.get("tool_calls") or []:
            out["tool_calls"] += 1
            args = tc.get("arguments") or {}
            cmd = str(args.get("command") or args.get("cmd") or "") if isinstance(args, dict) else str(args)
            if cmd and _TEST_CMD_RE.search(cmd):
                out["test_runs"] += 1
    out["approach_summary"] = last_agent_msg[:1600]
    return out


def _compute_cost(
    model: str,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    api_cost_usd: float,
    pricing: dict[str, Any],
) -> tuple[float, str]:
    if api_cost_usd and api_cost_usd > 0.0:
        return round(api_cost_usd, 6), "reported"
    models_tbl = pricing.get("models") or {}
    base_model = model.split("#", 1)[0]
    p = models_tbl.get(base_model)
    if not p:
        return 0.0, "unknown"
    # Note: in OpenCode step_finish, `input` is uncached input and `cache.read` is cached input.
    cost = (
        (input_tokens / 1_000_000.0) * float(p.get("input_per_1m") or 0.0)
        + (cached_tokens / 1_000_000.0) * float(p.get("cached_input_per_1m") or 0.0)
        + ((output_tokens + reasoning_tokens) / 1_000_000.0) * float(p.get("output_per_1m") or 0.0)
    )
    return round(cost, 6), ("free" if p.get("free_tier") else "estimated_list")


def parse_trial_dir(trial_dir: Path, pricing: dict[str, Any], *, release_tag: str = DEFAULT_RELEASE) -> dict[str, Any] | None:
    res_path = trial_dir / "result.json"
    if not res_path.is_file():
        return None
    try:
        res = json.loads(res_path.read_text(encoding="utf-8"))
    except ValueError:
        return None

    cfg = res.get("config") or {}
    agent_cfg = cfg.get("agent") or {}
    task_cfg = cfg.get("task") or {}
    agent_res = res.get("agent_result") or {}
    ver_res = res.get("verifier_result") or {}

    task_id = str(res.get("task_name") or Path(str(task_cfg.get("path") or trial_dir.name)).name)
    category = "rewind" if task_id.startswith("s3-") else "optimize"
    agent_name = str(agent_cfg.get("name") or (res.get("agent_info") or {}).get("name") or "unknown")
    raw_model = str(agent_cfg.get("model_name") or "unknown")
    kwargs = agent_cfg.get("kwargs") or {}
    effort = str(kwargs.get("reasoning_effort") or kwargs.get("variant") or (raw_model.split("#", 1)[1] if "#" in raw_model else "default"))
    model = raw_model.split("#", 1)[0]

    # Walltime (agent execution + total trial)
    a_exec = res.get("agent_execution") or {}
    v_exec = res.get("verifier") or {}
    a_start, a_end = _parse_iso(a_exec.get("started_at")), _parse_iso(a_exec.get("finished_at"))
    v_start, v_end = _parse_iso(v_exec.get("started_at")), _parse_iso(v_exec.get("finished_at"))
    t_start, t_end = _parse_iso(res.get("started_at")), _parse_iso(res.get("finished_at"))
    agent_wall_s = round(a_end - a_start, 2) if (a_start and a_end and a_end >= a_start) else None
    verifier_wall_s = round(v_end - v_start, 2) if (v_start and v_end and v_end >= v_start) else None
    total_wall_s = round(t_end - t_start, 2) if (t_start and t_end and t_end >= t_start) else agent_wall_s

    # Trajectory + raw OpenCode JSONL telemetry
    oc_stats = _extract_from_opencode_jsonl(trial_dir / "agent" / "opencode-v2.txt")
    atif_stats = _extract_from_atif_trajectory(trial_dir / "agent" / "trajectory.json")

    if oc_stats["found"] and (oc_stats["input_tokens"] + oc_stats["cached_tokens"] + oc_stats["output_tokens"] > 0):
        input_toks = oc_stats["input_tokens"]
        cached_toks = oc_stats["cached_tokens"]
        cache_write_toks = oc_stats["cache_write_tokens"]
        output_toks = oc_stats["output_tokens"]
        reasoning_toks = oc_stats["reasoning_tokens"]
        api_cost = oc_stats["api_cost_usd"]
    else:
        # Harbor ATIF / agent_result fallback (note: ATIF total_prompt_tokens may include cached)
        raw_in = int(agent_res.get("n_input_tokens") or atif_stats["input_tokens"] or 0)
        cached_toks = int(agent_res.get("n_cache_tokens") or atif_stats["cached_tokens"] or 0)
        input_toks = max(0, raw_in - cached_toks) if (raw_in >= cached_toks and cached_toks > 0) else raw_in
        cache_write_toks = 0
        output_toks = int(agent_res.get("n_output_tokens") or atif_stats["output_tokens"] or 0)
        reasoning_toks = 0
        api_cost = float(agent_res.get("cost_usd") or atif_stats["api_cost_usd"] or 0.0)

    turns = max(oc_stats["turns"], atif_stats["turns"])
    tool_calls = max(oc_stats["tool_calls"], atif_stats["tool_calls"])
    test_runs = max(oc_stats["test_runs"], atif_stats["test_runs"])
    approach_summary = oc_stats["approach_summary"] or atif_stats["approach_summary"]

    total_toks = input_toks + cached_toks + cache_write_toks + output_toks + reasoning_toks
    denom_in = input_toks + cached_toks
    cache_hit_rate = round(cached_toks / denom_in, 4) if denom_in > 0 else 0.0
    cost_usd, cost_source = _compute_cost(model, input_toks, cached_toks, output_toks, reasoning_toks, api_cost, pricing)

    # Verifier detailed metrics + solution complexity (prefer details.json, fall back to reward.json)
    reward_data: dict[str, Any] = {}
    for fname in ("details.json", "reward.json"):
        p = trial_dir / "verifier" / fname
        if p.is_file():
            try:
                reward_data = json.loads(p.read_text(encoding="utf-8"))
                break
            except ValueError:
                pass

    rewards_map = ver_res.get("rewards") or {}
    reward = float(reward_data.get("reward", rewards_map.get("reward") or 0.0))
    passed = bool(reward_data.get("passed")) if "passed" in reward_data else (reward > 0.0)
    solution = reward_data.get("solution") or _solution_stats_from_file(trial_dir / "verifier" / "submission.py")

    exc = res.get("exception_info")
    error = f"{exc.get('exception_type')}: {exc.get('exception_message')}" if isinstance(exc, dict) else None
    exc_type = exc.get("exception_type") if isinstance(exc, dict) else None

    oc_log = trial_dir / "agent" / "opencode-v2.txt"
    oc_raw_text = oc_log.read_text(encoding="utf-8", errors="replace") if oc_log.is_file() else ""
    oc_low = oc_raw_text.lower()
    rate_limited = (
        ("provider.free-limit" in oc_raw_text)
        or ("provider.quota" in oc_raw_text)
        or ("provider.rate-limit" in oc_raw_text)
        or ("FreeUsageLimitError" in oc_raw_text)
        or ("usage limit reached" in oc_low)
        or ("rate limit" in oc_low and exc_type in ("NonZeroAgentExitCodeError", "ApiRateLimitError"))
    )
    termination = classify_termination(exc_type, rate_limited=rate_limited, tool_calls=tool_calls)

    m_rel = re.match(r"^(bench-v\d+\.\d+\.\d+)\b", trial_dir.parent.name)
    trial_release = m_rel.group(1) if m_rel else release_tag

    return {
        "trial_id": str(res.get("trial_name") or trial_dir.name),
        "release": trial_release,
        "model": model,
        "effort": effort,
        "config_key": f"{model}#{effort}",
        "agent": agent_name,
        "task_id": task_id,
        "category": category,
        "passed": passed,
        "termination": termination,
        "rate_limited": rate_limited,
        "reward": round(reward, 6),
        "verifier_outcome": reward_data.get("outcome"),
        "ref_speedup": reward_data.get("ref_speedup"),
        "start_speedup": reward_data.get("start_speedup", reward_data.get("speedup")),
        "reference": reward_data.get("reference"),
        "reference_soft_min": reward_data.get("reference_soft_min"),
        "frontier_speedup": reward_data.get("frontier_speedup"),
        "frontier_soft_min": reward_data.get("frontier_soft_min"),
        "gap_closed": reward_data.get("gap_closed"),
        "start_soft_min": reward_data.get("start_soft_min", reward_data.get("baseline_soft_min")),
        "soft_min_eff": reward_data.get("soft_min_eff"),
        "short_horizon_soft_min": reward_data.get("short_horizon_soft_min"),
        "long_horizon_soft_min": reward_data.get("long_horizon_soft_min"),
        "b2_max": reward_data.get("b2_max"),
        "overhead_ratio": reward_data.get("overhead_ratio"),
        "by_class": reward_data.get("by_class") or {},
        "failed_targets": reward_data.get("failed_targets") or [],
        "agent_wall_s": agent_wall_s,
        "verifier_wall_s": verifier_wall_s,
        "total_wall_s": total_wall_s,
        "input_tokens": input_toks,
        "cached_tokens": cached_toks,
        "cache_write_tokens": cache_write_toks,
        "output_tokens": output_toks,
        "reasoning_tokens": reasoning_toks,
        "total_tokens": total_toks,
        "cache_hit_rate": cache_hit_rate,
        "cost_usd": cost_usd,
        "cost_source": cost_source,
        "turns": turns,
        "tool_calls": tool_calls,
        "test_runs": test_runs,
        "solution": solution,
        "approach_summary": approach_summary,
        "error": error,
        "started_at": res.get("started_at"),
        "finished_at": res.get("finished_at"),
    }


def scan_jobs(jobs_dirs: list[Path], pricing: dict[str, Any], *, release_tag: str) -> list[dict[str, Any]]:
    """Every attempt found under the given Harbor jobs directories."""
    trials: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for root in jobs_dirs:
        if not root.exists():
            continue
        for res_file in sorted(root.rglob("result.json")):
            trial_dir = res_file.parent
            # Skip job-level result.json (trial result.json has `task_name`)
            parsed = parse_trial_dir(trial_dir, pricing, release_tag=release_tag)
            if parsed is None or parsed["task_id"] == trial_dir.name and not (trial_dir / "verifier").exists() and not (trial_dir / "agent").exists():
                continue
            if parsed["trial_id"] in seen_ids:
                continue
            seen_ids.add(parsed["trial_id"])
            trials.append(parsed)
    return trials


def collect_jobs(jobs_dirs: list[Path], pricing: dict[str, Any], *, release_tag: str) -> dict[str, Any]:
    return aggregate_results(scan_jobs(jobs_dirs, pricing, release_tag=release_tag), pricing, release_tag=release_tag)


def load_task_meta(task_ids: list[str], tasks_dir: Path = TASKS_DIR) -> dict[str, dict[str, Any]]:
    """Describe each task from its own `task.toml`, so the page never restates task facts by hand."""
    meta: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        info: dict[str, Any] = {"category": "rewind" if task_id.startswith("s3-") else "optimize"}
        toml_path = tasks_dir / task_id / "task.toml"
        if toml_path.is_file():
            cfg = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            md = cfg.get("metadata") or {}
            info.update({
                "title": md.get("description"),
                "category": md.get("category") or info["category"],
                "lane": md.get("lane"),
                "difficulty": md.get("difficulty"),
                "agent_timeout_s": (cfg.get("agent") or {}).get("timeout_sec"),
            })
        meta[task_id] = info
    return meta


# An attempt that the agent did not get to finish is not evidence about the agent. Which
# attempts count as interrupted is decided from how the run ended, never from its score:
# keeping the interrupted runs that happened to score well would bias every average.
INTERRUPTED = frozenset({"provider_limit", "infrastructure", "agent_crash"})
_INFRA_EXCEPTIONS = frozenset({
    "EnvironmentStartTimeoutError", "VerifierOutputParseError", "VerifierTimeoutError",
    "RewardFileNotFoundError", "RewardFileEmptyError",
})
IMPROVED_MARGIN = 0.05  # same margin as harbor/verify.py; used only for trials graded before it reported an outcome
INVALID_FLOOR = 0.1  # a task with no valid attempt enters the geometric mean at this ratio
OUTCOMES = ("solved", "improved", "no_gain", "invalid", "unscorable")
SCORING_NOTES = {
    "reference": "Speed-ups are relative to a Stan-style reference sampler on the same seeds. "
                 "A speed-up task is solved when the result is valid, beats the starting code by 5% and matches the reference.",
    "start": "Earlier release. Speed-ups are relative to the starting code, containers were not sealed and agents had a one-hour limit. "
             "Not comparable with later releases.",
}


def classify_termination(exc_type: str | None, *, rate_limited: bool, tool_calls: int) -> str:
    """How the agent phase ended: finished, time_limit, agent_error, or one of INTERRUPTED."""
    if exc_type is None:
        return "finished"
    if exc_type == "AgentTimeoutError":
        return "time_limit"
    if rate_limited or exc_type == "ApiRateLimitError":
        return "provider_limit"
    if exc_type in _INFRA_EXCEPTIONS:
        return "infrastructure"
    if exc_type == "NonZeroAgentExitCodeError" and tool_calls == 0:
        return "agent_crash"
    return "agent_error"


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def classify_attempt(trial: dict[str, Any]) -> dict[str, Any]:
    """Normalise one attempt, old or new, to: termination, outcome, ratio fields.

    Trials collected before schema 2 carry `infra_error` and `speedup` (against the starting
    code) instead of `termination` and `ref_speedup`; they are mapped here so one aggregation
    serves every release.
    """
    t = dict(trial)
    if "termination" not in t:
        exc_type = str(t["error"]).split(":", 1)[0] if t.get("error") else None
        t["termination"] = classify_termination(exc_type, rate_limited=bool(t.get("rate_limited")), tool_calls=int(t.get("tool_calls") or 0))
    t.pop("infra_error", None)
    if "start_speedup" not in t:
        t["start_speedup"] = t.pop("speedup", None)
        t["start_soft_min"] = t.pop("baseline_soft_min", None)
        t.setdefault("ref_speedup", None)
    t.pop("speedup", None)

    if t["termination"] in INTERRUPTED:
        t.update({"outcome": "interrupted", "passed": False, "reward": 0.0, "ref_speedup": None, "start_speedup": None,
                  "frontier_speedup": None, "gap_closed": None})
        return t
    reported = t.get("verifier_outcome")
    start, ref = _num(t.get("start_speedup")), _num(t.get("ref_speedup"))
    if reported in OUTCOMES:
        outcome = reported
    elif t["category"] == "rewind":
        outcome = "solved" if t.get("passed") else "invalid"
    elif not t.get("passed"):
        outcome = "invalid"
    elif start is None or start < 1.0 + IMPROVED_MARGIN:
        outcome = "no_gain"
    elif ref is None:
        outcome = "improved"  # no reference was scored for this release
    else:
        outcome = "solved" if ref >= 1.0 else "improved"
    t["outcome"] = outcome
    return t


def _geomean(values: list[float]) -> float | None:
    return round(math.exp(sum(math.log(v) for v in values) / len(values)), 4) if values else None


def _summarise_task(rows: list[dict[str, Any]], scoring: str) -> dict[str, Any]:
    """All attempts of one setting on one task."""
    scored = [r for r in rows if r["outcome"] != "interrupted"]
    key = "ref_speedup" if scoring == "reference" else "start_speedup"
    success = ("solved",) if scoring == "reference" or rows[0]["category"] == "rewind" else ("solved", "improved")
    # An invalid or unscorable attempt earned no speed-up: it counts as zero in the task mean.
    ratios = [(_num(r.get(key)) or 0.0) if r["outcome"] not in ("invalid", "unscorable") else 0.0 for r in scored]
    # Share of the distance from the starting code to the lane's frozen frontier that was closed
    # (0 for an invalid attempt, as for unchanged code); only where the verifier reported it.
    gaps = [0.0 if r["outcome"] in ("invalid", "unscorable") else _num(r.get("gap_closed")) for r in scored]
    gaps = [g for g in gaps if g is not None]
    counts = {o: sum(1 for r in scored if r["outcome"] == o) for o in OUTCOMES}
    n_ok = sum(counts[o] for o in success)
    best = max(scored, key=lambda r: (r["outcome"] in success, _num(r.get(key)) or 0.0), default=None)
    return {
        "category": rows[0]["category"],
        "attempts": len(scored),
        "interrupted": len(rows) - len(scored),
        "outcomes": {o: n for o, n in counts.items() if n},
        "solve_rate": round(n_ok / len(scored), 4) if scored else None,
        "ratio_mean": round(sum(ratios) / len(ratios), 4) if ratios else None,
        "ratio_min": round(min(ratios), 4) if ratios else None,
        "ratio_max": round(max(ratios), 4) if ratios else None,
        "gap_closed_mean": round(sum(gaps) / len(gaps), 4) if gaps else None,
        "best_trial": best["trial_id"] if best else None,
        "trial_ids": [r["trial_id"] for r in rows],
    }


def summarise_setting(rows: list[dict[str, Any]], tasks: list[str], scoring: str, label: str) -> dict[str, Any]:
    """One leaderboard row: every attempt one model and effort setting made in one release."""
    per_task = {tid: _summarise_task([r for r in rows if r["task_id"] == tid], scoring) for tid in tasks if any(r["task_id"] == tid for r in rows)}
    done = {tid: s for tid, s in per_task.items() if s["attempts"]}
    fixes = [s for s in done.values() if s["category"] == "rewind"]
    opts = [s for s in done.values() if s["category"] == "optimize"]
    scored = [r for r in rows if r["outcome"] != "interrupted"]

    def total(field: str) -> float:
        return sum(float(r.get(field) or 0) for r in rows)

    def mean(field: str, src: list[dict[str, Any]] = scored) -> float | None:
        vals = [float(r[field]) for r in src if _num(r.get(field)) is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    sol = [r["solution"] for r in scored if isinstance(r.get("solution"), dict)]
    tot_in, tot_cached = total("input_tokens"), total("cached_tokens")
    return {
        "config_key": rows[0]["config_key"],
        "model": rows[0]["model"],
        "label": label,
        "effort": rows[0]["effort"],
        "agents": sorted({str(r.get("agent") or "unknown") for r in rows}),
        "n_tasks": len(tasks),
        "tasks_run": len(done),
        "complete": len(done) == len(tasks),
        "attempts": len(scored),
        "interrupted": len(rows) - len(scored),
        "attempts_per_task": max((s["attempts"] for s in done.values()), default=0),
        # Expected number of tasks solved by one attempt at each: the ranking key.
        "solved": round(sum(s["solve_rate"] for s in done.values()), 3),
        "fix_solved": round(sum(s["solve_rate"] for s in fixes), 3),
        "fix_total": sum(1 for t in tasks if t.startswith("s3-")),
        "opt_solved": round(sum(s["solve_rate"] for s in opts), 3),
        "opt_total": sum(1 for t in tasks if not t.startswith("s3-")),
        "opt_valid": round(sum(1 - (s["outcomes"].get("invalid", 0) + s["outcomes"].get("unscorable", 0)) / s["attempts"] for s in opts), 3),
        "opt_geomean": _geomean([max(s["ratio_mean"], INVALID_FLOOR) for s in opts]),
        "opt_gap_closed": (lambda g: round(sum(g) / len(g), 4) if g else None)([s["gap_closed_mean"] for s in opts if s["gap_closed_mean"] is not None]),
        "tasks": per_task,
        "total_cost_usd": round(total("cost_usd"), 5),
        "total_tokens": int(total("total_tokens")),
        "input_tokens": int(tot_in),
        "cached_tokens": int(tot_cached),
        "output_tokens": int(total("output_tokens")),
        "reasoning_tokens": int(total("reasoning_tokens")),
        "cache_hit_rate": round(tot_cached / (tot_in + tot_cached), 4) if (tot_in + tot_cached) > 0 else 0.0,
        "total_wall_s": round(total("agent_wall_s"), 1),
        "max_wall_s": max((float(r["agent_wall_s"]) for r in rows if _num(r.get("agent_wall_s")) is not None), default=None),
        "time_limit_hits": sum(1 for r in rows if r["termination"] == "time_limit"),
        "mean_turns": mean("turns"),
        "mean_tool_calls": mean("tool_calls"),
        "mean_test_runs": mean("test_runs"),
        "mean_loc": mean("loc", sol),
        "mean_cyclomatic": mean("cyclomatic_complexity", sol),
    }


def summarise_release(release: str, trials: list[dict[str, Any]], pricing: dict[str, Any]) -> dict[str, Any]:
    """Leaderboard and attempts for one benchmark release."""
    # A release is scored against the reference once any of its attempts carries a reference ratio.
    scoring = "reference" if any(_num(t.get("ref_speedup")) is not None for t in trials) else "start"
    tasks = sorted({t["task_id"] for t in trials})
    by_cfg: dict[str, list[dict[str, Any]]] = {}
    for t in trials:
        by_cfg.setdefault(t["config_key"], []).append(t)
    labels = pricing.get("models") or {}
    board = [summarise_setting(rows, tasks, scoring, labels.get(rows[0]["model"], {}).get("label") or rows[0]["model"]) for _, rows in sorted(by_cfg.items())]
    # Settings with a result on every task come first; among them more tasks solved, then the larger
    # typical speed-up, then the cheaper sweep.
    board.sort(key=lambda r: (not r["complete"], -r["solved"], -(r["opt_geomean"] or 0.0), r["total_cost_usd"]))
    for idx, row in enumerate(board, start=1):
        row["rank"] = idx
    return {
        "release": release,
        "scoring": scoring,
        "note": SCORING_NOTES[scoring],
        "tasks": tasks,
        "task_meta": load_task_meta(tasks),
        "leaderboard": board,
        "trials": trials,
    }


def _release_order(tag: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", tag))


def aggregate_results(trials: list[dict[str, Any]], pricing: dict[str, Any], *, release_tag: str) -> dict[str, Any]:
    """Site data: one self-contained block per release, newest first. Releases are never pooled."""
    by_release: dict[str, list[dict[str, Any]]] = {}
    for t in map(classify_attempt, trials):
        by_release.setdefault(str(t.get("release") or release_tag), []).append(t)
    releases = [summarise_release(tag, rows, pricing) for tag, rows in sorted(by_release.items(), key=lambda kv: _release_order(kv[0]), reverse=True)]
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "current_release": releases[0]["release"] if releases else release_tag,
        "invalid_floor": INVALID_FLOOR,
        "releases": releases,
    }


def _json_safe(value: Any) -> Any:
    """Replace NaN and infinities with null: browsers reject them, Python writes them by default."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_site_data(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _flatten_trials(data: dict[str, Any]) -> list[dict[str, Any]]:
    if "releases" in data:
        return [t for rel in data["releases"] for t in rel["trials"]]
    return list(data.get("trials") or [])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collect Harbor benchmark results into the site's benchmark data file")
    ap.add_argument("--jobs", type=Path, action="append", default=[], help="Harbor jobs directory to scan (repeatable)")
    ap.add_argument("--input-json", type=Path, help="Data file whose attempts are kept (default: the output file, if it exists)")
    ap.add_argument("--fresh", action="store_true", help="Start from nothing instead of adding to the existing data file")
    ap.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    ap.add_argument("--release", default=DEFAULT_RELEASE, help="Release tag for job directories whose name does not start with one")
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = ap.parse_args(argv)

    pricing = json.loads(args.pricing.read_text(encoding="utf-8")) if args.pricing.is_file() else {"models": {}}
    # Attempts already published are kept: a sweep adds to the record, it does not replace it.
    prior = None if args.fresh else (args.input_json or args.out_json)
    trials: dict[str, dict[str, Any]] = {}
    if prior and prior.is_file():
        for t in _flatten_trials(json.loads(prior.read_text(encoding="utf-8"))):
            trials[f"{t.get('release')}/{t['trial_id']}"] = t
    if args.jobs or not trials:
        for t in scan_jobs(args.jobs or [ROOT / "jobs"], pricing, release_tag=args.release):
            trials[f"{t.get('release')}/{t['trial_id']}"] = t
    data = aggregate_results(list(trials.values()), pricing, release_tag=args.release)

    write_site_data(data, args.out_json)
    n = sum(len(r["trials"]) for r in data["releases"])
    print(f"wrote {args.out_json} ({n} attempts, {len(data['releases'])} releases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
