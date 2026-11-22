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


def parse_trial_dir(trial_dir: Path, pricing: dict[str, Any], *, release_tag: str = "bench-v0.1.0") -> dict[str, Any] | None:
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

    # Distinguish real agent timeouts / late rate-limits (partial work on disk may be a real pass!)
    # from 0-edit crashes where an untouched optimization baseline would falsely score 1.0.
    oc_raw_text = (trial_dir / "agent" / "opencode-v2.txt").read_text(encoding="utf-8", errors="replace") if (trial_dir / "agent" / "opencode-v2.txt").is_file() else ""
    oc_low = oc_raw_text.lower()
    rate_limited = (
        ("provider.free-limit" in oc_raw_text)
        or ("provider.quota" in oc_raw_text)
        or ("provider.rate-limit" in oc_raw_text)
        or ("FreeUsageLimitError" in oc_raw_text)
        or ("usage limit reached" in oc_low)
        or ("rate limit" in oc_low and exc_type in ("NonZeroAgentExitCodeError", "ApiRateLimitError"))
    )
    infra_error = bool(
        (exc_type == "NonZeroAgentExitCodeError" and tool_calls == 0)
        or (rate_limited and tool_calls == 0)
        or (rate_limited and not passed)
        or exc_type in ("EnvironmentStartTimeoutError", "VerifierOutputParseError")
    )
    if infra_error:
        passed = False
        reward = 0.0

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
        "infra_error": infra_error,
        "rate_limited": rate_limited,
        "reward": round(reward, 6),
        "speedup": reward_data.get("speedup"),
        "soft_min_eff": reward_data.get("soft_min_eff"),
        "baseline_soft_min": reward_data.get("baseline_soft_min"),
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


def collect_jobs(jobs_dirs: list[Path], pricing: dict[str, Any], *, release_tag: str) -> dict[str, Any]:
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

    return aggregate_results(trials, pricing, release_tag=release_tag)


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


def aggregate_results(trials: list[dict[str, Any]], pricing: dict[str, Any], *, release_tag: str) -> dict[str, Any]:
    tasks = sorted({t["task_id"] for t in trials})
    by_cfg: dict[str, list[dict[str, Any]]] = {}
    for t in trials:
        by_cfg.setdefault(t["config_key"], []).append(t)

    leaderboard: list[dict[str, Any]] = []
    for cfg_key, rows in sorted(by_cfg.items()):
        model = rows[0]["model"]
        effort = rows[0]["effort"]
        label = (pricing.get("models") or {}).get(model, {}).get("label") or model
        rewinds = [r for r in rows if r["category"] == "rewind"]
        opts = [r for r in rows if r["category"] == "optimize"]
        n_pass = sum(1 for r in rows if r["passed"])
        rw_pass = sum(1 for r in rewinds if r["passed"])
        opt_pass = sum(1 for r in opts if r["passed"])
        opt_pos = [float(r["reward"]) for r in opts if r["passed"] and float(r["reward"]) > 0]
        opt_geomean = round(math.exp(sum(math.log(x) for x in opt_pos) / len(opt_pos)), 4) if opt_pos else 0.0
        mean_reward = round(sum(float(r["reward"]) for r in rows) / len(rows), 4) if rows else 0.0

        wall_vals = [r["agent_wall_s"] for r in rows if isinstance(r.get("agent_wall_s"), (int, float))]
        loc_vals = [r["solution"]["loc"] for r in rows if isinstance((r.get("solution") or {}).get("loc"), (int, float))]
        cyc_vals = [
            r["solution"]["cyclomatic_complexity"]
            for r in rows
            if isinstance((r.get("solution") or {}).get("cyclomatic_complexity"), (int, float))
        ]
        tot_in = sum(int(r.get("input_tokens") or 0) for r in rows)
        tot_cached = sum(int(r.get("cached_tokens") or 0) for r in rows)
        tot_out = sum(int(r.get("output_tokens") or 0) for r in rows)
        tot_reas = sum(int(r.get("reasoning_tokens") or 0) for r in rows)
        tot_toks = sum(int(r.get("total_tokens") or 0) for r in rows)
        tot_cost = round(sum(float(r.get("cost_usd") or 0.0) for r in rows), 5)

        leaderboard.append({
            "config_key": cfg_key,
            "model": model,
            "label": label,
            "effort": effort,
            "agents": sorted({str(r.get("agent") or "unknown") for r in rows}),
            "n_trials": len(rows),
            "passed": n_pass,
            "infra_errors": sum(1 for r in rows if r.get("infra_error")),
            "pass_rate": round(n_pass / len(rows), 4) if rows else 0.0,
            "rewind_passed": rw_pass,
            "rewind_total": len(rewinds),
            "opt_passed": opt_pass,
            "opt_total": len(opts),
            "opt_geomean_speedup": opt_geomean,
            "mean_reward": mean_reward,
            "total_cost_usd": tot_cost,
            "mean_cost_usd": round(tot_cost / len(rows), 5) if rows else 0.0,
            "total_tokens": tot_toks,
            "input_tokens": tot_in,
            "cached_tokens": tot_cached,
            "output_tokens": tot_out,
            "reasoning_tokens": tot_reas,
            "cache_hit_rate": round(tot_cached / (tot_in + tot_cached), 4) if (tot_in + tot_cached) > 0 else 0.0,
            "total_wall_s": round(sum(wall_vals), 1) if wall_vals else 0.0,
            "mean_wall_s": round(sum(wall_vals) / len(wall_vals), 1) if wall_vals else 0.0,
            "mean_turns": round(sum(int(r.get("turns") or 0) for r in rows) / len(rows), 1) if rows else 0.0,
            "mean_tool_calls": round(sum(int(r.get("tool_calls") or 0) for r in rows) / len(rows), 1) if rows else 0.0,
            "mean_test_runs": round(sum(int(r.get("test_runs") or 0) for r in rows) / len(rows), 1) if rows else 0.0,
            "mean_loc": round(sum(loc_vals) / len(loc_vals), 1) if loc_vals else None,
            "mean_cyclomatic": round(sum(cyc_vals) / len(cyc_vals), 1) if cyc_vals else None,
        })

    leaderboard.sort(key=lambda r: (-r["mean_reward"], -r["pass_rate"], r["total_cost_usd"]))
    for idx, row in enumerate(leaderboard, start=1):
        row["rank"] = idx

    return {
        "schema_version": 1,
        "release": release_tag,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tasks": tasks,
        "task_meta": load_task_meta(tasks),
        "leaderboard": leaderboard,
        "trials": trials,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collect Harbor benchmark results into the site's benchmark data file")
    ap.add_argument("--jobs", type=Path, action="append", default=[], help="Harbor jobs directory to scan (repeatable)")
    ap.add_argument("--input-json", type=Path, help="Existing data file whose trials are re-aggregated without scanning jobs/")
    ap.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    ap.add_argument("--release", default="bench-v0.1.0")
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = ap.parse_args(argv)

    pricing = json.loads(args.pricing.read_text(encoding="utf-8")) if args.pricing.is_file() else {"models": {}}
    if args.input_json and args.input_json.is_file():
        prior = json.loads(args.input_json.read_text(encoding="utf-8"))
        data = aggregate_results(prior["trials"], pricing, release_tag=prior.get("release") or args.release)
    else:
        jobs_dirs = args.jobs or [ROOT / "jobs"]
        data = collect_jobs(jobs_dirs, pricing, release_tag=args.release)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json} ({len(data['trials'])} trials, {len(data['leaderboard'])} configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
