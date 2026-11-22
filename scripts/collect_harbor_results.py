#!/usr/bin/env python3
"""Collect Harbor job outputs into benchmarks/results.json and render benchmarks/index.html.

Reads Harbor trial directories (`result.json`, `agent/trajectory.json`,
`agent/opencode-v2.txt`, `verifier/reward.json`, `verifier/submission.py`),
extracts exact walltime, token usage (input, cached, cache_write, output, reasoning),
price, turns, tool calls, test runs, solution size & complexity, approach summary,
and all verifier metrics, and builds a standalone interactive benchmark page with
Price & Token Pareto curves.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICING = ROOT / "benchmarks" / "pricing.json"
DEFAULT_OUT_JSON = ROOT / "benchmarks" / "results.json"
DEFAULT_OUT_HTML = ROOT / "benchmarks" / "index.html"

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

    return {
        "trial_id": str(res.get("trial_name") or trial_dir.name),
        "release": release_tag,
        "model": model,
        "effort": effort,
        "config_key": f"{model}#{effort}",
        "agent": agent_name,
        "task_id": task_id,
        "category": category,
        "passed": passed,
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
            "n_trials": len(rows),
            "passed": n_pass,
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
        "leaderboard": leaderboard,
        "trials": trials,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>apin Agent Benchmark — Harbor Leaderboard &amp; Pareto Curves</title>
<style>
  :root {
    --bg: #0b0f14; --surface: #121821; --surface2: #18202c; --border: #263244;
    --text: #e6edf3; --muted: #8b9bb0; --accent: #58a6ff; --green: #3fb950;
    --amber: #d29922; --red: #f85149; --purple: #bc8cff; --teal: #39c5cf;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }
  .wrap { max-width: 1260px; margin: 0 auto; padding: 28px 20px 64px; }
  header { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-end; gap: 12px; border-bottom: 1px solid var(--border); padding-bottom: 18px; margin-bottom: 24px; }
  h1 { margin: 0; font-size: 1.75rem; letter-spacing: -0.02em; }
  .sub { color: var(--muted); font-size: 0.92rem; margin-top: 4px; }
  .pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; background: var(--surface2); border: 1px solid var(--border); color: var(--accent); }
  .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; margin-bottom: 24px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
  .card h2 { margin: 0 0 12px; font-size: 1.05rem; display: flex; justify-content: space-between; align-items: center; }
  .hint { color: var(--muted); font-size: 0.8rem; font-weight: 400; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; align-items: center; }
  select, input { background: var(--surface2); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; font-size: 0.86rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
  th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }
  th { color: var(--muted); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em; }
  tr:hover td { background: rgba(88, 166, 255, 0.05); }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .badge { display: inline-block; padding: 2px 7px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
  .b-low { background: rgba(57,197,207,0.15); color: var(--teal); }
  .b-high { background: rgba(88,166,255,0.15); color: var(--accent); }
  .b-max { background: rgba(188,140,255,0.15); color: var(--purple); }
  .b-def { background: rgba(139,155,176,0.15); color: var(--muted); }
  .cell-btn { cursor: pointer; border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; font-weight: 600; font-size: 0.8rem; background: var(--surface2); color: var(--text); min-width: 68px; text-align: center; }
  .cell-pass { border-color: rgba(63,185,80,0.5); background: rgba(63,185,80,0.14); color: var(--green); }
  .cell-win { border-color: rgba(88,166,255,0.6); background: rgba(88,166,255,0.18); color: #8ec5ff; }
  .cell-fail { border-color: rgba(248,81,73,0.45); background: rgba(248,81,73,0.12); color: var(--red); }
  .table-wrap { overflow-x: auto; }
  svg.chart { width: 100%; height: 290px; display: block; }
  .tooltip { position: fixed; pointer-events: none; background: #0d131c; border: 1px solid var(--accent); border-radius: 8px; padding: 8px 10px; font-size: 0.8rem; max-width: 300px; display: none; z-index: 99; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
  .drawer { margin-top: 20px; display: none; }
  .drawer.open { display: block; }
  .kv-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 12px 0; }
  .kv { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
  .kv .k { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; }
  .kv .v { font-size: 1.02rem; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }
  pre.summary { background: #080c11; border: 1px solid var(--border); border-radius: 8px; padding: 12px; white-space: pre-wrap; font-size: 0.82rem; color: #c9d1d9; max-height: 260px; overflow-y: auto; }
  .bar-row { display: flex; align-items: center; gap: 10px; margin: 4px 0; font-size: 0.82rem; }
  .bar-track { flex: 1; height: 8px; background: var(--surface2); border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--accent); }
  .empty { color: var(--muted); text-align: center; padding: 36px 12px; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>apin Agent Benchmark</h1>
      <div class="sub">Containerized Harbor suite for Bayesian posterior approximators · exact-draw invariance &amp; dual-horizon soft-min (p = −4) speedup</div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <span class="pill" id="releasePill">release</span>
      <span class="pill" id="countPill">0 trials</span>
      <a class="pill" href="https://github.com/Numenomics/apin" target="_blank" rel="noopener">GitHub</a>
      <a class="pill" href="https://apin-flywheel.m0hawk.workers.dev" target="_blank" rel="noopener">Flywheel</a>
    </div>
  </header>

  <div class="controls">
    <label>Effort:
      <select id="effortFilter"><option value="all">All efforts</option></select>
    </label>
    <label>Model:
      <select id="modelFilter"><option value="all">All models</option></select>
    </label>
    <span class="hint" id="genAt"></span>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Price Pareto Frontier <span class="hint">Mean Reward vs. Total USD Cost (dashed = Pareto frontier)</span></h2>
      <svg id="priceChart" class="chart" viewBox="0 0 540 290"></svg>
    </div>
    <div class="card">
      <h2>Token Efficiency Pareto <span class="hint">Mean Reward vs. Total Tokens (cached + uncached + output)</span></h2>
      <svg id="tokenChart" class="chart" viewBox="0 0 540 290"></svg>
    </div>
  </div>

  <div class="card" style="margin-bottom:24px">
    <h2>Model × Effort Leaderboard <span class="hint">Ranked by mean reward, pass rate, then cost</span></h2>
    <div class="table-wrap"><table id="lbTable"></table></div>
  </div>

  <div class="card">
    <h2>Task × Model Matrix <span class="hint">Click any cell to inspect tokens, walltime, complexity, by-class metrics &amp; approach summary</span></h2>
    <div class="table-wrap"><table id="matrixTable"></table></div>
  </div>

  <div class="card drawer" id="detailDrawer">
    <h2 id="detailTitle">Run Details <button class="cell-btn" onclick="document.getElementById('detailDrawer').classList.remove('open')">Close</button></h2>
    <div class="kv-grid" id="detailKvs"></div>
    <div id="detailClasses"></div>
    <h3 style="font-size:0.9rem;margin:14px 0 6px;color:var(--muted)">Approach &amp; Solution Summary</h3>
    <pre class="summary" id="detailSummary"></pre>
  </div>
</div>
<div class="tooltip" id="tip"></div>

<script>
const DATA = __BENCHMARK_DATA_JSON__;

const COLORS = {
  "zai-coding-plan/glm-5.3": "#58a6ff",
  "zai-coding-plan/glm-5.3-flash": "#3fb950",
  "opencode/tidepool-alpha-free": "#bc8cff",
};
function modelColor(m) { return COLORS[m] || "#39c5cf"; }
function fmtTok(n) {
  if (!n) return "0";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}
function fmtUsd(x) { return x === 0 ? "$0.00" : "$" + Number(x).toFixed(x < 0.01 ? 4 : 3); }
function fmtSec(s) {
  if (s == null) return "—";
  if (s >= 60) return Math.floor(s / 60) + "m " + Math.round(s % 60) + "s";
  return Number(s).toFixed(1) + "s";
}
function effortBadge(e) {
  const cls = e === "low" ? "b-low" : e === "high" ? "b-high" : e === "max" ? "b-max" : "b-def";
  return `<span class="badge ${cls}">${e}</span>`;
}
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

document.getElementById("releasePill").textContent = DATA.release || "dev";
document.getElementById("countPill").textContent = `${DATA.trials.length} trials · ${DATA.tasks.length} tasks`;
document.getElementById("genAt").textContent = DATA.generated_at ? `Updated ${DATA.generated_at}` : "";

const effSel = document.getElementById("effortFilter");
const modSel = document.getElementById("modelFilter");
[...new Set(DATA.leaderboard.map(r => r.effort))].sort().forEach(e => effSel.insertAdjacentHTML("beforeend", `<option value="${esc(e)}">${esc(e)}</option>`));
[...new Set(DATA.leaderboard.map(r => r.model))].sort().forEach(m => modSel.insertAdjacentHTML("beforeend", `<option value="${esc(m)}">${esc(m)}</option>`));

function filteredBoard() {
  return DATA.leaderboard.filter(r =>
    (effSel.value === "all" || r.effort === effSel.value) &&
    (modSel.value === "all" || r.model === modSel.value)
  );
}

function drawScatter(svgId, rows, xFn, xFmt, xLabel) {
  const svg = document.getElementById(svgId);
  svg.innerHTML = "";
  if (!rows.length) {
    svg.innerHTML = `<text x="270" y="145" text-anchor="middle" fill="#8b9bb0" font-size="13">No benchmark runs yet</text>`;
    return;
  }
  const W = 540, H = 290, padL = 54, padR = 24, padT = 22, padB = 42;
  const xs = rows.map(xFn), ys = rows.map(r => r.mean_reward);
  const xMin = 0, xMax = Math.max(...xs, 1e-6) * 1.12;
  const yMin = 0, yMax = Math.max(...ys, 1.0) * 1.15;
  const sx = x => padL + ((x - xMin) / (xMax - xMin)) * (W - padL - padR);
  const sy = y => H - padB - ((y - yMin) / (yMax - yMin)) * (H - padT - padB);

  let html = "";
  for (let i = 0; i <= 4; i++) {
    const yv = (yMax * i) / 4, yp = sy(yv);
    html += `<line x1="${padL}" y1="${yp}" x2="${W - padR}" y2="${yp}" stroke="#263244" stroke-dasharray="3,3"/>`;
    html += `<text x="${padL - 8}" y="${yp + 4}" text-anchor="end" fill="#8b9bb0" font-size="10">${yv.toFixed(2)}</text>`;
    const xv = (xMax * i) / 4, xp = sx(xv);
    html += `<line x1="${xp}" y1="${padT}" x2="${xp}" y2="${H - padB}" stroke="#1c2533"/>`;
    html += `<text x="${xp}" y="${H - padB + 16}" text-anchor="middle" fill="#8b9bb0" font-size="10">${xFmt(xv)}</text>`;
  }
  html += `<text x="${(padL + W - padR) / 2}" y="${H - 8}" text-anchor="middle" fill="#8b9bb0" font-size="11">${xLabel}</text>`;
  html += `<text x="14" y="${(padT + H - padB) / 2}" text-anchor="middle" fill="#8b9bb0" font-size="11" transform="rotate(-90 14 ${(padT + H - padB) / 2})">Mean Reward</text>`;

  // Pareto frontier (minimize x, maximize y)
  const sorted = [...rows].sort((a, b) => xFn(a) - xFn(b) || b.mean_reward - a.mean_reward);
  const frontier = [];
  let bestY = -1;
  for (const r of sorted) {
    if (r.mean_reward > bestY) { frontier.push(r); bestY = r.mean_reward; }
  }
  if (frontier.length >= 2) {
    const pts = frontier.map(r => `${sx(xFn(r)).toFixed(1)},${sy(r.mean_reward).toFixed(1)}`).join(" ");
    html += `<polyline points="${pts}" fill="none" stroke="#58a6ff" stroke-width="1.8" stroke-dasharray="5,4" opacity="0.85"/>`;
  }

  rows.forEach((r, idx) => {
    const cx = sx(xFn(r)), cy = sy(r.mean_reward), col = modelColor(r.model);
    const rad = r.effort === "max" ? 7.5 : r.effort === "high" ? 6 : 4.8;
    html += `<circle cx="${cx}" cy="${cy}" r="${rad}" fill="${col}" stroke="#0b0f14" stroke-width="1.5" data-idx="${idx}" style="cursor:pointer"/>`;
    html += `<text x="${cx + 9}" y="${cy + 3}" fill="#c9d1d9" font-size="10">${esc(r.label)} (${esc(r.effort)})</text>`;
  });
  svg.innerHTML = html;

  const tip = document.getElementById("tip");
  svg.querySelectorAll("circle").forEach(c => {
    c.addEventListener("mousemove", ev => {
      const r = rows[Number(c.dataset.idx)];
      tip.style.display = "block";
      tip.style.left = (ev.clientX + 14) + "px";
      tip.style.top = (ev.clientY - 12) + "px";
      tip.innerHTML = `<strong>${esc(r.label)} (${esc(r.effort)})</strong><br>` +
        `Mean reward: <b>${r.mean_reward.toFixed(4)}</b> · Pass: ${r.passed}/${r.n_trials}<br>` +
        `Opt geomean: ${r.opt_geomean_speedup.toFixed(3)}x<br>` +
        `Cost: ${fmtUsd(r.total_cost_usd)} · Tokens: ${fmtTok(r.total_tokens)} (${(r.cache_hit_rate*100).toFixed(0)}% cached)<br>` +
        `Walltime: ${fmtSec(r.total_wall_s)} · Turns: ${r.mean_turns} · Tools: ${r.mean_tool_calls}`;
    });
    c.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  });
}

function renderLeaderboard(rows) {
  const tbl = document.getElementById("lbTable");
  if (!rows.length) { tbl.innerHTML = `<tr><td class="empty">No runs match the current filter.</td></tr>`; return; }
  let html = `<thead><tr>
    <th>#</th><th>Model</th><th>Effort</th><th class="num">Mean Reward</th>
    <th class="num">Rewinds</th><th class="num">Optimize</th><th class="num">Opt Geomean</th>
    <th class="num">Cost</th><th class="num">Tokens (Cached)</th>
    <th class="num">Wall</th><th class="num">Turns / Tools / Tests</th><th class="num">LOC / Cyc</th>
  </tr></thead><tbody>`;
  for (const r of rows) {
    html += `<tr>
      <td>${r.rank}</td>
      <td><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${modelColor(r.model)};margin-right:6px"></span>${esc(r.label)}</td>
      <td>${effortBadge(r.effort)}</td>
      <td class="num"><b>${r.mean_reward.toFixed(4)}</b></td>
      <td class="num">${r.rewind_passed}/${r.rewind_total}</td>
      <td class="num">${r.opt_passed}/${r.opt_total}</td>
      <td class="num">${r.opt_geomean_speedup ? r.opt_geomean_speedup.toFixed(3) + "x" : "—"}</td>
      <td class="num">${fmtUsd(r.total_cost_usd)}</td>
      <td class="num">${fmtTok(r.total_tokens)} <span class="hint">(${(r.cache_hit_rate*100).toFixed(0)}%)</span></td>
      <td class="num">${fmtSec(r.total_wall_s)}</td>
      <td class="num">${r.mean_turns} / ${r.mean_tool_calls} / ${r.mean_test_runs}</td>
      <td class="num">${r.mean_loc ?? "—"} / ${r.mean_cyclomatic ?? "—"}</td>
    </tr>`;
  }
  tbl.innerHTML = html + "</tbody>";
}

function showDetail(t) {
  const d = document.getElementById("detailDrawer");
  d.classList.add("open");
  document.getElementById("detailTitle").innerHTML =
    `<span>${esc(t.model)} (${esc(t.effort)}) — <code>${esc(t.task_id)}</code></span>` +
    `<button class="cell-btn" onclick="document.getElementById('detailDrawer').classList.remove('open')">Close</button>`;
  const sol = t.solution || {};
  const kvs = [
    ["Reward", t.reward.toFixed(4)],
    ["Passed", t.passed ? "YES" : "NO"],
    ["Cost (USD)", `${fmtUsd(t.cost_usd)} (${t.cost_source})`],
    ["Walltime (Agent)", fmtSec(t.agent_wall_s)],
    ["Input / Cached", `${fmtTok(t.input_tokens)} / ${fmtTok(t.cached_tokens)} (${(t.cache_hit_rate*100).toFixed(0)}%)`],
    ["Output / Reasoning", `${fmtTok(t.output_tokens)} / ${fmtTok(t.reasoning_tokens)}`],
    ["Turns / Tools / Tests", `${t.turns} / ${t.tool_calls} / ${t.test_runs}`],
    ["LOC / Magic Consts", `${sol.loc ?? "—"} / ${sol.magic_constants ?? "—"}`],
    ["AST / Cyclomatic", `${sol.ast_nodes ?? "—"} / ${sol.cyclomatic_complexity ?? "—"}`],
    ["b²_max / Overhead", `${t.b2_max ?? "—"} / ${t.overhead_ratio != null ? t.overhead_ratio + "x" : "—"}`],
  ];
  document.getElementById("detailKvs").innerHTML = kvs.map(([k, v]) => `<div class="kv"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");

  const clsDiv = document.getElementById("detailClasses");
  const byClass = t.by_class || {};
  const entries = Object.entries(byClass).sort((a, b) => a[1] - b[1]);
  if (entries.length) {
    const mx = Math.max(...entries.map(e => e[1]), 1e-9);
    clsDiv.innerHTML = `<h3 style="font-size:0.9rem;margin:12px 0 6px;color:var(--muted)">Per-Class Efficiency (weakest first — dominates p=−4 soft-min)</h3>` +
      entries.map(([k, v]) => `<div class="bar-row"><span style="width:160px">${esc(k)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, (v/mx)*100).toFixed(1)}%"></div></div><span class="num" style="width:80px">${Number(v).toExponential(2)}</span></div>`).join("");
  } else if ((t.failed_targets || []).length) {
    clsDiv.innerHTML = `<div style="color:var(--red);font-size:0.84rem;margin-top:8px">Failed targets: ${esc(t.failed_targets.join("; "))}</div>`;
  } else {
    clsDiv.innerHTML = "";
  }
  document.getElementById("detailSummary").textContent = t.approach_summary || t.error || "(no summary captured)";
  d.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderMatrix(rows) {
  const tbl = document.getElementById("matrixTable");
  if (!rows.length || !DATA.tasks.length) { tbl.innerHTML = `<tr><td class="empty">No task matrix data yet.</td></tr>`; return; }
  const lookup = {};
  for (const t of DATA.trials) { lookup[t.config_key + "||" + t.task_id] = t; }
  let html = `<thead><tr><th>Task</th><th>Type</th>` + rows.map(r => `<th class="num">${esc(r.label)}<br>${effortBadge(r.effort)}</th>`).join("") + `</tr></thead><tbody>`;
  for (const tid of DATA.tasks) {
    const cat = tid.startsWith("s3-") ? "rewind" : "optimize";
    html += `<tr><td><code>${esc(tid)}</code></td><td class="hint">${cat}</td>`;
    for (const r of rows) {
      const t = lookup[r.config_key + "||" + tid];
      if (!t) {
        html += `<td class="num"><span class="hint">—</span></td>`;
      } else {
        const cls = !t.passed ? "cell-fail" : (cat === "optimize" && t.reward > 1.02 ? "cell-win" : "cell-pass");
        const txt = cat === "rewind" ? (t.passed ? "PASS" : "FAIL") : (t.passed ? t.reward.toFixed(3) + "x" : "FAIL");
        html += `<td class="num"><button class="cell-btn ${cls}" data-key="${esc(r.config_key + "||" + tid)}">${txt}</button></td>`;
      }
    }
    html += `</tr>`;
  }
  tbl.innerHTML = html + "</tbody>";
  tbl.querySelectorAll("button[data-key]").forEach(btn => {
    btn.addEventListener("click", () => showDetail(lookup[btn.dataset.key]));
  });
}

function render() {
  const rows = filteredBoard();
  drawScatter("priceChart", rows, r => r.total_cost_usd, fmtUsd, "Total USD Cost");
  drawScatter("tokenChart", rows, r => r.total_tokens, fmtTok, "Total Tokens");
  renderLeaderboard(rows);
  renderMatrix(rows);
}

effSel.addEventListener("change", render);
modSel.addEventListener("change", render);
render();
</script>
</body>
</html>
"""


def render_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, indent=None, ensure_ascii=False).replace("</script>", "<\\/script>")
    return HTML_TEMPLATE.replace("__BENCHMARK_DATA_JSON__", payload)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collect Harbor benchmark results and build benchmarks/index.html")
    ap.add_argument("--jobs", type=Path, action="append", default=[], help="Harbor jobs directory to scan (repeatable)")
    ap.add_argument("--input-json", type=Path, help="Existing results.json to re-render without scanning jobs/")
    ap.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    ap.add_argument("--release", default="bench-v0.1.0")
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-html", type=Path, default=DEFAULT_OUT_HTML)
    args = ap.parse_args(argv)

    pricing = json.loads(args.pricing.read_text(encoding="utf-8")) if args.pricing.is_file() else {"models": {}}
    if args.input_json and args.input_json.is_file():
        data = json.loads(args.input_json.read_text(encoding="utf-8"))
    else:
        jobs_dirs = args.jobs or [ROOT / "jobs"]
        data = collect_jobs(jobs_dirs, pricing, release_tag=args.release)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(render_html(data), encoding="utf-8")
    print(f"wrote {args.out_json} ({len(data['trials'])} trials, {len(data['leaderboard'])} configs) and {args.out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
