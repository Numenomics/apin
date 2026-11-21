#!/usr/bin/env python3
"""Seed apin Flywheel D1 SQL (`flywheel/seed.sql`) and static dashboard snapshot
(`flywheel/public/snapshot.json`) from local autoresearch ledgers and real-Stan score CSVs.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

KEPT_DECISIONS = {"keep", "keep_baseline", "keep_simpler"}

# Explicit mapping from real-Stan score CSV labels to (lane, run_id, target_selector)
# target_selector is either "final_kept" or "baseline".
SCORE_LABEL_TARGETS: dict[str, tuple[str, str, str]] = {
    "a0_glm53_dual": ("A0", "glm53_a0_dual_1h", "final_kept"),
    "a1_glm53_dual": ("A1", "glm53_a1_dual_1h", "final_kept"),
    "a2_glm53_dual": ("A2", "glm53_a2_dual_1h", "final_kept"),
    "a3_glm53_symp": ("A3", "glm53_a3_symp_1h", "final_kept"),
    "a4_glm53_dual": ("A4", "glm53_a4_dual_1h", "final_kept"),
    "a3_glm53_pdmp_1h": ("A3", "glm53_pdmp_1h", "T0003"),
    "a0_glm53_groundup_1h": ("A0", "glm53_a0_groundup_1h", "final_kept"),
    "a1_glm53_1h": ("A1", "glm53_a1_1h", "final_kept"),
    "a2_muse13_1h": ("A2", "muse13_a2_1h", "final_kept"),
    "a3_deepseek41_1h": ("A3", "deepseek41_a3_1h", "final_kept"),
    "a1_fisher_lr": ("A1", "e1_v1", "final_kept"),
    "a1_glm53": ("A1", "glm53_flash_a1", "final_kept"),
    "a1_diag": ("A1", "e1_v1", "baseline"),
}

SCORE_CSV_PATHS = (
    Path("scores/dual_5lane_L1/per_model.csv"),
    Path("scores/multilane_1h_all/per_model.csv"),
    Path("scores/a1_real_with_glm53/per_model.csv"),
)


def infer_agent_model(run_id: str) -> str:
    lower = run_id.lower()
    if "glm53" in lower:
        return "GLM-5.3-Flash"
    if "muse13" in lower:
        return "Muse-1.3"
    if "deepseek41" in lower:
        return "DeepSeek-4.1"
    if lower.startswith("e1"):
        return "E1-reference"
    return run_id


def _to_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _make_title(trial_id: str | None, hypothesis: str | None, run_id: str) -> str:
    prefix = f"{trial_id}: " if trial_id else ""
    if not hypothesis:
        return f"{prefix}{run_id}"
    # Strip trailing pipes/clauses if overly verbose, keep first ~72 chars cleanly.
    cleaned = re.sub(r"\s+", " ", hypothesis).strip()
    if len(cleaned) > 76:
        cleaned = cleaned[:73].rstrip(" ,;:-") + "..."
    return f"{prefix}{cleaned}"


def load_real_stan_scores(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Load and aggregate real-Stan per-model CSV scores keyed by sampler label."""
    aggregated: dict[str, dict[str, Any]] = {}

    for rel_path in SCORE_CSV_PATHS:
        csv_path = repo_root / rel_path
        if not csv_path.exists():
            continue

        grouped: dict[str, list[dict[str, str]]] = {}
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                label = (row.get("label") or "").strip()
                if not label:
                    continue
                grouped.setdefault(label, []).append(row)

        for label, rows in grouped.items():
            # Earlier CSVs in SCORE_CSV_PATHS take precedence if duplicate label exists
            if label in aggregated:
                continue
            passed = 0
            total = len(rows)
            ess_values: list[float] = []
            overhead_values: list[float] = []
            per_model: dict[str, dict[str, Any]] = {}

            for r in rows:
                model = (r.get("model") or "").strip()
                status = (r.get("status") or "").strip()
                ess = _to_float(r.get("ess_per_cost"))
                oh = _to_float(r.get("overhead_ratio"))
                if status == "right":
                    passed += 1
                    if ess is not None:
                        ess_values.append(ess)
                if oh is not None:
                    overhead_values.append(oh)
                per_model[model] = {
                    "status": status,
                    "reason": (r.get("reason") or "").strip() or None,
                    "ess_per_cost": ess,
                    "overhead_ratio": oh,
                }

            mean_ess = sum(ess_values) / len(ess_values) if ess_values else None
            mean_oh = sum(overhead_values) / len(overhead_values) if overhead_values else None
            aggregated[label] = {
                "label": label,
                "source_csv": str(rel_path),
                "real_stan_passed": passed,
                "real_stan_total": total,
                "real_ess_per_cost": round(mean_ess, 8) if mean_ess is not None else None,
                "real_overhead_ratio": round(mean_oh, 4) if mean_oh is not None else None,
                "models": per_model,
            }

    return aggregated


def collect_nodes(repo_root: Path) -> list[dict[str, Any]]:
    """Parse all local autoresearch ledgers into normalized Flywheel node dicts
    and merge real-Stan scores where available.
    """
    lanes_dir = repo_root / "autoresearch" / "lanes"
    ledger_files = sorted(lanes_dir.glob("*/runs/*/ledger.jsonl")) if lanes_dir.exists() else []

    # Load all ledgers first, then sort runs chronologically by first timestamp so cross-run
    # SHA lineage links point strictly backwards in time.
    parsed_runs: list[tuple[str, str, str, list[dict[str, Any]]]] = []
    for ledger_path in ledger_files:
        run_id = ledger_path.parent.name
        lane_dir = ledger_path.parent.parent.parent.name
        rows: list[dict[str, Any]] = []
        with ledger_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            continue
        first_ts = str(rows[0].get("timestamp") or "")
        lane = str(rows[0].get("lane") or lane_dir)
        parsed_runs.append((first_ts, lane, run_id, rows))

    parsed_runs.sort(key=lambda item: (item[0], item[1], item[2]))

    nodes: list[dict[str, Any]] = []
    # Track last kept node id by (lane, code_sha) for cross-run lineage continuity
    kept_by_lane_sha: dict[tuple[str, str], str] = {}
    # Track nodes per (lane, run_id) for real-Stan score attachment
    run_nodes_map: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for _first_ts, lane, run_id, rows in parsed_runs:
        agent_model = infer_agent_model(run_id)
        current_incumbent_id: str | None = None
        run_nodes: list[dict[str, Any]] = []

        for idx, row in enumerate(rows, start=1):
            trial_id = str(row.get("trial_id") or f"T{idx:04d}")
            node_lane = str(row.get("lane") or lane)
            node_id = f"{node_lane}:{run_id}:{trial_id}"
            decision = str(row.get("decision") or "unknown")
            code_sha = row.get("candidate_sha") or row.get("code_sha")
            code_sha_str = str(code_sha) if code_sha else None

            if current_incumbent_id is not None:
                parent_id = current_incumbent_id
            elif code_sha_str and (node_lane, code_sha_str) in kept_by_lane_sha:
                parent_id = kept_by_lane_sha[(node_lane, code_sha_str)]
            else:
                parent_id = None

            dims_val = row.get("dims")
            if isinstance(dims_val, list):
                dims = [int(x) for x in dims_val if isinstance(x, (int, float))]
            elif row.get("dim") is not None:
                dims = [int(row["dim"])]
            else:
                dims = None

            codex_raw = row.get("codex_refs")
            codex_refs = (
                [str(x) for x in codex_raw if str(x).strip()]
                if isinstance(codex_raw, list)
                else []
            )

            extra_metrics = {
                k: row[k]
                for k in (
                    "loc",
                    "magic_constants",
                    "wall_s",
                    "queue_wait_s",
                    "archive_cells",
                    "tier0_ok",
                    "tier1_ok",
                    "tier0_info",
                    "incumbent_soft_min",
                    "median_adapt_cost",
                    "judge_digest",
                    "seed",
                    "repeats",
                    "k_steps",
                    "adapt_draws",
                    "horizon_steps",
                    "ast_sha",
                )
                if k in row
            }

            hypothesis = str(row.get("hypothesis") or "").strip() or None
            created_at = str(row.get("timestamp") or "1970-01-01T00:00:00+00:00")

            node: dict[str, Any] = {
                "id": node_id,
                "parent_id": parent_id,
                "lane": node_lane,
                "run_id": run_id,
                "trial_id": trial_id,
                "agent_model": str(row.get("agent_model") or agent_model),
                "author": str(row.get("author") or agent_model),
                "title": _make_title(trial_id, hypothesis, run_id),
                "hypothesis": hypothesis,
                "decision": decision,
                "soft_min_eff": _to_float(row.get("soft_min_eff")),
                "ratio_vs_incumbent": _to_float(row.get("ratio_vs_incumbent")),
                "short_horizon_eff": _to_float(row.get("short_horizon_soft_min")),
                "long_horizon_eff": _to_float(row.get("long_horizon_soft_min")),
                "real_stan_passed": None,
                "real_stan_total": None,
                "real_ess_per_cost": None,
                "overhead_ratio": _to_float(row.get("overhead_ratio")),
                "b2_max": _to_float(row.get("b2_max")),
                "dims": dims,
                "codex_refs": codex_refs,
                "code_sha": code_sha_str,
                "metrics": extra_metrics,
                "metrics_json": json.dumps(extra_metrics, sort_keys=True),
                "verified": 0,
                "created_at": created_at,
            }

            nodes.append(node)
            run_nodes.append(node)

            if decision in KEPT_DECISIONS:
                current_incumbent_id = node_id
                if code_sha_str:
                    kept_by_lane_sha[(node_lane, code_sha_str)] = node_id

        run_nodes_map[(lane, run_id)] = run_nodes

    # Merge real-Stan scores
    real_scores = load_real_stan_scores(repo_root)
    for label, score in real_scores.items():
        target = SCORE_LABEL_TARGETS.get(label)
        target_node: dict[str, Any] | None = None

        if target:
            t_lane, t_run, selector = target
            candidates = run_nodes_map.get((t_lane, t_run), [])
            if selector == "baseline":
                baselines = [n for n in candidates if n["decision"] == "keep_baseline"]
                target_node = baselines[0] if baselines else (candidates[0] if candidates else None)
            else:
                kept = [n for n in candidates if n["decision"] in KEPT_DECISIONS]
                target_node = kept[-1] if kept else (candidates[-1] if candidates else None)
        else:
            # Generic fallback: match label against (lane, run_id)
            for (r_lane, r_run), candidates in run_nodes_map.items():
                if label in (r_run, f"{r_lane.lower()}_{r_run}"):
                    kept = [n for n in candidates if n["decision"] in KEPT_DECISIONS]
                    target_node = kept[-1] if kept else (candidates[-1] if candidates else None)
                    break

        if target_node is not None:
            target_node["real_stan_passed"] = score["real_stan_passed"]
            target_node["real_stan_total"] = score["real_stan_total"]
            target_node["real_ess_per_cost"] = score["real_ess_per_cost"]
            target_node["verified"] = 1
            metrics = dict(target_node["metrics"])
            metrics["real_stan"] = {
                "label": score["label"],
                "source_csv": score["source_csv"],
                "real_overhead_ratio": score["real_overhead_ratio"],
                "models": score["models"],
            }
            target_node["metrics"] = metrics
            target_node["metrics_json"] = json.dumps(metrics, sort_keys=True)

    nodes.sort(key=lambda n: (n["created_at"], n["id"]))
    return nodes


def compute_frontier(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_lane: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        by_lane.setdefault(node["lane"], []).append(node)

    frontier: dict[str, dict[str, Any]] = {}
    for lane in sorted(by_lane):
        lane_nodes = by_lane[lane]
        eligible = [n for n in lane_nodes if n["verified"] == 1 or n["decision"] in KEPT_DECISIONS]
        pool = eligible if eligible else lane_nodes
        if pool:
            pool.sort(
                key=lambda n: (
                    n["soft_min_eff"] if n["soft_min_eff"] is not None else float("-inf"),
                    n["verified"],
                    n["real_stan_passed"] if n["real_stan_passed"] is not None else -1,
                    n["created_at"],
                    n["id"],
                ),
                reverse=True,
            )
            frontier[lane] = pool[0]
    return frontier


def compute_alignment(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alignment: list[dict[str, Any]] = []
    for n in nodes:
        if (
            n["soft_min_eff"] is not None
            and n["real_stan_passed"] is not None
            and n["real_stan_total"] is not None
            and n["real_stan_total"] > 0
        ):
            alignment.append(
                {
                    "id": n["id"],
                    "lane": n["lane"],
                    "run_id": n["run_id"],
                    "trial_id": n["trial_id"],
                    "agent_model": n["agent_model"],
                    "title": n["title"],
                    "decision": n["decision"],
                    "soft_min_eff": n["soft_min_eff"],
                    "ratio_vs_incumbent": n["ratio_vs_incumbent"],
                    "real_stan_passed": n["real_stan_passed"],
                    "real_stan_total": n["real_stan_total"],
                    "real_stan_pass_rate": n["real_stan_passed"] / n["real_stan_total"],
                    "real_ess_per_cost": n["real_ess_per_cost"],
                    "overhead_ratio": n["overhead_ratio"],
                    "verified": n["verified"],
                }
            )
    return alignment


def build_snapshot(
    nodes: list[dict[str, Any]], claims: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    node_ids = {n["id"] for n in nodes}
    edges = [
        {
            "source": n["parent_id"],
            "target": n["id"],
            "lane": n["lane"],
            "decision": n["decision"],
        }
        for n in nodes
        if n["parent_id"] is not None and n["parent_id"] in node_ids
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "frontier": compute_frontier(nodes),
        "claims": list(claims) if claims else [],
        "alignment": compute_alignment(nodes),
    }


def _sql_literal(val: Any) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, (int, float)):
        if isinstance(val, float) and not math.isfinite(val):
            return "NULL"
        return repr(val)
    text = str(val).replace("'", "''")
    return f"'{text}'"


def render_seed_sql(nodes: list[dict[str, Any]]) -> str:
    columns = (
        "id",
        "parent_id",
        "lane",
        "run_id",
        "trial_id",
        "agent_model",
        "author",
        "title",
        "hypothesis",
        "decision",
        "soft_min_eff",
        "ratio_vs_incumbent",
        "short_horizon_eff",
        "long_horizon_eff",
        "real_stan_passed",
        "real_stan_total",
        "real_ess_per_cost",
        "overhead_ratio",
        "b2_max",
        "dims",
        "codex_refs",
        "code_sha",
        "metrics_json",
        "verified",
        "created_at",
    )
    lines = [
        "-- Auto-generated by flywheel/seed_from_apin.py (idempotent D1 seed)",
        "BEGIN TRANSACTION;",
    ]
    col_sql = ", ".join(columns)
    for n in nodes:
        values = [
            _sql_literal(n["id"]),
            _sql_literal(n["parent_id"]),
            _sql_literal(n["lane"]),
            _sql_literal(n["run_id"]),
            _sql_literal(n["trial_id"]),
            _sql_literal(n["agent_model"]),
            _sql_literal(n["author"]),
            _sql_literal(n["title"]),
            _sql_literal(n["hypothesis"]),
            _sql_literal(n["decision"]),
            _sql_literal(n["soft_min_eff"]),
            _sql_literal(n["ratio_vs_incumbent"]),
            _sql_literal(n["short_horizon_eff"]),
            _sql_literal(n["long_horizon_eff"]),
            _sql_literal(n["real_stan_passed"]),
            _sql_literal(n["real_stan_total"]),
            _sql_literal(n["real_ess_per_cost"]),
            _sql_literal(n["overhead_ratio"]),
            _sql_literal(n["b2_max"]),
            _sql_literal(json.dumps(n["dims"]) if n["dims"] is not None else None),
            _sql_literal(json.dumps(n["codex_refs"])),
            _sql_literal(n["code_sha"]),
            _sql_literal(n["metrics_json"]),
            _sql_literal(int(n["verified"])),
            _sql_literal(n["created_at"]),
        ]
        lines.append(f"INSERT OR REPLACE INTO nodes ({col_sql}) VALUES ({', '.join(values)});")
    lines.append("COMMIT;")
    lines.append("")
    return "\n".join(lines)


def generate_seed_artifacts(repo_root: Path, flywheel_dir: Path | None = None) -> dict[str, Any]:
    target_dir = flywheel_dir if flywheel_dir is not None else repo_root / "flywheel"
    public_dir = target_dir / "public"
    public_dir.mkdir(parents=True, exist_ok=True)

    nodes = collect_nodes(repo_root)
    snapshot = build_snapshot(nodes)
    seed_sql = render_seed_sql(nodes)

    sql_path = target_dir / "seed.sql"
    snapshot_path = public_dir / "snapshot.json"
    sql_path.write_text(seed_sql, encoding="utf-8")
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    return {
        "nodes": len(nodes),
        "edges": len(snapshot["edges"]),
        "lanes": sorted(snapshot["frontier"].keys()),
        "alignment": len(snapshot["alignment"]),
        "sql_path": str(sql_path),
        "snapshot_path": str(snapshot_path),
    }


def main() -> None:
    flywheel_dir = Path(__file__).resolve().parent
    repo_root = flywheel_dir.parent
    summary = generate_seed_artifacts(repo_root, flywheel_dir)
    print(
        f"Seeded apin Flywheel: {summary['nodes']} nodes, {summary['edges']} edges, "
        f"lanes={summary['lanes']}, alignment={summary['alignment']}\n"
        f"  SQL:      {summary['sql_path']}\n"
        f"  Snapshot: {summary['snapshot_path']}"
    )


if __name__ == "__main__":
    main()
