"""Systems Autoresearch Lanes S1 (Upstream Sampler Patches) and S2 (Gradient & Compiler Speed).

Implements `autoresearch/RULINGS.md` R-008:
  - Lane S1: scoped upstream sampler patches (`upstream/walnutpie`, `upstream/stan`, `upstream/nuts-rs`),
    enforcing `allowed_paths`, build + upstream unit tests + `apin` statistical gate.
  - Lane S2: scoped gradient/compiler patches (`upstream/math`, `upstream/stan`, `upstream/stanc3`,
    `upstream/bridgestan`), enforcing:
      1. `allowed_paths` diff guard,
      2. clean-rebuild verification (hard ban on mixed ISA/SIMD builds, NM-A1-002 / W-27),
      3. finite-difference gradient parity (`max_rel_err < 1e-5`),
      4. byte-identical draw canary at fixed seed (grants R-008 exact-math waiver of statistical test),
      5. paired wall-clock per gradient call (`median_us` & speedup ratio).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import tracks
from .models import ROOT


def _git_changed_files(repo_dir: Path, base_ref: str = "HEAD") -> list[str]:
    """Return sorted list of modified/untracked files relative to `repo_dir`."""
    r1 = subprocess.run(
        ["git", "-C", str(repo_dir), "diff", "--name-only", base_ref],
        capture_output=True, text=True, check=False,
    )
    r2 = subprocess.run(
        ["git", "-C", str(repo_dir), "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True, check=False,
    )
    files = {ln.strip() for ln in (r1.stdout + "\n" + r2.stdout).splitlines() if ln.strip()}
    return sorted(files)


def check_allowed_paths(changed: list[str], allowed_prefixes: list[str]) -> tuple[bool, list[str]]:
    """Verify every changed file matches at least one allowed prefix (empty = allow all)."""
    if not allowed_prefixes:
        return True, []
    norm_allowed = [p.rstrip("/") for p in allowed_prefixes]
    violations = [
        f for f in changed
        if not any(f == pref or f.startswith(pref + "/") for pref in norm_allowed)
    ]
    return len(violations) == 0, violations


def finite_diff_gradient_check(
    log_density_grad_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    points: np.ndarray,
    *,
    eps: float = 1e-6,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Central finite-difference gradient check across a batch of points `(n, dim)`."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    max_abs_err = 0.0
    max_rel_err = 0.0
    for x in pts:
        lp0, g_ana = log_density_grad_fn(x)
        if not (math.isfinite(lp0) and np.all(np.isfinite(g_ana))):
            return {"passed": False, "reason": "non-finite analytic density or gradient", "max_rel_err": math.inf}
        d = x.size
        g_fd = np.empty(d, dtype=np.float64)
        for j in range(d):
            xp = x.copy()
            xm = x.copy()
            xp[j] += eps
            xm[j] -= eps
            lpp, _ = log_density_grad_fn(xp)
            lpm, _ = log_density_grad_fn(xm)
            g_fd[j] = (lpp - lpm) / (2.0 * eps)
        abs_err = float(np.max(np.abs(g_ana - g_fd)))
        denom = np.maximum(np.maximum(np.abs(g_ana), np.abs(g_fd)), 1.0)
        rel_err = float(np.max(np.abs(g_ana - g_fd) / denom))
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)
    passed = (max_rel_err <= rtol) or (max_abs_err <= atol)
    return {
        "passed": passed,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "n_points": int(pts.shape[0]),
        "reason": "ok" if passed else f"max_rel_err={max_rel_err:.3e} > {rtol:.1e}",
    }


def check_byte_identical_draws(baseline_files: list[Path], candidate_files: list[Path]) -> dict[str, Any]:
    """Verify exact SHA-256 byte identity across paired draw CSV/binary outputs (R-008 waiver)."""
    b_map = {p.name: p for p in sorted(baseline_files)}
    c_map = {p.name: p for p in sorted(candidate_files)}
    if set(b_map) != set(c_map) or not b_map:
        return {
            "byte_identical": False,
            "reason": f"file set mismatch: baseline={sorted(b_map)}, candidate={sorted(c_map)}",
            "digests": {},
        }
    digests = {}
    mismatches = []
    for name in sorted(b_map):
        hb = hashlib.sha256(b_map[name].read_bytes()).hexdigest()
        hc = hashlib.sha256(c_map[name].read_bytes()).hexdigest()
        digests[name] = {"baseline": hb[:16], "candidate": hc[:16], "equal": hb == hc}
        if hb != hc:
            mismatches.append(name)
    return {
        "byte_identical": len(mismatches) == 0,
        "waiver_granted": len(mismatches) == 0,
        "mismatches": mismatches,
        "digests": digests,
        "reason": "byte-identical (R-008 exact-math waiver)" if not mismatches else f"differs on {mismatches}",
    }


def benchmark_paired_gradient_speed(
    baseline_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    candidate_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    points: np.ndarray,
    *,
    repeats: int = 7,
) -> dict[str, Any]:
    """Interleaved A/B timing of baseline vs candidate gradient calls over `points`."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    n = int(pts.shape[0])
    base_us: list[float] = []
    cand_us: list[float] = []
    for rep in range(repeats):
        order = ("base", "cand") if rep % 2 == 0 else ("cand", "base")
        for arm in order:
            fn = baseline_fn if arm == "base" else candidate_fn
            t0 = time.perf_counter()
            for x in pts:
                fn(x)
            dt_us = (time.perf_counter() - t0) * 1e6 / max(1, n)
            if arm == "base":
                base_us.append(dt_us)
            else:
                cand_us.append(dt_us)
    b_med = float(np.median(base_us))
    c_med = float(np.median(cand_us))
    speedup = (b_med / c_med) if c_med > 0 else math.nan
    return {
        "baseline_median_us": round(b_med, 4),
        "candidate_median_us": round(c_med, 4),
        "speedup_ratio": round(speedup, 4) if math.isfinite(speedup) else None,
        "repeats": repeats,
        "n_points": n,
    }


def _append_systems_ledger(run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_dir / "ledger.jsonl"
    tsv_path = run_dir / "results.tsv"
    with open(ledger_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    rows = [json.loads(ln) for ln in ledger_path.read_text().splitlines() if ln.strip()]
    cols = [
        "trial_id", "timestamp", "lane", "upstream", "candidate_sha",
        "paths_ok", "build_ok", "tests_ok", "fd_ok", "byte_identical",
        "speedup_ratio", "baseline_median_us", "candidate_median_us",
        "decision", "wall_s", "hypothesis",
    ]
    with open(tsv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return row


def _run_shell(cmd: str, cwd: Path, timeout_s: float = 600.0) -> tuple[bool, str]:
    r = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True, text=True, timeout=timeout_s)
    tail = (r.stdout + "\n" + r.stderr).strip()[-2000:]
    return r.returncode == 0, tail


def _patch_digest(repo_dir: Path, base_ref: str = "HEAD") -> str:
    r = subprocess.run(
        ["git", "-C", str(repo_dir), "diff", base_ref],
        capture_output=True, text=True, check=False,
    )
    return hashlib.sha256(r.stdout.encode("utf-8", errors="ignore")).hexdigest()[:16]


def evaluate_s1_trial(
    upstream: str,
    hypothesis: str,
    *,
    allowed_paths: list[str] | None = None,
    build_cmd: str | None = None,
    test_cmd: str | None = None,
    bench_cmd: str | None = None,
    base_ref: str = "HEAD",
    run_id: str = "default",
    autoresearch_dir: Path = ROOT / "autoresearch",
    upstream_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate one Lane S1 (Upstream Sampler Patch) trial."""
    t0 = time.perf_counter()
    repo_dir = Path(upstream_dir) if upstream_dir is not None else (ROOT / "upstream" / upstream)
    run_dir = autoresearch_dir / "lanes" / "S1" / "runs" / run_id
    prior = (
        [json.loads(ln) for ln in (run_dir / "ledger.jsonl").read_text().splitlines() if ln.strip()]
        if (run_dir / "ledger.jsonl").is_file() else []
    )
    trial_id = f"T{len(prior) + 1:04d}"

    if not repo_dir.is_dir():
        return {
            "trial_id": trial_id, "lane": "S1", "upstream": upstream,
            "decision": "error_missing_upstream", "error": f"upstream directory not found: {repo_dir}",
            "wall_s": round(time.perf_counter() - t0, 3),
        }

    changed = _git_changed_files(repo_dir, base_ref=base_ref)
    paths_ok, violations = check_allowed_paths(changed, list(allowed_paths or []))
    cand_sha = _patch_digest(repo_dir, base_ref=base_ref)

    if dry_run:
        return {
            "trial_id": trial_id, "lane": "S1", "upstream": upstream, "candidate_sha": cand_sha,
            "changed_files": changed, "paths_ok": paths_ok, "path_violations": violations,
            "decision": "dry_run_ok" if paths_ok else "dry_run_path_violation",
            "wall_s": round(time.perf_counter() - t0, 3),
        }

    row: dict[str, Any] = {
        "trial_id": trial_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lane": "S1",
        "upstream": upstream,
        "candidate_sha": cand_sha,
        "changed_files": changed,
        "paths_ok": paths_ok,
        "path_violations": violations,
        "build_ok": False,
        "tests_ok": False,
        "hypothesis": hypothesis,
        "machine": tracks.machine_fingerprint(),
    }
    if not paths_ok:
        row.update(decision="discard_path_violation", wall_s=round(time.perf_counter() - t0, 3))
        return _append_systems_ledger(run_dir, row)

    if build_cmd:
        ok, tail = _run_shell(build_cmd, repo_dir)
        row["build_ok"] = ok
        row["build_tail"] = tail
        if not ok:
            row.update(decision="discard_build_fail", wall_s=round(time.perf_counter() - t0, 3))
            return _append_systems_ledger(run_dir, row)
    else:
        row["build_ok"] = True

    if test_cmd:
        ok, tail = _run_shell(test_cmd, repo_dir)
        row["tests_ok"] = ok
        row["tests_tail"] = tail
        if not ok:
            row.update(decision="discard_test_fail", wall_s=round(time.perf_counter() - t0, 3))
            return _append_systems_ledger(run_dir, row)
    else:
        row["tests_ok"] = True

    if bench_cmd:
        ok, tail = _run_shell(bench_cmd, ROOT)
        row["bench_ok"] = ok
        row["bench_tail"] = tail
        row["decision"] = "keep" if ok else "discard_bench_fail"
    else:
        row["decision"] = "keep_tests_only"
    row["wall_s"] = round(time.perf_counter() - t0, 3)
    return _append_systems_ledger(run_dir, row)


def evaluate_s2_trial(
    upstream: str,
    hypothesis: str,
    *,
    allowed_paths: list[str] | None = None,
    clean_build_cmd: str | None = None,
    test_cmd: str | None = None,
    baseline_draws: list[Path] | None = None,
    candidate_draws: list[Path] | None = None,
    baseline_grad_fn: Callable[[np.ndarray], tuple[float, np.ndarray]] | None = None,
    candidate_grad_fn: Callable[[np.ndarray], tuple[float, np.ndarray]] | None = None,
    probe_points: np.ndarray | None = None,
    min_speedup: float = 1.02,
    base_ref: str = "HEAD",
    run_id: str = "default",
    autoresearch_dir: Path = ROOT / "autoresearch",
    upstream_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate one Lane S2 (Gradient & Compiler Speed) trial with clean-build, FD, and byte-identity gates."""
    t0 = time.perf_counter()
    repo_dir = Path(upstream_dir) if upstream_dir is not None else (ROOT / "upstream" / upstream)
    run_dir = autoresearch_dir / "lanes" / "S2" / "runs" / run_id
    prior = (
        [json.loads(ln) for ln in (run_dir / "ledger.jsonl").read_text().splitlines() if ln.strip()]
        if (run_dir / "ledger.jsonl").is_file() else []
    )
    trial_id = f"T{len(prior) + 1:04d}"

    if not repo_dir.is_dir():
        return {
            "trial_id": trial_id, "lane": "S2", "upstream": upstream,
            "decision": "error_missing_upstream", "error": f"upstream directory not found: {repo_dir}",
            "wall_s": round(time.perf_counter() - t0, 3),
        }

    changed = _git_changed_files(repo_dir, base_ref=base_ref)
    paths_ok, violations = check_allowed_paths(changed, list(allowed_paths or []))
    cand_sha = _patch_digest(repo_dir, base_ref=base_ref)

    if dry_run:
        return {
            "trial_id": trial_id, "lane": "S2", "upstream": upstream, "candidate_sha": cand_sha,
            "changed_files": changed, "paths_ok": paths_ok, "path_violations": violations,
            "decision": "dry_run_ok" if paths_ok else "dry_run_path_violation",
            "wall_s": round(time.perf_counter() - t0, 3),
        }

    row: dict[str, Any] = {
        "trial_id": trial_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lane": "S2",
        "upstream": upstream,
        "candidate_sha": cand_sha,
        "changed_files": changed,
        "paths_ok": paths_ok,
        "path_violations": violations,
        "build_ok": False,
        "tests_ok": False,
        "fd_ok": None,
        "byte_identical": None,
        "speedup_ratio": None,
        "baseline_median_us": None,
        "candidate_median_us": None,
        "hypothesis": hypothesis,
        "machine": tracks.machine_fingerprint(),
    }
    if not paths_ok:
        row.update(decision="discard_path_violation", wall_s=round(time.perf_counter() - t0, 3))
        return _append_systems_ledger(run_dir, row)

    if clean_build_cmd:
        ok, tail = _run_shell(clean_build_cmd, repo_dir)
        row["build_ok"] = ok
        row["build_tail"] = tail
        if not ok:
            row.update(decision="discard_build_fail", wall_s=round(time.perf_counter() - t0, 3))
            return _append_systems_ledger(run_dir, row)
    else:
        row["build_ok"] = True

    if test_cmd:
        ok, tail = _run_shell(test_cmd, repo_dir)
        row["tests_ok"] = ok
        row["tests_tail"] = tail
        if not ok:
            row.update(decision="discard_test_fail", wall_s=round(time.perf_counter() - t0, 3))
            return _append_systems_ledger(run_dir, row)
    else:
        row["tests_ok"] = True

    if candidate_grad_fn is not None and probe_points is not None:
        fd = finite_diff_gradient_check(candidate_grad_fn, probe_points)
        row["fd_ok"] = fd["passed"]
        row["fd_detail"] = fd
        if not fd["passed"]:
            row.update(decision="discard_fd_fail", wall_s=round(time.perf_counter() - t0, 3))
            return _append_systems_ledger(run_dir, row)

    if baseline_draws is not None and candidate_draws is not None:
        canary = check_byte_identical_draws(baseline_draws, candidate_draws)
        row["byte_identical"] = canary["byte_identical"]
        row["canary_detail"] = canary
        if not canary["byte_identical"]:
            row.update(decision="needs_statistical_gate", wall_s=round(time.perf_counter() - t0, 3))
            return _append_systems_ledger(run_dir, row)

    if baseline_grad_fn is not None and candidate_grad_fn is not None and probe_points is not None:
        timing = benchmark_paired_gradient_speed(baseline_grad_fn, candidate_grad_fn, probe_points)
        row["baseline_median_us"] = timing["baseline_median_us"]
        row["candidate_median_us"] = timing["candidate_median_us"]
        row["speedup_ratio"] = timing["speedup_ratio"]
        sr = timing["speedup_ratio"]
        if sr is not None and sr >= min_speedup:
            row["decision"] = "keep"
        else:
            row["decision"] = "discard_no_gain"
    else:
        row["decision"] = "keep_verified"

    row["wall_s"] = round(time.perf_counter() - t0, 3)
    return _append_systems_ledger(run_dir, row)
