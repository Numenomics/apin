"""Command-line entry point: `apin list`, `apin fetch-data`, `apin run`, `apin score`,
`apin reference build`, `apin costs build`, `apin codex validate`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import (
    autoresearch, cost, grid, invariance, models, queue, records, reference,
    rewind, sampler, score, shapes, systems_lanes, tracks,
)
from .models import ROOT


def _opts(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--opt needs KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def _validate_codex(index_path: Path) -> int:
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML is required for `apin codex validate`")
    data = yaml.safe_load(index_path.read_text())
    assert data.get("schema_version") == 1, "expected schema_version: 1"
    repos = data.get("repositories") or []
    repo_ids = {r["id"] for r in repos}
    assert len(repo_ids) == len(repos), "duplicate repo id in INDEX.yaml"
    tier3 = {r["id"] for r in repos if r.get("tier") == 3}
    req_fields = {
        "id", "card", "axis", "lanes", "phase", "requires", "state_schema",
        "invariance_obligations", "cost", "incompatible_with", "source",
        "reuse_policy", "evidence", "test_ids",
    }
    cards = data.get("cards") or []
    for c in cards:
        missing = req_fields - set(c.keys())
        if missing:
            raise SystemExit(f"card {c.get('id')!r} missing fields: {sorted(missing)}")
        cpath = index_path.parent / c["card"]
        if not cpath.is_file():
            raise SystemExit(f"missing card file: {cpath}")
        srepo = c["source"]["repo"]
        if srepo not in repo_ids:
            raise SystemExit(f"card {c['id']!r} references unknown repo {srepo!r}")
        if srepo in tier3:
            raise SystemExit(f"card {c['id']!r} references license-quarantined Tier 3 repo {srepo!r}")
    print(f"codex valid: {len(repos)} repositories, {len(cards)} cards ({index_path})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="apin")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list samplers and models")
    sub.add_parser("fetch-data", help="download and verify any model data not shipped in data/")

    r = sub.add_parser("run", help="run one sampler over models x reps")
    r.add_argument("sampler", help="adapter name (`apin list` shows them)")
    r.add_argument("--models", default="all", help="'all', a family, 'public', 'held-out', or a comma list")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--label", help="name of this arm in runs/ (default: the sampler name)")
    r.add_argument("--cmdstan", help="CmdStan tree (default: $CMDSTAN)")
    r.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE", help="adapter option")
    r.add_argument("--runs", type=Path, default=ROOT / "runs")
    r.add_argument("--build", type=Path, default=ROOT / "build")
    r.add_argument("--chains", type=int, help="default: the model's budget in models/manifest.json")
    r.add_argument("--warmup", type=int, help="default: the model's budget")
    r.add_argument("--draws", type=int, help="draws per chain; default: the model's budget")
    r.add_argument("--base-seed", type=int, default=grid.BASE_SEED,
                   help="rep r uses BASE_SEED + 1000 r; the benchmark's scorer keeps its own secret (BENCH.md)")
    r.add_argument("--no-queue", action="store_true", help="bypass the host FIFO evaluation lock")

    s = sub.add_parser("score", help="score finished runs")
    s.add_argument("--runs", type=Path, default=ROOT / "runs")
    s.add_argument("--out", type=Path, default=ROOT / "scores")
    s.add_argument("--labels", help="comma list of labels to include")
    s.add_argument("--baseline", help="label to compute speed against")
    s.add_argument("--models", default="all", help="the grid: models with a reference among these")
    s.add_argument("--reps", type=int, default=3, help="reps in the grid; a rep that is not there is wrong")
    s.add_argument("--warmup", type=int, help="override manifest warmup budget when scoring custom runs")
    s.add_argument("--draws", type=int, help="override manifest draws budget when scoring custom runs")
    s.add_argument("--base-seed", type=int, default=grid.BASE_SEED)
    s.add_argument("--build", type=Path, default=ROOT / "build")
    s.add_argument("--costs", type=Path, default=cost.COSTS_PATH, help="path to frozen models/costs.json")
    s.add_argument("--track", choices=("L1", "L2", "L5"), default="L1",
                   help="leaderboard track: L1 (cost units), L2 (wall-clock ESS/s + host fingerprint), L5 (density-only)")
    s.add_argument("--submission", action="append", default=[], metavar="LABEL=PATH",
                   help="the submission file of a label; every rep of it must record its sha256 (repeatable)")

    ref = sub.add_parser("reference", help="reference posteriors for the correctness gate")
    ref_sub = ref.add_subparsers(dest="ref_cmd", required=True)
    rb = ref_sub.add_parser("build", help="summarise posteriordb's reference draws, or build one with CmdStan")
    rb.add_argument("--models", default="all", help="'all', a family, or a comma list")
    rb.add_argument("--cmdstan", help="CmdStan tree for local builds (default: $CMDSTAN)")
    rb.add_argument("--build", type=Path, default=ROOT / "build")
    rb.add_argument("--adapt-delta", type=float, help=f"default {reference.LOCAL['adapt_delta']}")
    rb.add_argument("--keep-going", action="store_true", help="continue after a model fails")
    rb.add_argument("--rerun", action="store_true",
                    help="rerun CmdStan for local references even when complete chain CSVs are in build/reference")

    c = sub.add_parser("costs", help="target-calibrated evaluation cost weights (r_m)")
    c_sub = c.add_subparsers(dest="cost_cmd", required=True)
    cb = c_sub.add_parser("build", help="measure r_grad via BridgeStan at raw reference draws")
    cb.add_argument("--models", default="all", help="'all', a family, 'public', 'held-out', or a comma list")
    cb.add_argument("--build", type=Path, default=ROOT / "build")
    cb.add_argument("--bridgestan", help="BridgeStan source tree (default: $BRIDGESTAN or package tree)")
    cb.add_argument("--out", type=Path, default=cost.COSTS_PATH)
    cb.add_argument("--draws", type=int, default=cost.DEFAULT_DRAWS)
    cb.add_argument("--repeats", type=int, default=cost.DEFAULT_REPEATS)
    cb.add_argument("--include-hvp", action="store_true")
    cb.add_argument("--include-hess", action="store_true")

    cx = sub.add_parser("codex", help="sampler & optimization codex tools")
    cx_sub = cx.add_subparsers(dest="codex_cmd", required=True)
    cv = cx_sub.add_parser("validate", help="validate codex/INDEX.yaml schema and card links")
    cv.add_argument("--index", type=Path, default=ROOT / "codex" / "INDEX.yaml")

    inv = sub.add_parser("invariance", help="run Slice 4 analytic exact-draw invariance & empirical efficiency check")
    inv.add_argument("--sampler", type=Path, required=True)
    inv.add_argument("--dim", type=int, default=6)
    inv.add_argument("--chains", type=int, default=64)
    inv.add_argument("--steps", type=int, default=20)
    inv.add_argument("--seed", type=int, default=1)

    rec = sub.add_parser("records", help="validate PR submission records")
    rec_sub = rec.add_subparsers(dest="rec_cmd", required=True)
    rv = rec_sub.add_parser("validate", help="validate a records/<release>/<track>/<date>_<slug>/ directory")
    rv.add_argument("path", type=Path)

    ar = sub.add_parser("autoresearch", help="run Karpathy-ratchet + MAP-Elites autoresearch trials")
    ar_sub = ar.add_subparsers(dest="ar_cmd", required=True)
    ae = ar_sub.add_parser("eval", help="evaluate current lane submission.py through Tier 0 + Tier 1")
    ae.add_argument("--lane", default="A1")
    ae.add_argument("--hypothesis", required=True)
    ae.add_argument("--codex-ref", action="append", default=[])
    ae.add_argument("--run-id", default="default")
    ae.add_argument("--seed", type=int)
    ae.add_argument("--dim", type=int, help="single-dimension override (e.g. --dim 16)")
    ae.add_argument("--dims", default="6,20", help="comma-separated Tier 1 dimension ladder (default: 6,20)")
    ae.add_argument("--repeats", type=int, default=2, help="paired Tier 1 suites per trial (default: 2)")
    ae.add_argument("--horizon-steps", type=int, default=500,
                    help="sampling horizon for amortizing adapt_cost in Tier 1 (default: 500)")
    ae.add_argument("--submission", type=Path, help="override submission file path")
    ae.add_argument("--allow-duplicate", action="store_true")
    ae.add_argument("--dry-run", action="store_true", help="lint & check duplicates without writing a ledger row")
    ae.add_argument("--no-queue", action="store_true", help="bypass the host FIFO evaluation lock")

    as1 = ar_sub.add_parser("eval-s1", help="evaluate Lane S1 upstream sampler patch")
    as1.add_argument("--upstream", required=True, help="upstream submodule name (e.g. walnutpie, stan, nuts-rs)")
    as1.add_argument("--hypothesis", required=True)
    as1.add_argument("--allowed-path", action="append", default=[], dest="allowed_paths")
    as1.add_argument("--build-cmd")
    as1.add_argument("--test-cmd")
    as1.add_argument("--bench-cmd")
    as1.add_argument("--run-id", default="default")
    as1.add_argument("--dry-run", action="store_true")

    as2 = ar_sub.add_parser("eval-s2", help="evaluate Lane S2 gradient/compiler speed patch")
    as2.add_argument("--upstream", required=True, help="upstream submodule name (e.g. math, stan, stanc3, bridgestan)")
    as2.add_argument("--hypothesis", required=True)
    as2.add_argument("--allowed-path", action="append", default=[], dest="allowed_paths")
    as2.add_argument("--clean-build-cmd")
    as2.add_argument("--test-cmd")
    as2.add_argument("--run-id", default="default")
    as2.add_argument("--dry-run", action="store_true")

    as3 = ar_sub.add_parser("eval-s3", help="run Lane S3 rewind calibration & agent grading (AUARC)")
    as3.add_argument("--mode", choices=("base", "reference", "custom"), default="reference")
    as3.add_argument("--tasks", type=Path, default=rewind.DEFAULT_TASKS_PATH)
    as3.add_argument("--submission", action="append", default=[], metavar="TASK_ID=PATH",
                     help="custom submission for one rewind task (repeatable)")
    as3.add_argument("--run-id", default="calibration")
    as3.add_argument("--out", type=Path)

    tr = sub.add_parser("track", help="run specialized benchmark tracks (L3 stress, L4 unadjusted, L5 density-only)")
    tr_sub = tr.add_subparsers(dest="track_cmd", required=True)
    tr_run = tr_sub.add_parser("run", help="run a submission on Track L3, L4, or L5")
    tr_run.add_argument("--track", choices=("L3", "L4", "L5"), required=True)
    tr_run.add_argument("--sampler", type=Path, required=True)
    tr_run.add_argument("--lane", help="optional lane wrapper (e.g. A1, A4)")
    tr_run.add_argument("--dims", default="6,12", help="comma-separated dimensions (default: 6,12)")
    tr_run.add_argument("--seed", type=int, default=20261121)
    tr_run.add_argument("--chains", type=int, default=48)
    tr_run.add_argument("--out", type=Path)

    a = ap.parse_args(argv)
    if a.cmd == "list":
        print("samplers:", ", ".join(sorted(sampler.registry())))
        costs_tbl = cost.load()
        for m in models.load().values():
            rm = costs_tbl.get(m.name, {}).get("r_grad")
            rm_s = f"r_grad={rm:.2f}" if rm is not None else "r_grad=MISSING"
            print(f"  {m.name:45s} {m.family:14s} data={'shipped' if m.data_shipped else 'fetched':8s} "
                  f"reference={m.reference}{'' if reference.path(m.name).is_file() else ' (not built)'} {rm_s}")
        return 0
    if a.cmd == "fetch-data":
        for m in models.load().values():
            print(f"{m.name}: {m.data_file()}")
        return 0
    if a.cmd == "run":
        settings = sampler.Settings(cmdstan=a.cmdstan, build_dir=a.build, options=_opts(a.opt))
        s_obj = sampler.get(a.sampler, settings)
        with queue.eval_slot(f"run:{a.label or a.sampler}:{a.models}", enabled=not a.no_queue):
            done = grid.run_grid(s_obj, models.select(a.models), a.reps, a.runs, a.label, a.chains, a.warmup, a.draws,
                                 base_seed=a.base_seed)
        failed = list(a.runs.glob(f"{a.label or a.sampler}/*/rep*/FAILED"))
        print(f"{len(done)} reps done, {len(failed)} failed")
        return 1 if failed else 0
    if a.cmd == "score":
        g = score.declared(a.models, a.reps, a.base_seed, a.build, costs_path=a.costs,
                           warmup=a.warmup, draws=a.draws)
        subs = dict(x.split("=", 1) for x in a.submission)
        try:
            score.score(a.runs, a.out, g, a.baseline, a.labels.split(",") if a.labels else None,
                        submissions={k: Path(v) for k, v in subs.items()}, track=a.track)
        except ValueError as e:
            raise SystemExit(f"apin score: {e}") from None
        return 0
    if a.cmd == "reference":
        settings = {"adapt_delta": a.adapt_delta} if a.adapt_delta else None
        failed = []
        for m in models.select(a.models):
            try:
                reference.build(m, m.reference, a.cmdstan, a.build, settings=settings, rerun=a.rerun)
            except Exception as e:
                if not a.keep_going:
                    raise
                failed.append(m.name)
                print(f"{m.name}: FAILED: {e}")
        return 1 if failed else 0
    if a.cmd == "costs":
        cost.build(
            models.select(a.models), build_dir=a.build, bridgestan_root=a.bridgestan,
            out_path=a.out, n_draws=a.draws, repeats=a.repeats,
            include_hvp=a.include_hvp, include_hess=a.include_hess,
        )
        return 0
    if a.cmd == "codex":
        return _validate_codex(a.index)
    if a.cmd == "invariance":
        suite = shapes.make_suite(a.seed, dim=a.dim)
        failed = 0
        for i, tgt in enumerate(suite):
            r = invariance.check_target_invariance(tgt, a.sampler, seed=a.seed + 100 * i,
                                                   n_chains=a.chains, k_steps=a.steps)
            status = "PASS" if r.passed else "FAIL"
            print(f"{status:4s} {tgt.family:20s} rot={str(tgt.rotated):5s} max_z={r.max_z:5.2f}/{r.z_threshold:4.2f} "
                  f"b2={r.b2_max:.4g} eff={r.empirical_efficiency:.3g} {r.reason}")
            failed += 0 if r.passed else 1
        return 1 if failed else 0
    if a.cmd == "records":
        meta = records.validate_record_dir(a.path)
        print(f"record valid: {meta['entry_id']} ({meta['track']}, {meta['license']})")
        return 0
    if a.cmd == "track":
        dims_tuple = tuple(int(x.strip()) for x in a.dims.split(",") if x.strip())
        if a.track == "L3":
            rep = tracks.run_track_l3(
                a.sampler, dims=dims_tuple, seed=a.seed, n_chains=a.chains, lane=a.lane, out_dir=a.out,
            )
            print(tracks.format_track_markdown(rep))
            return 0 if rep["passed_all_invariance"] else 1
        if a.track == "L4":
            rep = tracks.run_track_l4(
                a.sampler, dims=dims_tuple, seed=a.seed, n_chains=a.chains, lane=a.lane, out_dir=a.out,
            )
            print(tracks.format_track_markdown(rep))
            return 0
        if a.track == "L5":
            rep = tracks.run_track_l5(
                a.sampler, dims=dims_tuple, seed=a.seed, n_chains=a.chains, out_dir=a.out,
            )
            print(tracks.format_track_markdown(rep))
            return 0 if (rep.get("passed_density_only") and rep.get("passed_all_invariance")) else 1
        return 2
    if a.cmd == "autoresearch":
        if a.ar_cmd == "eval-s1":
            row = systems_lanes.evaluate_s1_trial(
                a.upstream, a.hypothesis, allowed_paths=a.allowed_paths,
                build_cmd=a.build_cmd, test_cmd=a.test_cmd, bench_cmd=a.bench_cmd,
                run_id=a.run_id, dry_run=a.dry_run,
            )
            print(f"{row['trial_id']} [S1:{a.upstream}] decision={row['decision']} paths_ok={row.get('paths_ok')} wall={row['wall_s']}s")
            return 0 if row["decision"] in ("dry_run_ok", "keep", "keep_tests_only") else 1
        if a.ar_cmd == "eval-s2":
            row = systems_lanes.evaluate_s2_trial(
                a.upstream, a.hypothesis, allowed_paths=a.allowed_paths,
                clean_build_cmd=a.clean_build_cmd, test_cmd=a.test_cmd,
                run_id=a.run_id, dry_run=a.dry_run,
            )
            print(f"{row['trial_id']} [S2:{a.upstream}] decision={row['decision']} paths_ok={row.get('paths_ok')} wall={row['wall_s']}s")
            return 0 if row["decision"] in ("dry_run_ok", "keep", "keep_verified") else 1
        if a.ar_cmd == "eval-s3":
            custom = dict(x.split("=", 1) for x in a.submission)
            rep = rewind.evaluate_s3_suite(
                {k: Path(v) for k, v in custom.items()} if custom else None,
                mode=a.mode, tasks_path=a.tasks, run_id=a.run_id, out_dir=a.out,
            )
            print(f"[S3:{a.mode}] passed={rep['passed']}/{rep['total']} pass_rate={rep['pass_rate']} auarc={rep['auarc']}")
            for r in rep["results"]:
                print(f"  {'PASS' if r['passed'] else 'FAIL'} {r['task_id']:32s} eff={r['emp_eff']:.3g} ({r['reason']})")
            return 0 if (rep["passed"] == rep["total"] if a.mode == "reference" else True) else 1
        dims_tuple = (a.dim,) if a.dim is not None else tuple(int(x.strip()) for x in a.dims.split(",") if x.strip())
        row = autoresearch.evaluate_lane_trial(
            a.lane, a.hypothesis, codex_refs=a.codex_ref, seed=a.seed,
            dim=dims_tuple[0], dims=dims_tuple, repeats=a.repeats, horizon_steps=a.horizon_steps,
            run_id=a.run_id, allow_duplicate=a.allow_duplicate,
            submission_path=a.submission, dry_run=a.dry_run, use_queue=not a.no_queue,
        )
        if a.dry_run:
            print(f"{row['trial_id']} [{row['lane']}] dry_run={row['decision']} "
                  f"dup={row['is_duplicate']} lint_ok={row['lint_ok']} loc={row['loc']} "
                  f"consts={row['magic_constants']}")
            return 0 if row["decision"] == "dry_run_ok" else 2
        print(f"{row['trial_id']} [{row['lane']}] decision={row['decision']} "
              f"tier0={row['tier0_ok']} tier1={row['tier1_ok']} "
              f"soft_min={row.get('soft_min_eff')} ratio={row.get('ratio_vs_incumbent')} "
              f"dims={','.join(str(x) for x in (row.get('dims') or []))} repeats={row.get('repeats')} "
              f"qwait={row.get('queue_wait_s')}s wall={row['wall_s']}s")
        # pi-autoresearch / external-harness compatible METRIC lines
        for mkey, rkey in (
            ("soft_min_eff", "soft_min_eff"),
            ("ratio_vs_incumbent", "ratio_vs_incumbent"),
            ("short_horizon_soft_min", "short_horizon_soft_min"),
            ("long_horizon_soft_min", "long_horizon_soft_min"),
            ("median_adapt_cost", "median_adapt_cost"),
            ("overhead_ratio", "overhead_ratio"),
            ("b2_max", "b2_max"),
        ):
            val = row.get(rkey)
            if val is not None:
                print(f"METRIC {mkey}={val}")
        for cls_name, cls_val in sorted((row.get("by_class") or {}).items()):
            print(f"CLASS_METRIC {cls_name}={cls_val}")
        return 0 if row["tier0_ok"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
