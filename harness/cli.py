"""apin command line.

  apin list                         samplers and models
  apin fetch-data                   download the data files that are not shipped
  apin run --sampler S [--models M] [--reps N] [--label L] [--cmdstan DIR] [--opt k=v ...]
  apin score [--runs DIR] [--out DIR] [--baseline LABEL]
  apin reference build [--models M] [--cmdstan DIR] [--adapt-delta X]
                                    write models/reference/<model>.json (summaries, no draws)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import grid, models, reference, sampler, score
from .models import ROOT


def _opts(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--opt expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="apin")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list samplers and models")
    sub.add_parser("fetch-data", help="download the data files that are not shipped")

    r = sub.add_parser("run", help="run a sampler over models x reps")
    r.add_argument("--sampler", required=True)
    r.add_argument("--models", default="all", help="'all', a family, 'public', 'held-out', or a comma list")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--label", help="name of this arm in runs/ (default: the sampler name)")
    r.add_argument("--cmdstan", help="CmdStan tree (default: $CMDSTAN)")
    r.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE", help="adapter option")
    r.add_argument("--runs", type=Path, default=ROOT / "runs")
    r.add_argument("--build", type=Path, default=ROOT / "build")
    r.add_argument("--chains", type=int, default=4)
    r.add_argument("--warmup", type=int, default=1000)
    r.add_argument("--draws", type=int, default=1000)
    r.add_argument("--base-seed", type=int, default=grid.BASE_SEED,
                   help="rep r uses BASE_SEED + 1000 r; the benchmark's scorer keeps its own secret (BENCH.md)")

    s = sub.add_parser("score", help="score finished runs")
    s.add_argument("--runs", type=Path, default=ROOT / "runs")
    s.add_argument("--out", type=Path, default=ROOT / "scores")
    s.add_argument("--labels", help="comma list of labels to include")
    s.add_argument("--baseline", help="label to compute ratios against")

    ref = sub.add_parser("reference", help="reference posteriors for the correctness gate")
    ref_sub = ref.add_subparsers(dest="ref_cmd", required=True)
    rb = ref_sub.add_parser("build", help="summarise posteriordb's reference draws, or build one with CmdStan")
    rb.add_argument("--models", default="all", help="'all', a family, or a comma list")
    rb.add_argument("--cmdstan", help="CmdStan tree for local builds (default: $CMDSTAN)")
    rb.add_argument("--build", type=Path, default=ROOT / "build")
    rb.add_argument("--adapt-delta", type=float, help=f"default {reference.LOCAL['adapt_delta']}")
    rb.add_argument("--keep-going", action="store_true", help="continue after a model fails")

    a = ap.parse_args(argv)
    if a.cmd == "list":
        print("samplers:", ", ".join(sorted(sampler.registry())))
        for m in models.load().values():
            print(f"  {m.name:45s} {m.family:14s} data={'shipped' if m.data_shipped else 'fetched':8s} "
                  f"reference={m.reference}{'' if reference.path(m.name).is_file() else ' (not built)'}")
        return 0
    if a.cmd == "fetch-data":
        for m in models.load().values():
            print(f"{m.name}: {m.data_file()}")
        return 0
    if a.cmd == "run":
        settings = sampler.Settings(cmdstan=a.cmdstan, build_dir=a.build, options=_opts(a.opt))
        s_obj = sampler.get(a.sampler, settings)
        done = grid.run_grid(s_obj, models.select(a.models), a.reps, a.runs, a.label, a.chains, a.warmup, a.draws,
                             base_seed=a.base_seed)
        failed = list(a.runs.glob(f"{a.label or a.sampler}/*/rep*/FAILED"))
        print(f"{len(done)} reps done, {len(failed)} failed")
        return 1 if failed else 0
    if a.cmd == "score":
        score.score(a.runs, a.out, a.labels.split(",") if a.labels else None, a.baseline)
        return 0
    if a.cmd == "reference":
        settings = {"adapt_delta": a.adapt_delta} if a.adapt_delta else None
        failed = []
        for m in models.select(a.models):
            try:
                reference.build(m, m.reference, a.cmdstan, a.build, settings=settings)
            except Exception as e:
                if not a.keep_going:
                    raise
                failed.append(m.name)
                print(f"{m.name}: FAILED: {e}")
        return 1 if failed else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
