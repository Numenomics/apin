"""Command-line entry point: `apin list`, `apin fetch-data`, `apin run`, `apin score`,
`apin reference build`, `apin costs build`, `apin codex validate`."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from . import (
    autoresearch, cost, flywheel, grid, invariance, models, queue, records, reference,
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



def _claim_or_release(a: argparse.Namespace) -> int:
    """Reserve or free a task on the Flywheel claim ledger; prints what to do next."""
    try:
        if a.ar_cmd == "claim":
            res = flywheel.claim_task(a.task, a.lane, a.agent, a.summary, a.ttl_minutes)
        else:
            res = flywheel.release_claim(a.claim, a.status)
    except flywheel.FlywheelError as exc:
        if exc.status == 409:
            print(f"{a.task} is already claimed. Pick another task: uv run python scripts/choose_research_task.py --list", file=sys.stderr)
        else:
            print(f"The Flywheel refused the request: {exc}", file=sys.stderr)
        return 1
    if res is None:
        print("Nothing was claimed. " + NO_TOKEN_HINT.format(path=flywheel.token_path()), file=sys.stderr)
        return 1
    claim = res.get("claim") or res
    if a.ar_cmd == "claim":
        print(f"Claimed {a.task} in lane {a.lane} until {claim.get('expires_at', 'the TTL runs out')}.")
        print(f"When you finish: uv run apin autoresearch release --claim {claim.get('id', '<claim id>')}")
    else:
        print(f"Claim {a.claim} marked {a.status}.")
    return 0

NO_TOKEN_HINT = (
    "No Flywheel token found (APIN_FLYWHEEL_TOKEN or {path}). "
    "Run `uv run apin autoresearch register --handle <you>` once, then retry."
)


def _no_token() -> int:
    print(NO_TOKEN_HINT.format(path=flywheel.token_path()), file=sys.stderr)
    return 1


def _refusal(exc: flywheel.FlywheelError) -> str:
    body = exc.body if isinstance(exc.body, dict) else {}
    if body.get("error"):
        extra = {k: v for k, v in body.items() if k != "error"}
        return f"{body['error']} (HTTP {exc.status}){' ' + json.dumps(extra) if extra else ''}"
    return str(exc)


def _safe_name(node_id: str) -> str:
    """A file or directory name for a trial id written by someone else: never `.`, `..` or empty."""
    import hashlib
    import re

    name = re.sub(r"[^A-Za-z0-9_.-]", "_", node_id).strip(".")
    return name or "trial_" + hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:12]


def _local_lane(lane: object) -> str:
    """A lane name from the Flywheel, checked before it is used as a directory name.

    The server is not trusted with paths: only a lane that exists in this checkout is accepted,
    so a crafted value such as `../..` or an absolute path cannot move a write elsewhere."""
    name = str(lane or "")
    lanes_dir = ROOT / "autoresearch" / "lanes"
    if not name.isalnum() or not (lanes_dir / name).is_dir():
        raise ValueError(f"the Flywheel named lane {name!r}, which this checkout does not have")
    return name


def _split_ids(values: list[str] | None) -> list[str]:
    return [x.strip() for v in (values or []) for x in v.split(",") if x.strip()]


def _fetch_source(node_id: str) -> tuple[dict, str]:
    """Fetch a trial's source and check it against its sha256."""
    import hashlib

    res = flywheel.get_source(node_id) or {}
    source = str(res.get("source") or "")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if not source or digest != res.get("sha256"):
        raise flywheel.FlywheelError(f"source for {node_id} failed its sha256 check", status=None)
    return res, source


def _ar_register(a: argparse.Namespace) -> int:
    path = flywheel.token_path()
    if path.exists() and not a.force:
        print(f"A token is already saved at {path}. Pass --force to replace it "
              "(the old token keeps working until a maintainer revokes it).", file=sys.stderr)
        return 1
    try:
        res = flywheel.register(a.handle, agent=a.agent, model=a.model, invite_code=a.invite_code)
    except flywheel.FlywheelError as exc:
        print(f"Registration refused: {_refusal(exc)}", file=sys.stderr)
        return 1
    token = res.get("token")
    if not token:
        print(f"Registration returned no token: {res}", file=sys.stderr)
        return 1
    saved = flywheel.save_token(token, force=True)
    flywheel.save_identity(str(res.get("handle") or a.handle), a.agent, a.model)
    url = flywheel.read_url()
    print(f"Registered `{res.get('handle')}` on {url} at tier `{res.get('role')}` "
          f"({res.get('hourly_limit')} writes/hour"
          + (f", {res['daily_write_limit']} writes/day" if res.get("daily_write_limit") else "") + ").")
    print(f"Token, shown once and saved to {saved} (mode 0600): {token}")
    if url != flywheel.DEFAULT_PUBLIC_URL:
        print(f"This token is for {url}: keep APIN_FLYWHEEL_URL={url} set when you use it.")
    if not a.model:
        print("No --model was given, so your trials will not name a model on the site. "
              "Set APIN_AGENT_MODEL=<model> in your shell to fix that; a handle cannot be registered twice.")
    print("Every `uv run apin autoresearch eval` from this checkout now posts its trial and source; nothing to export.")
    return 0


def _ar_fetch(a: argparse.Namespace) -> int:
    try:
        res, source = _fetch_source(a.node)
        lane = None
        if a.out is None:
            lane = _local_lane(((flywheel.get_node(a.node) or {}).get("node") or {}).get("lane"))
    except flywheel.FlywheelError as exc:
        print(f"Could not fetch {a.node}: {_refusal(exc)}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Could not place {a.node}: {exc}. Pass --out to choose the file yourself.", file=sys.stderr)
        return 1
    out = a.out or (ROOT / "autoresearch" / "lanes" / str(lane) / "runs" / a.run_id / "candidates"
                    / f"{_safe_name(a.node)}.py")
    if out.exists() and not a.force:
        print(f"{out} exists; pass --force to overwrite it.", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(source, encoding="utf-8")
    print(f"Wrote {res.get('size')} bytes of {a.node} (sha256 {str(res.get('sha256'))[:16]}) to {out}")
    if lane:
        # The id comes from another contributor, so it is quoted before anyone pastes it into a shell.
        print(f"To build on it: copy it over autoresearch/lanes/{lane}/submission.py, then "
              f"`uv run apin autoresearch eval --lane {lane} --fork-of {shlex.quote(a.node)} --hypothesis \"...\"`.")
    return 0


def _print_note(n: dict) -> None:
    lane = n.get("lane") or "-"
    print(f"{n.get('id')}  [{lane}] {n.get('kind')}/{n.get('state')}  by {n.get('author')}  {n.get('updated_at', '')[:19]}")
    print(f"    {n.get('title')}")
    if n.get("body"):
        body = " ".join(str(n["body"]).split())
        print(f"    {body[:200]}{'…' if len(body) > 200 else ''}")
    if n.get("node_ids"):
        print(f"    trials: {', '.join(n['node_ids'])}")


def _ar_note(a: argparse.Namespace) -> int:
    try:
        if a.note_cmd == "list":
            res = flywheel.list_notes(lane=a.lane, kind=a.kind, state=a.state, author=a.author, limit=a.limit) or {}
            notes = res.get("notes") or []
            if a.json_output:
                print(json.dumps(notes, indent=1))
            elif not notes:
                print("No notes match.")
            for n in [] if a.json_output else notes:
                _print_note(n)
            return 0
        if flywheel._resolve_token() is None:
            return _no_token()
        if a.note_cmd == "add":
            body = a.body_file.read_text(encoding="utf-8") if a.body_file else a.body
            res = flywheel.add_note(a.kind, a.title, lane=a.lane, body=body, node_ids=_split_ids(a.node))
            note = (res or {}).get("note") or {}
            print(f"Recorded {note.get('kind')} {note.get('id')} [{note.get('lane') or '-'}]: {note.get('title')}")
            return 0
        res = flywheel.update_note(a.note, state=a.state, reason=a.reason)
        note = (res or {}).get("note") or {}
        print(f"{note.get('id')} is now {note.get('state')}.")
        return 0
    except flywheel.FlywheelError as exc:
        print(f"The Flywheel refused the request: {_refusal(exc)}", file=sys.stderr)
        return 1


def _ar_set_state(a: argparse.Namespace) -> int:
    if flywheel._resolve_token() is None:
        return _no_token()
    try:
        res = flywheel.set_node_state(a.node, a.state, a.reason) or {}
    except flywheel.FlywheelError as exc:
        print(f"The Flywheel refused the request: {_refusal(exc)}", file=sys.stderr)
        return 1
    node = res.get("node") or {}
    print(f"{node.get('id', a.node)} is now {node.get('state')}.")
    return 0


def _ar_search(a: argparse.Namespace) -> int:
    try:
        res = flywheel.search(a.text, lane=a.lane, kind=a.kind, limit=a.limit, offset=a.offset) or {}
    except flywheel.FlywheelError as exc:
        print(f"Search failed: {_refusal(exc)}", file=sys.stderr)
        return 1
    results = res.get("results") or []
    if a.json_output:
        print(json.dumps(res, indent=1))
        return 0
    if not results:
        print(f"No trials or notes match {a.text!r}.")
    for r in results:
        if r.get("kind") == "node":
            head = f"trial {r.get('id')}  {r.get('decision')}/{r.get('state')}  soft_min={r.get('soft_min_eff')}"
            head += "  [source]" if r.get("has_source") else ""
        else:
            head = f"note  {r.get('id')}  [{r.get('lane') or '-'}] {r.get('note_kind')}/{r.get('state')}"
        print(f"{head}  by {r.get('author')}")
        print(f"    {r.get('title') or ''}")
        print(f"    {r.get('snippet') or ''}")
    if res.get("next_offset") is not None:
        print(f"More: --offset {res['next_offset']}")
    return 0


def _ar_verify_next(a: argparse.Namespace) -> int:
    if flywheel._resolve_token() is None:
        return _no_token()
    try:
        res = flywheel.verify_claim(lane=a.lane, ttl_minutes=a.ttl_minutes) or {}
    except flywheel.FlywheelError as exc:
        if exc.status == 404:
            print(f"Nothing to verify{' in lane ' + a.lane if a.lane else ''} right now.")
            return 0
        print(f"Could not lease a trial: {_refusal(exc)}", file=sys.stderr)
        return 1
    lease, node = res.get("lease") or {}, res.get("node") or {}
    node_id = str(lease.get("node_id"))
    try:
        lane = _local_lane(lease.get("lane") or node.get("lane"))
    except ValueError as exc:
        print(f"Cannot rerun {node_id}: {exc}. The lease expires on its own.", file=sys.stderr)
        return 1
    print(f"Leased {node_id} (lane {lane}, claimed soft_min={node.get('soft_min_eff')}) until {lease.get('lease_expires_at')}.")
    try:
        _, source = _fetch_source(node_id)
    except flywheel.FlywheelError as exc:
        print(f"Could not fetch its source: {_refusal(exc)}. The lease expires on its own.", file=sys.stderr)
        return 1
    work = ROOT / "autoresearch" / "lanes" / lane / "runs" / a.run_id / _safe_name(node_id)
    work.mkdir(parents=True, exist_ok=True)
    cand = work / "candidate.py"
    cand.write_text(source, encoding="utf-8")
    dims = tuple(int(x) for x in (node.get("dims") or [])) or (6, 20)
    try:
        outcome = autoresearch.rerun_for_verification(
            cand, lane, claimed_soft_min=node.get("soft_min_eff"), dims=dims, repeats=a.repeats,
            tolerance=a.tolerance, work_dir=work / "work", use_queue=not a.no_queue,
        )
    except Exception as exc:  # an evaluator crash is not evidence against the trial
        print(f"The rerun crashed ({type(exc).__name__}: {exc}); no result posted. "
              f"The lease expires at {lease.get('lease_expires_at')}.", file=sys.stderr)
        return 1
    try:
        flywheel.verify_result(node_id, str(lease.get("lease_token")), outcome["passed"],
                               metrics=outcome["metrics"], note=outcome.get("reason"))
    except flywheel.FlywheelError as exc:
        print(f"The Flywheel refused the result: {_refusal(exc)}", file=sys.stderr)
        return 1
    verdict = "replicated" if outcome["passed"] else "disputed"
    print(f"{node_id}: rerun {'PASSED' if outcome['passed'] else 'FAILED'} ({outcome.get('reason')}); "
          f"trial marked {verdict}. Candidate and work files: {work}")
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
    s.add_argument("--strict", action="store_true",
                   help="strict verification gate (alpha_right=0.001, alpha_wrong=0.001, alpha_calibration=0.01)")
    s.add_argument("--target-ess", type=float, default=score.DEFAULT_TARGET_ESS,
                   help="target worst-series ESS for time-to-ESS / cost-to-ESS speedrun metrics (default: 1000)")
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
    inv.add_argument("--chains", type=int, help="number of parallel exact-start chains (default: 64, or 256 with --strict)")
    inv.add_argument("--steps", type=int, help="steps per chain (default: 20, or 40 with --strict)")
    inv.add_argument("--alpha", type=float, help="calibrated omnibus alpha (default: 0.001, or 1e-4 with --strict)")
    inv.add_argument("--strict", action="store_true",
                     help="strict verification mode: longer chains + more chains + tighter alpha (256 chains x 40 steps, alpha=1e-4)")
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
    ae.add_argument("--json", action="store_true", dest="json_output",
                    help="emit a single Auto-research-agent compatible JSON object on stdout")
    ae.add_argument("--combines", action="append", default=[], metavar="NODE_ID[,NODE_ID]",
                    help="Flywheel trial(s) whose mechanism this candidate reuses (posted as `combines` parents)")
    ae.add_argument("--fork-of", action="append", default=[], metavar="NODE_ID",
                    help="Flywheel trial whose source this candidate started from (see `apin autoresearch fetch`)")

    ab = ar_sub.add_parser("brief", help="print a zero-synthesis verbatim brief (Flywheel frontier, claims, dead ends, stall check)")
    ab.add_argument("--lane", default="all", help="lane name (A0..A4) or 'all' for a cross-lane overview (default: all)")
    ab.add_argument("--run-id", help="optional local run_id to include stall & weakest-class diagnostics")
    ab.add_argument("--offline", action="store_true", help="skip live Flywheel fetch (local files only)")
    ab.add_argument("--max-failures", type=int, default=8)

    ac = ar_sub.add_parser("claim", help="reserve a task on the Flywheel so nobody else runs the same idea (needs a token: `apin autoresearch register`)")
    ac.add_argument("--task", required=True, help="task id from scripts/choose_research_task.py --list, or a short slug of your own")
    ac.add_argument("--lane", required=True, help="lane the task belongs to (A0..A4, S1..S3)")
    ac.add_argument("--agent", required=True, help="your handle, or the model doing the work")
    ac.add_argument("--summary", default="", help="one sentence on what you plan to try")
    ac.add_argument("--ttl-minutes", type=int, default=60, help="how long the claim holds before it expires on its own")
    arl = ar_sub.add_parser("release", help="mark a claim done or abandoned so the task is free again")
    arl.add_argument("--claim", required=True, help="claim id printed by `apin autoresearch claim`")
    arl.add_argument("--status", choices=("done", "abandoned"), default="done")

    arg = ar_sub.add_parser("register", help="register a handle on the Flywheel and save your token (no invite needed)")
    arg.add_argument("--handle", required=True, help="2-40 chars: letters, digits, _ . -")
    arg.add_argument("--agent", help="the agent harness you run, e.g. claude-code")
    arg.add_argument("--model", help="the model doing the work")
    arg.add_argument("--invite-code", help="community invite code: contributor tier at once instead of provisional")
    arg.add_argument("--force", action="store_true", help="replace an already saved token")

    afe = ar_sub.add_parser("fetch", help="download a Flywheel trial's source to a candidate file")
    afe.add_argument("--node", required=True, help="trial id, e.g. A1:run:T0004")
    afe.add_argument("--out", type=Path, help="default: autoresearch/lanes/<lane>/runs/<run-id>/candidates/<node>.py")
    afe.add_argument("--run-id", default="default", help="run directory for the default --out (default: default)")
    afe.add_argument("--force", action="store_true", help="overwrite an existing file (needed for submission.py)")

    an = ar_sub.add_parser("note", help="record and browse hypotheses and insights")
    an_sub = an.add_subparsers(dest="note_cmd", required=True)
    ana = an_sub.add_parser("add", help="record a hypothesis or an insight")
    ana.add_argument("--kind", choices=("hypothesis", "insight"), required=True)
    ana.add_argument("--title", required=True)
    ana.add_argument("--lane")
    body = ana.add_mutually_exclusive_group()
    body.add_argument("--body")
    body.add_argument("--body-file", type=Path)
    ana.add_argument("--node", action="append", default=[], metavar="NODE_ID[,NODE_ID]", help="linked trial id(s)")
    anl = an_sub.add_parser("list", help="list notes, newest first")
    anl.add_argument("--lane")
    anl.add_argument("--kind", choices=("hypothesis", "insight"))
    anl.add_argument("--state")
    anl.add_argument("--author")
    anl.add_argument("--limit", type=int, default=20)
    anl.add_argument("--json", action="store_true", dest="json_output")
    ans = an_sub.add_parser("set-state", help="hypothesis: open|supported|refuted|needs-replication; insight: open|superseded|retracted")
    ans.add_argument("--note", required=True, help="note id")
    ans.add_argument("--state", required=True)
    ans.add_argument("--reason")

    ass = ar_sub.add_parser("set-state", help="retract your trial, or flag another's as needs-replication / disputed")
    ass.add_argument("--node", required=True)
    ass.add_argument("--state", required=True, choices=("retracted", "needs-replication", "disputed", "unverified", "replicated", "verified"))
    ass.add_argument("--reason", help="required when flagging someone else's trial")

    asr = ar_sub.add_parser("search", help="search Flywheel trials and notes")
    asr.add_argument("text")
    asr.add_argument("--lane")
    asr.add_argument("--kind", choices=("node", "note"))
    asr.add_argument("--limit", type=int, default=20)
    asr.add_argument("--offset", type=int, default=0)
    asr.add_argument("--json", action="store_true", dest="json_output")

    avn = ar_sub.add_parser("verify-next", help="lease the next queued trial, rerun it on fresh seeds, post the result")
    avn.add_argument("--lane")
    avn.add_argument("--repeats", type=int, default=2)
    avn.add_argument("--tolerance", type=float, default=autoresearch.REPLICATION_TOLERANCE,
                     help="pass when the rerun reaches (1 - tolerance) x the claimed soft_min_eff (default: 0.25)")
    avn.add_argument("--ttl-minutes", type=int, default=120, help="lease length")
    avn.add_argument("--run-id", default="verify", help="run directory for candidates and work files")
    avn.add_argument("--no-queue", action="store_true", help="bypass the host FIFO evaluation lock")

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
                        submissions={k: Path(v) for k, v in subs.items()}, track=a.track,
                        strict=a.strict, target_ess=a.target_ess)
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
        n_chains = a.chains if a.chains is not None else (256 if a.strict else 64)
        k_steps = a.steps if a.steps is not None else (40 if a.strict else 20)
        alpha = a.alpha if a.alpha is not None else (1e-4 if a.strict else 0.001)
        if a.strict:
            print(f"[invariance: STRICT] dim={a.dim} chains={n_chains} steps={k_steps} alpha={alpha:g}")
        suite = shapes.make_suite(a.seed, dim=a.dim)
        failed = 0
        for i, tgt in enumerate(suite):
            r = invariance.check_target_invariance(tgt, a.sampler, seed=a.seed + 100 * i,
                                                   n_chains=n_chains, k_steps=k_steps, alpha=alpha)
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
        if a.ar_cmd == "brief":
            print(autoresearch.build_research_brief(
                a.lane, run_id=a.run_id, offline=a.offline, max_failures=a.max_failures,
            ), end="")
            return 0
        if a.ar_cmd in ("claim", "release"):
            return _claim_or_release(a)
        handlers = {
            "register": _ar_register, "fetch": _ar_fetch, "note": _ar_note, "set-state": _ar_set_state,
            "search": _ar_search, "verify-next": _ar_verify_next,
        }
        if a.ar_cmd in handlers:
            return handlers[a.ar_cmd](a)
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
            combines=_split_ids(a.combines), fork_of=_split_ids(a.fork_of),
        )
        if a.json_output:
            print(json.dumps(autoresearch.format_evaluator_json(row, run_id=a.run_id)))
            if a.dry_run:
                return 0 if row["decision"] == "dry_run_ok" else 2
            return 0 if row["tier0_ok"] else 1
        if a.dry_run:
            print(f"{row['trial_id']} [{row['lane']}] dry_run={row['decision']} "
                  f"dup={row['is_duplicate']} lint_ok={row['lint_ok']} loc={row['loc']} "
                  f"consts={row['magic_constants']}")
            return 0 if row["decision"] == "dry_run_ok" else 2
        inc_note = " [WARNING: paired incumbent failed Tier 1 on fresh seeds — UNPAIRED keep; re-verify]" if row.get("incumbent_ok") is False else ""
        print(f"{row['trial_id']} [{row['lane']}] decision={row['decision']} "
              f"tier0={row['tier0_ok']} tier1={row['tier1_ok']} "
              f"soft_min={row.get('soft_min_eff')} inc={row.get('incumbent_soft_min')} ratio={row.get('ratio_vs_incumbent')} "
              f"dims={','.join(str(x) for x in (row.get('dims') or []))} repeats={row.get('repeats')} "
              f"qwait={row.get('queue_wait_s')}s wall={row['wall_s']}s{inc_note}")
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
