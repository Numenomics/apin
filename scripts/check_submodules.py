#!/usr/bin/env python3
"""Check `.gitmodules` pins against remote repositories and `upstream/STATUS.md`,
and optionally report how many commits each fork is behind its upstream default branch.

Usage:
  python scripts/check_submodules.py --check-pins
  python scripts/check_submodules.py --drift-report
"""

from __future__ import annotations

import argparse
import configparser
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Canonical upstream repos corresponding to each submodule in upstream/STATUS.md
UPSTREAM_REPOS: dict[str, tuple[str, str]] = {
    "upstream/cmdstan": ("stan-dev/cmdstan", "develop"),
    "upstream/stan": ("stan-dev/stan", "develop"),
    "upstream/math": ("stan-dev/math", "develop"),
    "upstream/stanc3": ("stan-dev/stanc3", "master"),
    "upstream/bridgestan": ("roualdes/bridgestan", "main"),
    "upstream/walnutpie": ("flatironinstitute/walnutpie", "main"),
    "upstream/posteriordb": ("stan-dev/posteriordb", "master"),
}


def parse_gitmodules() -> dict[str, dict[str, str]]:
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / ".gitmodules")
    out: dict[str, dict[str, str]] = {}
    for sec in cfg.sections():
        path = cfg[sec].get("path", "").strip()
        url = cfg[sec].get("url", "").strip()
        branch = cfg[sec].get("branch", "").strip()
        if path:
            out[path] = {"url": url, "branch": branch}
    return out


def pinned_shas() -> dict[str, str]:
    r = subprocess.run(
        ["git", "-C", str(ROOT), "ls-tree", "HEAD", "upstream/"],
        capture_output=True, text=True, check=True,
    )
    pins: dict[str, str] = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[1] == "commit":
            pins[parts[3]] = parts[2]
    return pins


def github_owner_repo(url: str) -> str | None:
    m = re.search(r"github\.com[:/]([^/]+/[^/.]+?)(?:\.git)?$", url)
    return m.group(1) if m else None


def gh_api(path: str, token: str | None = None) -> tuple[int, dict | None]:
    url = f"https://api.github.com{path}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "apin-ci"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except urllib.error.URLError:
        return 0, None


def check_pins(token: str | None = None, verify_remote: bool = False) -> int:
    mods = parse_gitmodules()
    pins = pinned_shas()
    status_text = (ROOT / "upstream" / "STATUS.md").read_text(encoding="utf-8")
    errors = 0

    for path, info in sorted(mods.items()):
        sha = pins.get(path)
        if not sha:
            print(f"FAIL {path}: present in .gitmodules but missing gitlink in HEAD")
            errors += 1
            continue
        in_status = sha in status_text or sha[:7] in status_text
        if not in_status:
            print(f"WARN {path}: pinned SHA {sha[:12]} not found in upstream/STATUS.md")
        repo = github_owner_repo(info["url"])
        remote_note = "local-only"
        if verify_remote and repo:
            code, _ = gh_api(f"/repos/{repo}/commits/{sha}", token=token)
            if code == 200:
                remote_note = f"reachable on {repo}"
            elif code == 403:
                remote_note = f"rate-limited checking {repo} (skipped)"
            else:
                print(f"FAIL {path}: pinned SHA {sha} NOT reachable on {repo} (HTTP {code})")
                errors += 1
                continue
        print(f"OK   {path:22s} {sha[:12]} ({remote_note})")

    return 1 if errors else 0


def drift_report(token: str | None = None) -> int:
    mods = parse_gitmodules()
    pins = pinned_shas()
    lines = [
        "## Weekly Upstream Submodule Drift Report",
        "",
        "| submodule | fork pin | upstream repo | upstream head | behind / ahead | status |",
        "|---|---|---|---|---|---|",
    ]
    for path, info in sorted(mods.items()):
        sha = pins.get(path, "")
        fork_repo = github_owner_repo(info["url"]) or ""
        up_repo, up_branch = UPSTREAM_REPOS.get(path, ("", ""))
        up_head = "—"
        behind_ahead = "—"
        note = "ok"
        if fork_repo and sha:
            c_pin, _ = gh_api(f"/repos/{fork_repo}/commits/{sha}", token=token)
            if c_pin != 200:
                note = f"WARNING: pin HTTP {c_pin} on {fork_repo}"
        if up_repo and up_branch:
            c_up, data_up = gh_api(f"/repos/{up_repo}/commits/{up_branch}", token=token)
            if c_up == 200 and data_up:
                up_head = str(data_up.get("sha", ""))[:10]
            # Compare upstream branch against fork branch if fork has a branch
            branch = info.get("branch")
            if fork_repo and branch and fork_repo != up_repo:
                fork_owner = fork_repo.split("/")[0]
                c_cmp, cmp_data = gh_api(
                    f"/repos/{up_repo}/compare/{up_branch}...{fork_owner}:{branch}",
                    token=token,
                )
                if c_cmp == 200 and cmp_data:
                    behind = cmp_data.get("behind_by", "?")
                    ahead = cmp_data.get("ahead_by", "?")
                    behind_ahead = f"-{behind} / +{ahead}"
        lines.append(
            f"| `{path}` | `{sha[:10]}` | `{up_repo}` (`{up_branch}`) | `{up_head}` | {behind_ahead} | {note} |"
        )
    print("\n".join(lines))
    return 0


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--check-pins", action="store_true")
    ap.add_argument("--verify-remote", action="store_true")
    ap.add_argument("--drift-report", action="store_true")
    a = ap.parse_args()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if a.drift_report:
        return drift_report(token=token)
    return check_pins(token=token, verify_remote=a.verify_remote)


if __name__ == "__main__":
    sys.exit(main())
