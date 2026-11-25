"""Tests for Linux Landlock + seccomp-BPF worker confinement (`harness/sandbox.py`)."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import pytest

from harness import ipc, logdensity, sandbox, shapes

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples" / "samplers"


def _linux_sandbox_available() -> bool:
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        return False
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from harness import sandbox; "
            "r = sandbox.confine_worker(Path(sandbox.__file__), mode='strict'); assert r['active']",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def test_normal_submission_runs_under_strict_sandbox(tmp_path, monkeypatch):
    """A legitimate sampler (hmc.py) succeeds cleanly with APIN_SANDBOX=strict on Linux."""
    if not _linux_sandbox_available():
        pytest.skip("Linux Landlock/seccomp unavailable in this environment")
    monkeypatch.setenv(sandbox.ENV_SANDBOX, "strict")
    tgt = shapes.IllCondGaussian(dim=3, seed=7, rotated=True, kappa=10.0)
    budget = logdensity.Budget(chains=1, draws=20, max_evaluations=50_000, max_seconds=30.0)
    draws, summary = ipc.execute_run_isolated(
        tgt, EXAMPLES / "hmc.py", budget, run_seed=11, work_dir=tmp_path, replay_count=3,
    )
    assert draws.shape == (1, 20, 3)
    assert summary["replay"]["replayed_singles"] == 3


def test_sandbox_blocks_secret_file_read_file_write_and_network_bypass(tmp_path, monkeypatch):
    """Even when a submission passes AST import lint (e.g. pure NumPy file I/O) or bypasses it,
    Landlock + seccomp block secret file reads, file writes, and network sockets."""
    if not _linux_sandbox_available():
        pytest.skip("Linux Landlock/seccomp unavailable in this environment")
    monkeypatch.setenv(sandbox.ENV_SANDBOX, "strict")

    secret = tmp_path / "secret.txt"
    secret.write_text("1.0 2.0\n")
    smuggle = tmp_path / "smuggled.npy"

    # 1. File read outside allowed roots is blocked by Landlock even via allowed numpy I/O
    sub_read = tmp_path / "evil_read.py"
    sub_read.write_text(
        "import numpy as np\n"
        "def adapt(oracle, budget, rng):\n"
        f"    np.loadtxt({str(secret)!r})\n"
        "    x = np.zeros(oracle.dim)\n"
        "    oracle.log_density(x)\n"
        "    return {}, (x, None), [1.0], 'none'\n"
        "def transition(tuning, chain, oracle, rng):\n"
        "    return chain.position, None\n"
    )
    tgt = shapes.IllCondGaussian(dim=2, seed=1)
    counter = logdensity.Counter(tgt, max_evaluations=1000, max_seconds=10.0)
    budget = logdensity.Budget(1, 5, 1000, 10.0)
    with pytest.raises(RuntimeError, match="Permission"):
        ipc.run_isolated_adapt(counter, sub_read, budget, run_seed=1)

    # 2. File write to /tmp or tmp_path is blocked by Landlock even via allowed numpy I/O
    sub_write = tmp_path / "evil_write.py"
    sub_write.write_text(
        "import numpy as np\n"
        "def adapt(oracle, budget, rng):\n"
        f"    np.save({str(smuggle)!r}, np.zeros(2))\n"
        "    x = np.zeros(oracle.dim)\n"
        "    oracle.log_density(x)\n"
        "    return {}, (x, None), [1.0], 'none'\n"
        "def transition(tuning, chain, oracle, rng):\n"
        "    return chain.position, None\n"
    )
    with pytest.raises(RuntimeError, match="Permission"):
        ipc.run_isolated_adapt(counter, sub_write, budget, run_seed=2)
    assert not smuggle.exists()

    # 3. IPv4 / IPv6 network socket creation is blocked by seccomp-BPF even if AST lint is bypassed
    net_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import socket; from pathlib import Path; from harness import sandbox; "
            f"sandbox.confine_worker(Path({str(sub_read)!r}), mode='strict'); "
            "socket.socket(socket.AF_INET, socket.SOCK_STREAM)",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert net_probe.returncode != 0
    assert "Operation not permitted" in net_probe.stderr or "Permission" in net_probe.stderr
