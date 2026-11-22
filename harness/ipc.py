"""Out-of-process brokered oracle IPC and clean phase worker isolation.

Runs untrusted submission code in dedicated child processes (`OMP_NUM_THREADS=1`)
connected to the parent harness over a private Unix domain `socketpair`:
  - The parent owns `Counter` and the target model (`AnalyticTarget` or `BridgeStan`).
  - The child NEVER imports `shapes`, `bridgestan`, or `Counter`—closure introspection,
    `gc.get_objects()`, and stack inspection inside the submission process cannot reach
    evaluation counters, budgets, or `AnalyticTarget.exact_draws()`.
  - `adapt()`, `transition()` (sampling/invariance), and `replay()` each execute in a
    brand-new clean child process so module globals and caches never cross phases.
  - For Lane A1, the transition worker runs ONLY `harness.judge.a1_transition` and
    never even imports the user's `submission.py`.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import judge, logdensity, sandbox
from .logdensity import (
    Budget,
    BudgetExceeded,
    CapabilityError,
    Chain,
    ContractError,
    Counter,
    MAX_TUNING_BYTES,
    Oracle,
    REPLAY_STEPS,
    adapt_rng,
    canonical_digest,
    choose_replay_steps,
    decode_canonical,
    deserialize_tree,
    encode_canonical,
    normalize_aux_spec,
    normalize_chain,
    serialize_tree,
    step_rng,
    validate_aux,
    validate_betas,
)
from .models import ROOT

# Framing tags
TAG_CMD_ADAPT = 1
TAG_CMD_INIT_TRANSITION = 2
TAG_CMD_STEP = 3
TAG_CMD_CHECK_TUNING = 4
TAG_CMD_QUIT = 5

TAG_OP_DENSITY = 10
TAG_OP_GRAD = 11
TAG_OP_HVP = 12
TAG_OP_HESS = 13

TAG_VAL_DENSITY = 20
TAG_VAL_GRAD = 21
TAG_VAL_HESS = 22

TAG_RESP_OK = 30
TAG_RESP_ERR = 31

TAG_ERR_BUDGET = 40
TAG_ERR_CAP = 41
TAG_ERR_VALUE = 42
TAG_ERR_RUNTIME = 43

_HDR = struct.Struct("<BI")


def _send_msg(sock: socket.socket, tag: int, payload: bytes = b"") -> None:
    sock.sendall(_HDR.pack(tag, len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    if n == 0:
        return b""
    chunks = bytearray()
    while len(chunks) < n:
        part = sock.recv(n - len(chunks))
        if not part:
            raise EOFError("IPC peer closed connection unexpectedly")
        chunks.extend(part)
    return bytes(chunks)


def _recv_msg(sock: socket.socket) -> tuple[int, bytes]:
    hdr = _recv_exact(sock, _HDR.size)
    tag, length = _HDR.unpack(hdr)
    payload = _recv_exact(sock, length)
    return tag, payload


def _worker_env() -> dict[str, str]:
    return {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def _raise_remote_error(err_kind: str, msg: str) -> None:
    mapping = {
        "BudgetExceeded": BudgetExceeded,
        "CapabilityError": CapabilityError,
        "ContractError": ContractError,
        "ImportError": ImportError,
        "AttributeError": AttributeError,
        "ValueError": ValueError,
    }
    cls = mapping.get(err_kind, RuntimeError)
    raise cls(msg)


def _serve_until_response(sock: socket.socket, counter: Counter) -> Any:
    """Service oracle RPCs from the child worker until it returns TAG_RESP_OK or TAG_RESP_ERR."""
    dim = counter.dim
    while True:
        tag, payload = _recv_msg(sock)
        if tag == TAG_OP_DENSITY:
            try:
                theta = np.frombuffer(payload, dtype=np.float64)
                lp = counter.log_density(theta)
                _send_msg(sock, TAG_VAL_DENSITY, struct.pack("<d", float(lp)))
            except BudgetExceeded as e:
                _send_msg(sock, TAG_ERR_BUDGET, str(e).encode("utf-8"))
            except CapabilityError as e:
                _send_msg(sock, TAG_ERR_CAP, str(e).encode("utf-8"))
            except ValueError as e:
                _send_msg(sock, TAG_ERR_VALUE, str(e).encode("utf-8"))
            except Exception as e:
                _send_msg(sock, TAG_ERR_RUNTIME, f"{type(e).__name__}: {e}".encode("utf-8"))
        elif tag == TAG_OP_GRAD:
            try:
                theta = np.frombuffer(payload, dtype=np.float64)
                lp, g = counter.log_density_gradient(theta)
                g_arr = np.ascontiguousarray(g, dtype=np.float64)
                _send_msg(sock, TAG_VAL_GRAD, struct.pack("<d", float(lp)) + g_arr.tobytes())
            except BudgetExceeded as e:
                _send_msg(sock, TAG_ERR_BUDGET, str(e).encode("utf-8"))
            except CapabilityError as e:
                _send_msg(sock, TAG_ERR_CAP, str(e).encode("utf-8"))
            except ValueError as e:
                _send_msg(sock, TAG_ERR_VALUE, str(e).encode("utf-8"))
            except Exception as e:
                _send_msg(sock, TAG_ERR_RUNTIME, f"{type(e).__name__}: {e}".encode("utf-8"))
        elif tag == TAG_OP_HVP:
            try:
                half = dim * 8
                theta = np.frombuffer(payload[:half], dtype=np.float64)
                v = np.frombuffer(payload[half:], dtype=np.float64)
                lp, hv = counter.hvp(theta, v)
                hv_arr = np.ascontiguousarray(hv, dtype=np.float64)
                _send_msg(sock, TAG_VAL_GRAD, struct.pack("<d", float(lp)) + hv_arr.tobytes())
            except BudgetExceeded as e:
                _send_msg(sock, TAG_ERR_BUDGET, str(e).encode("utf-8"))
            except CapabilityError as e:
                _send_msg(sock, TAG_ERR_CAP, str(e).encode("utf-8"))
            except ValueError as e:
                _send_msg(sock, TAG_ERR_VALUE, str(e).encode("utf-8"))
            except Exception as e:
                _send_msg(sock, TAG_ERR_RUNTIME, f"{type(e).__name__}: {e}".encode("utf-8"))
        elif tag == TAG_OP_HESS:
            try:
                theta = np.frombuffer(payload, dtype=np.float64)
                lp, g, h = counter.hess(theta)
                g_arr = np.ascontiguousarray(g, dtype=np.float64)
                h_arr = np.ascontiguousarray(h, dtype=np.float64)
                _send_msg(sock, TAG_VAL_HESS, struct.pack("<d", float(lp)) + g_arr.tobytes() + h_arr.tobytes())
            except BudgetExceeded as e:
                _send_msg(sock, TAG_ERR_BUDGET, str(e).encode("utf-8"))
            except CapabilityError as e:
                _send_msg(sock, TAG_ERR_CAP, str(e).encode("utf-8"))
            except ValueError as e:
                _send_msg(sock, TAG_ERR_VALUE, str(e).encode("utf-8"))
            except Exception as e:
                _send_msg(sock, TAG_ERR_RUNTIME, f"{type(e).__name__}: {e}".encode("utf-8"))
        elif tag == TAG_RESP_OK:
            return decode_canonical(payload)
        elif tag == TAG_RESP_ERR:
            err_kind, msg = decode_canonical(payload)
            _raise_remote_error(err_kind, msg)
        else:
            raise RuntimeError(f"unexpected IPC tag {tag}")


def _spawn_worker(mode: str, dim: int, block_shape: tuple[tuple[int, ...], ...] | None,
                  timeout_s: float) -> tuple[subprocess.Popen, socket.socket]:
    parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    parent_sock.settimeout(max(5.0, float(timeout_s) + 10.0))
    os.set_inheritable(child_sock.fileno(), True)
    bs_json = json.dumps([list(x) for x in block_shape] if block_shape is not None else None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "harness.ipc", mode, str(child_sock.fileno()), str(int(dim)), bs_json],
        cwd=ROOT,
        env=_worker_env(),
        pass_fds=(child_sock.fileno(),),
        close_fds=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    child_sock.close()
    return proc, parent_sock


def _close_worker(proc: subprocess.Popen, sock: socket.socket) -> str:
    try:
        _send_msg(sock, TAG_CMD_QUIT)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass
    stderr_text = ""
    try:
        _, err = proc.communicate(timeout=5.0)
        stderr_text = (err or b"").decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        proc.kill()
        _, err = proc.communicate()
        stderr_text = (err or b"").decode("utf-8", errors="replace")
    return stderr_text


def run_isolated_adapt(
    counter: Counter,
    submission_path: Path,
    budget: Budget,
    run_seed: int,
    *,
    lane: str | None = None,
    seed_seq: list[int] | None = None,
) -> tuple[Any, np.ndarray, Any, bool, np.ndarray, dict, str]:
    """Run adapt() in a clean child process over brokered IPC.

    Parent validates all outputs, checks chain0 provenance against parent `counter`,
    verifies no swallowed BudgetExceeded, and returns:
      (tuning, pos_2d, aux, is_1d, betas, aux_spec, tuning_sha256)
    """
    sub_path = Path(submission_path).resolve()
    logdensity.check_imports(sub_path.read_text())
    proc, sock = _spawn_worker("adapt", counter.dim, counter._block_shape, budget.max_seconds)
    try:
        req = {
            "submission": str(sub_path),
            "lane": lane,
            "budget": [int(budget.chains), int(budget.draws), int(budget.max_evaluations), float(budget.max_seconds)],
            "seed_seq": list(seed_seq) if seed_seq is not None else [int(run_seed), 0],
        }
        _send_msg(sock, TAG_CMD_ADAPT, encode_canonical(req))
        try:
            res = _serve_until_response(sock, counter)
        except socket.timeout as e:
            proc.kill()
            raise BudgetExceeded(f"adapt() worker timed out after {budget.max_seconds} s") from e
        except EOFError as e:
            err_out = _close_worker(proc, sock)
            raise RuntimeError(f"adapt() worker crashed: {err_out[-1500:]}") from e
    finally:
        _close_worker(proc, sock)

    tuning = res["tuning"]
    pos_2d, aux, is_1d = normalize_chain((res["position"], res["aux"]), counter.dim)
    walkers = pos_2d.shape[0]
    betas = validate_betas(res["betas"], walkers)
    aux_spec = normalize_aux_spec(res["aux_spec"])
    validate_aux(aux_spec, aux, walkers, counter.dim, is_1d)
    for w in range(walkers):
        if logdensity._row_bytes(pos_2d[w]) not in counter.step_seen:
            raise ContractError(
                f"adapt() returned chain0 walker {w} that was never evaluated with a finite log density"
            )
    tuning_sha = canonical_digest(tuning)
    if counter.exceeded:
        raise BudgetExceeded(f"adapt() went past the budget ({counter.exceeded}) and caught the error")
    return tuning, pos_2d, aux, is_1d, betas, aux_spec, tuning_sha


class IsolatedTransitionSession:
    """Clean post-adapt child process serving transition() steps over brokered IPC."""

    def __init__(
        self,
        counter: Counter,
        submission_path: Path,
        tuning: Any,
        tuning_sha256: str,
        walkers: int,
        is_1d: bool,
        aux_spec: dict,
        *,
        lane: str | None = None,
        timeout_s: float = 300.0,
    ):
        self.counter = counter
        self.tuning_sha256 = tuning_sha256
        self.walkers = int(walkers)
        self.is_1d = bool(is_1d)
        self.aux_spec = aux_spec
        self._closed = False
        sub_path = Path(submission_path).resolve()
        if lane != "A1":
            logdensity.check_imports(sub_path.read_text())
        self.proc, self.sock = _spawn_worker("transition", counter.dim, counter._block_shape, timeout_s)
        try:
            init_req = {
                "submission": str(sub_path),
                "lane": lane,
                "tuning": tuning,
                "is_1d": self.is_1d,
            }
            _send_msg(self.sock, TAG_CMD_INIT_TRANSITION, encode_canonical(init_req))
            _serve_until_response(self.sock, counter)
        except Exception:
            self.close()
            raise

    def step(self, pos_2d: np.ndarray, aux: Any, seed_seq: list[int],
             check_provenance: bool = True) -> tuple[np.ndarray, Any]:
        self.counter.reset_step()
        req = {
            "position": pos_2d[0] if self.is_1d else pos_2d,
            "aux": aux,
            "seed_seq": [int(s) for s in seed_seq],
        }
        _send_msg(self.sock, TAG_CMD_STEP, encode_canonical(req))
        try:
            res = _serve_until_response(self.sock, self.counter)
        except socket.timeout as e:
            self.proc.kill()
            raise BudgetExceeded("transition() worker timed out") from e
        except EOFError as e:
            err_out = self.close()
            raise RuntimeError(f"transition() worker crashed: {err_out[-1500:]}") from e
        pos_next, aux_next, _ = normalize_chain(
            (res["position"], res["aux"]), self.counter.dim,
            expected_walkers=self.walkers, expected_is_1d=self.is_1d,
        )
        validate_aux(self.aux_spec, aux_next, self.walkers, self.counter.dim, self.is_1d)
        if check_provenance:
            self.counter.check_step_provenance(pos_2d, pos_next)
        return pos_next, aux_next

    def verify_tuning_frozen(self) -> None:
        _send_msg(self.sock, TAG_CMD_CHECK_TUNING)
        sha = _serve_until_response(self.sock, self.counter)
        if sha != self.tuning_sha256:
            raise ContractError("tuning was mutated during transition() (SHA-256 mismatch)")

    def close(self) -> str:
        if self._closed:
            return ""
        self._closed = True
        return _close_worker(self.proc, self.sock)

    def __enter__(self) -> "IsolatedTransitionSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def run_adapt_phase_isolated(
    model,
    submission_path: Path,
    budget: Budget,
    run_seed: int,
    bundle_dir: Path,
    *,
    capabilities: set[str] | None = None,
    lane: str | None = None,
) -> dict:
    """Isolated brokered equivalent of logdensity.run_adapt_phase."""
    limits = Budget(int(budget.chains), int(budget.draws), int(budget.max_evaluations), float(budget.max_seconds))
    started = logdensity._now()
    counter = Counter(model, limits.max_evaluations, limits.max_seconds, started=started, capabilities=capabilities)
    tuning, pos_2d, aux, is_1d, betas, aux_spec, tuning_sha = run_isolated_adapt(
        counter, submission_path, limits, run_seed, lane=lane,
    )
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    sha2, tuning_bytes = serialize_tree(tuning, bundle_dir / "tuning", max_bytes=MAX_TUNING_BYTES)
    assert sha2 == tuning_sha
    serialize_tree({"position": pos_2d, "aux": aux}, bundle_dir / "chain0")
    seconds = logdensity._now() - started
    if seconds > limits.max_seconds:
        raise BudgetExceeded(f"adapt() took {seconds:.1f} s, exceeding {limits.max_seconds} s")
    meta = {
        "tuning_sha256": tuning_sha,
        "tuning_bytes": tuning_bytes,
        "walkers": int(pos_2d.shape[0]),
        "is_1d": is_1d,
        "betas": betas.tolist(),
        "aux_spec": aux_spec,
        "dim": counter.dim,
        "density_evals": counter.density,
        "gradient_evals": counter.gradient,
        "hvp_evals": counter.hvp_calls,
        "hess_evals": counter.hess_calls,
        "seconds": seconds,
        "eval_s": counter.eval_s,
    }
    (bundle_dir / "adapt_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def run_sample_phase_isolated(
    model,
    submission_path: Path,
    budget: Budget,
    run_seed: int,
    bundle_dir: Path,
    adapt_meta: dict,
    *,
    capabilities: set[str] | None = None,
    replay_count: int = REPLAY_STEPS,
    lane: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Isolated brokered equivalent of logdensity.run_sample_phase."""
    limits = Budget(int(budget.chains), int(budget.draws), int(budget.max_evaluations), float(budget.max_seconds))
    prior_evals = (
        adapt_meta["density_evals"] + adapt_meta["gradient_evals"]
        + adapt_meta["hvp_evals"] + adapt_meta["hess_evals"]
    )
    remaining_s = max(0.0, limits.max_seconds - float(adapt_meta["seconds"]))
    started = logdensity._now()
    bundle_dir = Path(bundle_dir)
    tuning = deserialize_tree(bundle_dir / "tuning")
    if canonical_digest(tuning) != adapt_meta["tuning_sha256"]:
        raise ContractError("deserialized tuning digest mismatch before sampling")
    c0 = deserialize_tree(bundle_dir / "chain0")
    pos_2d = np.array(c0["position"], dtype=np.float64)
    aux = c0["aux"]
    walkers = int(adapt_meta["walkers"])
    is_1d = bool(adapt_meta["is_1d"])
    aux_spec = adapt_meta["aux_spec"]

    counter = Counter(
        model, limits.max_evaluations, remaining_s, started=started,
        capabilities=capabilities, prior_evaluations=prior_evals,
    )
    draws = np.empty((walkers, limits.draws, counter.dim), dtype=np.float64)
    singles, win_steps = choose_replay_steps(limits.draws, run_seed, count=replay_count)
    need_single = set(singles)
    win_start = win_steps[0] if win_steps else -1
    win_end = win_steps[-1] if win_steps else -1
    ckpt_dir = bundle_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    saved_singles: list[dict] = []
    win_start_saved = False
    win_out_digests: list[tuple[str, str]] = []

    with IsolatedTransitionSession(
        counter, submission_path, tuning, adapt_meta["tuning_sha256"],
        walkers, is_1d, aux_spec, lane=lane, timeout_s=max(10.0, remaining_s),
    ) as sess:
        for step in range(limits.draws):
            if step in need_single:
                serialize_tree({"position": pos_2d, "aux": aux}, ckpt_dir / f"in_{step}")
            if step == win_start and not win_start_saved:
                serialize_tree({"position": pos_2d, "aux": aux}, ckpt_dir / f"win_in_{step}")
                win_start_saved = True
            next_pos, next_aux = sess.step(pos_2d, aux, [int(run_seed), 1, int(step)])
            draws[:, step, :] = next_pos
            pos_2d, aux = next_pos, next_aux
            if step in need_single:
                saved_singles.append({
                    "step": step,
                    "pos_sha": canonical_digest(pos_2d),
                    "aux_sha": canonical_digest(aux),
                })
            if win_start <= step <= win_end:
                win_out_digests.append((canonical_digest(pos_2d), canonical_digest(aux)))
        sess.verify_tuning_frozen()

    seconds = logdensity._now() - started
    if counter.exceeded:
        raise BudgetExceeded(f"transition() went past the budget ({counter.exceeded}) and caught the error")
    if seconds > remaining_s:
        raise BudgetExceeded(
            f"run took {adapt_meta['seconds'] + seconds:.1f} s, exceeding {limits.max_seconds} s"
        )
    meta = {
        "density_evals": counter.density,
        "gradient_evals": counter.gradient,
        "hvp_evals": counter.hvp_calls,
        "hess_evals": counter.hess_calls,
        "seconds": seconds,
        "eval_s": counter.eval_s,
        "replay_singles": saved_singles,
        "replay_window": {"steps": win_steps, "digests": win_out_digests},
    }
    (bundle_dir / "sample_meta.json").write_text(json.dumps(meta, indent=1))
    return draws, meta


def run_replay_phase_isolated(
    model,
    submission_path: Path,
    run_seed: int,
    bundle_dir: Path,
    adapt_meta: dict,
    sample_meta: dict,
    *,
    capabilities: set[str] | None = None,
    lane: str | None = None,
) -> dict:
    """Isolated brokered equivalent of logdensity.run_replay_phase."""
    bundle_dir = Path(bundle_dir)
    tuning = deserialize_tree(bundle_dir / "tuning")
    if canonical_digest(tuning) != adapt_meta["tuning_sha256"]:
        raise ContractError("replay: tuning SHA-256 mismatch")
    walkers = int(adapt_meta["walkers"])
    is_1d = bool(adapt_meta["is_1d"])
    aux_spec = adapt_meta["aux_spec"]
    ckpt_dir = bundle_dir / "checkpoints"

    counter = Counter(model, max_evaluations=10**9, max_seconds=1e9, capabilities=capabilities)
    with IsolatedTransitionSession(
        counter, submission_path, tuning, adapt_meta["tuning_sha256"],
        walkers, is_1d, aux_spec, lane=lane, timeout_s=300.0,
    ) as sess:
        for rec in sample_meta["replay_singles"]:
            step = int(rec["step"])
            c_in = deserialize_tree(ckpt_dir / f"in_{step}")
            pos_in = np.array(c_in["position"], dtype=np.float64)
            aux_in = c_in["aux"]
            pos_out, aux_out = sess.step(pos_in, aux_in, [int(run_seed), 1, step], check_provenance=False)
            if canonical_digest(pos_out) != rec["pos_sha"] or canonical_digest(aux_out) != rec["aux_sha"]:
                raise ContractError(f"replay mismatch at step {step}: transition() is not deterministic/stateless")

        win = sample_meta.get("replay_window") or {}
        win_steps = win.get("steps") or []
        if win_steps:
            c_in = deserialize_tree(ckpt_dir / f"win_in_{win_steps[0]}")
            pos_cur = np.array(c_in["position"], dtype=np.float64)
            aux_cur = c_in["aux"]
            for step, (want_pos, want_aux) in zip(win_steps, win["digests"]):
                pos_cur, aux_cur = sess.step(pos_cur, aux_cur, [int(run_seed), 1, int(step)], check_provenance=False)
                if canonical_digest(pos_cur) != want_pos or canonical_digest(aux_cur) != want_aux:
                    raise ContractError(f"replay window mismatch at step {step}: hidden state across steps")
        sess.verify_tuning_frozen()

    return {"replayed_singles": len(sample_meta["replay_singles"]), "replayed_window": len(win_steps)}


def execute_run_isolated(
    model,
    submission_path: Path,
    budget: Budget,
    run_seed: int,
    work_dir: Path,
    *,
    capabilities: set[str] | None = None,
    replay_count: int = REPLAY_STEPS,
    lane: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Execute adapt -> sample -> replay in three separate out-of-process workers."""
    bundle_dir = Path(work_dir) / f"run_{run_seed}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    adapt_meta = run_adapt_phase_isolated(
        model, submission_path, budget, run_seed, bundle_dir, capabilities=capabilities, lane=lane,
    )
    draws, sample_meta = run_sample_phase_isolated(
        model, submission_path, budget, run_seed, bundle_dir, adapt_meta,
        capabilities=capabilities, replay_count=replay_count, lane=lane,
    )
    replay_meta = run_replay_phase_isolated(
        model, submission_path, run_seed, bundle_dir, adapt_meta, sample_meta,
        capabilities=capabilities, lane=lane,
    )
    total_s = adapt_meta["seconds"] + sample_meta["seconds"]
    eval_s = adapt_meta["eval_s"] + sample_meta["eval_s"]
    summary = {
        "adapt": adapt_meta,
        "sample": sample_meta,
        "replay": replay_meta,
        "walkers": adapt_meta["walkers"],
        "betas": adapt_meta["betas"],
        "tuning_sha256": adapt_meta["tuning_sha256"],
        "seconds": total_s,
        "eval_s": eval_s,
        "overhead_s": max(0.0, total_s - eval_s),
        "density_evals": adapt_meta["density_evals"] + sample_meta["density_evals"],
        "gradient_evals": adapt_meta["gradient_evals"] + sample_meta["gradient_evals"],
        "hvp_evals": adapt_meta["hvp_evals"] + sample_meta["hvp_evals"],
        "hess_evals": adapt_meta["hess_evals"] + sample_meta["hess_evals"],
    }
    return draws, summary


# ---------------------------------------------------------------------------
# Child worker entry point: python -m harness.ipc <mode> <fd> <dim> <block_shape_json>
# ---------------------------------------------------------------------------

def _make_remote_oracle(sock: socket.socket, dim: int,
                        block_shape: tuple[tuple[int, ...], ...] | None) -> Oracle:
    def _check_err(tag: int, payload: bytes) -> None:
        if tag == TAG_ERR_BUDGET:
            raise BudgetExceeded(payload.decode("utf-8"))
        if tag == TAG_ERR_CAP:
            raise CapabilityError(payload.decode("utf-8"))
        if tag == TAG_ERR_VALUE:
            raise ValueError(payload.decode("utf-8"))
        if tag == TAG_ERR_RUNTIME:
            raise RuntimeError(payload.decode("utf-8"))

    def _vec(theta) -> bytes:
        x = np.ascontiguousarray(theta, dtype=np.float64)
        if x.shape != (dim,):
            raise ValueError(f"theta must have shape ({dim},), got {x.shape}")
        return x.tobytes()

    def log_density(theta) -> float:
        _send_msg(sock, TAG_OP_DENSITY, _vec(theta))
        tag, payload = _recv_msg(sock)
        if tag == TAG_VAL_DENSITY:
            return float(struct.unpack("<d", payload)[0])
        _check_err(tag, payload)
        raise RuntimeError(f"unexpected oracle reply tag {tag}")

    def log_density_gradient(theta) -> tuple[float, np.ndarray]:
        _send_msg(sock, TAG_OP_GRAD, _vec(theta))
        tag, payload = _recv_msg(sock)
        if tag == TAG_VAL_GRAD:
            lp = float(struct.unpack_from("<d", payload, 0)[0])
            g = np.frombuffer(payload[8:], dtype=np.float64).copy()
            return lp, g
        _check_err(tag, payload)
        raise RuntimeError(f"unexpected oracle reply tag {tag}")

    def hvp(theta, v) -> tuple[float, np.ndarray]:
        _send_msg(sock, TAG_OP_HVP, _vec(theta) + _vec(v))
        tag, payload = _recv_msg(sock)
        if tag == TAG_VAL_GRAD:
            lp = float(struct.unpack_from("<d", payload, 0)[0])
            hv = np.frombuffer(payload[8:], dtype=np.float64).copy()
            return lp, hv
        _check_err(tag, payload)
        raise RuntimeError(f"unexpected oracle reply tag {tag}")

    def hess(theta) -> tuple[float, np.ndarray, np.ndarray]:
        _send_msg(sock, TAG_OP_HESS, _vec(theta))
        tag, payload = _recv_msg(sock)
        if tag == TAG_VAL_HESS:
            lp = float(struct.unpack_from("<d", payload, 0)[0])
            g_end = 8 + dim * 8
            g = np.frombuffer(payload[8:g_end], dtype=np.float64).copy()
            h = np.frombuffer(payload[g_end:], dtype=np.float64).reshape((dim, dim)).copy()
            return lp, g, h
        _check_err(tag, payload)
        raise RuntimeError(f"unexpected oracle reply tag {tag}")

    return Oracle(dim, log_density, log_density_gradient, hvp, hess, block_shape)


def _worker_main(argv: list[str]) -> int:
    mode = argv[0]
    fd = int(argv[1])
    dim = int(argv[2])
    raw_bs = json.loads(argv[3])
    block_shape = tuple(tuple(x) for x in raw_bs) if raw_bs is not None else None
    sock = socket.socket(fileno=fd)
    oracle = _make_remote_oracle(sock, dim, block_shape)

    try:
        if mode == "adapt":
            tag, payload = _recv_msg(sock)
            if tag != TAG_CMD_ADAPT:
                return 2
            req = decode_canonical(payload)
            try:
                sub_path = Path(req["submission"])
                lane = req["lane"]
                b = req["budget"]
                budget = Budget(int(b[0]), int(b[1]), int(b[2]), float(b[3]))
                rng = np.random.default_rng(np.random.SeedSequence(req["seed_seq"]))
                sandbox.confine_worker(sub_path)
                if lane == "A1":
                    raw = judge.run_a1_adapt(sub_path, oracle, budget, rng)
                elif lane == "A4":
                    raw = judge.run_a4_adapt(sub_path, oracle, budget, rng)
                else:
                    adapt_fn, _ = logdensity.load_submission(sub_path)
                    raw = adapt_fn(oracle, budget, rng)
                if not isinstance(raw, tuple) or len(raw) != 4:
                    raise ContractError("adapt() must return a 4-tuple (tuning, chain0, betas, aux_spec)")
                tuning, chain0_raw, betas_raw, aux_spec_raw = raw
                if isinstance(chain0_raw, Chain):
                    pos_raw, aux_raw = chain0_raw.position, chain0_raw.aux
                elif isinstance(chain0_raw, tuple) and len(chain0_raw) == 2:
                    pos_raw, aux_raw = chain0_raw
                else:
                    raise ContractError("chain0 must be a Chain(position, aux) or (position, aux) 2-tuple")
                out = {
                    "tuning": tuning,
                    "position": np.asarray(pos_raw, dtype=np.float64),
                    "aux": aux_raw,
                    "betas": np.asarray(betas_raw, dtype=np.float64).tolist(),
                    "aux_spec": aux_spec_raw,
                }
                _send_msg(sock, TAG_RESP_OK, encode_canonical(out))
            except Exception as e:
                _send_msg(sock, TAG_RESP_ERR, encode_canonical((type(e).__name__, str(e))))
            return 0

        if mode == "transition":
            transition_fn = None
            tuning = None
            while True:
                try:
                    tag, payload = _recv_msg(sock)
                except EOFError:
                    return 0
                if tag == TAG_CMD_QUIT:
                    return 0
                if tag == TAG_CMD_INIT_TRANSITION:
                    try:
                        req = decode_canonical(payload)
                        lane = req["lane"]
                        tuning = req["tuning"]
                        if lane == "A1":
                            # Never even import submission.py inside the A1 transition worker!
                            transition_fn = judge.a1_transition
                        elif lane == "A4":
                            sub_path = Path(req["submission"])
                            sandbox.confine_worker(sub_path)
                            transition_fn = judge.make_a4_transition(sub_path)
                        else:
                            sub_path = Path(req["submission"])
                            sandbox.confine_worker(sub_path)
                            _, transition_fn = logdensity.load_submission(sub_path)
                        _send_msg(sock, TAG_RESP_OK, encode_canonical(True))
                    except Exception as e:
                        _send_msg(sock, TAG_RESP_ERR, encode_canonical((type(e).__name__, str(e))))
                        return 0
                elif tag == TAG_CMD_STEP:
                    try:
                        req = decode_canonical(payload)
                        rng = np.random.default_rng(np.random.SeedSequence(req["seed_seq"]))
                        chain_in = Chain(np.asarray(req["position"], dtype=np.float64), req["aux"])
                        raw_next = transition_fn(tuning, chain_in, oracle, rng)
                        if isinstance(raw_next, Chain):
                            pos_raw, aux_raw = raw_next.position, raw_next.aux
                        elif isinstance(raw_next, tuple) and len(raw_next) == 2:
                            pos_raw, aux_raw = raw_next
                        else:
                            raise ContractError("transition() must return Chain or (position, aux) 2-tuple")
                        out = {"position": np.asarray(pos_raw, dtype=np.float64), "aux": aux_raw}
                        _send_msg(sock, TAG_RESP_OK, encode_canonical(out))
                    except Exception as e:
                        _send_msg(sock, TAG_RESP_ERR, encode_canonical((type(e).__name__, str(e))))
                elif tag == TAG_CMD_CHECK_TUNING:
                    try:
                        _send_msg(sock, TAG_RESP_OK, encode_canonical(canonical_digest(tuning)))
                    except Exception as e:
                        _send_msg(sock, TAG_RESP_ERR, encode_canonical((type(e).__name__, str(e))))
                else:
                    return 2
        return 2
    finally:
        sock.close()


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
