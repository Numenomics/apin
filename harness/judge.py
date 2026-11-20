"""Read-only judge module (Karpathy 3-file contract) & Lane A1 fixed HMC transition kernel.

Contract (`autoresearch/lanes/A1/program.md` & `ARCHITECTURE.md` §2.2):
  - `harness/judge.py` is read-only to autoresearch agents; its digest (together with
    `logdensity.py`, `ipc.py`, `shapes.py`, `invariance.py`, and `autoresearch.py`) is
    recorded on every trial row in `ledger.jsonl`.
  - Lane A1 submissions (`autoresearch/lanes/A1/submission.py`) implement ONLY:

        def adapt_metric(oracle, budget, rng) -> tuple[metric_tuning, x0]

    They may NOT define `transition()` or `adapt()`. The harness owns the verified,
    reversible HMC transition kernel (`a1_transition`), so A1 wins are strictly attributable
    to metric estimation and warmup efficiency rather than kernel or transition changes.

Supported `metric_tuning` schemas for `a1_transition`:
  1. Diagonal metric:
       {"eps": float, "inv_m": 1D ndarray (dim,)}
  2. Dense SPD Cholesky metric (M^{-1} = L @ L.T):
       {"eps": float, "inv_m_chol": 2D lower-triangular ndarray (dim, dim)}
  3. Low-rank + diagonal metric (M^{-1} = D^{1/2} (I + U diag(eigs - 1) U^T) D^{1/2}):
       {"eps": float, "inv_m_diag": 1D ndarray (dim,),
        "low_rank_u": 2D orthonormal ndarray (dim, k),
        "low_rank_eigs": 1D positive ndarray (k,)}
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.linalg import solve_triangular

from . import logdensity
from .logdensity import ContractError

A1_PATH_LENGTH = 2.0
A1_MAX_STEPS = 256
A1_EPS_MIN = 1e-8
A1_EPS_MAX = 10.0


def judge_digest() -> str:
    """Deterministic SHA-256 prefix over all read-only judge modules."""
    from . import autoresearch, invariance, ipc, shapes

    h = hashlib.sha256()
    for mod in (logdensity, ipc, shapes, invariance, sys.modules[__name__], autoresearch):
        h.update(Path(mod.__file__).read_bytes())
    return h.hexdigest()[:16]


def validate_a1_metric_tuning(metric_tuning: Any, dim: int) -> dict[str, Any]:
    """Validate and normalize Lane A1 `metric_tuning` returned by `adapt_metric()`."""
    if not isinstance(metric_tuning, dict):
        raise ContractError(f"Lane A1 metric_tuning must be a dict, got {type(metric_tuning).__name__}")
    if "eps" not in metric_tuning:
        raise ContractError("Lane A1 metric_tuning must include positive finite 'eps'")
    eps = float(metric_tuning["eps"])
    if not math.isfinite(eps) or not (A1_EPS_MIN <= eps <= A1_EPS_MAX):
        raise ContractError(f"Lane A1 'eps' must be finite in [{A1_EPS_MIN}, {A1_EPS_MAX}], got {eps!r}")

    has_diag = "inv_m" in metric_tuning
    has_chol = "inv_m_chol" in metric_tuning
    has_lr = "inv_m_diag" in metric_tuning
    modes = int(has_diag) + int(has_chol) + int(has_lr)
    if modes != 1:
        raise ContractError(
            "Lane A1 metric_tuning must specify exactly one of: "
            "'inv_m' (diagonal), 'inv_m_chol' (dense Cholesky), or 'inv_m_diag' (+ optional low-rank)"
        )

    out: dict[str, Any] = {"eps": eps}
    if has_diag:
        inv_m = np.asarray(metric_tuning["inv_m"], dtype=np.float64)
        if inv_m.shape != (dim,) or not np.all(np.isfinite(inv_m)) or not np.all(inv_m > 0.0):
            raise ContractError(f"Lane A1 'inv_m' must be strictly positive finite shape ({dim},)")
        out["kind"] = "diag"
        out["inv_m"] = inv_m
    elif has_chol:
        chol = np.asarray(metric_tuning["inv_m_chol"], dtype=np.float64)
        if chol.shape != (dim, dim) or not np.all(np.isfinite(chol)):
            raise ContractError(f"Lane A1 'inv_m_chol' must be finite shape ({dim}, {dim})")
        if not np.allclose(np.triu(chol, k=1), 0.0, atol=1e-10):
            raise ContractError("Lane A1 'inv_m_chol' must be lower-triangular")
        if not np.all(np.diag(chol) > 0.0):
            raise ContractError("Lane A1 'inv_m_chol' must have strictly positive diagonal")
        out["kind"] = "chol"
        out["inv_m_chol"] = np.tril(chol)
    else:
        inv_d = np.asarray(metric_tuning["inv_m_diag"], dtype=np.float64)
        if inv_d.shape != (dim,) or not np.all(np.isfinite(inv_d)) or not np.all(inv_d > 0.0):
            raise ContractError(f"Lane A1 'inv_m_diag' must be strictly positive finite shape ({dim},)")
        u = metric_tuning.get("low_rank_u")
        eigs = metric_tuning.get("low_rank_eigs")
        if (u is None) != (eigs is None):
            raise ContractError("Lane A1 low-rank metric requires both 'low_rank_u' and 'low_rank_eigs'")
        if u is None:
            out["kind"] = "diag"
            out["inv_m"] = inv_d
        else:
            u_arr = np.asarray(u, dtype=np.float64)
            e_arr = np.asarray(eigs, dtype=np.float64)
            if u_arr.ndim != 2 or u_arr.shape[0] != dim or u_arr.shape[1] < 1 or u_arr.shape[1] > dim:
                raise ContractError(f"Lane A1 'low_rank_u' must have shape ({dim}, k) with 1 <= k <= {dim}")
            if e_arr.shape != (u_arr.shape[1],) or not np.all(np.isfinite(e_arr)) or not np.all(e_arr > 0.0):
                raise ContractError("Lane A1 'low_rank_eigs' must be strictly positive finite shape (k,)")
            if not np.all(np.isfinite(u_arr)):
                raise ContractError("Lane A1 'low_rank_u' must be finite")
            gram = u_arr.T @ u_arr
            if not np.allclose(gram, np.eye(u_arr.shape[1]), atol=1e-5, rtol=1e-5):
                raise ContractError("Lane A1 'low_rank_u' columns must be orthonormal (U^T U = I)")
            out["kind"] = "low_rank"
            out["inv_m_diag"] = inv_d
            out["low_rank_u"] = u_arr
            out["low_rank_eigs"] = e_arr
    return out


def _a1_steps(eps: float, rng: np.random.Generator) -> int:
    n = max(1, min(A1_MAX_STEPS, math.ceil(A1_PATH_LENGTH / eps)))
    return int(rng.integers(max(1, n // 2), n + n // 2 + 1))


def _sample_momentum_and_apply(tuning: dict[str, Any], rng: np.random.Generator):
    """Return (p, apply_inv_m, kinetic) for exact p ~ N(0, M) and v = M^{-1} p."""
    kind = tuning["kind"]
    if kind == "diag":
        inv_m = tuning["inv_m"]
        p = rng.standard_normal(inv_m.size) / np.sqrt(inv_m)

        def apply_inv_m(mom: np.ndarray) -> np.ndarray:
            return inv_m * mom

        def kinetic(mom: np.ndarray) -> float:
            return 0.5 * float(np.sum(inv_m * mom * mom))

        return p, apply_inv_m, kinetic

    if kind == "chol":
        chol = tuning["inv_m_chol"]
        z = rng.standard_normal(chol.shape[0])
        # M^{-1} = L L^T => L^T p = z gives Cov(p) = (L L^T)^{-1} = M
        p = solve_triangular(chol, z, lower=True, trans="T")

        def apply_inv_m(mom: np.ndarray) -> np.ndarray:
            return chol @ (chol.T @ mom)

        def kinetic(mom: np.ndarray) -> float:
            w = chol.T @ mom
            return 0.5 * float(w @ w)

        return p, apply_inv_m, kinetic

    # low_rank: M^{-1} = D^{1/2} (I + U diag(eigs - 1) U^T) D^{1/2}
    inv_d = tuning["inv_m_diag"]
    sqrt_d = np.sqrt(inv_d)
    u = tuning["low_rank_u"]
    eigs = tuning["low_rank_eigs"]
    z = rng.standard_normal(inv_d.size)
    inv_sqrt_delta = (1.0 / np.sqrt(eigs)) - 1.0
    p = (z + u @ (inv_sqrt_delta * (u.T @ z))) / sqrt_d
    eig_delta = eigs - 1.0

    def apply_inv_m(mom: np.ndarray) -> np.ndarray:
        w = sqrt_d * mom
        return sqrt_d * (w + u @ (eig_delta * (u.T @ w)))

    def kinetic(mom: np.ndarray) -> float:
        return 0.5 * float(mom @ apply_inv_m(mom))

    return p, apply_inv_m, kinetic


def a1_transition(tuning: dict[str, Any], chain: logdensity.Chain, oracle: logdensity.Oracle,
                  rng: np.random.Generator) -> tuple[np.ndarray, None]:
    """Harness-owned verified reversible HMC transition kernel for Lane A1."""
    x = chain.position
    lp, g = oracle.log_density_gradient(x)
    if not (math.isfinite(lp) and np.all(np.isfinite(g))):
        return x, None
    eps = float(tuning["eps"])
    n = _a1_steps(eps, rng)
    p, apply_inv_m, kinetic = _sample_momentum_and_apply(tuning, rng)
    h0 = lp - kinetic(p)
    x1, g1, lp1 = x, g, lp
    p1 = p + 0.5 * eps * g1
    for i in range(n):
        x1 = x1 + eps * apply_inv_m(p1)
        lp1, g1 = oracle.log_density_gradient(x1)
        if not (math.isfinite(lp1) and np.all(np.isfinite(g1))):
            return x, None
        if i < n - 1:
            p1 = p1 + eps * g1
    p1 = p1 + 0.5 * eps * g1
    h1 = lp1 - kinetic(p1)
    accept = math.exp(min(0.0, h1 - h0)) if math.isfinite(h1) else 0.0
    if rng.uniform() < accept:
        return x1, None
    return x, None


def load_a1_submission(path: Path) -> Callable:
    """Load a Lane A1 submission; enforce narrow `adapt_metric` seam and forbid `transition`/`adapt`."""
    path = Path(path)
    logdensity.check_imports(path.read_text())
    name = "apin_a1_submission"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "transition") or hasattr(mod, "adapt"):
        raise ContractError(
            f"Lane A1 submission {path.name} must define ONLY adapt_metric(oracle, budget, rng) "
            "and may not define transition() or adapt()"
        )
    fn = getattr(mod, "adapt_metric", None)
    if not callable(fn):
        raise AttributeError(
            f"Lane A1 submission {path} must define adapt_metric(oracle, budget, rng) -> (metric_tuning, x0)"
        )
    return fn


def run_a1_adapt(path: Path, oracle: logdensity.Oracle, budget: logdensity.Budget,
                 rng: np.random.Generator) -> tuple[dict[str, Any], tuple[np.ndarray, None], list[float], str]:
    """Execute Lane A1 `adapt_metric()` and wrap into standard kernel-v2 4-tuple."""
    adapt_metric_fn = load_a1_submission(path)
    raw = adapt_metric_fn(oracle, budget, rng)
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise ContractError("Lane A1 adapt_metric() must return a 2-tuple (metric_tuning, x0)")
    metric_tuning, x0_raw = raw
    tuning = validate_a1_metric_tuning(metric_tuning, oracle.dim)
    x0 = np.array(x0_raw, dtype=np.float64, copy=True)
    if x0.shape != (oracle.dim,) or not np.all(np.isfinite(x0)):
        raise ContractError(f"Lane A1 x0 must be finite 1D array of shape ({oracle.dim},), got {x0.shape}")
    return tuning, (x0, None), [1.0], "none"


def resolve_lane_submission(path: Path, lane: str | None) -> tuple[Callable, Callable]:
    """Resolve a submission path into (adapt, transition), locking Lane A1 to `a1_transition`."""
    if lane == "A1":
        p = Path(path)
        return (lambda oracle, budget, rng: run_a1_adapt(p, oracle, budget, rng)), a1_transition
    return logdensity.load_submission(Path(path))
