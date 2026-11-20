"""The Slice 2 kernel ABI: Markov chain samplers behind adapt() and transition().

A submission is one Python file defining two entry points:

    def adapt(oracle: Oracle, budget: Budget, rng: np.random.Generator)
        -> tuple[tuning, chain0, betas, aux_spec]

    def transition(tuning, chain: Chain, oracle: Oracle, rng: np.random.Generator)
        -> Chain | tuple[np.ndarray, Any]

Contract highlights (BENCH.md, PROTOCOL.md, apin/ARCHITECTURE.md):
  - `tuning` is a canonical plain tree (dict/list/tuple/str/int/float/bool/None/ndarray),
    serialized canonically (no pickle) and SHA-256 hashed before and after sampling.
  - `chain = Chain(position, aux)` separates parameter positions from typed auxiliary Markov
    state (e.g. persistent momentum, unit-sphere velocity, NRPT DEO swap parity).
    Single-walker kernels may use 1D `position` of shape `(dim,)` with `betas = [1.0]`;
    K-walker / tempered kernels use 2D `(K, dim)` with `betas` of length K in `(0, 1]`
    and at least one `beta == 1.0`.
  - `aux_spec` declares the stationary law of `aux` from a finite harness vocabulary
    (`"none"`, `"normal"`, `"unit_sphere"`, `"uniform_discrete"`, `"product"`).
  - `oracle` is input-blind and explicitly metered; higher-order methods (`hvp`, `hess`)
    and `block_shape` metadata are disabled unless explicitly enabled as track capabilities.
  - Process & RNG isolation: each independent run executes `adapt()` in one process,
    serializes `(tuning, chain0, betas, aux_spec)`, then runs `transition()` in a clean
    child process with derived per-step RNGs `(run_seed, 1, step_index)`, per-step finite
    provenance checking, frozen-`tuning` SHA-256 verification, and unbilled bit-exact
    replay of ~20 steps (including a short contiguous window) in a third fresh process.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

ALLOWED_IMPORTS = {
    "numpy", "scipy", "math", "__future__", "typing", "dataclasses",
    "functools", "itertools", "collections",
}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "breakpoint",
    "input", "globals", "vars",
}
ALLOWED_AUX_KINDS = ("none", "normal", "unit_sphere", "uniform_discrete", "product")
MAX_TUNING_BYTES = 16 * 1024 * 1024  # 16 MiB serialized tuning cap
REPLAY_STEPS = 20
REPLAY_WINDOW = 3


class BudgetExceeded(RuntimeError):
    pass


class CapabilityError(RuntimeError):
    pass


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class Budget:
    """Per-run sampling budget handed to adapt() and visible to the submission."""

    chains: int  # total independent runs in the rep (informational)
    draws: int  # post-warmup transition() calls for this run
    max_evaluations: int  # total oracle calls for this run (adapt + transition combined)
    max_seconds: float  # wall-clock limit for this run (import + adapt + transition)


@dataclass(frozen=True)
class Chain:
    """Evolving Markov state: parameter position(s) plus typed auxiliary state."""

    position: np.ndarray
    aux: Any = None


class Oracle:
    """Input-blind, explicitly metered target oracle.

    Exposes only `dim`, `block_shape` (None unless enabled), and plain bound-free callables
    for `log_density`, `log_density_gradient`, `hvp`, and `hess`.
    """

    __slots__ = ("dim", "block_shape", "log_density", "log_density_gradient", "hvp", "hess")

    def __init__(self, dim: int, log_density: Callable, log_density_gradient: Callable,
                 hvp: Callable, hess: Callable, block_shape: tuple[tuple[int, ...], ...] | None = None):
        object.__setattr__(self, "dim", int(dim))
        object.__setattr__(self, "block_shape", block_shape)
        object.__setattr__(self, "log_density", log_density)
        object.__setattr__(self, "log_density_gradient", log_density_gradient)
        object.__setattr__(self, "hvp", hvp)
        object.__setattr__(self, "hess", hess)

    def __setattr__(self, name, value):
        raise AttributeError("Oracle is immutable")


_now = time.perf_counter  # the harness's clock; tests replace it


# ---------------------------------------------------------------------------
# Canonical tree validation, hashing, and pickle-free serialization
# ---------------------------------------------------------------------------

def _hash_canonical(obj: Any, h: "hashlib._Hash", arrays: list[np.ndarray] | None = None) -> Any:
    """Walk a canonical tree, feeding exact typed bytes into `h` and optionally encoding a JSON manifest."""
    if obj is None:
        h.update(b"N;")
        return None
    if type(obj) is bool:
        h.update(b"B1;" if obj else b"B0;")
        return obj
    if type(obj) is int:
        raw = str(obj).encode("ascii")
        h.update(b"I" + raw + b";")
        return obj
    if type(obj) is float:
        if not math.isfinite(obj):
            raise ContractError(f"non-finite float {obj!r} in canonical state")
        # Normalize -0.0 to +0.0 for canonical stability
        val = 0.0 if obj == 0.0 else obj
        h.update(b"F" + struct.pack("<d", val))
        return val
    if type(obj) is str:
        raw = obj.encode("utf-8")
        h.update(b"S" + struct.pack("<Q", len(raw)) + raw)
        return obj
    if isinstance(obj, np.generic):
        return _hash_canonical(obj.item(), h, arrays)
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            raise ContractError("non-finite float array in canonical state")
        if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_):
            raise ContractError(f"unsupported ndarray dtype {arr.dtype!r} in canonical state")
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.copy()
            arr[arr == 0.0] = 0.0
        dt = arr.dtype.str.encode("ascii")
        shape_b = json.dumps(list(arr.shape)).encode("ascii")
        raw = arr.tobytes()
        h.update(b"A" + struct.pack("<QQQ", len(dt), len(shape_b), len(raw)) + dt + shape_b + raw)
        if arrays is not None:
            idx = len(arrays)
            arrays.append(arr)
            return {"__ndarray__": idx}
        return None
    if type(obj) is list:
        h.update(b"L" + struct.pack("<Q", len(obj)))
        encoded = [_hash_canonical(v, h, arrays) for v in obj]
        return {"__list__": encoded} if arrays is not None else None
    if type(obj) is tuple:
        h.update(b"T" + struct.pack("<Q", len(obj)))
        encoded = [_hash_canonical(v, h, arrays) for v in obj]
        return {"__tuple__": encoded} if arrays is not None else None
    if type(obj) is dict:
        for k in obj:
            if type(k) is not str:
                raise ContractError(f"canonical dict keys must be str, got {type(k).__name__}")
        items = sorted(obj.items(), key=lambda kv: kv[0])
        h.update(b"D" + struct.pack("<Q", len(items)))
        out = {}
        for k, v in items:
            kb = k.encode("utf-8")
            h.update(struct.pack("<Q", len(kb)) + kb)
            out[k] = _hash_canonical(v, h, arrays)
        return {"__dict__": out} if arrays is not None else None
    raise ContractError(f"unsupported type {type(obj).__name__!r} in canonical tuning/aux state")


def _decode_canonical(node: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(node, dict):
        if "__ndarray__" in node:
            return arrays[f"a{node['__ndarray__']}"].copy()
        if "__list__" in node:
            return [_decode_canonical(v, arrays) for v in node["__list__"]]
        if "__tuple__" in node:
            return tuple(_decode_canonical(v, arrays) for v in node["__tuple__"])
        if "__dict__" in node:
            return {k: _decode_canonical(v, arrays) for k, v in node["__dict__"].items()}
    return node


def canonical_digest(obj: Any) -> str:
    """Deterministic SHA-256 hex digest of a canonical tree."""
    h = hashlib.sha256()
    _hash_canonical(obj, h)
    return h.hexdigest()


def serialize_tree(obj: Any, path_prefix: Path, max_bytes: int | None = None) -> tuple[str, int]:
    """Serialize canonical tree to `<prefix>.json` + `<prefix>.npz` (no pickle). Returns (sha256, bytes)."""
    h = hashlib.sha256()
    arrays: list[np.ndarray] = []
    manifest = _hash_canonical(obj, h, arrays)
    meta_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    arr_bytes = sum(a.nbytes for a in arrays)
    total_bytes = len(meta_bytes) + arr_bytes
    if max_bytes is not None and total_bytes > max_bytes:
        raise ContractError(f"serialized state is {total_bytes} bytes, exceeding cap {max_bytes}")
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    path_prefix.with_suffix(".json").write_bytes(meta_bytes)
    np.savez(path_prefix.with_suffix(".npz"), **{f"a{i}": a for i, a in enumerate(arrays)})
    return h.hexdigest(), total_bytes


def deserialize_tree(path_prefix: Path) -> Any:
    """Load canonical tree serialized by serialize_tree()."""
    manifest = json.loads(path_prefix.with_suffix(".json").read_text())
    with np.load(path_prefix.with_suffix(".npz"), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return _decode_canonical(manifest, arrays)


# ---------------------------------------------------------------------------
# Chain, betas, and aux_spec validation
# ---------------------------------------------------------------------------

def normalize_chain(raw: Any, dim: int, expected_walkers: int | None = None,
                    expected_is_1d: bool | None = None) -> tuple[np.ndarray, Any, bool]:
    """Validate and normalize a Chain or (position, aux) pair -> (pos_2d, aux, is_1d)."""
    if isinstance(raw, Chain):
        pos_raw, aux = raw.position, raw.aux
    elif isinstance(raw, tuple) and len(raw) == 2:
        pos_raw, aux = raw
    else:
        raise ContractError("chain must be a Chain(position, aux) or (position, aux) 2-tuple")
    pos = np.array(pos_raw, dtype=float, copy=True)
    if pos.ndim == 1:
        if pos.shape != (dim,):
            raise ContractError(f"1D chain position must have shape ({dim},), got {pos.shape}")
        is_1d = True
        pos_2d = pos[None, :]
    elif pos.ndim == 2:
        if pos.shape[1] != dim or pos.shape[0] < 1:
            raise ContractError(f"2D chain position must have shape (K>=1, {dim}), got {pos.shape}")
        is_1d = False
        pos_2d = pos
    else:
        raise ContractError(f"chain position must be 1D or 2D, got shape {pos.shape}")
    if not np.all(np.isfinite(pos_2d)):
        raise ContractError("chain position contains non-finite values")
    if expected_walkers is not None and pos_2d.shape[0] != expected_walkers:
        raise ContractError(f"chain walker count changed from {expected_walkers} to {pos_2d.shape[0]}")
    if expected_is_1d is not None and is_1d != expected_is_1d:
        raise ContractError("chain position dimensionality (1D vs 2D) changed between steps")
    # Ensure aux is canonical
    canonical_digest(aux)
    return pos_2d, aux, is_1d


def validate_betas(betas: Any, walkers: int) -> np.ndarray:
    """Validate inverse temperatures: 1D length-K floats in (0, 1] with at least one == 1.0."""
    arr = np.array(betas, dtype=float)
    if arr.ndim != 1 or len(arr) != walkers:
        raise ContractError(f"betas must be 1D of length {walkers}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)) or not np.all((arr > 0.0) & (arr <= 1.0)):
        raise ContractError(f"all betas must lie in (0, 1], got {arr.tolist()}")
    if not np.any(arr == 1.0):
        raise ContractError(f"at least one beta must equal 1.0, got {arr.tolist()}")
    return arr


def normalize_aux_spec(spec: Any) -> dict:
    """Normalize shorthand aux_spec ('none') to canonical dict and validate vocabulary."""
    if spec is None or spec == "none":
        return {"kind": "none"}
    if not isinstance(spec, dict) or "kind" not in spec:
        raise ContractError(f"aux_spec must be 'none' or a dict with 'kind', got {spec!r}")
    kind = spec["kind"]
    if kind not in ALLOWED_AUX_KINDS:
        raise ContractError(f"unknown aux_spec kind {kind!r}; allowed: {', '.join(ALLOWED_AUX_KINDS)}")
    # Ensure spec itself is canonical
    canonical_digest(spec)
    return spec


def validate_aux(aux_spec: dict, aux: Any, walkers: int, dim: int, is_1d: bool) -> None:
    """Verify `aux` value conforms to declared `aux_spec`."""
    kind = aux_spec["kind"]
    if kind == "none":
        if aux is not None:
            raise ContractError("aux must be None when aux_spec is 'none'")
        return
    if kind == "normal":
        arr = np.asarray(aux, dtype=float)
        want = (dim,) if is_1d else (walkers, dim)
        if arr.shape != want or not np.all(np.isfinite(arr)):
            raise ContractError(f"normal aux must be finite array of shape {want}, got {arr.shape}")
        return
    if kind == "unit_sphere":
        arr = np.asarray(aux, dtype=float)
        want_d = int(aux_spec.get("dim", dim))
        want = (want_d,) if is_1d else (walkers, want_d)
        if arr.shape != want or not np.all(np.isfinite(arr)):
            raise ContractError(f"unit_sphere aux must have shape {want}, got {arr.shape}")
        norms = np.linalg.norm(arr, axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
            raise ContractError(f"unit_sphere aux must have unit norm, got norms {norms}")
        return
    if kind == "uniform_discrete":
        k = int(aux_spec.get("k", 0))
        if k < 2:
            raise ContractError(f"uniform_discrete aux_spec requires integer k >= 2, got {k!r}")
        if type(aux) is not int or not (0 <= aux < k):
            raise ContractError(f"uniform_discrete(k={k}) aux must be int in [0, {k}), got {aux!r}")
        return
    if kind == "product":
        parts = aux_spec.get("parts")
        if not isinstance(parts, (list, tuple)) or not parts:
            raise ContractError("product aux_spec requires non-empty 'parts' list")
        if not isinstance(aux, (list, tuple)) or len(aux) != len(parts):
            raise ContractError(f"product aux must have {len(parts)} elements")
        for sub_spec, sub_aux in zip(parts, aux):
            validate_aux(normalize_aux_spec(sub_spec), sub_aux, walkers, dim, is_1d)
        return


# ---------------------------------------------------------------------------
# Metered Oracle Counter
# ---------------------------------------------------------------------------

def _row_bytes(row: np.ndarray) -> bytes:
    x = np.ascontiguousarray(row, dtype=np.float64).copy()
    x[x == 0.0] = 0.0
    return x.tobytes()


class Counter:
    """Wraps a BridgeStan model, enforces limits & capabilities, tracks per-step finite provenance."""

    def __init__(self, model, max_evaluations: int, max_seconds: float, started: float | None = None,
                 capabilities: set[str] | None = None, block_shape: tuple[tuple[int, ...], ...] | None = None,
                 prior_evaluations: int = 0):
        self._model = model
        self.dim = int(model.param_unc_num())
        self.max_evaluations = int(max_evaluations)
        self.max_seconds = float(max_seconds)
        self.prior_evaluations = int(prior_evaluations)
        self.capabilities = set(capabilities or ())
        self._block_shape = block_shape if "block_shape" in self.capabilities else None
        self.density = self.gradient = self.hvp_calls = self.hess_calls = 0
        self.eval_s = 0.0
        self.exceeded = ""
        self.t0 = _now() if started is None else started
        self.step_seen: set[bytes] = set()

    @property
    def total_calls(self) -> int:
        return self.density + self.gradient + self.hvp_calls + self.hess_calls

    def reset_step(self) -> None:
        self.step_seen.clear()

    def oracle(self) -> Oracle:
        """Return an input-blind Oracle whose callables expose no Counter or model attributes."""
        _dens = self.log_density
        _grad = self.log_density_gradient
        _hvp = self.hvp
        _hess = self.hess

        def log_density(theta) -> float:
            return _dens(theta)

        def log_density_gradient(theta) -> tuple[float, np.ndarray]:
            return _grad(theta)

        def hvp(theta, v) -> tuple[float, np.ndarray]:
            return _hvp(theta, v)

        def hess(theta) -> tuple[float, np.ndarray, np.ndarray]:
            return _hess(theta)

        return Oracle(self.dim, log_density, log_density_gradient, hvp, hess, self._block_shape)

    def _charge(self) -> None:
        if self.exceeded:
            raise BudgetExceeded(self.exceeded)
        if self.prior_evaluations + self.total_calls >= self.max_evaluations:
            self.exceeded = f"more than {self.max_evaluations} evaluations"
        elif _now() - self.t0 > self.max_seconds:
            self.exceeded = f"more than {self.max_seconds} s"
        if self.exceeded:
            raise BudgetExceeded(self.exceeded)

    def _point(self, theta) -> np.ndarray:
        x = np.array(theta, dtype=np.float64, copy=True)
        if x.shape != (self.dim,):
            raise ValueError(f"theta must have shape ({self.dim},), got {x.shape}")
        return x

    def _record(self, x: np.ndarray, lp: float) -> None:
        if math.isfinite(lp):
            self.step_seen.add(_row_bytes(x))

    def log_density(self, theta) -> float:
        self._charge()
        self.density += 1
        x = self._point(theta)
        t_start = _now()
        try:
            lp = float(self._model.log_density(x, propto=False, jacobian=True))
        except RuntimeError:
            self.eval_s += _now() - t_start
            return -math.inf
        self.eval_s += _now() - t_start
        self._record(x, lp)
        return lp

    def log_density_gradient(self, theta) -> tuple[float, np.ndarray]:
        self._charge()
        self.gradient += 1
        x = self._point(theta)
        t_start = _now()
        try:
            lp, grad = self._model.log_density_gradient(x, propto=False, jacobian=True)
        except RuntimeError:
            self.eval_s += _now() - t_start
            return -math.inf, np.full(self.dim, math.nan)
        self.eval_s += _now() - t_start
        lp = float(lp)
        self._record(x, lp)
        return lp, np.array(grad, dtype=np.float64)

    def hvp(self, theta, v) -> tuple[float, np.ndarray]:
        if "hvp" not in self.capabilities:
            raise CapabilityError("oracle.hvp is disabled; declare and price capability 'hvp' first")
        self._charge()
        self.hvp_calls += 1
        x = self._point(theta)
        vec = self._point(v)
        t_start = _now()
        try:
            lp, hv = self._model.log_density_hessian_vector_product(x, vec, propto=False, jacobian=True)
        except RuntimeError:
            self.eval_s += _now() - t_start
            return -math.inf, np.full(self.dim, math.nan)
        self.eval_s += _now() - t_start
        lp = float(lp)
        self._record(x, lp)
        return lp, np.array(hv, dtype=np.float64)

    def hess(self, theta) -> tuple[float, np.ndarray, np.ndarray]:
        if "hess" not in self.capabilities:
            raise CapabilityError("oracle.hess is disabled; declare and price capability 'hess' first")
        self._charge()
        self.hess_calls += 1
        x = self._point(theta)
        t_start = _now()
        try:
            lp, g, h = self._model.log_density_hessian(x, propto=False, jacobian=True)
        except RuntimeError:
            self.eval_s += _now() - t_start
            return -math.inf, np.full(self.dim, math.nan), np.full((self.dim, self.dim), math.nan)
        self.eval_s += _now() - t_start
        lp = float(lp)
        self._record(x, lp)
        return lp, np.array(g, dtype=np.float64), np.array(h, dtype=np.float64)

    def check_step_provenance(self, prev_pos: np.ndarray, next_pos: np.ndarray) -> None:
        """Every walker position in next_pos not already in prev_pos must have been evaluated finitely in this call."""
        prev_keys = {_row_bytes(prev_pos[w]) for w in range(prev_pos.shape[0])}
        for w in range(next_pos.shape[0]):
            key = _row_bytes(next_pos[w])
            if key not in prev_keys and key not in self.step_seen:
                raise ContractError(
                    f"transition() returned walker {w} position that was neither in chain.position "
                    "nor evaluated with a finite log density during this call"
                )


# ---------------------------------------------------------------------------
# Submission loading & AST lint
# ---------------------------------------------------------------------------

def check_imports(source: str) -> None:
    """Reject imports outside ALLOWED_IMPORTS and builtins in FORBIDDEN_NAMES. A lint, not a sandbox."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ImportError(f"line {node.lineno}: {node.id!r} is not allowed")
        if isinstance(node, ast.Import):
            roots = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]] if node.level == 0 else ["<relative>"]
        else:
            continue
        bad = [r for r in roots if r not in ALLOWED_IMPORTS]
        if bad:
            raise ImportError(f"line {node.lineno}: import of {bad[0]!r} is not allowed "
                              f"(allowed: {', '.join(sorted(ALLOWED_IMPORTS))})")


def load_submission(path: Path) -> tuple[Callable, Callable]:
    """Load submission file and return (adapt, transition)."""
    path = Path(path)
    check_imports(path.read_text())
    name = "apin_submission"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses and typing look the module up there (S11)
    spec.loader.exec_module(mod)
    adapt_fn = getattr(mod, "adapt", None)
    transition_fn = getattr(mod, "transition", None)
    if not callable(adapt_fn) or not callable(transition_fn):
        raise AttributeError(
            f"{path} must define both adapt(oracle, budget, rng) and transition(tuning, chain, oracle, rng)"
        )
    return adapt_fn, transition_fn


def _resolve_submission(submission: Path | str | tuple[Callable, Callable]) -> tuple[Callable, Callable]:
    if isinstance(submission, (str, Path)):
        return load_submission(Path(submission))
    if isinstance(submission, tuple) and len(submission) == 2 and all(callable(f) for f in submission):
        return submission
    raise TypeError("submission must be a file path or (adapt, transition) pair")


# ---------------------------------------------------------------------------
# Derived RNGs & replay selection
# ---------------------------------------------------------------------------

def adapt_rng(run_seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(run_seed), 0]))


def step_rng(run_seed: int, step_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(run_seed), 1, int(step_index)]))


def choose_replay_steps(draws: int, run_seed: int, count: int = REPLAY_STEPS,
                        window: int = REPLAY_WINDOW) -> tuple[list[int], list[int]]:
    """Choose deterministic single-step indices and one short contiguous window for unbilled replay."""
    if draws <= 0:
        return [], []
    rng = np.random.default_rng(np.random.SeedSequence([int(run_seed), 2]))
    win_len = min(window, draws)
    win_start = int(rng.integers(0, draws - win_len + 1))
    win_steps = list(range(win_start, win_start + win_len))
    k = min(count, draws)
    singles = sorted(int(i) for i in rng.choice(draws, size=k, replace=False))
    return singles, win_steps


# ---------------------------------------------------------------------------
# Phase runners (adapt, sample, replay) + in-process helper
# ---------------------------------------------------------------------------

def _make_chain_arg(pos_2d: np.ndarray, aux: Any, is_1d: bool) -> Chain:
    pos = pos_2d[0].copy() if is_1d else pos_2d.copy()
    return Chain(position=pos, aux=aux)


def run_adapt_phase(model, submission: Path | str | tuple[Callable, Callable], budget: Budget, run_seed: int,
                    bundle_dir: Path, capabilities: set[str] | None = None) -> dict:
    """Execute adapt() for one run, validate outputs, serialize canonical bundle, return phase stats."""
    limits = Budget(int(budget.chains), int(budget.draws), int(budget.max_evaluations), float(budget.max_seconds))
    started = _now()
    adapt_fn, _ = _resolve_submission(submission)
    counter = Counter(model, limits.max_evaluations, limits.max_seconds, started=started, capabilities=capabilities)
    raw = adapt_fn(
        counter.oracle(),
        Budget(limits.chains, limits.draws, limits.max_evaluations, limits.max_seconds),
        adapt_rng(run_seed),
    )
    if not isinstance(raw, tuple) or len(raw) != 4:
        raise ContractError("adapt() must return a 4-tuple (tuning, chain0, betas, aux_spec)")
    tuning, chain0_raw, betas_raw, aux_spec_raw = raw
    pos_2d, aux, is_1d = normalize_chain(chain0_raw, counter.dim)
    walkers = pos_2d.shape[0]
    betas = validate_betas(betas_raw, walkers)
    aux_spec = normalize_aux_spec(aux_spec_raw)
    validate_aux(aux_spec, aux, walkers, counter.dim, is_1d)
    # Require chain0 positions to have been evaluated finitely during adapt()
    for w in range(walkers):
        if _row_bytes(pos_2d[w]) not in counter.step_seen:
            raise ContractError(
                f"adapt() returned chain0 walker {w} that was never evaluated with a finite log density"
            )
    tuning_sha, tuning_bytes = serialize_tree(tuning, bundle_dir / "tuning", max_bytes=MAX_TUNING_BYTES)
    serialize_tree({"position": pos_2d, "aux": aux}, bundle_dir / "chain0")
    seconds = _now() - started
    if counter.exceeded:
        raise BudgetExceeded(f"adapt() went past the budget ({counter.exceeded}) and caught the error")
    if seconds > limits.max_seconds:
        raise BudgetExceeded(f"adapt() took {seconds:.1f} s, exceeding {limits.max_seconds} s")
    meta = {
        "tuning_sha256": tuning_sha,
        "tuning_bytes": tuning_bytes,
        "walkers": walkers,
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


def run_sample_phase(model, submission: Path | str | tuple[Callable, Callable], budget: Budget, run_seed: int,
                     bundle_dir: Path, adapt_meta: dict, capabilities: set[str] | None = None,
                     replay_count: int = REPLAY_STEPS) -> tuple[np.ndarray, dict]:
    """Execute transition() draws in a clean environment from serialized tuning, recording replay checkpoints."""
    limits = Budget(int(budget.chains), int(budget.draws), int(budget.max_evaluations), float(budget.max_seconds))
    prior_evals = (adapt_meta["density_evals"] + adapt_meta["gradient_evals"]
                   + adapt_meta["hvp_evals"] + adapt_meta["hess_evals"])
    remaining_s = max(0.0, limits.max_seconds - float(adapt_meta["seconds"]))
    started = _now()
    _, transition_fn = _resolve_submission(submission)
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
    oracle = counter.oracle()
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

    for step in range(limits.draws):
        if step in need_single:
            serialize_tree({"position": pos_2d, "aux": aux}, ckpt_dir / f"in_{step}")
        if step == win_start and not win_start_saved:
            serialize_tree({"position": pos_2d, "aux": aux}, ckpt_dir / f"win_in_{step}")
            win_start_saved = True
        prev_pos = pos_2d
        counter.reset_step()
        chain_in = _make_chain_arg(pos_2d, aux, is_1d)
        raw_next = transition_fn(tuning, chain_in, oracle, step_rng(run_seed, step))
        next_pos, next_aux, _ = normalize_chain(raw_next, counter.dim, expected_walkers=walkers, expected_is_1d=is_1d)
        validate_aux(aux_spec, next_aux, walkers, counter.dim, is_1d)
        counter.check_step_provenance(prev_pos, next_pos)
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

    if canonical_digest(tuning) != adapt_meta["tuning_sha256"]:
        raise ContractError("tuning was mutated during transition() (SHA-256 mismatch)")
    seconds = _now() - started
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


def run_replay_phase(model, submission: Path | str | tuple[Callable, Callable], run_seed: int,
                     bundle_dir: Path, adapt_meta: dict, sample_meta: dict,
                     capabilities: set[str] | None = None) -> dict:
    """Replay selected single steps and the short window in a fresh environment; assert bit-exact match."""
    _, transition_fn = _resolve_submission(submission)
    tuning = deserialize_tree(bundle_dir / "tuning")
    if canonical_digest(tuning) != adapt_meta["tuning_sha256"]:
        raise ContractError("replay: tuning SHA-256 mismatch")
    walkers = int(adapt_meta["walkers"])
    is_1d = bool(adapt_meta["is_1d"])
    aux_spec = adapt_meta["aux_spec"]
    ckpt_dir = bundle_dir / "checkpoints"

    # Unbilled generous counter for replay
    counter = Counter(model, max_evaluations=10**9, max_seconds=1e9, capabilities=capabilities)
    oracle = counter.oracle()

    for rec in sample_meta["replay_singles"]:
        step = int(rec["step"])
        c_in = deserialize_tree(ckpt_dir / f"in_{step}")
        pos_in = np.array(c_in["position"], dtype=np.float64)
        aux_in = c_in["aux"]
        counter.reset_step()
        raw_next = transition_fn(tuning, _make_chain_arg(pos_in, aux_in, is_1d), oracle, step_rng(run_seed, step))
        pos_out, aux_out, _ = normalize_chain(raw_next, counter.dim, expected_walkers=walkers, expected_is_1d=is_1d)
        validate_aux(aux_spec, aux_out, walkers, counter.dim, is_1d)
        if canonical_digest(pos_out) != rec["pos_sha"] or canonical_digest(aux_out) != rec["aux_sha"]:
            raise ContractError(f"replay mismatch at step {step}: transition() is not deterministic/stateless")

    win = sample_meta.get("replay_window") or {}
    win_steps = win.get("steps") or []
    if win_steps:
        c_in = deserialize_tree(ckpt_dir / f"win_in_{win_steps[0]}")
        pos_cur = np.array(c_in["position"], dtype=np.float64)
        aux_cur = c_in["aux"]
        for step, (want_pos, want_aux) in zip(win_steps, win["digests"]):
            counter.reset_step()
            raw_next = transition_fn(tuning, _make_chain_arg(pos_cur, aux_cur, is_1d), oracle, step_rng(run_seed, step))
            pos_cur, aux_cur, _ = normalize_chain(
                raw_next, counter.dim, expected_walkers=walkers, expected_is_1d=is_1d
            )
            if canonical_digest(pos_cur) != want_pos or canonical_digest(aux_cur) != want_aux:
                raise ContractError(f"replay window mismatch at step {step}: hidden state across steps")

    if canonical_digest(tuning) != adapt_meta["tuning_sha256"]:
        raise ContractError("replay: tuning was mutated during replay")
    return {"replayed_singles": len(sample_meta["replay_singles"]), "replayed_window": len(win_steps)}


def execute_run(model, submission: Path | str | tuple[Callable, Callable], budget: Budget, run_seed: int,
                work_dir: Path, capabilities: set[str] | None = None,
                replay_count: int = REPLAY_STEPS) -> tuple[np.ndarray, dict]:
    """Run adapt -> canonical serialize -> sample -> fresh-reload replay in-process (for unit tests)."""
    bundle_dir = Path(work_dir) / f"run_{run_seed}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    adapt_meta = run_adapt_phase(model, submission, budget, run_seed, bundle_dir, capabilities=capabilities)
    draws, sample_meta = run_sample_phase(
        model, submission, budget, run_seed, bundle_dir, adapt_meta,
        capabilities=capabilities, replay_count=replay_count,
    )
    replay_meta = run_replay_phase(
        model, submission, run_seed, bundle_dir, adapt_meta, sample_meta, capabilities=capabilities
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


def write_constrained(model, draws: np.ndarray, betas: list[float], out: Path, seed: int) -> None:
    """Write constrained CSVs for cold walkers (beta == 1.0). draws shape: (runs, walkers, draws, dim)."""
    names = list(model.param_names(include_tp=True, include_gq=True))
    rng = model.new_rng(seed)
    cold = [w for w, b in enumerate(betas) if b == 1.0]
    for r in range(draws.shape[0]):
        for w_idx, w in enumerate(cold):
            fname = f"chain_{r}.csv" if len(cold) == 1 else f"chain_{r}_w{w_idx}.csv"
            with open(out / fname, "w") as fh:
                fh.write(",".join(names) + "\n")
                for theta in draws[r, w]:
                    try:
                        if not np.all(np.isfinite(theta)):
                            raise ValueError
                        row = model.param_constrain(theta, include_tp=True, include_gq=True, rng=rng)
                    except Exception:
                        row = np.full(len(names), math.nan)
                    fh.write(",".join(repr(float(v)) for v in row) + "\n")


# ---------------------------------------------------------------------------
# Subprocess CLI: python -m harness.logdensity <adapt|sample|replay|constrain> spec.json
# ---------------------------------------------------------------------------

def _open_bridgestan(spec: dict):
    import bridgestan

    return bridgestan.StanModel(
        spec["library"], spec["data"], seed=int(spec["seed"]), capture_stan_prints=True, warn=False
    )


def main(argv: list[str]) -> int:
    phase = argv[0]
    spec = json.loads(Path(argv[1]).read_text())
    caps = set(spec.get("capabilities") or ())
    bundle_dir = Path(spec["bundle_dir"])
    if phase == "adapt":
        model = _open_bridgestan(spec)
        budget = Budget(spec["chains"], spec["draws"], spec["max_evaluations"], spec["max_seconds"])
        run_adapt_phase(model, Path(spec["submission"]), budget, int(spec["run_seed"]), bundle_dir, capabilities=caps)
        return 0
    if phase == "sample":
        model = _open_bridgestan(spec)
        budget = Budget(spec["chains"], spec["draws"], spec["max_evaluations"], spec["max_seconds"])
        adapt_meta = json.loads((bundle_dir / "adapt_meta.json").read_text())
        draws, _ = run_sample_phase(
            model, Path(spec["submission"]), budget, int(spec["run_seed"]), bundle_dir, adapt_meta,
            capabilities=caps, replay_count=int(spec.get("replay_steps", REPLAY_STEPS)),
        )
        np.save(bundle_dir / "draws.npy", draws)
        return 0
    if phase == "replay":
        model = _open_bridgestan(spec)
        adapt_meta = json.loads((bundle_dir / "adapt_meta.json").read_text())
        sample_meta = json.loads((bundle_dir / "sample_meta.json").read_text())
        rep = run_replay_phase(
            model, Path(spec["submission"]), int(spec["run_seed"]), bundle_dir,
            adapt_meta, sample_meta, capabilities=caps,
        )
        (bundle_dir / "replay_meta.json").write_text(json.dumps(rep, indent=1))
        return 0
    if phase == "constrain":
        model = _open_bridgestan(spec)
        t0 = time.perf_counter()
        arr = np.load(Path(spec["draws_file"]))
        write_constrained(model, arr["draws"], arr["betas"].tolist(), Path(spec["out_dir"]), int(spec["seed"]))
        Path(spec["result_file"]).write_text(json.dumps({"constrain_s": time.perf_counter() - t0}))
        return 0
    raise SystemExit(f"unknown logdensity phase: {phase!r}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
