"""The log-density seam: run a sampling algorithm that is handed a log density and its gradient.

A submission is one Python file that defines

    def sample(problem: Problem, budget: Budget, rng: numpy.random.Generator) -> numpy.ndarray

and returns unconstrained draws of shape (budget.chains, budget.draws, problem.dim).
BENCH.md gives the rules; PROTOCOL.md, "The log-density seam", the contract.

The harness, not the submission, counts every call to `log_density` and
`log_density_gradient`, a call at a rejected point too. A call past
`budget.max_evaluations` raises BudgetExceeded, and the run fails even if the
submission catches it. The clock runs from loading the submission to converting
what it returns; a run longer than `budget.max_seconds` fails, at the next call
or at the end. Every returned draw must be a point the submission evaluated with
a finite log density (the provenance check). A point the model rejects (Stan
raises a domain error) has log density -inf and a NaN gradient, which is how
Stan's own samplers read it; the submission never sees the exception. The
harness then maps the draws to the constrained space with BridgeStan and writes
the chain CSVs that the scorer reads.

The submission gets plain functions and its own copy of the budget; the limits
the harness enforces are copied before the submission runs. This keeps an honest
submission from changing them by accident. It does not stop one that sets out to
(BENCH.md, rule 5).

`run(spec)` does one run in the current process; the logdensity adapter calls
it in a child process (`python -m harness.logdensity spec.json`) so that a
hung or crashing submission cannot take the grid with it.
"""

from __future__ import annotations

import array
import ast
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ALLOWED_IMPORTS = {"numpy", "scipy", "math", "__future__", "typing", "dataclasses", "functools", "itertools",
                   "collections"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "breakpoint", "input", "globals", "vars"}


@dataclass(frozen=True)
class Problem:
    """What the sampler may know about the target. theta is a float array of length dim."""

    dim: int
    log_density: Callable[[np.ndarray], float]
    log_density_gradient: Callable[[np.ndarray], tuple[float, np.ndarray]]
    initial_points: np.ndarray  # (chains, dim), one starting point per chain


@dataclass(frozen=True)
class Budget:
    chains: int  # independent chains to return
    draws: int  # draws per chain to return
    max_evaluations: int  # calls to log_density plus log_density_gradient, all chains, warmup included
    max_seconds: float  # wall time of the whole sample() call


class BudgetExceeded(RuntimeError):
    pass


_now = time.perf_counter  # the harness's clock; tests replace it


class Counter:
    """Wraps a BridgeStan model's log density and enforces the budget.

    The limits are copied when the counter is made, so changing the Budget object the submission
    was handed changes nothing. The submission gets plain functions from callbacks(), not this
    object. Every call is charged, a rejected one too, but only a point with a finite log density
    is recorded for the provenance check."""

    def __init__(self, model, budget: Budget, started: float | None = None):
        self._model = model
        self.dim = model.param_unc_num()
        self.max_evaluations, self.max_seconds = int(budget.max_evaluations), float(budget.max_seconds)
        self.density = self.gradient = 0
        self.exceeded = ""
        self.t0 = _now() if started is None else started
        self.seen = array.array("q")  # hash of every point with a finite density, in call order

    def callbacks(self) -> tuple[Callable, Callable]:
        """(log_density, log_density_gradient) as plain functions, with no attribute that leads
        back to the counter or the model. Python's introspection (a closure's cells, the garbage
        collector, the call stack) still does; BENCH.md says what that means."""
        density, gradient = self.log_density, self.log_density_gradient

        def log_density(theta) -> float:
            return density(theta)

        def log_density_gradient(theta) -> tuple[float, np.ndarray]:
            return gradient(theta)

        return log_density, log_density_gradient

    def first_call(self, draws: np.ndarray) -> np.ndarray:
        """For each draw (..., dim), the index of the first finite evaluation of it, or -1 if none."""
        flat = draws.reshape(-1, self.dim)
        hashes = np.fromiter((hash(np.ascontiguousarray(x).tobytes()) for x in flat), dtype=np.int64,
                             count=len(flat))
        seen, first = np.unique(np.frombuffer(self.seen, dtype=np.int64), return_index=True)
        out = np.full(len(hashes), -1)
        if len(seen):
            k = np.minimum(np.searchsorted(seen, hashes), len(seen) - 1)
            hit = seen[k] == hashes
            out[hit] = first[k[hit]]
        return out.reshape(draws.shape[:-1])

    def _charge(self) -> None:
        if self.exceeded:
            raise BudgetExceeded(self.exceeded)
        if self.density + self.gradient >= self.max_evaluations:
            self.exceeded = f"more than {self.max_evaluations} evaluations"
        elif _now() - self.t0 > self.max_seconds:
            self.exceeded = f"more than {self.max_seconds} s"
        if self.exceeded:
            raise BudgetExceeded(self.exceeded)

    def log_density(self, theta) -> float:
        self._charge()
        self.density += 1
        x = self._point(theta)
        try:
            lp = float(self._model.log_density(x, propto=False, jacobian=True))
        except RuntimeError:  # Stan rejected the point (a domain error): density zero, as Stan's samplers read it
            return -math.inf
        self._record(x, lp)
        return lp

    def log_density_gradient(self, theta) -> tuple[float, np.ndarray]:
        self._charge()
        self.gradient += 1
        x = self._point(theta)
        try:
            lp, grad = self._model.log_density_gradient(x, propto=False, jacobian=True)
        except RuntimeError:
            return -math.inf, np.full(self.dim, math.nan)
        lp = float(lp)
        self._record(x, lp)
        return lp, np.array(grad, dtype=float)

    def _point(self, theta) -> np.ndarray:
        x = np.array(theta, dtype=float)  # a copy: the submission cannot change it after the call
        if x.shape != (self.dim,):
            raise ValueError(f"theta must have shape ({self.dim},), got {x.shape}")
        return x

    def _record(self, x: np.ndarray, lp: float) -> None:
        if math.isfinite(lp):
            self.seen.append(hash(x.tobytes()))


def check_imports(source: str) -> None:
    """Reject imports outside ALLOWED_IMPORTS and the builtins in FORBIDDEN_NAMES. A lint, not a sandbox."""
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


def load_submission(path: Path) -> Callable:
    path = Path(path)
    check_imports(path.read_text())
    name = "apin_submission"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses and typing look the module up there (S11)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "sample", None)
    if not callable(fn):
        raise AttributeError(f"{path} defines no function sample(problem, budget, rng)")
    return fn


def initial_points(rng: np.random.Generator, chains: int, dim: int) -> np.ndarray:
    """Stan's default: uniform on (-2, 2) in the unconstrained space."""
    return rng.uniform(-2.0, 2.0, size=(chains, dim))


def execute(model, submission: Path | Callable, budget: Budget, seed: int) -> tuple[np.ndarray, Counter, float]:
    """Load the submission (a file, or a function), run it on a model with the BridgeStan methods
    used here, and check what it returns.

    The clock runs from loading the submission (its module-level code) to converting its return
    value to an array; an overrun at the end fails the run as one during a call does. Returns the
    draws (chains, draws, dim), the counter and those seconds. Raises BudgetExceeded when the
    submission went past the budget, ValueError when the draws have the wrong shape or include a
    point the submission never evaluated with a finite log density.
    """
    limits = Budget(int(budget.chains), int(budget.draws), int(budget.max_evaluations), float(budget.max_seconds))
    dim = model.param_unc_num()
    seeds = np.random.SeedSequence(seed).spawn(2)
    inits = initial_points(np.random.default_rng(seeds[0]), limits.chains, dim)
    started = _now()
    fn = load_submission(submission) if isinstance(submission, (str, Path)) else submission
    counter = Counter(model, limits, started)
    log_density, log_density_gradient = counter.callbacks()
    problem = Problem(dim, log_density, log_density_gradient, inits.copy())
    raw = fn(problem, Budget(*(getattr(limits, f) for f in ("chains", "draws", "max_evaluations", "max_seconds"))),
             np.random.default_rng(seeds[1]))
    draws = np.array(raw, dtype=float)
    seconds = _now() - started
    if counter.exceeded:
        raise BudgetExceeded(f"the submission went past the budget ({counter.exceeded}) and caught the error")
    if seconds > limits.max_seconds:
        raise BudgetExceeded(f"the submission took {seconds:.1f} s, more than {limits.max_seconds} s, counted from "
                             "loading it to converting its return value")
    want = (limits.chains, limits.draws, dim)
    if draws.shape != want:
        raise ValueError(f"sample() returned shape {draws.shape}, expected {want}")
    check_provenance(counter.first_call(draws))
    return draws, counter, seconds


def check_provenance(first: np.ndarray) -> None:
    """first: (chains, draws), the first call that evaluated each draw, -1 for none (Counter.first_call).

    Every draw must have been evaluated; within a chain the draws must keep the order in which they were
    first evaluated; and no point may appear in two chains. These rule out draws the submission did not
    pay for, shuffled chains (which hide autocorrelation from ESS) and copied chains.
    """
    if (first < 0).any():
        c, i = map(int, np.argwhere(first < 0)[0])
        raise ValueError(f"{int((first < 0).sum())} returned draws were never evaluated (first: chain {c}, "
                         f"draw {i}); every draw must be a point passed to log_density or log_density_gradient")
    back = np.diff(first, axis=1) < 0
    if back.any():
        c, i = map(int, np.argwhere(back)[0])
        raise ValueError(f"chain {c}: draw {i + 1} was evaluated before draw {i}; "
                         "each chain must return its draws in the order the sampler visited them")
    owner: dict[int, int] = {}
    for c, row in enumerate(first):
        for k in np.unique(row).tolist():
            if owner.setdefault(k, c) != c:
                raise ValueError(f"chains {owner[k]} and {c} share a draw; the chains must be distinct")


def run(spec: dict) -> dict:
    """One run. spec: library, data, submission, seed, chains, draws, max_evaluations, max_seconds, out_dir.

    Writes out_dir/chain_<c>.csv and returns the counts and times.
    """
    import bridgestan

    model = bridgestan.StanModel(spec["library"], spec["data"], seed=spec["seed"], capture_stan_prints=True,
                                 warn=False)
    budget = Budget(spec["chains"], spec["draws"], spec["max_evaluations"], spec["max_seconds"])
    draws, counter, seconds = execute(model, Path(spec["submission"]), budget, spec["seed"])
    t1 = time.perf_counter()
    write_constrained(model, draws, Path(spec["out_dir"]), spec["seed"])
    return {"seconds": seconds, "density_evals": counter.density, "gradient_evals": counter.gradient,
            "evaluations": counter.density + counter.gradient, "constrain_s": time.perf_counter() - t1,
            "dim": counter.dim}


def write_constrained(model, draws: np.ndarray, out: Path, seed: int) -> None:
    """Constrained parameters, transformed parameters and generated quantities, one CSV per chain.
    A draw that cannot be mapped (non-finite, or outside the support) is written as NaN."""
    names = list(model.param_names(include_tp=True, include_gq=True))
    rng = model.new_rng(seed)
    for c, chain in enumerate(draws):
        with open(out / f"chain_{c}.csv", "w") as fh:
            fh.write(",".join(names) + "\n")
            for theta in chain:
                try:
                    if not np.all(np.isfinite(theta)):
                        raise ValueError
                    row = model.param_constrain(theta, include_tp=True, include_gq=True, rng=rng)
                except Exception:
                    row = np.full(len(names), math.nan)
                fh.write(",".join(repr(float(v)) for v in row) + "\n")


def main(argv: list[str]) -> int:
    spec = json.loads(Path(argv[0]).read_text())
    result = run(spec)
    Path(spec["out_dir"], "evaluations.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
