"""Target-calibrated evaluation cost weights: cost = n_density + r_grad * n_gradient (+ hvp/hess).

For each target model m, r_grad = max(1.0, median_t(grad_m) / median_t(density_m)) is measured
once through BridgeStan at raw reference draws (posteriordb draw zips or build/reference/<model>/),
floored at 1.0, and frozen in models/costs.json for a benchmark release.

Missing weights are an explicit error, never silently defaulted to 1.0.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np

from . import cmdstan, reference, stan_library
from .models import Model, ROOT

COSTS_PATH = ROOT / "models" / "costs.json"
DEFAULT_DRAWS = 64
DEFAULT_REPEATS = 7


def _raw_reference_draws(model: Model, build_dir: Path, param_names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Load raw constrained reference draws for `model` as name -> 1D array of draws."""
    if model.reference == "posteriordb":
        ref_name = reference.posteriordb_name(model)
        if not ref_name:
            raise LookupError(f"{model.name}: posteriordb has no reference posterior for {model.posterior}")
        zpath = reference.PDB / "reference_posteriors" / "draws" / "draws" / f"{ref_name}.json.zip"
        if not zpath.is_file():
            raise FileNotFoundError(f"{model.name}: posteriordb reference draws missing at {zpath}")
        with zipfile.ZipFile(zpath) as z:
            chains = json.loads(z.read(f"{ref_name}.json"))
        draws = {reference.stan_name(v): np.concatenate([np.asarray(c[v], dtype=float) for c in chains])
                 for v in chains[0]}
        if model.name in reference.DERIVED:
            extra = reference.DERIVED[model.name]({k: v[:, None] for k, v in draws.items()})
            draws.update({k: v[:, 0] for k, v in extra.items()})
        return draws
    if model.reference == "local":
        ref_dir = build_dir / "reference" / model.name
        csvs = sorted(ref_dir.glob("ref_*.csv")) or sorted(ref_dir.glob("chain_*.csv"))
        if not csvs:
            raise FileNotFoundError(f"{model.name}: local reference CSVs missing in {ref_dir}")
        files = [cmdstan.read_csv(p, columns=param_names) for p in csvs[:4]]
        cols = [c for c in files[0].columns if not c.endswith("__")]
        return {c: np.concatenate([f.column(c) for f in files]) for c in cols}
    raise ValueError(f"{model.name}: no reference source ({model.reference!r}) for cost measurement")


def _unconstrain_draws(bs_model, raw_draws: dict[str, np.ndarray], n_draws: int) -> np.ndarray:
    """Map up to n_draws evenly spaced constrained draws into BridgeStan unconstrained space."""
    param_names = list(bs_model.param_names(include_tp=False, include_gq=False))
    missing = [n for n in param_names if n not in raw_draws]
    if missing:
        raise KeyError(f"reference draws missing unconstrained parameters: {missing[:5]}")
    total = len(raw_draws[param_names[0]])
    idx = np.linspace(0, total - 1, min(n_draws, total), dtype=int)
    pts = []
    for i in idx:
        constrained = np.array([float(raw_draws[n][i]) for n in param_names], dtype=float)
        try:
            u = np.asarray(bs_model.param_unconstrain(constrained), dtype=float)
        except RuntimeError:
            continue
        if np.all(np.isfinite(u)):
            pts.append(u)
    if not pts:
        raise RuntimeError("could not unconstrain any reference draw")
    return np.stack(pts)


def _median_us_per_call(fn, points: np.ndarray, repeats: int, min_batch_s: float = 0.02) -> float:
    """Measure median wall microseconds per call over `points`, calibrating inner loops to beat timer jitter."""
    for x in points[: min(4, len(points))]:
        fn(x)
    # Calibrate loop count so each repeat takes at least min_batch_s
    loops = 1
    while True:
        t0 = time.perf_counter_ns()
        for _ in range(loops):
            for x in points:
                fn(x)
        elapsed_s = (time.perf_counter_ns() - t0) * 1e-9
        if elapsed_s >= min_batch_s or loops >= 4096:
            break
        loops = min(4096, max(loops * 2, int(loops * (min_batch_s / max(elapsed_s, 1e-6)))))
    samples = []
    calls = loops * len(points)
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        for _ in range(loops):
            for x in points:
                fn(x)
        elapsed_ns = time.perf_counter_ns() - t0
        samples.append((elapsed_ns / calls) / 1000.0)
    return float(np.median(samples))


def measure_model(model: Model, build_dir: Path = ROOT / "build", bridgestan_root: str | Path | None = None,
                  n_draws: int = DEFAULT_DRAWS, repeats: int = DEFAULT_REPEATS,
                  include_hvp: bool = False, include_hess: bool = False) -> dict:
    """Measure r_grad (and optionally r_hvp, r_hess) for one model via BridgeStan at reference draws."""
    import bridgestan

    bs_tree = stan_library.tree(bridgestan_root)
    so = stan_library.build(bs_tree, model.stan_file, build_dir)
    data_path = model.data_file()
    bs_model = bridgestan.StanModel(str(so), str(data_path), seed=1, capture_stan_prints=True, warn=False)
    raw = _raw_reference_draws(model, build_dir, list(bs_model.param_names(include_tp=False, include_gq=False)))
    pts = _unconstrain_draws(bs_model, raw, n_draws)

    dens_us = _median_us_per_call(lambda x: bs_model.log_density(x, propto=False, jacobian=True), pts, repeats)
    grad_us = _median_us_per_call(lambda x: bs_model.log_density_gradient(x, propto=False, jacobian=True), pts, repeats)
    r_grad = max(1.0, grad_us / dens_us) if dens_us > 0 else 1.0

    r_hvp = None
    hvp_us = None
    if include_hvp:
        rng = np.random.default_rng(0)
        vecs = rng.standard_normal(pts.shape)
        hvp_us = _median_us_per_call(
            lambda i: bs_model.log_density_hessian_vector_product(pts[i], vecs[i], propto=False, jacobian=True),
            np.arange(len(pts)), repeats,
        )
        r_hvp = max(1.0, hvp_us / dens_us) if dens_us > 0 else 1.0

    r_hess = None
    hess_us = None
    if include_hess:
        hess_us = _median_us_per_call(
            lambda x: bs_model.log_density_hessian(x, propto=False, jacobian=True), pts, repeats
        )
        r_hess = max(1.0, hess_us / dens_us) if dens_us > 0 else 1.0

    entry = {
        "r_grad": round(float(r_grad), 4),
        "r_hvp": round(float(r_hvp), 4) if r_hvp is not None else None,
        "r_hess": round(float(r_hess), 4) if r_hess is not None else None,
        "median_density_us": round(float(dens_us), 4),
        "median_grad_us": round(float(grad_us), 4),
        "n_draws": int(len(pts)),
        "dim": int(bs_model.param_unc_num()),
        "stan_sha256": hashlib.sha256(model.stan_file.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "bridgestan_key": so.parent.name,
    }
    if hvp_us is not None:
        entry["median_hvp_us"] = round(float(hvp_us), 4)
    if hess_us is not None:
        entry["median_hess_us"] = round(float(hess_us), 4)
    return entry


def build(models_list: list[Model], build_dir: Path = ROOT / "build", bridgestan_root: str | Path | None = None,
          out_path: Path = COSTS_PATH, n_draws: int = DEFAULT_DRAWS, repeats: int = DEFAULT_REPEATS,
          include_hvp: bool = False, include_hess: bool = False, log=print) -> dict[str, dict]:
    """Measure cost weights for models_list and update out_path (models/costs.json)."""
    data = load(out_path) if out_path.is_file() else {}
    for m in models_list:
        if m.reference == "none":
            continue
        rec = measure_model(m, build_dir, bridgestan_root, n_draws=n_draws, repeats=repeats,
                            include_hvp=include_hvp, include_hess=include_hess)
        data[m.name] = rec
        log(f"{m.name:45s} r_grad={rec['r_grad']:6.3f}  dens={rec['median_density_us']:8.2f}us  "
            f"grad={rec['median_grad_us']:8.2f}us  (n={rec['n_draws']}, d={rec['dim']})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n")
    return data


def load(path: Path = COSTS_PATH) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def weights_for(model_name: str, costs: dict[str, dict] | None = None, path: Path = COSTS_PATH) -> dict[str, float]:
    """Return cost weights for model_name; missing entry is an explicit error."""
    table = load(path) if costs is None else costs
    if model_name not in table:
        raise KeyError(f"missing cost weight r_m for model {model_name!r} in {path}; run `apin costs build`")
    row = table[model_name]
    r_grad = row.get("r_grad")
    if r_grad is None or not math.isfinite(r_grad) or r_grad < 1.0:
        raise ValueError(f"invalid r_grad {r_grad!r} for model {model_name!r} (must be finite and >= 1.0)")
    out = {"r_grad": float(r_grad)}
    for k in ("r_hvp", "r_hess"):
        v = row.get(k)
        if v is not None:
            if not math.isfinite(v) or v < 1.0:
                raise ValueError(f"invalid {k} {v!r} for model {model_name!r}")
            out[k] = float(v)
    return out


def compute_cost(density: int, gradient: int, weights: dict[str, float] | None,
                 hvp: int = 0, hess: int = 0) -> float:
    """Compute target-calibrated cost = n_density + r_grad * n_gradient + r_hvp * n_hvp + r_hess * n_hess."""
    if weights is None or "r_grad" not in weights:
        raise ValueError("missing r_m cost weights; never defaulted to 1.0")
    r_grad = float(weights["r_grad"])
    if not math.isfinite(r_grad) or r_grad < 1.0:
        raise ValueError(f"invalid r_grad {r_grad!r}")
    total = float(density) + r_grad * float(gradient)
    if hvp:
        r_hvp = weights.get("r_hvp")
        if r_hvp is None or not math.isfinite(r_hvp) or r_hvp < 1.0:
            raise ValueError("model invoked hvp without a frozen r_hvp weight")
        total += float(r_hvp) * float(hvp)
    if hess:
        r_hess = weights.get("r_hess")
        if r_hess is None or not math.isfinite(r_hess) or r_hess < 1.0:
            raise ValueError("model invoked hess without a frozen r_hess weight")
        total += float(r_hess) * float(hess)
    return total
