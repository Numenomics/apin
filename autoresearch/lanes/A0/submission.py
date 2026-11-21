"""Lane A0 ground-up sampler v4: windowed dense-whitened HMC + preconditioned RWM.

Designed from first principles against the dual-horizon, two-dimension Tier 1 metric:
the scored quantity is min-functional Var/MSE within a short late window after a handful
of transitions from overdispersed starts, per cost unit (gradients billed 2x densities),
so the kernel must (a) kill initialization bias within ~10 transitions in d=20, including
under *rotated* ill-conditioning that defeats any diagonal metric, and (b) keep the
per-transition oracle-call cost small.

Warmup (adapt) is a single HMC engine with Stan-style expanding metric windows —
Hamiltonian flow escapes bad initializations by following the gradient field, unlike
random-walk warmup which can sit frozen on curved ridges (hybrid Rosenbrock) and poison
the metric estimate. Metric switches at 45%/75% of the window; dual averaging restarts
at each switch, re-scaled by the median mass diagonal. The dense metric is a
MAD-prewhitened, shrunk, eigenvalue-guarded sample covariance, so one matrix fixes
rotated ill-conditioning that any diagonal metric cannot see.

Kernel (transition): each step draws fresh momentum p ~ N(0, M) (M = dense mass,
declared exactly as the aux law) and flips a biased coin:
  - RWM branch (cost ~2): whitened random-walk Metropolis, sigma = 2.4/sqrt(d).
  - HMC branch: doubling-jitter leapfrog trajectory (L in {1,2,4,...,16}, P(L=2k) ~
    0.55^k) with exact Metropolis correction on the Hamiltonian error; short steps
    carry the steady-state cost, rare long trajectories erase dispersed-start bias.
Both branches leave pi(x) x N(0, M) exactly invariant; the mixture does too.

requires = ["grad"]
"""

from __future__ import annotations

import math

import numpy as np

DELTA = 0.8     # high acceptance target: keeps eps stable in tight-curvature zones
Q_RWM = 0.25    # probability of the preconditioned random-walk branch
L_MAX = 16      # leapfrog step cap for the doubling jitter (transition kernel)
L_WARM = 3      # fixed warmup trajectory length
GROW = 0.55     # doubling-jitter growth probability
RW_SCALE = 2.4  # whitened RWM scale multiplier (sigma = RW_SCALE / sqrt(d))


def _finite_init(oracle, rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        x = rng.uniform(-2.0, 2.0, size=oracle.dim)
        if math.isfinite(oracle.log_density(x)):
            return x
    return np.zeros(oracle.dim, dtype=float)


def _dense_cov(buf: np.ndarray) -> np.ndarray:
    """Robust dense covariance: MAD pre-whitening, shrinkage, guarded eigenvalue floor.

    Guards matter: a warmup that has not escaped a ridge yet yields near-constant draws,
    and a naively floored MAD would produce a ~1e-22 covariance whose inverse mass makes
    the kernel effectively frozen (a stuck kernel). Absolute MAD floor, degenerate-trace
    fallback, and a relative eigenvalue floor (condition <= 1e4) keep the metric sane.
    """
    med = np.median(buf, axis=0)
    mad = np.median(np.abs(buf - med), axis=0) * 1.4826
    mad = np.maximum(mad, 1e-4)
    y = (buf - med) / mad
    d = buf.shape[1]
    s = np.cov(y, rowvar=False)
    s = (s + s.T) / 2.0
    lam = min(0.4, max(0.15, d / (2.5 * buf.shape[0])))
    s = (1.0 - lam) * s + (lam * float(np.trace(s)) / d) * np.eye(d)
    w, v = np.linalg.eigh(s)
    w = np.maximum(w, max(1e-12, 1e-4 * float(np.max(np.abs(w)))))
    cov = (v * w) @ v.T
    cov = mad[:, None] * cov * mad[None, :]
    if float(np.trace(cov)) / d < 1e-8:
        cov = np.eye(d) * 0.01
    return (cov + cov.T) / 2.0


def _factor(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor a dense covariance into (mass M = C^-1, chol(M), chol(C))."""
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, max(1e-12, 1e-6 * float(np.max(np.abs(w)))))
    cov = (v * w) @ v.T
    mass = (v * (1.0 / w)) @ v.T
    mass = (mass + mass.T) / 2.0
    return mass, np.linalg.cholesky(mass), np.linalg.cholesky(cov)


def _gradient_slide(oracle, x, lp, grad):
    """Backtracked gradient ascent to a high-density region before sampling.

    Curved ridges (hybrid Rosenbrock) and funnels have tiny typical sets; a chain that
    starts (or wanders) off them sees near-flat density, which fools acceptance-based
    step-size calibration. Climbing first puts the warmup where the geometry lives.
    """
    eta = 0.25 / math.sqrt(x.size)
    for _ in range(15):
        moved = False
        for _ in range(3):
            y = x + eta * grad
            lpy = oracle.log_density(y)
            if math.isfinite(lpy) and lpy > lp:
                lp, grad = lpy, oracle.log_density_gradient(y)[1]
                x = y
                eta *= 1.5
                moved = True
                break
            eta *= 0.3
            if eta < 1e-10:
                return x, lp, grad
        if not moved:
            break
    return x, lp, grad


def _hmc_acc(oracle, x, lp, grad, eps, cov, mass, chol, rng, n_steps):
    """Run one leapfrog trajectory; return (accepted, acc_prob, lpn, gn, xn, pe).

    Momentum is drawn p ~ N(0, M) with M the mass, so the kinetic energy is
    0.5 * p^T M^-1 p = 0.5 * p^T cov p (v3's `p*p*inv_mass` convention, dense).
    """
    p = chol @ rng.standard_normal(x.size)
    h0 = -lp + 0.5 * float(p @ (cov @ p))
    xn, gn, pe = x, grad, p + 0.5 * eps * grad
    for step in range(n_steps):
        xn = xn + eps * (cov @ pe)
        lpn, gn = oracle.log_density_gradient(xn)
        if not (math.isfinite(lpn) and bool(np.all(np.isfinite(gn)))):
            return False, 0.0, lp, grad, x, pe
        if step < n_steps - 1:
            pe = pe + eps * gn
    pe = pe + 0.5 * eps * gn
    h1 = -lpn + 0.5 * float(pe @ (cov @ pe))
    acc = math.exp(min(0.0, h0 - h1))
    ok = math.isfinite(h1) and rng.uniform() < acc
    return ok, acc, lpn, gn, xn, pe


def _reasonable_eps(oracle, x, lp, grad, cov, mass, chol, rng):
    """Hoffman-Gelman 'find reasonable epsilon': double/halve until acc crosses 0.5."""
    eps = 1.0 / math.sqrt(x.size)
    _, acc, _, _, _, _ = _hmc_acc(oracle, x, lp, grad, eps, cov, mass, chol, rng, L_WARM)
    direction = 1.0 if acc > 0.5 else -1.0
    for _ in range(12):
        eps *= 2.0 ** direction
        if eps > 16.0 or eps < 1e-7:
            break
        _, acc, _, _, _, _ = _hmc_acc(oracle, x, lp, grad, eps, cov, mass, chol, rng, L_WARM)
        if (acc > 0.5) != (direction > 0):
            break
    return min(8.0, max(1e-6, eps * 0.5 ** direction))


def adapt(oracle, budget, rng):
    """Windowed from-scratch warmup -> (tuning, chain0, betas, aux_spec)."""
    d = oracle.dim
    x = _finite_init(oracle, rng)
    lp, grad = oracle.log_density_gradient(x)
    x, lp, grad = _gradient_slide(oracle, x, lp, grad)
    n_warm = 70 + 5 * d
    n1 = int(0.3 * n_warm)
    n2 = int(0.62 * n_warm)
    buf_at = max(1, int(0.4 * n1))
    cov = np.eye(d)
    mass = np.eye(d)
    chol = np.eye(d)
    eps = _reasonable_eps(oracle, x, lp, grad, cov, mass, chol, rng)
    log_eps = math.log(eps)
    buf: list[np.ndarray] = []
    ring: list[tuple[np.ndarray, float, np.ndarray]] = []
    for i in range(1, n_warm + 1):
        ok, acc, lpn, gn, xn, _ = _hmc_acc(
            oracle, x, lp, grad, eps, cov, mass, chol, rng, L_WARM
        )
        if ok:
            x, lp, grad = xn, lpn, gn
        # Robbins-Monro step-size adaptation: bounded per-iteration movement (gain
        # 1/sqrt(i+10)). Dual averaging's sqrt(t)/gamma term amplifies the acceptance
        # oscillation that curved targets (funnel, ridges) produce and can collapse
        # eps to a frozen kernel.
        log_eps = min(4.0, max(-12.0, log_eps + (acc - DELTA) / math.sqrt(i + 10.0)))
        eps = math.exp(log_eps)
        if buf_at < i <= n2:
            buf.append(x.copy())
        if i > n2 and i % 8 == 0:
            ring.append((x.copy(), lp, grad))
            if len(ring) > 3:
                ring.pop(0)
        if i == n1 or i == n2:
            arr = np.asarray(buf)
            raw_mad = np.median(np.abs(arr - np.median(arr, axis=0)), axis=0)
            if buf and float(np.median(raw_mad)) > 0.02:
                # Trust the dense metric only when the chain has genuinely spread out;
                # a neck-trapped buffer would scale the metric absurdly wrong.
                cov = _dense_cov(arr)
                mass, chol, _ = _factor(cov)
            buf = []

    eps = math.exp(min(4.0, max(-12.0, log_eps)))
    # Measured calibration sweep over a ring of recent warmup positions (stored with
    # their lp/grad, so only the sweep trajectories cost oracle calls): RM may
    # under-converge in the short final window, and a single-endpoint measurement is
    # fooled by position-dependent acceptance (curved ridges). Pick the LARGEST eps
    # whose mean acceptance stays >= DELTA across the ring; fall back to the smallest.
    if not ring:
        ring.append((x, lp, grad))
    best, best_acc = None, -1.0
    for mult in (0.5, 0.75, 1.15, 1.75):
        cand = eps * mult
        acc_sum = 0.0
        for px, plp, pgrad in ring:
            acc_sum += _hmc_acc(oracle, px, plp, pgrad, cand, cov, mass, chol, rng, L_WARM)[1]
        mean_acc = acc_sum / len(ring)
        if mean_acc >= DELTA:
            best, best_acc = cand, mean_acc
        elif best is None and mean_acc > best_acc:
            best, best_acc = cand, mean_acc
    eps = min(4.0, max(1e-6, best if best is not None else eps * 0.5))
    mass, chol, cov_chol = _factor(cov)
    aux0 = chol @ rng.standard_normal(d)
    tuning = {
        "eps": float(eps),
        "l_max": int(L_MAX),
        "grow": float(GROW),
        "q_rwm": float(Q_RWM),
        "rwm_scale": float(RW_SCALE / math.sqrt(d)),
        "mass": [[float(vv) for vv in row] for row in mass],
        "cov": [[float(vv) for vv in row] for row in cov],
        "chol": [[float(vv) for vv in row] for row in chol],
        "cov_chol": [[float(vv) for vv in row] for row in cov_chol],
    }
    aux_spec = {"kind": "normal", "cov": tuning["mass"]}
    return tuning, (x, aux0), [1.0], aux_spec


def transition(tuning, chain, oracle, rng):
    """One mixture transition: fresh N(0, M) momentum, then whitened RWM or jittered HMC."""
    x = np.asarray(chain.position, dtype=float)
    d = x.size
    eps = float(tuning["eps"])
    l_max = int(tuning["l_max"])
    grow = float(tuning["grow"])
    q_rwm = float(tuning["q_rwm"])
    cov = np.asarray(tuning["cov"], dtype=float)
    chol = np.asarray(tuning["chol"], dtype=float)
    cov_chol = np.asarray(tuning["cov_chol"], dtype=float)

    p0 = chol @ rng.standard_normal(d)
    if rng.uniform() < q_rwm:
        lp = oracle.log_density(x)
        y = x + float(tuning["rwm_scale"]) * (cov_chol @ rng.standard_normal(d))
        lpy = oracle.log_density(y)
        if math.isfinite(lpy) and math.log(rng.uniform() + 1e-300) < lpy - lp:
            return y, -p0
        return x, -p0

    lp, grad = oracle.log_density_gradient(x)
    h0 = -lp + 0.5 * float(p0 @ (cov @ p0))
    n = 1
    while n < l_max and rng.uniform() < grow:
        n *= 2
    xn, gn, pe = x, grad, p0 + 0.5 * eps * grad
    for step in range(n):
        xn = xn + eps * (cov @ pe)
        lpn, gn = oracle.log_density_gradient(xn)
        if not (math.isfinite(lpn) and bool(np.all(np.isfinite(gn)))):
            return x, -p0
        if step < n - 1:
            pe = pe + eps * gn
    pe = pe + 0.5 * eps * gn
    h1 = -lpn + 0.5 * float(pe @ (cov @ pe))
    if math.isfinite(h1) and math.log(rng.uniform() + 1e-300) < h0 - h1:
        return xn, -pe
    return x, -p0
