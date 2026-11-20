"""Slice 4 parametric analytic shape block: fresh targets with exact draws, exact truth,
latent-coordinate evaluation, aligned & rotated parametrisations, and exact tempered sampling.

Implemented families (Council B2 & D3):
  - gaussian_illcond   Ill-conditioned Gaussian (log-spaced eigenvalues, kappa in [1e2, 1e4])
  - funnel             Neal's funnel: v ~ N(0, 9), x_i | v ~ N(0, exp(v)) (latent neck v < -6)
  - banana             Haario/Rosenbrock banana with exact sequential draws
  - hybrid_rosenbrock  Block-scaled hybrid Rosenbrock (Pagani et al. 2022) with exact draws
  - student_t          Multivariate/independent Student-t; nu <= 2 tests ONLY defined functionals
                       (quantiles/median/IQR/log-density), never variance/second moment!
  - mixture_crossable  2-component Gaussian mixture (sep <= 3.5 SD) + exact tempered rejection envelope
  - mixture_separated  Unequal-weight separated mixture (sep >= 6 SD, multimodal column) + exact tempered envelope

Every shape exposes the BridgeStan-compatible oracle methods (`param_unc_num`, `log_density`,
`log_density_gradient`) in the *observed* (permuted/signed/shifted or rotated) coordinates while
evaluating functionals in the *unrotated latent* coordinates known only to the harness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import norm, t as student_t_dist


def _random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed random orthogonal matrix via QR with positive diagonal R."""
    a = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs[None, :]


def _random_signed_perm(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Axis-aligned permutation + random sign flip matrix (preserves axis alignment)."""
    perm = rng.permutation(dim)
    signs = rng.choice([-1.0, 1.0], size=dim)
    q = np.zeros((dim, dim), dtype=float)
    q[np.arange(dim), perm] = signs
    return q


class AnalyticTarget:
    """Base class for analytic targets under an affine wrapper x = shift + scale * (Q @ z)."""

    family: str = ""
    multimodal: bool = False

    def __init__(self, dim: int, seed: int, rotated: bool = False, scale_stress: float = 1.0):
        self.dim = int(dim)
        self.seed = int(seed)
        self.rotated = bool(rotated)
        self.scale_stress = float(scale_stress)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7919]))
        self._Q = _random_orthogonal(dim, rng) if rotated else _random_signed_perm(dim, rng)
        self._shift = rng.uniform(-2.5, 2.5, size=dim)
        self._scale = float(scale_stress)
        self._log_jac_inv = -dim * math.log(abs(self._scale))

    def param_unc_num(self) -> int:
        return self.dim

    def to_latent(self, x: np.ndarray) -> np.ndarray:
        """Map observed x (..., dim) -> latent z (..., dim)."""
        y = (np.asarray(x, dtype=float) - self._shift) / self._scale
        return y @ self._Q  # since x = shift + scale * (z @ Q^T)

    def from_latent(self, z: np.ndarray) -> np.ndarray:
        """Map latent z (..., dim) -> observed x (..., dim)."""
        return self._shift + self._scale * (np.asarray(z, dtype=float) @ self._Q.T)

    def _latent_log_density_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        raise NotImplementedError

    def _latent_exact_draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def _latent_tempered_draws(self, n: int, beta: float, rng: np.random.Generator) -> np.ndarray | None:
        return None

    def log_density(self, x: np.ndarray, propto: bool = False, jacobian: bool = True) -> float:
        lp, _ = self.log_density_gradient(x, propto=propto, jacobian=jacobian)
        return lp

    def log_density_gradient(self, x: np.ndarray, propto: bool = False, jacobian: bool = True) -> tuple[float, np.ndarray]:
        z = self.to_latent(np.asarray(x, dtype=float))
        lp_z, gz = self._latent_log_density_grad(z)
        if not math.isfinite(lp_z) or not np.all(np.isfinite(gz)):
            return -math.inf, np.full(self.dim, math.nan)
        lp = float(lp_z + self._log_jac_inv)
        gx = (gz @ self._Q.T) / self._scale
        return lp, np.asarray(gx, dtype=float)

    def exact_draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.from_latent(self._latent_exact_draws(n, rng))

    def exact_tempered_draws(self, n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
        if beta == 1.0:
            return self.exact_draws(n, rng)
        z = self._latent_tempered_draws(n, float(beta), rng)
        if z is None:
            raise NotImplementedError(f"{self.family} does not support exact tempered draws for beta={beta}")
        return self.from_latent(z)

    def latent_functionals(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """Compute named scalar functionals on latent draws z of shape (..., dim)."""
        raise NotImplementedError

    def exact_truth(self) -> dict[str, tuple[float, float]]:
        """Return {functional_name: (exact_mean, exact_sd)} for every functional in latent_functionals()."""
        raise NotImplementedError


class IllCondGaussian(AnalyticTarget):
    family = "gaussian_illcond"

    def __init__(self, dim: int, seed: int, rotated: bool = False, scale_stress: float = 1.0, kappa: float = 100.0):
        super().__init__(dim, seed, rotated=rotated, scale_stress=scale_stress)
        self.stds = np.geomspace(1.0, math.sqrt(kappa), dim) if dim > 1 else np.array([1.0])
        self.inv_var = 1.0 / (self.stds ** 2)
        self._log_norm = -0.5 * (dim * math.log(2.0 * math.pi) + 2.0 * float(np.sum(np.log(self.stds))))

    def _latent_log_density_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        lp = self._log_norm - 0.5 * float(np.sum((z ** 2) * self.inv_var))
        return lp, -z * self.inv_var

    def _latent_exact_draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n, self.dim)) * self.stds[None, :]

    def _latent_tempered_draws(self, n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n, self.dim)) * (self.stds / math.sqrt(beta))[None, :]

    def latent_functionals(self, z: np.ndarray) -> dict[str, np.ndarray]:
        u = z / self.stds
        lp = self._log_norm - 0.5 * np.sum(u ** 2, axis=-1)
        out = {
            "u0": u[..., 0],
            "u_last": u[..., -1],
            "u0_sq": u[..., 0] ** 2,
            "u_last_sq": u[..., -1] ** 2,
            "rad_sq": np.mean(u ** 2, axis=-1),
            "lp": lp,
        }
        return out

    def exact_truth(self) -> dict[str, tuple[float, float]]:
        d = self.dim
        return {
            "u0": (0.0, 1.0),
            "u_last": (0.0, 1.0),
            "u0_sq": (1.0, math.sqrt(2.0)),
            "u_last_sq": (1.0, math.sqrt(2.0)),
            "rad_sq": (1.0, math.sqrt(2.0 / d)),
            "lp": (self._log_norm - 0.5 * d, math.sqrt(0.5 * d)),
        }


class NealsFunnel(AnalyticTarget):
    """Neal's funnel: v = z[0] ~ N(0, 3^2), z[1:] | v ~ N(0, exp(v))."""

    family = "funnel"

    def __init__(self, dim: int, seed: int, rotated: bool = False, scale_stress: float = 1.0, v_sd: float = 3.0):
        if dim < 2:
            raise ValueError("Neal's funnel requires dim >= 2")
        super().__init__(dim, seed, rotated=rotated, scale_stress=scale_stress)
        self.v_sd = float(v_sd)

    def _latent_log_density_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        v = float(z[0])
        if v < -40.0 or v > 40.0:
            return -math.inf, np.full(self.dim, math.nan)
        x = z[1:]
        inv_ev = math.exp(-v)
        m = self.dim - 1
        lp = -0.5 * (v / self.v_sd) ** 2 - math.log(self.v_sd * math.sqrt(2 * math.pi))
        lp += -0.5 * m * (v + math.log(2 * math.pi)) - 0.5 * inv_ev * float(x @ x)
        g = np.empty(self.dim, dtype=float)
        g[0] = -v / (self.v_sd ** 2) - 0.5 * m + 0.5 * inv_ev * float(x @ x)
        g[1:] = -x * inv_ev
        return lp, g

    def _latent_exact_draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        v = rng.normal(0.0, self.v_sd, size=(n, 1))
        x = rng.standard_normal((n, self.dim - 1)) * np.exp(0.5 * v)
        return np.concatenate([v, x], axis=1)

    def latent_functionals(self, z: np.ndarray) -> dict[str, np.ndarray]:
        v = z[..., 0]
        # Scaled radial statistic x / exp(v/2) ~ N(0, I) exactly, plus explicit neck indicator v < -6!
        u = z[..., 1:] * np.exp(-0.5 * np.clip(v[..., None], -40.0, 40.0))
        return {
            "v": v,
            "v_sq": v ** 2,
            "neck_v_lt_m6": (v < -6.0).astype(float),
            "neck_v_lt_m3": (v < -3.0).astype(float),
            "u0": u[..., 0],
            "u_rad_sq": np.mean(u ** 2, axis=-1),
        }

    def exact_truth(self) -> dict[str, tuple[float, float]]:
        p6 = float(norm.cdf(-6.0 / self.v_sd))
        p3 = float(norm.cdf(-3.0 / self.v_sd))
        m = self.dim - 1
        return {
            "v": (0.0, self.v_sd),
            "v_sq": (self.v_sd ** 2, math.sqrt(2.0) * (self.v_sd ** 2)),
            "neck_v_lt_m6": (p6, math.sqrt(p6 * (1.0 - p6))),
            "neck_v_lt_m3": (p3, math.sqrt(p3 * (1.0 - p3))),
            "u0": (0.0, 1.0),
            "u_rad_sq": (1.0, math.sqrt(2.0 / m)),
        }


class Banana(AnalyticTarget):
    """Curved banana: z[0] ~ N(0, s0^2), z[1] | z[0] ~ N(b*(z[0]^2 - s0^2), s1^2), z[2:] ~ N(0, I)."""

    family = "banana"

    def __init__(self, dim: int, seed: int, rotated: bool = False, scale_stress: float = 1.0,
                 s0: float = 2.0, s1: float = 0.5, b: float = 0.25):
        if dim < 2:
            raise ValueError("Banana requires dim >= 2")
        super().__init__(dim, seed, rotated=rotated, scale_stress=scale_stress)
        self.s0, self.s1, self.b = float(s0), float(s1), float(b)

    def _latent_log_density_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        z0, z1 = float(z[0]), float(z[1])
        mu1 = self.b * (z0 * z0 - self.s0 * self.s0)
        r1 = z1 - mu1
        lp = -0.5 * (z0 / self.s0) ** 2 - 0.5 * (r1 / self.s1) ** 2
        g = np.empty(self.dim, dtype=float)
        g[0] = -z0 / (self.s0 ** 2) + (r1 / (self.s1 ** 2)) * (2.0 * self.b * z0)
        g[1] = -r1 / (self.s1 ** 2)
        if self.dim > 2:
            rest = z[2:]
            lp -= 0.5 * float(rest @ rest)
            g[2:] = -rest
        return lp, g

    def _latent_exact_draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        z0 = rng.normal(0.0, self.s0, size=(n, 1))
        z1 = self.b * (z0 ** 2 - self.s0 ** 2) + rng.normal(0.0, self.s1, size=(n, 1))
        if self.dim == 2:
            return np.concatenate([z0, z1], axis=1)
        rest = rng.standard_normal((n, self.dim - 2))
        return np.concatenate([z0, z1, rest], axis=1)

    def latent_functionals(self, z: np.ndarray) -> dict[str, np.ndarray]:
        z0, z1 = z[..., 0], z[..., 1]
        r1 = (z1 - self.b * (z0 ** 2 - self.s0 ** 2)) / self.s1
        out = {"z0": z0, "z0_sq": z0 ** 2, "z1": z1, "r1": r1, "r1_sq": r1 ** 2}
        return out

    def exact_truth(self) -> dict[str, tuple[float, float]]:
        # E[z1] = 0, Var(z1) = 2 * b^2 * s0^4 + s1^2
        sd_z1 = math.sqrt(2.0 * (self.b ** 2) * (self.s0 ** 4) + self.s1 ** 2)
        return {
            "z0": (0.0, self.s0),
            "z0_sq": (self.s0 ** 2, math.sqrt(2.0) * (self.s0 ** 2)),
            "z1": (0.0, sd_z1),
            "r1": (0.0, 1.0),
            "r1_sq": (1.0, math.sqrt(2.0)),
        }


class HybridRosenbrock(AnalyticTarget):
    """Block-scaled hybrid Rosenbrock (Pagani et al. 2022): z0 ~ N(mu, 1/(2a)),
    z_j | z_{j-1} ~ N(z_{j-1}^2, 1/(2b)) in chains of length 2."""

    family = "hybrid_rosenbrock"

    def __init__(self, dim: int, seed: int, rotated: bool = False, scale_stress: float = 1.0,
                 a: float = 0.5, b: float = 10.0, mu: float = 1.0):
        if dim < 2:
            raise ValueError("HybridRosenbrock requires dim >= 2")
        super().__init__(dim, seed, rotated=rotated, scale_stress=scale_stress)
        self.a, self.b, self.mu = float(a), float(b), float(mu)
        self.s0 = 1.0 / math.sqrt(2.0 * self.a)
        self.sb = 1.0 / math.sqrt(2.0 * self.b)

    def _latent_log_density_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        z0 = float(z[0])
        rest = z[1:]
        diff = rest - z0 * z0
        lp = -self.a * (z0 - self.mu) ** 2 - self.b * float(diff @ diff)
        g = np.empty(self.dim, dtype=float)
        g[0] = -2.0 * self.a * (z0 - self.mu) + 4.0 * self.b * z0 * float(np.sum(diff))
        g[1:] = -2.0 * self.b * diff
        return lp, g

    def _latent_exact_draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        z0 = rng.normal(self.mu, self.s0, size=(n, 1))
        rest = (z0 ** 2) + rng.normal(0.0, self.sb, size=(n, self.dim - 1))
        return np.concatenate([z0, rest], axis=1)

    def latent_functionals(self, z: np.ndarray) -> dict[str, np.ndarray]:
        z0 = z[..., 0]
        r = (z[..., 1:] - (z0[..., None] ** 2)) / self.sb
        return {
            "z0": z0,
            "z0_sq": (z0 - self.mu) ** 2,
            "r0": r[..., 0],
            "r_rad_sq": np.mean(r ** 2, axis=-1),
        }

    def exact_truth(self) -> dict[str, tuple[float, float]]:
        m = self.dim - 1
        return {
            "z0": (self.mu, self.s0),
            "z0_sq": (self.s0 ** 2, math.sqrt(2.0) * (self.s0 ** 2)),
            "r0": (0.0, 1.0),
            "r_rad_sq": (1.0, math.sqrt(2.0 / m)),
        }


class StudentT(AnalyticTarget):
    """Independent Student-t(nu). Council B2: for nu <= 2, variance is infinite so ONLY
    bounded quantiles / indicators / median-based functionals and log-density are tested."""

    family = "student_t"

    def __init__(self, dim: int, seed: int, rotated: bool = False, scale_stress: float = 1.0, nu: float = 3.0):
        super().__init__(dim, seed, rotated=rotated, scale_stress=scale_stress)
        self.nu = float(nu)
        self._log_c = float(gammaln(0.5 * (self.nu + 1.0)) - gammaln(0.5 * self.nu) - 0.5 * math.log(self.nu * math.pi))
        self._q05 = float(student_t_dist.ppf(0.05, self.nu))
        self._q95 = float(student_t_dist.ppf(0.95, self.nu))
        self._q25 = float(student_t_dist.ppf(0.25, self.nu))
        self._q75 = float(student_t_dist.ppf(0.75, self.nu))

    def _latent_log_density_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        t = 1.0 + (z ** 2) / self.nu
        lp = self.dim * self._log_c - 0.5 * (self.nu + 1.0) * float(np.sum(np.log(t)))
        g = -(self.nu + 1.0) * z / (self.nu + z ** 2)
        return lp, g

    def _latent_exact_draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_t(self.nu, size=(n, self.dim))

    def latent_functionals(self, z: np.ndarray) -> dict[str, np.ndarray]:
        z0 = z[..., 0]
        out = {
            "z0_lt_q05": (z0 <= self._q05).astype(float),
            "z0_lt_q95": (z0 <= self._q95).astype(float),
            "z0_in_iqr": ((z0 >= self._q25) & (z0 <= self._q75)).astype(float),
            "z0_pos": (z0 > 0.0).astype(float),
        }
        # Only include finite-variance moments when mathematically defined (nu > 2, square needs nu > 4)
        if self.nu > 2.0:
            out["z0_mean"] = z0
        if self.nu > 4.0:
            out["z0_sq"] = z0 ** 2
        return out

    def exact_truth(self) -> dict[str, tuple[float, float]]:
        out = {
            "z0_lt_q05": (0.05, math.sqrt(0.05 * 0.95)),
            "z0_lt_q95": (0.95, math.sqrt(0.95 * 0.05)),
            "z0_in_iqr": (0.50, 0.50),
            "z0_pos": (0.50, 0.50),
        }
        if self.nu > 2.0:
            out["z0_mean"] = (0.0, math.sqrt(self.nu / (self.nu - 2.0)))
        if self.nu > 4.0:
            v = self.nu / (self.nu - 2.0)
            m4 = 3.0 * (self.nu ** 2) / ((self.nu - 2.0) * (self.nu - 4.0))
            out["z0_sq"] = (v, math.sqrt(m4 - v ** 2))
        return out


class GaussianMixture(AnalyticTarget):
    """2-component isotropic Gaussian mixture along z[0], with exact rejection-sampled tempered draws
    for any beta in (0, 1] via the envelope (sum a_i)^beta <= sum a_i^beta (Council B2)."""

    def __init__(self, dim: int, seed: int, rotated: bool = False, scale_stress: float = 1.0,
                 separation: float = 3.0, weights: tuple[float, float] = (0.5, 0.5), multimodal: bool = False):
        super().__init__(dim, seed, rotated=rotated, scale_stress=scale_stress)
        self.family = "mixture_separated" if multimodal else "mixture_crossable"
        self.multimodal = multimodal
        self.sep = float(separation)
        self.w = np.asarray(weights, dtype=float)
        self.w = self.w / self.w.sum()
        self.log_w = np.log(self.w)
        self.mus = np.zeros((2, dim), dtype=float)
        self.mus[0, 0] = -0.5 * self.sep
        self.mus[1, 0] = +0.5 * self.sep
        self._log_norm = -0.5 * dim * math.log(2.0 * math.pi)

    def _comp_log_terms(self, z: np.ndarray) -> np.ndarray:
        diffs = z[None, :] - self.mus  # (2, dim)
        return self.log_w + self._log_norm - 0.5 * np.sum(diffs ** 2, axis=-1)

    def _latent_log_density_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        terms = self._comp_log_terms(z)
        lp = float(logsumexp(terms))
        resp = np.exp(terms - lp)
        g = -(resp[:, None] * (z[None, :] - self.mus)).sum(axis=0)
        return lp, g

    def _latent_exact_draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        comps = rng.choice(2, size=n, p=self.w)
        return self.mus[comps] + rng.standard_normal((n, self.dim))

    def _latent_tempered_draws(self, n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
        """Exact draws from pi(z)^beta using envelope q(z) proportional to sum_i w_i^beta N(mu_i, I)^beta."""
        # Each component N(mu_i, I)^beta is proportional to N(mu_i, beta^{-1} I) with same normalizing factor
        env_w = self.w ** beta
        env_w = env_w / env_w.sum()
        sd = 1.0 / math.sqrt(beta)
        out = []
        collected = 0
        while collected < n:
            batch = max(n - collected, 64) * 2
            comps = rng.choice(2, size=batch, p=env_w)
            cand = self.mus[comps] + sd * rng.standard_normal((batch, self.dim))
            # log a_i = log_w_i + log N_i(z)
            diffs = cand[:, None, :] - self.mus[None, :, :]  # (batch, 2, dim)
            log_a = self.log_w[None, :] + self._log_norm - 0.5 * np.sum(diffs ** 2, axis=-1)
            log_target = beta * logsumexp(log_a, axis=-1)
            log_env = logsumexp(beta * log_a, axis=-1)  # >= log_target since beta in (0, 1]
            acc = np.log(rng.uniform(size=batch)) <= (log_target - log_env)
            if acc.any():
                out.append(cand[acc])
                collected += int(acc.sum())
        return np.concatenate(out, axis=0)[:n]

    def latent_functionals(self, z: np.ndarray) -> dict[str, np.ndarray]:
        z0 = z[..., 0]
        return {
            "z0": z0,
            "mode1_weight": (z0 > 0.0).astype(float),
            "z0_sq": z0 ** 2,
        }

    def exact_truth(self) -> dict[str, tuple[float, float]]:
        m1 = float(self.w[0] * self.mus[0, 0] + self.w[1] * self.mus[1, 0])
        m2 = float(self.w[0] * (1.0 + self.mus[0, 0] ** 2) + self.w[1] * (1.0 + self.mus[1, 0] ** 2))
        v1 = m2 - m1 ** 2
        # Fourth moment of mixture of unit-variance normals: sum w_i (mu_i^4 + 6 mu_i^2 + 3)
        m4 = float(sum(self.w[i] * (self.mus[i, 0] ** 4 + 6.0 * self.mus[i, 0] ** 2 + 3.0) for i in range(2)))
        p_pos = float(self.w[0] * norm.sf(-self.mus[0, 0]) + self.w[1] * norm.sf(-self.mus[1, 0]))
        return {
            "z0": (m1, math.sqrt(v1)),
            "mode1_weight": (p_pos, math.sqrt(p_pos * (1.0 - p_pos))),
            "z0_sq": (m2, math.sqrt(m4 - m2 ** 2)),
        }


def make_suite(seed: int, dim: int = 10, include_multimodal: bool = False,
               scale_stress: float = 1.0) -> list[AnalyticTarget]:
    """Build a fresh suite containing BOTH axis-aligned and rotated instances per class (Council D3)."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 2**31 - 1, size=20).tolist()
    suite: list[AnalyticTarget] = [
        IllCondGaussian(dim, seeds[0], rotated=False, scale_stress=scale_stress, kappa=100.0),
        IllCondGaussian(dim, seeds[1], rotated=True, scale_stress=scale_stress, kappa=100.0),
        NealsFunnel(dim, seeds[2], rotated=False, scale_stress=scale_stress),
        NealsFunnel(dim, seeds[3], rotated=True, scale_stress=scale_stress),
        Banana(dim, seeds[4], rotated=False, scale_stress=scale_stress),
        Banana(dim, seeds[5], rotated=True, scale_stress=scale_stress),
        HybridRosenbrock(dim, seeds[6], rotated=False, scale_stress=scale_stress),
        HybridRosenbrock(dim, seeds[7], rotated=True, scale_stress=scale_stress),
        StudentT(dim, seeds[8], rotated=False, scale_stress=scale_stress, nu=1.5),
        StudentT(dim, seeds[9], rotated=True, scale_stress=scale_stress, nu=3.0),
        GaussianMixture(dim, seeds[10], rotated=False, scale_stress=scale_stress, separation=3.0),
        GaussianMixture(dim, seeds[11], rotated=True, scale_stress=scale_stress, separation=3.0),
    ]
    if include_multimodal:
        suite.append(GaussianMixture(dim, seeds[12], rotated=False, scale_stress=scale_stress,
                                     separation=7.0, weights=(0.3, 0.7), multimodal=True))
    return suite
