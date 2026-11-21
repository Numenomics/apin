"""Lane A4 candidate: robust whitened transport with gated triangular un-warping + SAS tails.

Latent z -> theta map composition (all closed-form, O(d^2)):
  1. elementwise sinh-arcsinh tail warp  u_i = sinh(delta_i * arcsinh(z_i))
  2. gated triangular child map          y_0 = u_0
                                         y_j = exp(a_j u_0 + b_j) * u_j
                                               + c_j u_0^2 + m_j u_0 + g_j
  3. affine whitening                    theta = loc + Lp @ y

Every nonlinear term is fitted from short latent-space HMC warmup draws and gated by a
conservative t-statistic, so near-Gaussian targets keep the map effectively affine:
rank-correlation PCA whitening handles rotated ill-conditioned Gaussians, the log-scale
regression un-funnels scale-mixture targets, the quadratic mean shift straightens
banana/Rosenbrock curvature, and the tail warp Gaussianizes heavy-tailed marginals.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, rankdata

requires = ["grad"]

_MED_ABS_N = math.log(0.6744897501960817)  # log median(|N(0,1)|)
_LOG_TWO = math.log(2.0)
_QUANT_P = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
_QUANT_Z = norm.ppf(_QUANT_P)
_EXP_CLIP = 300.0


# ---------------------------------------------------------------------------
# Transport map (forward / pullback_grad / inverse)
# ---------------------------------------------------------------------------

def _sas_log_terms(delta, z):
    """Per-coordinate log|ds_i/dz_i| and d/dz_i of it for u_i = sinh(delta_i asinh z_i)."""
    uu = delta * np.arcsinh(z)
    inv_r = 1.0 / np.sqrt(1.0 + z * z)
    log_sp = np.log(np.abs(delta)) + np.logaddexp(uu, -uu) - _LOG_TWO - 0.5 * np.log1p(z * z)
    dlog_sp = delta * np.tanh(uu) * inv_r - z * inv_r * inv_r
    return log_sp, dlog_sp


def _scale_factors(t, u0):
    """exp(a_j u_0 + b_j) clipped for numerical safety."""
    return np.exp(np.clip(t["tri_a"] * u0 + t["tri_b"], -_EXP_CLIP, _EXP_CLIP))


def forward(transport_tuning, z):
    t = transport_tuning
    z_arr = np.asarray(z, dtype=np.float64)
    u = np.sinh(t["delta"] * np.arcsinh(z_arr))
    e = _scale_factors(t, u[0])
    y = u.copy()
    y[1:] = e[1:] * u[1:] + t["tri_c"][1:] * u[0] ** 2 + t["tri_m"][1:] * u[0] + t["tri_g"][1:]
    theta = t["loc"] + t["chol"] @ y
    log_sp, _ = _sas_log_terms(t["delta"], z_arr)
    ldj = t["logdet0"] + float(np.sum(log_sp)) + float(np.sum(t["tri_a"] * u[0] + t["tri_b"]))
    return theta, ldj


def pullback_grad(transport_tuning, z, grad_theta):
    t = transport_tuning
    z_arr = np.asarray(z, dtype=np.float64)
    g = np.asarray(grad_theta, dtype=np.float64)
    log_sp, dlog_sp = _sas_log_terms(t["delta"], z_arr)
    sp = np.exp(log_sp)
    u = np.sinh(t["delta"] * np.arcsinh(z_arr))
    e = _scale_factors(t, u[0])
    v = t["chol"].T @ g
    u0 = u[0]
    # d y_j / d u_0 = a_j e_j u_j + 2 c_j u_0 + m_j  (j >= 1); triangular => M^T v:
    coeff0 = t["tri_a"][1:] * e[1:] * u[1:] + 2.0 * t["tri_c"][1:] * u0 + t["tri_m"][1:]
    w = e * v
    w[0] = v[0] + float(np.dot(coeff0, v[1:]))
    gz = sp * w
    gz[0] += sp[0] * float(np.sum(t["tri_a"]))
    return gz + dlog_sp


def inverse(transport_tuning, theta):
    t = transport_tuning
    th = np.asarray(theta, dtype=np.float64)
    w = t["inv_chol"] @ (th - t["loc"])
    u = np.empty_like(w)
    u[0] = w[0]
    e_inv = np.exp(np.clip(-(t["tri_a"] * w[0] + t["tri_b"]), -_EXP_CLIP, _EXP_CLIP))
    quad = t["tri_c"] * (w[0] ** 2) + t["tri_m"] * w[0] + t["tri_g"]
    u[1:] = (w[1:] - quad[1:]) * e_inv[1:]
    arg = np.arcsinh(np.clip(u, -1e100, 1e100)) / t["delta"]
    return np.sinh(np.clip(arg, -340.0, 340.0))


# ---------------------------------------------------------------------------
# Tuning constructors
# ---------------------------------------------------------------------------

def _affine_tuning(loc, chol, inv_chol, logdet0):
    d = loc.size
    return {
        "eps": 0.85,
        "loc": np.asarray(loc, dtype=np.float64),
        "chol": np.asarray(chol, dtype=np.float64),
        "inv_chol": np.asarray(inv_chol, dtype=np.float64),
        "logdet0": float(logdet0),
        "delta": np.ones(d, dtype=np.float64),
        "tri_a": np.zeros(d, dtype=np.float64),
        "tri_b": np.zeros(d, dtype=np.float64),
        "tri_c": np.zeros(d, dtype=np.float64),
        "tri_m": np.zeros(d, dtype=np.float64),
        "tri_g": np.zeros(d, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Map fitting from warmup draws
# ---------------------------------------------------------------------------

def _fit_affine(S):
    n, d = S.shape
    loc = np.median(S, axis=0)
    X = S - loc
    s = 1.4826 * np.median(np.abs(X), axis=0)
    s = np.clip(s, 1e-6, 1e12)
    if n >= d + 6:
        ranks = rankdata(X, axis=0)
        ranks -= 0.5 * (n + 1)
        corr = np.corrcoef(ranks, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        corr = 2.0 * np.sin(np.pi * np.clip(corr, -0.999, 0.999) / 6.0)
        cov = corr * np.outer(s, s)
        lam = n / (n + d + 6.0)
        cov = lam * cov + (1.0 - lam) * np.diag(s * s)
    else:
        cov = np.diag(s * s)
    cov += (1e-12 * float(np.mean(np.diag(cov)))) * np.eye(d)
    vals, vecs = np.linalg.eigh(cov)
    top = float(np.max(vals))
    vals = np.clip(vals, max(1e-300, top * 1e-10), None)
    root = np.sqrt(vals)
    chol = vecs @ np.diag(root)
    inv_chol = np.diag(1.0 / root) @ vecs.T
    return _affine_tuning(loc, chol, inv_chol, float(np.sum(np.log(root))))


def _parent_scan(W, LW):
    """Pick the scale-parent coordinate by aggregate log-scale slope significance."""
    n, d = W.shape
    best_c, best_score, cache = -1, -1.0, {}
    for c in range(d):
        xc = W[:, c] - np.mean(W[:, c])
        sxx = float(xc @ xc)
        if sxx <= 1e-12:
            continue
        centred = LW - np.mean(LW, axis=0)
        slope = (xc @ centred) / sxx
        resid = centred - np.outer(xc, slope)
        sig2 = np.sum(resid * resid, axis=0) / max(1, n - 2)
        tt = slope / (np.sqrt(sig2 / sxx) + 1e-30)
        tt[c] = 0.0
        score = float(np.sum(np.clip(tt - 4.0, 0.0, None)))
        cache[c] = (slope, tt)
        if score > best_score:
            best_c, best_score = c, score
    return best_c, cache


def _heteroscedastic_direction(W, LW):
    """Aggregate ridge direction along which child log-scales vary (Householder frame rotation)."""
    n, d = W.shape
    centred = LW - np.mean(LW, axis=0)
    G = W.T @ W + (1e-2 * n) * np.eye(d)
    acc = np.zeros(d)
    votes = 0.0
    try:
        for j in range(d):
            bj = np.linalg.solve(G, W.T @ centred[:, j])
            pred = W @ bj
            ss_fit = float(pred @ pred)
            sse = float(np.sum((centred[:, j] - pred) ** 2))
            r2 = ss_fit / max(1e-300, ss_fit + sse)
            F = (r2 / max(1, d - 1)) / max(1e-12, (1.0 - r2) / max(1, n - d))
            if F > 16.0 and np.all(np.isfinite(bj)):
                unit = bj / (float(np.linalg.norm(bj)) + 1e-300)
                if votes > 0.0 and float(unit @ acc) < 0.0:
                    unit = -unit
                acc += unit * math.sqrt(F - 16.0)
                votes += 1.0
    except Exception:
        return np.eye(d), False
    if votes == 0.0:
        return np.eye(d), False
    b = acc / (float(np.linalg.norm(acc)) + 1e-300)
    e0 = np.zeros(d)
    e0[0] = 1.0
    u = b - e0
    nu = float(u @ u)
    if not np.all(np.isfinite(b)) or nu < 1e-12:
        return np.eye(d), False
    Q = np.eye(d) - 2.0 * np.outer(u, u) / nu
    return Q, True


def _fit_nonlinear(t, S):
    W = (S - t["loc"]) @ t["inv_chol"].T
    LW = np.clip(np.log(np.abs(W) + 1e-12), -12.0, 12.0)
    n, d = W.shape
    c, cache = _parent_scan(W, LW)
    direct = False
    if c >= 0:
        slope, tt = cache[c]
        sig = tt > 4.0
        sig[c] = False
        cnt = int(np.sum(sig))
        need = max(2, int(round(0.25 * (d - 1))))
        direct = bool(cnt >= need and np.median(slope[sig]) >= 0.08)
    if not direct:
        Q, found = _heteroscedastic_direction(W, LW)
        if found:
            # Rotate the whitening frame so the scale-parent direction becomes latent coord 0.
            t["chol"] = t["chol"] @ Q
            t["inv_chol"] = Q.T @ t["inv_chol"]
            W = W @ Q
            LW = np.clip(np.log(np.abs(W) + 1e-12), -12.0, 12.0)
            c, cache = _parent_scan(W, LW)
    if c < 0:
        return t
    slope, tt = cache[c]
    sig = tt > 4.0
    sig[c] = False
    cnt = int(np.sum(sig))
    need = max(2, int(round(0.25 * (d - 1))))
    tri_on = bool(cnt >= need and np.median(slope[sig]) >= 0.08) if cnt else False

    # Quadratic mean shift (banana / Rosenbrock straightening), gated per child.
    wc = W[:, c]
    M = np.column_stack([np.ones(n), wc, wc * wc])
    G = M.T @ M + (1e-9 * float(np.trace(M.T @ M)) / 3.0) * np.eye(3)
    B = np.linalg.solve(G, M.T @ W)
    resid_q = W - M @ B
    sig2q = np.sum(resid_q * resid_q, axis=0) / max(1, n - 3)
    tc = B[2] / (np.sqrt(sig2q * G[2, 2]) + 1e-30)
    quad_on = tc > 4.0
    t["tri_c"] = np.where(quad_on, B[2], 0.0)
    t["tri_m"] = np.where(quad_on, B[1], 0.0)
    t["tri_g"] = np.where(quad_on, B[0], 0.0)

    # Log-scale child map (funnel un-warping), median-centred so latent child median matches |N(0,1)|.
    tri_a = np.zeros(d)
    tri_b = np.zeros(d)
    if tri_on:
        for j in range(d):
            if j != c and sig[j] and 0.05 <= slope[j] <= 3.0:
                tri_a[j] = slope[j]
                tri_b[j] = float(np.median(LW[:, j] - slope[j] * wc)) - _MED_ABS_N
        if not np.any(tri_a > 0.0):
            tri_a[:] = 0.0
            tri_b[:] = 0.0
    t["tri_a"] = tri_a
    t["tri_b"] = tri_b

    # Tail-warp fit on TRI-normalized coordinates (marginal quantile matching).
    quad_pred = np.outer(wc * wc, t["tri_c"]) + np.outer(wc, t["tri_m"]) + t["tri_g"][None, :]
    U = (W - quad_pred) * np.exp(np.clip(-(np.outer(wc, tri_a) + tri_b[None, :]), -_EXP_CLIP, _EXP_CLIP))
    grid = np.exp(np.linspace(math.log(0.5), math.log(2.6), 27))
    delta = np.ones(d)
    for i in range(d):
        if tri_a[i] > 0.0 or quad_on[i]:
            continue  # tri-reshaped children are |eps|-shaped, not symmetric-heavy: keep SAS identity
        q = np.quantile(U[:, i], _QUANT_P)
        sse_id = float(np.sum((np.sinh(_QUANT_Z) - q) ** 2))
        sses = np.array([float(np.sum((np.sinh(g * _QUANT_Z) - q) ** 2)) for g in grid])
        k = int(np.argmin(sses))
        if k in (0, grid.size - 1):
            continue
        denom = sses[k + 1] - 2.0 * sses[k] + sses[k - 1]
        step = 0.5 * (grid[k + 1] - grid[k - 1]) * (sses[k + 1] - sses[k - 1]) / (denom + 1e-300)
        g_hat = float(np.clip(grid[k] - step, grid[k - 1], grid[k + 1]))
        if sses[k] < 0.92 * sse_id and abs(g_hat - 1.0) >= 0.04:
            delta[i] = float(np.clip(g_hat, 0.55, 2.5))
    t["delta"] = delta
    return t


def _fit_map(S):
    """Fit the full map; returns (tuning, meta) with scale-parent metadata for warmup targeting."""
    t = _fit_affine(S)
    meta = {"parent": -1}
    try:
        if S.shape[0] >= S.shape[1] + 6:
            t = _fit_nonlinear(t, S)
            c, _ = _parent_scan((S - t["loc"]) @ t["inv_chol"].T,
                                np.clip(np.log(np.abs((S - t["loc"]) @ t["inv_chol"].T) + 1e-12), -12.0, 12.0))
            meta["parent"] = int(c)
    except Exception:
        pass
    if not np.all(np.isfinite(t["chol"])) or not np.all(np.isfinite(t["inv_chol"])):
        t = _fit_affine(S[-8:])
        meta["parent"] = -1
    return t, meta


# ---------------------------------------------------------------------------
# Warmup: mode polish, curvature seeding, latent-space HMC rounds
# ---------------------------------------------------------------------------

def _mode_polish(oracle, x, lp, g):
    best_x, best_lp = x.copy(), lp
    alpha = 0.1 / max(1.0, float(np.max(np.abs(g))))
    for _ in range(12):
        if not np.all(np.isfinite(g)):
            break
        xn = best_x + alpha * g
        lpn, gn = oracle.log_density_gradient(xn)
        for _ in range(2):
            if math.isfinite(lpn) and np.all(np.isfinite(gn)):
                break
            alpha *= 0.35
            xn = best_x + alpha * g
            lpn, gn = oracle.log_density_gradient(xn)
        if not (math.isfinite(lpn) and np.all(np.isfinite(gn))):
            break
        if lpn <= best_lp:
            alpha *= 0.5
        else:
            step = xn - best_x
            dg = float(np.dot(step, g - gn))
            alpha = float(np.clip(abs(float(step @ step) / dg) if dg > 1e-12 else alpha * 2.0, 1e-6, 10.0))
            best_x, best_lp = xn, lpn
        g = gn
    return best_x, best_lp


def _curvature_scales(oracle, x, d):
    """Diagonal curvature probe at x -> per-coordinate latent scale (2d grad evals)."""
    h = 0.02
    s = np.ones(d)
    for i in range(d):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        _, gp = oracle.log_density_gradient(xp)
        _, gm = oracle.log_density_gradient(xm)
        if np.all(np.isfinite(gp)) and np.all(np.isfinite(gm)):
            curv = abs((gp[i] - gm[i]) / (2.0 * h))
            s[i] = 1.0 / math.sqrt(min(max(curv, 1e-4), 1e8))
    return np.clip(s, 1e-3, 1e6)


def _latent_round(oracle, t, z, lp_z, g_z, eps, rng, n_steps, collect_from):
    d = z.size
    samples = []
    for step in range(n_steps):
        n = int(math.ceil(2.0 / eps))
        n_inner = int(rng.integers(max(1, n // 2), n + n // 2 + 1))
        p = rng.standard_normal(d)
        h0 = lp_z - 0.5 * float(p @ p)
        z1, lp1, g1, p1 = z, lp_z, g_z, p + 0.5 * eps * g_z
        acc = 0.0
        ok = True
        for i in range(n_inner):
            z1 = z1 + eps * p1
            th1, ldj1 = forward(t, z1)
            lp_th1, g_th1 = oracle.log_density_gradient(th1)
            if not (math.isfinite(lp_th1) and math.isfinite(ldj1) and np.all(np.isfinite(g_th1))):
                ok = False
                break
            lp1 = float(lp_th1) + float(ldj1)
            g1 = pullback_grad(t, z1, g_th1)
            if not np.all(np.isfinite(g1)):
                ok = False
                break
            if i < n_inner - 1:
                p1 = p1 + eps * g1
        if ok:
            p1 = p1 + 0.5 * eps * g1
            h1 = lp1 - 0.5 * float(p1 @ p1)
            acc = math.exp(min(0.0, h1 - h0)) if math.isfinite(h1) else 0.0
            if rng.uniform() < acc:
                z, lp_z, g_z = z1, lp1, g1
        eps = float(np.clip(eps * math.exp(0.08 * (acc - 0.8)), 0.06, 1.0))
        if step >= collect_from:
            samples.append(forward(t, z)[0])
    return z, lp_z, g_z, eps, samples


def adapt_transport(oracle, budget, rng):
    d = int(oracle.dim)
    draws = max(20, int(getattr(budget, "draws", 100)))

    x = np.zeros(d, dtype=np.float64)
    lp, g = oracle.log_density_gradient(x)
    tries = 0
    while not (math.isfinite(lp) and np.all(np.isfinite(g))) and tries < 8:
        x = rng.normal(0.0, 0.5, size=d)
        lp, g = oracle.log_density_gradient(x)
        tries += 1
    if not (math.isfinite(lp) and np.all(np.isfinite(g))):
        x = np.zeros(d, dtype=np.float64)
        lp, g = oracle.log_density_gradient(x)

    x, lp = _mode_polish(oracle, x, lp, g)
    scale = _curvature_scales(oracle, x, d)
    tuning = _affine_tuning(x, np.diag(scale), np.diag(1.0 / scale), float(np.sum(np.log(scale))))

    def refresh(z, t):
        theta, ldj = forward(t, z)
        lp_th, g_th = oracle.log_density_gradient(theta)
        if math.isfinite(lp_th) and np.all(np.isfinite(g_th)) and math.isfinite(ldj):
            g_z = pullback_grad(t, z, g_th)
            if np.all(np.isfinite(g_z)):
                return float(lp_th) + float(ldj), g_z
        z = np.zeros(d, dtype=np.float64)
        theta, ldj = forward(t, z)
        lp_th, g_th = oracle.log_density_gradient(theta)
        return float(lp_th) + float(ldj), pullback_grad(t, z, g_th)

    z = np.zeros(d, dtype=np.float64)
    eps = 0.85
    samples: list[np.ndarray] = []
    parent = -1
    plans = [int(np.clip(draws // 3, 14, 30)), int(np.clip(draws // 2, 20, 50))]
    for n_steps in plans:
        lp_z, g_z = refresh(z, tuning)
        z, _, _, eps, new = _latent_round(
            oracle, tuning, z, lp_z, g_z, eps, rng, n_steps, max(2, n_steps // 4)
        )
        samples.extend(new)
        try:
            tuning, meta = _fit_map(np.asarray(samples, dtype=np.float64))
            parent = meta["parent"]
        except Exception:
            pass

    if parent >= 0:
        # Dispersed parent-anchored sub-chains: span the heteroscedastic axis so the
        # log-scale regression sees the full range (funnel neck to mouth).
        levels = np.array([-3.0, -1.5, 0.0, 1.5])
        for level in levels:
            zs = np.zeros(d, dtype=np.float64)
            zs[parent] = level
            theta_s = forward(tuning, zs)[0]
            lp_s = oracle.log_density_gradient(theta_s)[0]
            if not math.isfinite(lp_s):
                continue
            lp_z, g_z = refresh(zs, tuning)
            zs, _, _, eps, new = _latent_round(
                oracle, tuning, zs, lp_z, g_z, eps, rng, 10, 4
            )
            samples.extend(new)
        try:
            tuning, _ = _fit_map(np.asarray(samples, dtype=np.float64))
        except Exception:
            pass
    else:
        lp_z, g_z = refresh(z, tuning)
        n3 = int(np.clip(draws // 3, 14, 32))
        z, _, _, eps, new = _latent_round(oracle, tuning, z, lp_z, g_z, eps, rng, n3, max(2, n3 // 4))
        samples.extend(new)
        try:
            tuning, _ = _fit_map(np.asarray(samples, dtype=np.float64))
        except Exception:
            pass

    tuning["eps"] = float(eps)

    def _eps_for_region(z0, eps_hi):
        """Smallest eps in a geometric ladder reaching >=0.5 acceptance from z0 (stuck-kernel guard)."""
        lp0, g0 = refresh(z0, tuning)
        for frac in (1.0, 0.5, 0.25, 0.125, 0.0625):
            e = eps_hi * frac
            n_i = int(math.ceil(2.0 / max(e, 1e-3)))
            p = rng.standard_normal(z0.size)
            h0 = lp0 - 0.5 * float(p @ p)
            z1, lp1, g1, p1 = z0, lp0, g0, p + 0.5 * e * g0
            for i in range(n_i):
                z1 = z1 + e * p1
                th1, ldj1 = forward(tuning, z1)
                lp_th1, g_th1 = oracle.log_density_gradient(th1)
                if not (math.isfinite(lp_th1) and math.isfinite(ldj1) and np.all(np.isfinite(g_th1))):
                    break
                lp1 = float(lp_th1) + float(ldj1)
                g1 = pullback_grad(tuning, z1, g_th1)
                if not np.all(np.isfinite(g1)):
                    break
                if i < n_i - 1:
                    p1 = p1 + e * g1
            else:
                p1 = p1 + 0.5 * e * g1
                h1 = lp1 - 0.5 * float(p1 @ p1)
                if math.isfinite(h1) and math.exp(min(0.0, h1 - h0)) >= 0.5:
                    return float(np.clip(e, 0.06, 1.0))
        return 0.06

    eps = min(eps, _eps_for_region(rng.normal(0.0, 1.5, size=d), eps))
    eps = min(eps, _eps_for_region(rng.normal(0.0, 2.5, size=d), eps))
    tuning["eps"] = float(np.clip(eps, 0.06, 1.0))
    # Choose x0 as the best finite-density candidate: current latent position mapped under
    # the final fit can land off-support when late refits rotate/scale the map.
    candidates = [forward(tuning, z)[0]]
    if len(samples):
        candidates.append(np.asarray(samples[-1], dtype=np.float64))
    candidates.append(np.asarray(tuning["loc"], dtype=np.float64).copy())
    x0 = None
    best_lp = -math.inf
    for cand in candidates:
        lp_c, _ = oracle.log_density_gradient(cand)
        if math.isfinite(lp_c) and lp_c > best_lp:
            best_lp = lp_c
            x0 = cand
    if x0 is None:
        x0 = np.zeros(d, dtype=np.float64)
        oracle.log_density_gradient(x0)
    return tuning, x0
