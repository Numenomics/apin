"""CmdStan NUTS.

Options:
  stancflags=...   passed to stanc (e.g. --O1)
  cxxflags=...     prepended to CXXFLAGS for the model build
  args=...         extra sample arguments, space separated (e.g. "adapt delta=0.95")
"""

from __future__ import annotations

import math

from .. import cmdstan
from ..sampler import RunRequest, Sampler, Timing


class Adapter(Sampler):
    name = "cmdstan"

    @property
    def root(self):
        return cmdstan.locate(self.settings.cmdstan)

    def check(self) -> None:
        self.root

    def describe(self) -> dict:
        return {"cmdstan": str(self.root), "stanc": cmdstan.version(self.root),
                "stancflags": self.opt("stancflags", ""), "cxxflags": self.opt("cxxflags", ""),
                "args": self.opt("args", "")}

    def build(self, req: RunRequest):
        return cmdstan.compile_model(self.root, req.model.stan_file, self.settings.build_dir,
                                     self.opt("stancflags", ""), self.opt("cxxflags", ""))

    def run(self, req: RunRequest, inits: list | None = None) -> Timing:
        exe = self.build(req)
        extra = (self.opt("args") or "").split()
        csvs = [req.out_dir / f"chain_{c}.csv" for c in range(req.chains)]
        wall = cmdstan.run_chains(
            lambda c: cmdstan.sample_argv(exe, req.data_file, req.seed, c, csvs[c], req.warmup, req.draws,
                                          init=inits[c] if inits else None, extra=extra),
            req.chains,
        )
        return timing_from_csvs(csvs, wall)


def timing_from_csvs(csvs, wall: float) -> Timing:
    """Times and gradient counts from CmdStan's CSVs (`gradients`). When a count is not complete,
    `uncounted` says why and the run earns no per-evaluation score."""
    warm, samp, chain, grads, grads_w, div, uncounted, searches = [], [], [], 0, 0, 0, [], []
    for path in csvs:
        f = cmdstan.read_csv(path, columns=["n_leapfrog__", "divergent__", "stepsize__", "accept_stat__"])
        warm.append(f.warmup_s)
        samp.append(f.sampling_s)
        chain.append((f.warmup_s or 0.0) + (f.sampling_s or 0.0))
        g_w, g_s, ks, problem = gradients(f)
        if problem:
            uncounted.append(f"{path.name}: {problem}")
        grads = None if grads is None or g_s is None else grads + g_s
        grads_w = None if grads_w is None or g_w is None else grads_w + g_w
        searches.append(ks)
        d = f.column("divergent__")
        div += int(d.sum()) if d is not None else 0
    return Timing(wall_s=wall, chain_s=chain, warmup_s=warm, sampling_s=samp,
                  grads_sampling=grads, grads_warmup=grads_w, divergences=div, uncounted="; ".join(uncounted),
                  extra={"step_size_searches": searches})


SEARCH_TOLERANCE = 1e-4  # in log2 of the step size; the rebuilt trace matches CmdStan's to about 3e-7


def gradients(f: cmdstan.StanCsv) -> tuple[int | None, int | None, list[int] | None, str]:
    """(warmup gradients, sampling gradients, step-size search lengths, why not counted) of one
    CmdStan 2.39 NUTS chain, from the source (stan/src/stan/mcmc/hmc):

    - every transition evaluates the gradient at its start (base_nuts.hpp, `hamiltonian_.init`)
      and once per leapfrog step (`n_leapfrog__`);
    - every step-size search (base_hmc.hpp `init_stepsize`: once before warmup and once after each
      metric update) costs 2k + 4 gradients, where k is the number of times it halved or doubled
      the step size. k is found by rebuilding the dual averaging (stepsize_adaptation.hpp) from
      `accept_stat__` and `stepsize__`; a trace that does not rebuild leaves warmup uncounted.

    The initialisation attempts (services/util/initialize.hpp) are not counted: one density and
    one gradient per accepted attempt, at most 100 attempts, so 2 to 200 evaluations per chain."""
    cfg = f.config
    lf = f.column("n_leapfrog__")
    if cfg.get("thin", "1") != "1":
        return None, None, None, f"thin = {cfg['thin']}, so most transitions are not written"
    if lf is None:
        return None, None, None, "no n_leapfrog__ column"
    sampling = int(lf.sum()) + len(lf)
    lf_w = f.warmup_column("n_leapfrog__")
    if lf_w is None:
        return None, sampling, None, ""  # warmup not saved: unknown
    ks, problem = _searches(f, cfg)
    if ks is None:
        return None, sampling, None, f"step size searches not rebuilt: {problem}"
    return int(lf_w.sum()) + len(lf_w) + sum(2 * k + 4 for k in ks), sampling, ks, ""


def _searches(f: cmdstan.StanCsv, cfg: dict) -> tuple[list[int] | None, str]:
    eps, acc = f.warmup_column("stepsize__"), f.warmup_column("accept_stat__")
    try:
        if cfg["engine"] != "nuts" or cfg["engaged"] not in ("1", "true"):
            return None, "not adaptive NUTS"
        if float(cfg["stepsize_jitter"]) != 0:
            return None, "stepsize_jitter is not 0"
        n_warm, eps0 = int(cfg["num_warmup"]), float(cfg["stepsize"])
        delta, gamma, kappa, t0 = (float(cfg[k]) for k in ("delta", "gamma", "kappa", "t0"))
        ups = _metric_updates(n_warm, int(cfg["init_buffer"]), int(cfg["term_buffer"]), int(cfg["window"]))
    except (KeyError, ValueError) as e:
        return None, f"header lacks {e}"
    if eps is None or acc is None or len(eps) != n_warm or n_warm < 20 or ups[-1] >= n_warm - 1:
        return None, "stepsize__, accept_stat__ or warmup rows missing"

    def k_of(ratio):
        r = math.log2(ratio)
        return round(abs(r)), abs(r - round(r))

    k, off = k_of(eps[0] / eps0)
    ks, worst = [k], off
    mu, n, s_bar, ups = math.log(10 * eps0), 0, 0.0, set(ups)
    for t in range(n_warm - 1):
        n += 1
        eta = 1.0 / (n + t0)
        s_bar = (1 - eta) * s_bar + eta * (delta - min(acc[t], 1.0))
        learned = math.exp(mu - s_bar * math.sqrt(n) / gamma)
        if t in ups:
            k, off = k_of(eps[t + 1] / learned)
            ks.append(k)
            mu, n, s_bar = math.log(10 * eps[t + 1]), 0, 0.0
        else:
            off = abs(math.log2(eps[t + 1] / learned))
        worst = max(worst, off)
    if not worst < SEARCH_TOLERANCE:
        return None, f"the trace is {worst:.2g} off in log2"
    return ks, ""


def _metric_updates(n_warm: int, init_buffer: int, term_buffer: int, window: int) -> list[int]:
    """The warmup transitions (from 0) after which the metric is updated (windowed_adaptation.hpp)."""
    if init_buffer + window + term_buffer > n_warm:
        init_buffer, term_buffer = int(0.15 * n_warm), int(0.10 * n_warm)
        window = n_warm - (init_buffer + term_buffer)
    last = n_warm - term_buffer - 1
    size, nxt, out = window, init_buffer + window - 1, []
    for c in range(n_warm):
        if c == nxt:
            out.append(c)
            if nxt != last:
                size *= 2
                nxt = c + size
                if nxt != last and nxt + 2 * size >= n_warm - term_buffer:
                    nxt = last
    return out
