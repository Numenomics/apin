# Multi-Lane 1-Hour Autoresearch Dogfood Record (2026-11-21)

**Harness commit:** `746ca14` (`Numenomics/apin`, FIFO file-locked evaluation queue `harness/queue.py`, per-`--run-id` isolated `incumbent.py`/`archive/`, `horizon_steps=500`, `--repeats 2`, snapshot-at-start).
**Setup:** Four autonomous agents ran concurrently for ~1 hour on four isolated lanes (`A0`, `A1`, `A2`, `A3`) sharing the host FIFO eval lock (zero CPU contention, zero file collisions), then all four final incumbents were scored side-by-side on 5 public Stan posteriors (`blr`, `eight_schools_noncentered`, `gp_regr`, `garch11`, `eight_schools_centered`, 3 reps each) against `a1_diag` and `cmdstan`.

## Side-by-Side Comparison & Public-vs-Real Alignment

| Agent / Model | Lane | Trials / Keeps | Tier 1 (`dim=6`) Gain | `dim=16` Pass | Real Stan Gate (5 models) | Real `ESS/cost` vs. `a1_diag` [95% CI] | Real `gate_speed` vs. `a1_diag` [95% CI] | Winning Mechanism |
|---|---|---:|---:|:---:|---:|---:|---:|---|
| **Muse Spark 1.3** (`opencode-go/muse-spark-1.3-contributor`) | **A2** (Funnel/Multiscale) | **19 / 7** | ~2.0× | Yes (`0.00025`) | **4 / 5** (`8s_c` R-hat **1.014**) | `0.74×` `[0.68, 0.84]` | `0.87×` `[0.66, 1.16]` | Randomized-refresh GHMC (`aux_spec="normal"`, `P_FULL=0.5`, `cos φ=0.9`, momentum flip on reject) + 30% Barker fallback |
| **GLM-5.3-Flash** (`zai-coding-plan/glm-5.3-flash`) | **A1** (Metric) | 10 / 5 | **~2.1×** | Yes (`0.00082`) | **3 / 5** (`blr`, `gp_regr`, `garch11`) | **`1.99×`** `[1.23, 4.66]` (`p=0.000`) | **`3.89×`** `[1.95, 6.38]` (`p=0.000`) | MALA-floor warmup + `{0.5w, 0.75w}` schedule + **50/50 variance/MAD robust diagonal** (for `StudentT(1.5)`) |
| **DeepSeek V4.1 Flash** (`opencode-go/deepseek-v4.1-flash`) | **A3** (Integrators/Traj) | 5 (+25 probes) / 2 | **13.90×** | Yes (`0.00164`) | **1 / 5** (`gp_regr` only) | `1.13×` `[0.84, 1.44]` | `2.61×` `[1.62, 3.92]` | Surrogate multivariate-$t_4$ independence + local preconditioned RWM (0 transition gradients) |
| **GLM-5.3-Flash** (`zai-coding-plan/glm-5.3-flash`) | **A0** (Ground-up, no priors) | 6 / 2 | **3.32×** vs. RWM | Yes (`0.00125`) | **1 / 5** (`gp_regr` only) | `1.00×` `[0.92, 1.06]` | `2.48×` `[1.75, 4.39]` | Built from 43-LOC RWM stub: 50/50 MAD-metric HMC (geometric `L`, `aux_spec="normal"`) + preconditioned RWM |

*Baselines on the exact same 5 models × 3 reps: `a1_diag` passed **4 / 5** (`1.00×`), `cmdstan` passed **4 / 5** (`0.95×` `ESS/cost`, `14.7×` `ESS/s`). Zero `wrong` verdicts occurred across all four agents.*

## Key Lessons Driving Post-Dogfood Upgrades
1. **Low-dim (`dim=6`) Tier 1 allows density-only RWM / independence-Metropolis shortcuts:** Both `DeepSeek A3` (`13.9×` via zero-gradient surrogate-$t_4$ + RWM) and `GLM A0` (bare RWM beat pure HMC on `rosenbrock`/`banana`/`funnel`) won big at `dim=6` yet passed only `1 / 5` real Stan models (`gp_regr`, `D=3`). Fixed immediately by upgrading Tier 1's CLI default to a **two-dimension ladder (`--dims 6,20`)**.
2. **Honest warmup credit without horizon distortion:** Upgraded Tier 1 to a **dual-horizon geometric mean (`H_short=150` & `H_long=1000`)** with both horizons emitted as secondary `METRIC` monitors, so genuinely fast-warmup methods get real credit without kneecapping them or letting warmup starvation dominate long runs.
3. **Exact-draw global-balance & provenance gates caught every genuine bug on first try:** Caught `DeepSeek`'s broken $t$-proposal (`rad_sq z=5.65`), `GLM A0`'s malformed leapfrog endpoint (`provenance`), and `GLM A0`'s `N(0,M)` vs `N(0,M⁻¹)` momentum mismatch (`aux_rad_sq z=740`), while passing `Muse A2`'s non-reversible persistent-momentum GHMC on 18/19 trials.
4. **Two minor harness bugs fixed:** `normalize_aux_spec` now coerces `np.ndarray` fields to JSON-plain lists (preventing `json.dumps` crashes on `adapt_meta.json`), and `_gandy_scott_rank_check` uses a finite-sample calibrated threshold (fixing the `|z|=3.98 > 3.93` `T0001` baseline flake hit by both A2 and A3).

---

## Follow-Up Run on Upgraded Harness (`ca9d5af`): Black-Box PDMP Exploration (`glm53_pdmp_1h`, Lane A3)

Running `GLM-5.3-Flash` (`zai-coding-plan/glm-5.3-flash`) on task `a3-pdmp-blackbox` (`codex/cards/pdmp-boomerang-zigzag.md`, `--dims 6,20 --repeats 2`, dual-horizon `H=150/1000`, calibrated Gandy–Scott ranks):
- **Zero rank flakes on `T0001`**, `40–85 s/trial`, `aux_spec="normal"` worked cleanly out of the box.
- **Adjusted elliptical Boomerang orbit** (`z' = z cos t + u sin t`, `u' = z sin t - u cos t`, 1 density eval per mini-step, exact $\pi \otimes \mathcal{N}(0, I)$ global balance, zero thinning-bound risk) beat diagonal HMC **2–8× at the typical set** on funnel/banana/Rosenbrock, but pure Boomerang could not descend radially from overdispersed starts (`T0002` failed mixing at `d=20`).
- **Exactness counterexample caught by invariance suite:** Position-dependent gating (`if r² > gate: HMC else orbit`) broke MH reverse-proposal pairing (`lp` drift `z = 5.21`). Replacing it with a state-independent coin (`HMC p=0.6 | Boomerang orbit`, `T0003`) passed all exact-draw invariance checks cleanly (`max_z = 1.1–2.8`, Tier-1 long-horizon `4.52e-4` vs. baseline `4.22e-4`, dual-horizon ratio `0.93×`).
- **Real Stan posteriors (`a3_glm53_pdmp_1h`, T0003 snapshot, 5 models × 3 reps):** Ranked **#1 overall** among all 5-model arms — **4 / 5 passed** (`blr`, `eight_schools_noncentered`, `gp_regr 0.0583/cost` [**1.45× CmdStan**], `garch11 0.0172/cost` [**1.70× CmdStan**]; `eight_schools_centered` `unproven: R-hat 1.270`), achieving **`1.02× [0.75, 1.39]` `ESS/cost`** and **`1.47× [0.97, 2.16]` (`p = 0.035`) `gate_speed`** vs. `a1_diag` (and **~1.08× CmdStan NUTS in `ESS/cost`**), up from `1 / 5` on the earlier `dim=6`-only Lane A3 run.

