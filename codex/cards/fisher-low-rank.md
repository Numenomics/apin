# Card: `fisher-low-rank` — Low-Rank Fisher-Divergence Mass Matrix

- **Axis / Lanes / Phase:** Axis 1 (Global Metric) · Lanes `A1`, `A0` · Phase `adapt`
- **Requires:** `["grad"]` · **Aux state:** `none`
- **Source & Licence:** `nuts-rs` (`src/adapt_strategy/low_rank.rs`, MIT — `direct_port_with_attribution`); `AdvancedVI.jl` Batch-and-Match (MIT)

## Mathematical Idea

Standard Stan adaptation estimates only sample variance $\operatorname{diag}(\operatorname{Cov}(x))$, ignoring gradients. A Fisher-divergence estimator combines **draw covariance** $S_x = \operatorname{Cov}(x)$ and **score covariance** $S_g = \operatorname{Cov}(\nabla \log \pi(x))$ by solving for the geometric mean $\Sigma = S_x^{1/2} (S_x^{-1/2} S_g^{-1} S_x^{-1/2})^{1/2} S_x^{1/2}$, approximated in $O(dk)$ as diagonal + rank-$k$ factors $D^{1/2}(I + U \Lambda U^\top) D^{1/2}$ from a rolling sketch of $(x_t, g_t)$.

- **Warm-start pairing:** Initialize the diagonal/Woodbury structure from multi-path Pathfinder (`Pathfinder.jl`) before Fisher refinement.

## Invariance & Failure Modes

- **Invariant:** Metric MUST freeze inside `tuning` before `transition()`. Since the resulting Euclidean HMC/NUTS kernel is reversible with fixed $M$, standard Gandy–Scott invariance applies.
- **Funnel pathology (Council B-note):** Because diagonal Fisher scales as $\sqrt{\operatorname{Var}(x)/\operatorname{Var}(\nabla)}$, on Neal's funnel $\operatorname{Var}(\nabla_x) \propto \mathbb{E}[e^{-v}]$ is dominated by the neck, shrinking the metric everywhere else. Never expect global Fisher alone to solve funnels.
- **Known `apin` conflict (Experiment E1):** `nutpie` (arXiv:2603.18845) reports ~4× median speedup, whereas `apin` W-62..W-66 in `walnutpie` saw 0.037× geomean with 8/15 models harmed and inverted `window_cross_ratio` screening.

## Falsification & Verification Tests

1. **E1 replication:** Compare exact `nuts-rs` low-rank Fisher shrinkage/regularization against diagonal on the 21 core models (`blr`, `diamonds`, `gp_regr`, `radon`, `eight_schools_*`).
2. **Rotation & isotropic guard:** Must beat diagonal on rotated correlated Gaussians without regressing isotropic Gaussians or short-warmup targets by $> 5\%$, and without crashing `eight_schools_centered`.
3. **Overhead check:** Record `overhead_s / eval_s` for rank $k \in \{5, 10, 20\}$ across $d \in \{50, 250, 1000\}$.
