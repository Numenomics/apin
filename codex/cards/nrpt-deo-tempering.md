# Card: `nrpt-deo-tempering` — Non-Reversible Parallel Tempering (DEO + $\Lambda$ Ladder)

- **Axis / Lanes / Phase:** Axis 6 (Multimodality & Global Mixing) · Lane `A0` (Multimodal Column) · Phase `both`
- **Requires:** `["grad"]` (or `["density_only"]`) + K-point `betas` · **Aux state:** `uniform_discrete_2` (parity bit $s \in \{0, 1\}$)
- **Source & Licence:** `Estep-Bingham-Lab/nrpt` (Apache-2.0 — `clean_room_from_paper`); Syed et al. (2022, JRSS-B). *(Do NOT clone AGPL `Pigeons.jl`.)*

## Mathematical Idea

Reversible parallel tempering (random even/odd pair swaps) diffuses across $N$ temperatures in $O(N^2)$ steps. **Non-Reversible Parallel Tempering (NRPT)** alternates **Deterministic Even-Odd (DEO)** swap sweeps:
- When parity $s = 0$: attempt MH swaps on pairs $(0,1), (2,3), \dots$
- When parity $s = 1$: attempt MH swaps on pairs $(1,2), (3,4), \dots$
- Flip $s \leftarrow 1 - s$ after each sweep (lifting the chain into a time-homogeneous Markov kernel on $\prod_i \pi^{\beta_i} \otimes \operatorname{Unif}(\{0,1\})$).

Round-trip time scales as **$O(N)$** instead of $O(N^2)$. During `adapt()`, estimate the **global communication barrier** $\Lambda = \int_0^1 \lambda(\beta)\,d\beta$ from rejection rates $r_i$ and place inverse temperatures $0 < \beta_1 < \dots < \beta_K = 1$ to equalize adjacent swap rejection rates (target $\approx 0.23\text{–}0.40$).

## Invariance & Failure Modes

- **Invariant:** Lifted non-reversible kernel! Parity bit $s \in \{0,1\}$ MUST live in `chain.aux` (`aux_spec = uniform_discrete_2`), NEVER in a mutable module global (violates frozen-kernel replay). Both `betas` and per-walker local tuning MUST freeze after `adapt()`.
- **Tempered target:** Ensure the cold reference ($\beta_{\min}$) is a proper, easily mixed distribution (e.g., fitted Gaussian reference $\pi_0$, interpolating $\pi_\beta \propto \pi_0^{1-\beta} \pi^\beta$) rather than an improper flat $\pi^0$.
- **Invariance testing:** Gandy–Scott does not apply (non-reversible). Test via **joint/marginal exact-draw stationarity** on tempered Gaussian mixtures using the exact rejection envelope $\sum w_i^\beta \mathcal{N}_i^\beta$ (Council B2).

## Falsification & Verification Tests

1. **Tempered invariance (`tier2_tempered_mixture_invariance`):** Exact-draw joint stationarity on $\prod_k \pi^{\beta_k} \otimes \operatorname{Unif}(\{0,1\})$ for bimodal/trimodal mixtures.
2. **Mode-weight recovery (`tier2_multimodal_column`):** Recover unequal mode weights on separated mixtures (sep $> 6$ SD) where all single-chain kernels fail.
