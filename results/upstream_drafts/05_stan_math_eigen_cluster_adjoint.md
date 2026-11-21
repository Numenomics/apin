# Draft Issue — `stan-dev/math`

> **Reminder (Stan AI contribution policy, May 2026):**
> 1. Review why closed PR `stan-dev/math#3367` was closed unmerged before deciding whether to file a focused bug report (this issue draft) or a fresh rebased PR.
> 2. Write/edit in your own words and include an explicit AI-assistance disclosure per the Stan project policy.
> 3. Re-verify against current `stan-dev/math` `develop` (last verified at `0329c4cb68`, BridgeStan `2.9.0`).

---

**Suggested Title:** `rev` `eigenvectors_sym` / `eigendecompose_sym` reverse-mode adjoint returns `NaN` on repeated eigenvalues and O(1)–O(1e3) ISA-dependent errors on near-degenerate clusters (`kronecker_gp`)

## Summary

The reverse-mode adjoints of `eigenvectors_sym` and `eigendecompose_sym` compute the antisymmetric Cauchy matrix $F_{ij} = 1 / (w_j - w_i)$ ($i \neq j$) without guarding repeated or rounding-degenerate eigenvalue gaps $|w_j - w_i| \lesssim \epsilon_{\text{mach}} \|w\|_\infty$.

This produces two concrete bugs:
1. **Exact repeated eigenvalues ($\delta = 0$):** $F_{ij} = \pm\infty$ produces `NaN` gradients even when the downstream scalar functional is completely smooth and invariant to within-cluster eigenvector rotations (e.g. a 4-fold degenerate symmetric matrix, or `posteriordb`'s `kronecker_gp` evaluated at $\theta = 0$, where 435 of 438 gradient components become `NaN`).
2. **Rounding-degenerate clusters ($\delta \sim 10^{-16}\text{–}10^{-13}$):** On `posteriordb`'s `kronecker_gp` (jitter-pinned covariance clusters), the returned eigenvectors within the cluster are rounding-arbitrary, so $1/(w_j - w_i)$ amplifies floating-point noise by $10^{13}\text{–}10^{16}$. Measured gradients on `var1`/`bw1` disagree with finite differences by **30–52%** in every build and diverge by **O(1)–O(1e3)** across compiler/ISA flags (`-O3` vs `-march=native` / AVX2).

## Proposed Minimal-Norm (Cluster-Gauge) Fix

When $|w_i - w_j| < \tau$ with $\tau = \kappa \cdot \max(1, \|w\|_\infty) \cdot \epsilon_{\text{mach}}$ (e.g. $\kappa = 10^3$), set $\tilde{F}_{ij} = 0$ (the Fox–Kapoor / Nelson minimal-norm within-cluster gauge), gated behind a fast check on adjacent sorted eigenvalue gaps so well-separated spectra execute the exact existing code path bit-identically.

*(Note: `eigenvalues_sym` needs no change because its reverse adjoint $V \operatorname{diag}(\bar{g}_w) V^\top$ contains no $1/(w_j - w_i)$ division.)*

## Empirical Validation (from `Numenomics/apin` `results/records/cluster_adjoint_w40.md`)

- **Well-separated spectra:** 200 / 200 random symmetric matrices bit-identical to stock `stan-math`.
- **Exact degeneracy (`kronecker_gp` at $\theta = 0$):** `435/438 NaN` components $\to$ `0/438 NaN` (all finite).
- **Finite-difference consistency (`kronecker_gp` `var1`/`bw1`):** relative error drops from `30–52%` to `~1e-6` (FD truncation floor).
- **Cross-ISA gradient divergence (`kronecker_gp`):** drops from `1.16` relative to `7e-5` ($\kappa = 10^3$) / `3.1e-8` ($\kappa = 10^5$).
- **Downstream sampling (`kronecker_gp`):** median min bulk-ESS improves ~7× (`48 -> 368`) once HMC stops adapting to corrupted gradients.

## Minimal C++ Reproducer

```cpp
// Any symmetric matrix with repeated eigenvalue (e.g. diag(1, 1, 2) rotated by Q)
// and a rotation-invariant downstream scalar functional such as trace(V diag(f(w)) V^T)
// or u^T A^{-1} u computed via eigendecompose_sym(A).
// Stock stan-math returns NaN in A.adj() at exact multiplicity >= 2.
```
