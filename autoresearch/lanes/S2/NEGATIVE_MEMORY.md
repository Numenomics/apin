# Lane S2 Negative Memory (`NEGATIVE_MEMORY.md`)

> Seeded from `REPORT.md` §6 (`W-1..W-133`).

- **NM-S2-001 (Mixed ISA/SIMD builds, `W-27`):** Hard ban. Mixing `-march=native` / AVX translation units with precompiled stock objects causes silent ABI/alignment bugs or invalid timings. Always rebuild cleanly from scratch and verify finite-difference gradients.
- **NM-S2-002 (Eigen cluster adjoint aliasing):** See `results/upstream_drafts/05_stan_math_eigen_cluster_adjoint.md` before modifying Eigen reverse-mode expression templates.
