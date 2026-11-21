# Draft Issue — `pymc-devs/nutpie` (or `pymc-devs/nuts-rs`)

**Suggested Title:** `adaptation="low_rank"` adds 3–5× wall-clock overhead per gradient evaluation on medium/high-dimensional Stan models (`radon`, `bym2`)

---

## Summary

While benchmarking `nutpie 0.16.11` (`nuts-rs` bundled) with `adaptation="diag"` vs `adaptation="low_rank"` on `posteriordb` Stan models via BridgeStan 2.9.0, `low_rank` delivers large `ESS/gradient` speedups on correlated models (e.g. `144×` on `diamonds`, `13.5×` on `kidscore_momiq`, `8.6×` on `blr`), **but incurs a 3.2–5.2× wall-clock slowdown per gradient evaluation** on medium-to-high-dimensional models even when the total gradient count is essentially identical to `diag`.

## Measurements (Clean 12-core Linux Host, 4 chains × 1000 warmup + 5000 draws, 5 reps)

| Model | Unconstrained Dim | Arm | Wall Time (5 reps, s) | Total Gradients (warmup + sampling) | Wall Time / Gradient | Credited `ESS/cost` | Credited `ESS/s` |
|---|---:|---|---:|---:|---:|---:|---:|
| `radon_variable_intercept_slope_noncentered` | 176 | `cmdstan 2.39.0` | `9.30–10.91` (mean `10.18`) | `530k–649k` | `17.1 µs` | `0.00159` | `109 / s` |
| `radon_variable_intercept_slope_noncentered` | 176 | `nutpie` (`diag`) | `4.71–4.90` (mean `4.81`) | `333k–349k` | **`13.9 µs`** | `0.00280` | **`186 / s`** |
| `radon_variable_intercept_slope_noncentered` | 176 | `nutpie` (`low_rank`) | `24.67–29.51` (mean `26.24`) | `358k–369k` | **`72.3 µs` (5.2×)** | **`0.00295`** | **`38.1 / s` (0.20× vs `diag`)** |
| `bym2_offset_only` (non-stalled reps 0,2,3) | 3842 | `nutpie` (`diag`) | `76.15–77.78` (mean `77.1`) | `1.52M–1.54M` | **`50.3 µs`** | — (`unproven`) | — |
| `bym2_offset_only` (non-stalled reps 0,2,3) | 3842 | `nutpie` (`low_rank`) | `273.4–281.6` (mean `277.4`) | `1.63M–1.73M` | **`165 µs` (3.3×)** | — (`unproven`) | — |

On `radon_variable_intercept_slope_noncentered`, `low_rank` actually **beats** both CmdStan (`1.86×`) and `nutpie` `diag` (`1.05×`) in `ESS/gradient`, yet ends up **4.9× slower in wall-clock `ESS/s`** than `nutpie` `diag` (`38.1/s` vs `186/s`) and **2.9× slower** than CmdStan (`109/s`).

## Minimal Reproducer

```python
import time
import nutpie

# Any posteriordb checkout containing radon_variable_intercept_slope_noncentered
compiled = nutpie.compile_stan_model(filename="radon_variable_intercept_slope_noncentered.stan")
model = compiled.with_data(data_dict)

for adapt in ("diag", "low_rank"):
    t0 = time.perf_counter()
    tr = nutpie.sample(
        model,
        chains=4,
        tune=1000,
        draws=5000,
        seed=20260819,
        low_rank_modified_mass_matrix=(adapt == "low_rank"),
        save_warmup=True,
    )
    dt = time.perf_counter() - t0
    grads = int(tr.sample_stats["n_steps"].sum() + tr.warmup_sample_stats["n_steps"].sum())
    print(f"{adapt:10s}: {dt:6.2f}s, grads={grads}, us/grad={dt*1e6/grads:.1f}")
```

## Environment

- `nutpie`: `0.16.11` (PyPI wheel, bundled `nuts-rs`)
- `bridgestan`: `2.9.0`
- OS / CPU: Linux x86_64, 12-core host, `OMP_NUM_THREADS=1`
- Full benchmark table: `Numenomics/apin` (`results/records/clean_h2h_and_e1_2026-11-21.md`)

## Question / Hypothesis

Is the extra ~58 µs/grad on `D=176` coming from per-step low-rank metric application (`U diag(eigs) U^T`) or from repeated SVD / eigendecomposition updates during warmup windows? If it is mostly in warmup SVD recomputation or non-SIMD matvecs inside `nuts-rs`, profiling `radon` (`D=176`) should pinpoint it quickly.
