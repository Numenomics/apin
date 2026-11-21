# Draft Issue — `pymc-devs/nutpie` (or `pymc-devs/nuts-rs`)

**Suggested Title:** Single-chain treedepth-10 / 100%-divergence trapping on `arma11`, `lotka_volterra`, and `bym2_offset_only` (`nutpie 0.16.11` vs CmdStan 2.39.0)

---

## Summary

Across a 16-model × 5-rep clean-machine comparison of `nutpie 0.16.11` (`adaptation="diag"` and `adaptation="low_rank"`) against stock CmdStan 2.39.0 NUTS (identical `4 × (1000 warmup + 5000 draws)` budget on `posteriordb` Stan models), `nutpie` passed **10 / 16** models vs CmdStan's **14 / 16**. On three models where CmdStan passed all 5 reps cleanly with **0 divergences**, individual `nutpie` chains got trapped either at 100% divergences or at max treedepth (`1023` steps/draw) for the entire run.

## Concrete Failure Cases (5 Reps, Seeds `20260819 + 1000 * rep`)

### 1. `lotka_volterra` (ODE posterior)
- **CmdStan 2.39.0:** 5/5 reps passed, `18.5–21.8 s` per rep, `817k–938k` grads, **0 divergences**.
- **`nutpie` (`diag`):** Reps 0, 1, 4 finished in `10.1–11.1 s` (`~385k` grads), **but rep 2 took `3,153.6 s` (`6.42M` grads) and rep 3 took `2,003.5 s` (`6.42M` grads)** — matching one chain hitting max treedepth on every warmup + sampling draw (`~6000 × 1024 ≈ 6.14M`).
- **`nutpie` (`low_rank`):** Reps 0, 1 took `6.5 s`, **but rep 2 took `732.3 s` (`4.75M` grads), rep 3 took `1,758.6 s` (`6.21M` grads, `1,098` divs), and rep 4 hit `4,287` divergences**; overall `R-hat = 1.229` on `theta.3`.

### 2. `arma11` (4-parameter ARMA(1,1) time series)
- **CmdStan 2.39.0:** 5/5 reps passed, `0.39–0.46 s` per rep, `156k–178k` grads, **0 divergences** (at 50k draws: `3.25–3.38 s`, 5/5 passed).
- **`nutpie` (`diag`):** Rep 1 had **5,000 divergences** (1 entire chain stuck) and rep 3 had **10,000 divergences** (2 entire chains stuck); overall `R-hat = 1.525` on `sigma` (reproduces identically at 50k draws: `R-hat = 1.525` on `lp__`).
- **`nutpie` (`low_rank`):** Rep 1 had **5,000 divergences** and rep 3 spent **`12.20M` gradients (`55.6 s` vs `0.27 s`)**; overall `R-hat = 1.525` on `mu` (at 50k draws, rep 3 spiked to **`488.24 s`** vs CmdStan's `3.33 s` — **147× slower** — with `R-hat = 1.525`).

### 3. `bym2_offset_only` (`D = 3842` spatial ICAR/BYM2)
- **CmdStan 2.39.0:** 5/5 reps passed, `110–115 s` per rep, `1.74M–1.75M` grads, **0 divergences**.
- **`nutpie` (`diag`):** 4 reps took `74–78 s` (`~1.5M` grads), rep 1 spiked to **`941 s` (`7.13M` grads)**; overall `R-hat = 1.594` on `theta.231`.
- **`nutpie` (`low_rank`):** Reps 1 and 4 spiked to **`993 s` and `996 s` (`~7.2M` grads)**; overall `R-hat = 1.527` on `theta.936`.

### 4. Secondary observation: `low_rank` on funnels
- On `pilots`, CmdStan and `nutpie` (`diag`) both fail mixing cleanly (`R-hat ~ 1.10`, `unproven`), whereas `nutpie` (`low_rank`) reports fewer divergences (`164–1093` vs `1399–2251` in `diag`) while showing a statistically significant `lp__` tail bias (`below_q95` error `+0.204` reference SDs, `bound = 0.133`, `wrong` at the 0.1-SD gate). On `eight_schools_centered`, both CmdStan (`-0.274` SD) and `nutpie` (`low_rank`, `-0.298` SD) show the expected centered-funnel bias.

## Minimal Reproducer (`arma11` and `lotka_volterra`)

```python
import nutpie

# Using posteriordb arma11.stan + arma11.data.json (or lotka_volterra)
compiled = nutpie.compile_stan_model(filename="arma11.stan")
model = compiled.with_data(data_dict)

# Rep 1 (seed=20261819) and Rep 3 (seed=20263819) reproduce stuck chains reliably
for seed in (20261819, 20263819):
    tr = nutpie.sample(model, chains=4, tune=1000, draws=5000, seed=seed, save_warmup=True)
    divs = tr.sample_stats["diverging"].sum(dim="draw").values
    steps = tr.sample_stats["n_steps"].sum(dim="draw").values
    print(f"seed={seed}: divergences_per_chain={divs}, steps_per_chain={steps}")
```

## Environment

- `nutpie`: `0.16.11` (PyPI wheel, bundled `nuts-rs`)
- `bridgestan`: `2.9.0`, CmdStan `2.39.0`
- OS / CPU: Linux x86_64, 12-core host
- Full table & per-model diagnostics: `Numenomics/apin` (`results/records/clean_h2h_and_e1_2026-11-21.md`)
