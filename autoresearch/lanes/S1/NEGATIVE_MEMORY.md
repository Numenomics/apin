# Lane S1 Negative Memory (`NEGATIVE_MEMORY.md`)

> Seeded from `REPORT.md` §6 and `results/upstream_drafts/`.

- **NM-S1-001 (walnutpie macro-time abort on `arma11`):** Hard-aborting WALNUTS trajectories when macro step count hits cap without preserving detailed balance introduces systematic bias; see `results/upstream_drafts/04_walnutpie_arma11_macro_time_abort.md`.
- **NM-S1-002 (low-rank Fisher without strong shrinkage on `stan`):** Unregularized low-rank mass matrices degrade on small/diagonal posteriors (`W-62..W-66`). Always gate rank selection against diagonal baseline.
