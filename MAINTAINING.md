# Maintainer guide (`MAINTAINING.md`)

Operations runbook for the repository owner: Flywheel administration, verifying contributor submissions, rebuilding references and cost weights, and managing upstream fork submodules. Contributors should read [CONTRIBUTING.md](CONTRIBUTING.md) instead.

## 1. Flywheel administration (`https://apin-flywheel.m0hawk.workers.dev`)

Your Cloudflare secret `FLYWHEEL_API_KEY` is the root admin Bearer token. Export it locally as `APIN_FLYWHEEL_ADMIN_KEY` (never commit it).

### Mint, list, and revoke contributor keys
```bash
# Mint a personal revocable contributor key (returns raw `apfw_...` token ONCE)
curl -fsS -X POST https://apin-flywheel.m0hawk.workers.dev/api/admin/keys \
  -H "Authorization: Bearer $APIN_FLYWHEEL_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"handle":"contributor-github-handle","role":"contributor","hourly_limit":120}'

# List issued keys (metadata only; hashes are never exposed)
curl -fsS https://apin-flywheel.m0hawk.workers.dev/api/admin/keys \
  -H "Authorization: Bearer $APIN_FLYWHEEL_ADMIN_KEY"

# Revoke a key immediately
curl -fsS -X DELETE https://apin-flywheel.m0hawk.workers.dev/api/admin/keys/<key_id> \
  -H "Authorization: Bearer $APIN_FLYWHEEL_ADMIN_KEY"
```

### Open registration: promote, revoke and purge handles
Anyone can register a handle with `apin autoresearch register`. It starts as `provisional` (100 new trials a day, marked "new" on the site).
```bash
H=https://apin-flywheel.m0hawk.workers.dev
AUTH="Authorization: Bearer $APIN_FLYWHEEL_ADMIN_KEY"
curl -fsS "$H/api/admin/keys?role=provisional" -H "$AUTH"                 # who registered
curl -fsS -X POST "$H/api/admin/handles/<handle>/promote" -H "$AUTH"      # provisional -> contributor
curl -fsS -X POST "$H/api/admin/handles/<handle>/revoke" -H "$AUTH"       # stop the handle, keep its trials
curl -fsS -X POST "$H/api/admin/handles/<handle>/purge" -H "$AUTH"        # stop it and delete everything it posted
```
An invite code lets a new handle skip the provisional stage:
```bash
cd flywheel
npx wrangler secret put FLYWHEEL_INVITE_CODE      # set or rotate
npx wrangler secret delete FLYWHEEL_INVITE_CODE   # no invite codes; registration stays open and provisional
npx wrangler secret put FLYWHEEL_REGISTRATIONS_PER_DAY   # enter 0 to close registration entirely
```

### Verify a reproduced contributor node or remove spam
```bash
# Promote a node to verified=1 after reproducing real-Stan scores
curl -fsS -X POST "https://apin-flywheel.m0hawk.workers.dev/api/admin/nodes/$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' 'A1:run_id:T0004')/verify" \
  -H "Authorization: Bearer $APIN_FLYWHEEL_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"verified":true,"real_stan_passed":4,"real_stan_total":5,"real_ess_per_cost":0.024}'

# Delete a spam/invalid node
curl -fsS -X DELETE "https://apin-flywheel.m0hawk.workers.dev/api/admin/nodes/<url-encoded-node-id>" \
  -H "Authorization: Bearer $APIN_FLYWHEEL_ADMIN_KEY"
```

### Deploy the website and Flywheel, and keep costs at zero
One deploy ships both the API Worker and the whole website (`flywheel/public/`: overview, research frontier, agent benchmark, lane guide).
```bash
cd flywheel
npm install --legacy-peer-deps        # wrangler 4.16x declares an optional workers-types v5 peer
node --test tests/worker.test.mjs && npx tsc --noEmit
npx wrangler d1 export apin-flywheel --remote --output=../backup-$(date +%F).sql   # backup first (keep it out of git)
npx wrangler d1 execute apin-flywheel --remote --yes --file=./schema.sql   # base tables; idempotent
npx wrangler d1 migrations apply apin-flywheel --remote                  # applies flywheel/migrations/*.sql once each
npx wrangler deploy
```
- **Deploy right after applying `0005_speedup.sql`.** It drops `nodes.standard`, which the Worker before it reads and writes, so trial posts and lane charts fail until `wrangler deploy` finishes. Trials posted before it have no `yardstick`. They stay in the lists, but the charts start empty and the lanes have no steps until graded trials arrive.
- **Put schema changes in a new numbered file in `flywheel/migrations/`.** Never edit `schema.sql`, which is the base schema every migration builds on. Wrangler records applied files in the `d1_migrations` table, so `migrations apply` runs only new ones. Run `npx wrangler d1 migrations list apin-flywheel --remote` to see pending files. Apply a migration before deploying the Worker that needs it: from `0001_research_objects.sql` onwards, the Worker reads `nodes.state`, `sources`, `node_parents`, `notes` and `verify_queue`, from `0002_bounded_reads.sql` onwards `run_stats` and `note_links`, from `0004_frontier_steps.sql` onwards `nodes.frontier_step` and the step counters in `lane_stats`, from `0005_speedup.sql` onwards `nodes.speedup` and `nodes.yardstick`, and from `0006_real_gate.sql` onwards `nodes.real_gate`. Apply migrations before loading `seed.sql` as well.
- **Do not add virtual tables (FTS5).** D1 cannot export a database that has them, and the export above is the backup.
- **Abuse limits** (registration caps, provisional write and trial caps) are listed in the API reference in `flywheel/README.md`. Change one with `npx wrangler secret put FLYWHEEL_<NAME>`. `FLYWHEEL_REGISTRATIONS_PER_DAY=0` closes open registration. Set `FLYWHEEL_IP_SALT` once to any random string.
- **Publishing a benchmark sweep:** run `uv run python scripts/collect_harbor_results.py --jobs jobs/` (adds the new runs to `flywheel/public/data/benchmark.json` and keeps earlier releases), commit it, then deploy.
- **Reads are bounded. Keep them that way.** Since migration `0002_bounded_reads.sql`, no read loads the whole trial table into the Worker. The site uses `/api/overview`, `/api/lanes/:lane/chart`, `/api/nodes`, `/api/nodes/:id`, `/api/notes` and `/api/search`, each measured at about 0.3 to 2 ms of JavaScript whether the Flywheel holds 1,000 or 20,000 trials. `/api/dag` is kept for old clients and capped at the newest 300 trials (about 6 to 8 ms). Run `node --test tests/scale.test.mjs` after changing a read. It fails when a read's cost grows with the table. Counts come from `run_stats`, which triggers on `nodes` keep current. Never write trials with `INSERT OR REPLACE`: it deletes rows without firing the delete trigger and double-counts. If the counts ever drift, rebuild them: `npx wrangler d1 execute apin-flywheel --remote --command "DELETE FROM run_stats"`, then run the `REBUILD` statement at the end of `migrations/0002_bounded_reads.sql` the same way. If the frontier step flags or their counters drift, run the `BACKFILL` block of `migrations/0006_real_gate.sql` the same way (`seed_from_apin.py --backfill-sql` writes it to a file).
- **Loading local trials:** `.venv/bin/python flywheel/seed_from_apin.py` writes `flywheel/seed.sql` from the local ledgers. The file ends with the `BACKFILL` block of migration 0005, so the step flags are correct once it has run.
- **Lane and track descriptions** on the site live in one place, `flywheel/public/assets/site.js` (`LANES`, `TRACKS`). Update them when a lane's `program.md` changes its contract.
- **Cost ceiling:** Keep Flywheel on the Cloudflare Workers Free plan ($0/mo). The free plan enforces hard fail-closed limits (100k req/day, 5M D1 reads/day, 100k D1 writes/day, 10ms CPU/req) and cannot incur overage charges. Public `GET` routes also carry a 30-second anti-cache-bust edge cache.

## 2. Reviewing and promoting contributor PRs

1. Validate the record schema, imports, licence allowlist, and involutive `invariance.md`:
   ```bash
   uv run apin records validate records/<release>/<track>/<date>_<slug>
   ```
2. Reproduce on your own machine with a fresh `--base-seed`:
   ```bash
   uv run apin run logdensity --label cand --opt sampler=records/.../submission.py --models public --reps 5
   uv run apin score --models public --reps 5 --baseline cmdstan --track L1 --submission cand=records/.../submission.py
   ```
3. Merge the PR and call `POST /api/admin/nodes/<id>/verify` to set `verified = 1` on Flywheel.

## 3. Rebuilding references and calibrated cost weights

```bash
export CMDSTAN=/path/to/cmdstan
uv run apin reference build --models pilots        # rebuild local 20-chain reference summaries
uv run apin costs build --models all               # re-measure r_grad weights into models/costs.json
CMDSTAN=/path/to/cmdstan uv run pytest -m slow     # full end-to-end tests on real models
```

## 4. Upstream submodules and fork patch workflow (lane S2)

Submodules are pinned to owner fork branches (`upstream/STATUS.md`). Check out only what you need:
```bash
git submodule update --init upstream/math upstream/stan
```
1. Branch inside `upstream/<name>/` from the upstream base commit (`origin` is your fork).
2. Follow `upstream/<name>.md` for clean rebuilds (do not mix ISA or SIMD builds, see `W-27`).
3. Verify with `uv run apin autoresearch eval-s2 ...` (finite-difference gradient parity and byte-identical draw canary for the R-008 exact-math waiver).
4. Record measurements in `results/records/`, push the fork branch (or tag `apin/<name>-<label>` if detached), update the submodule pin and `upstream/STATUS.md`.
5. Only the owner submits upstream PRs to `stan-dev`, `flatironinstitute`, `pymc-devs` or `cmdstanr`. Agents never do.

## 5. Running a Benchmark Sweep (Harbor, release `bench-v0.3.0`)

The tasks need Harbor 0.27.0 or later and a Docker host where Harbor's egress-control sidecar works (it needs nftables, without it Harbor fails the trial instead of running it unrestricted). [harbor/README.md](harbor/README.md) explains what the container holds and the network policy of each phase.

```bash
git pull                                               # a plain checkout is enough; no submodules
uv tool install harbor==0.27.0
uv run python scripts/build_harbor_base.py --tasks     # sealed base + the 5 task images
uv run pytest -q tests/test_harbor_seal.py             # add -m slow to inspect the built images
eval "$(grep -E '^export ZAI_API_KEY=' ~/.bashrc)"     # load the key without printing it
harbor run -p harbor/tasks -a opencode-v2 -m zai-coding-plan/glm-5.3-flash \
  --ak reasoning_effort=high --api-key-name ZAI_API_KEY --allow-agent-host api.z.ai \
  --job-name bench-v0.3.0-glm-5.3-flash-high     # the release prefix is how the collector tags trials
uv run python scripts/collect_harbor_results.py --jobs jobs/
```

The agent may run for 24 h per task, so start sweeps in the background. Rebuild the base image after any change to `harness/`, the lane documents, a lane's `start.py` or its `yardstick.json`. A stale image still runs but measures the old code.

### Refreezing the scores

Every speed-up divides by a frozen score: each lane's public yardstick (`autoresearch/lanes/<LANE>/yardstick.json`) and the benchmark's hidden denominators (`harbor/tasks/<task>/tests/frozen_scores.json`). Refreeze both after a change to a module in the judge digest (`harness/judge.py::judge_digest`), to a lane's `start.py`, to a reference, to a seed, or to the NumPy or SciPy version in `uv.lock` (the digest does not cover the lock file, so nothing reports that case):

```bash
uv run python scripts/sync_harbor_tests.py              # copy start.py, verify.py and the references into harbor/tasks/
uv run python scripts/freeze_harbor_references.py --workers 4
uv run pytest -q tests/test_harbor_scoring.py           # fails while any frozen score is stale
```

The freeze copies each lane's `submission.py` to `tests/frontier.py` first (`--keep-frontier` skips that). `--public-only` or `--hidden-only` freezes one half. The single source of a lane's starting code is `autoresearch/lanes/<LANE>/start.py`. The sync script writes it to `environment/baseline.py` and `tests/baseline.py` of the task. A new release rotates `HIDDEN_VERIFIER_SEED` and `HIDDEN_CONFIRMATION_SEED` in `harbor/verify.py` and lists the retired ones in the comment above them.
