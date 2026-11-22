# Maintainer Guide (`MAINTAINING.md`)

Operations runbook for the repository owner: Flywheel administration, verifying contributor submissions, rebuilding references/costs, and managing upstream fork submodules. Contributors should read [CONTRIBUTING.md](CONTRIBUTING.md) instead.

---

## 1. Flywheel Administration (`https://apin-flywheel.m0hawk.workers.dev`)

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

### Enable / rotate / close self-serve invite registration (optional)
```bash
cd flywheel
npx wrangler secret put FLYWHEEL_INVITE_CODE      # enable or rotate
npx wrangler secret delete FLYWHEEL_INVITE_CODE   # close self-serve registration (returns 403)
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

### Deploy the website and Flywheel updates & cost safety
One deploy ships both the API Worker and the whole website (`flywheel/public/`: overview, research frontier, agent benchmark, lane guide).
```bash
cd flywheel
node --test tests/worker.test.mjs
npx wrangler d1 execute apin-flywheel --remote --yes --file=./schema.sql   # if schema.sql changed
npx wrangler deploy
```
- **Publishing a benchmark sweep:** run `uv run python scripts/collect_harbor_results.py --release <tag> --jobs jobs/` (writes `flywheel/public/data/benchmark.json`), commit it, then deploy.
- **Refreshing the offline snapshot:** `.venv/bin/python flywheel/seed_from_apin.py` rewrites `flywheel/public/snapshot.json`, which the site falls back to when the API is unreachable.
- **Lane and track descriptions** on the site live in one place, `flywheel/public/assets/site.js` (`LANES`, `TRACKS`). Update them when a lane's `program.md` changes its contract.
- **Zero-bill guarantee:** Keep Flywheel on the Cloudflare **Workers Free** plan ($0/mo). Free plans enforce hard fail-closed limits (100k req/day, 5M D1 reads/day, 100k D1 writes/day, 10ms CPU/req) and cannot incur overage charges. Public `GET` routes also carry a 30-second anti-cache-bust edge cache.

---

## 2. Reviewing & Promoting Contributor PRs

1. Validate the record schema, imports, license allowlist, and involutive `invariance.md`:
   ```bash
   uv run apin records validate records/<release>/<track>/<date>_<slug>
   ```
2. Reproduce on your own machine with a fresh `--base-seed`:
   ```bash
   uv run apin run logdensity --label cand --opt sampler=records/.../submission.py --models public --reps 5
   uv run apin score --models public --reps 5 --baseline cmdstan --track L1 --submission cand=records/.../submission.py
   ```
3. Merge the PR and call `POST /api/admin/nodes/<id>/verify` to flip `verified = 1` on Flywheel.

---

## 3. Rebuilding References & Calibrated Cost Weights

```bash
export CMDSTAN=/path/to/cmdstan
uv run apin reference build --models pilots        # rebuild local 20-chain reference summaries
uv run apin costs build --models all               # re-measure r_grad weights into models/costs.json
CMDSTAN=/path/to/cmdstan uv run pytest -m slow     # full end-to-end tests on real models
```

---

## 4. Upstream Submodules & Fork Patch Workflow (Lanes S1 / S2)

Submodules are pinned to owner fork branches (`upstream/STATUS.md`). Check out only what you need:
```bash
git submodule update --init upstream/math upstream/stan
```
1. Branch inside `upstream/<name>/` from the upstream base commit (`origin` is your fork).
2. Follow `upstream/<name>.md` for clean rebuilds (hard ban on mixed ISA/SIMD builds — `W-27`).
3. Verify via `uv run apin autoresearch eval-s1 ...` or `eval-s2 ...` (finite-difference gradient parity + byte-identical draw canary for the R-008 exact-math waiver).
4. Record measurements in `results/records/`, push the fork branch (or tag `apin/<name>-<label>` if detached), update the submodule pin and `upstream/STATUS.md`.
5. Upstream PRs to `stan-dev`, `flatironinstitute`, `pymc-devs`, or `cmdstanr` are always submitted personally by the owner — never by agents.
