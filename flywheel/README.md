# apin Flywheel

Lightweight, self-hosted research DAG, task claim ledger, alignment dashboard, and MCP endpoint for `apin`, running on Cloudflare's **$0/mo free tier** (Workers + D1 SQLite + Static Assets: 100k req/day, 5M D1 reads/day, 100k D1 writes/day, unlimited static bandwidth).

## 4-Command Cloudflare Free-Tier Deploy

Run from `flywheel/` after `npm install` (or `npx wrangler login`):

```bash
# 1. Create the D1 database (copy the returned database_id into wrangler.toml)
npx wrangler d1 create apin-flywheel

# 2. Apply schema and seed from local apin autoresearch ledgers + real-Stan scores
npx wrangler d1 execute apin-flywheel --remote --file=./schema.sql && \
  npx wrangler d1 execute apin-flywheel --remote --file=./seed.sql

# 3. Set the write bearer token (reads stay public; writes require Authorization: Bearer <key>)
npx wrangler secret put FLYWHEEL_API_KEY

# 4. Deploy Worker + static dashboard assets
npx wrangler deploy
```

Refresh `seed.sql` and `public/snapshot.json` from local runs at any time:

```bash
.venv/bin/python flywheel/seed_from_apin.py
```

## Local Preview & Tests

- **Pure static preview** (zero Cloudflare account needed; falls back cleanly to `public/snapshot.json`):
  ```bash
  python3 -m http.server 8788 --directory flywheel/public
  ```
- **Local Worker dev server**:
  ```bash
  cd flywheel && npm run dev
  ```
- **Worker + MCP unit tests** (runs against in-memory `node:sqlite`, no live Cloudflare login):
  ```bash
  node --test flywheel/tests/worker.test.mjs
  ```
- **Python client + seeder tests**:
  ```bash
  .venv/bin/pytest -q tests/test_flywheel.py
  ```

## API Overview

| Method & Path | Auth | Description |
| --- | --- | --- |
| `GET /api/health` | Public | `{ ok, nodes, active_claims, lanes }` |
| `GET /api/dag` | Public | Full `{ nodes, edges, frontier, claims, alignment }` payload |
| `GET /api/nodes` | Public | Filter by `lane`, `run_id`, `decision`, `verified`, `limit` |
| `GET /api/nodes/:id` | Public | Node details plus `parent` and `children` |
| `GET /api/frontier` | Public | Best kept/verified node per lane |
| `GET /api/claims` | Public | Active, unexpired claims (optional `lane` filter) |
| `POST /api/nodes` | Bearer (admin or contributor) | Create/upsert a trial node (`contributor` forced `verified=0`, stamped `author=handle`, cannot overwrite other authors) |
| `POST /api/claims` | Bearer (admin or contributor) | Lease a task (HTTP 409 on duplicate; only owner or admin may `force: true`) |
| `PATCH /api/claims/:id` | Bearer (owner or admin) | Mark a claim `done` or `abandoned` |
| `POST /api/register` | Invite code (`FLYWHEEL_INVITE_CODE`) | Self-register a personal revocable `contributor` token (`apfw_...`, rate-limited) |
| `GET /api/admin/keys` | Admin Bearer | List issued API keys (metadata only; hashes never exposed) |
| `POST /api/admin/keys` | Admin Bearer | Mint a personal revocable contributor/admin token (`{ handle, role, hourly_limit }`) |
| `DELETE /api/admin/keys/:id` | Admin Bearer | Revoke a key immediately |
| `POST /api/admin/nodes/:id/verify` | Admin Bearer | Promote a node to `verified=1` (optionally attaching reproduced real-Stan metrics) |
| `DELETE /api/admin/nodes/:id` | Admin Bearer | Delete a spam/invalid node |
| `POST /api/mcp` | Read public / Write Bearer | JSON-RPC 2.0 MCP endpoint (`initialize`, `tools/list`, `tools/call`) |

\*Auth is enforced whenever `FLYWHEEL_API_KEY` is set; open in local dev when unset. Contributor tokens are stored SHA-256 hashed in D1 `api_keys` and rate-limited per UTC hour (`rate_limits`).

## Live Instance & Contributor Onboarding

- **Public dashboard & read API:** `https://apin-flywheel.m0hawk.workers.dev`
- **Auto-streaming trials:** Export `APIN_FLYWHEEL_TOKEN=<your-token>` (URL defaults to `https://apin-flywheel.m0hawk.workers.dev`); every `uv run apin autoresearch eval` automatically posts kept + discarded trials with parent→child lineage.
- **Minting a contributor key (maintainer):**
  ```bash
  curl -fsS -X POST https://apin-flywheel.m0hawk.workers.dev/api/admin/keys \
    -H "Authorization: Bearer $APIN_FLYWHEEL_ADMIN_KEY" \
    -H "Content-Type: application/json" \
    -d '{"handle":"researcher-name","role":"contributor","hourly_limit":120}'
  ```
- **Optional self-serve registration:** Run `npx wrangler secret put FLYWHEEL_INVITE_CODE` to enable `POST /api/register` with `{ "handle": "...", "invite_code": "..." }` (leave unset to keep registration invite-only via `/api/admin/keys`).
- **Promoting winners into Git:** Only verified/notable submissions go into `Numenomics/apin` via GitHub PR (`records/<release>/<track>/<date>_<slug>/`); once reproduced, mark the Flywheel node verified via `POST /api/admin/nodes/<id>/verify`.

## MCP Client Config Snippet

Expose `flywheel_frontier`, `flywheel_list_nodes`, `flywheel_claim_task`, and `flywheel_post_node` to any HTTP MCP client:

```json
{
  "mcpServers": {
    "apin-flywheel": {
      "url": "https://apin-flywheel.<your-subdomain>.workers.dev/api/mcp",
      "headers": {
        "Authorization": "Bearer ${APIN_FLYWHEEL_TOKEN}"
      }
    }
  }
}
```

## Python Client (`harness/flywheel.py`)

Set environment variables to enable live syncing (cleanly no-ops when unset):

```bash
export APIN_FLYWHEEL_URL="https://apin-flywheel.<your-subdomain>.workers.dev"
export APIN_FLYWHEEL_TOKEN="<your-secret-token>"
```

```python
from harness import flywheel

frontier = flywheel.get_frontier()
claim = flywheel.claim_task("A3:pdmp-zigzag", lane="A3", agent="GLM-5.3-Flash")
flywheel.sync_ledger("autoresearch/lanes/A3/runs/glm53_pdmp_1h/ledger.jsonl", lane="A3", run_id="glm53_pdmp_1h")
flywheel.release_claim(claim["claim"]["id"], status="done")
```
