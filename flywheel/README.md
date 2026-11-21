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
| `POST /api/nodes` | Bearer* | Create/upsert a trial node (`lane`, `run_id`, `decision` required) |
| `POST /api/claims` | Bearer* | Lease a task (default 60m TTL; HTTP 409 on duplicate unless `force: true`) |
| `PATCH /api/claims/:id` | Bearer* | Mark a claim `done` or `abandoned` |
| `POST /api/mcp` | Read public / Write Bearer* | JSON-RPC 2.0 MCP endpoint (`initialize`, `tools/list`, `tools/call`) |

\*Auth is enforced whenever `FLYWHEEL_API_KEY` is set; open in local dev when unset.

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
