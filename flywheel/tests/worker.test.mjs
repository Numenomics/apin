import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import { handleRequest } from "../src/index.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCHEMA_SQL = fs.readFileSync(path.resolve(__dirname, "../schema.sql"), "utf8");
const MIGRATIONS_DIR = path.resolve(__dirname, "../migrations");
const MIGRATIONS_SQL = fs
  .readdirSync(MIGRATIONS_DIR)
  .filter((f) => f.endsWith(".sql"))
  .sort()
  .map((f) => fs.readFileSync(path.join(MIGRATIONS_DIR, f), "utf8"));

/**
 * Minimal in-memory Cloudflare D1Database shim backed by node:sqlite.
 */
function createD1Shim(preMigration) {
  const sqlite = new DatabaseSync(":memory:");
  sqlite.exec(SCHEMA_SQL);
  if (preMigration) sqlite.exec(preMigration);
  for (const sql of MIGRATIONS_SQL) sqlite.exec(sql);

  function makeStatement(query, boundValues = []) {
    return {
      bind(...values) {
        return makeStatement(query, values);
      },
      async first(colName) {
        const stmt = sqlite.prepare(query);
        const row = stmt.get(...boundValues);
        if (!row) return null;
        if (colName) return row[colName] ?? null;
        return { ...row };
      },
      async all() {
        const stmt = sqlite.prepare(query);
        const rows = stmt.all(...boundValues).map((r) => ({ ...r }));
        return { results: rows, success: true, meta: {} };
      },
      async run() {
        const stmt = sqlite.prepare(query);
        const info = stmt.run(...boundValues);
        return {
          results: [],
          success: true,
          meta: { changes: Number(info.changes ?? 0) },
        };
      },
    };
  }

  return {
    sqlite,
    db: {
      prepare(query) {
        return makeStatement(query);
      },
      // D1 runs a batch in one round trip; the shim runs the statements in order.
      async batch(statements) {
        const results = [];
        for (const stmt of statements) results.push(await stmt.all());
        return results;
      },
    },
  };
}

function makeEnv(options = {}) {
  const { sqlite, db } = createD1Shim(options.preMigration);
  const assets = {
    async fetch(req) {
      const url = new URL(req.url);
      return new Response(`STATIC:${url.pathname}`, {
        status: 200,
        headers: { "Content-Type": "text/plain" },
      });
    },
  };
  return {
    sqlite,
    env: {
      DB: db,
      ASSETS: assets,
      FLYWHEEL_API_KEY: options.apiKey,
      // Tests without an API key act as the local maintainer, which needs the explicit opt-in.
      FLYWHEEL_DEV_OPEN_ADMIN: options.apiKey || options.closed ? undefined : "1",
    },
  };
}

async function reqJson(env, method, pathname, body = undefined, headers = {}) {
  const init = {
    method,
    headers: { ...headers },
  };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const resp = await handleRequest(new Request(`https://flywheel.test${pathname}`, init), env);
  const text = await resp.text();
  let json = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = text;
  }
  return { status: resp.status, headers: resp.headers, json };
}

describe("apin Flywheel Worker", () => {
  it("serves static assets for non-/api routes and handles CORS preflight", async () => {
    const { env } = makeEnv();
    const assetResp = await handleRequest(new Request("https://flywheel.test/index.html"), env);
    assert.equal(assetResp.status, 200);
    assert.equal(await assetResp.text(), "STATIC:/index.html");

    const preflight = await handleRequest(
      new Request("https://flywheel.test/api/dag", { method: "OPTIONS" }),
      env
    );
    assert.equal(preflight.status, 204);
    assert.equal(preflight.headers.get("Access-Control-Allow-Origin"), "*");
  });

  it("handles full node lifecycle, DAG edges, frontier, alignment, and filters", async () => {
    const { env } = makeEnv();

    // Empty health check
    const health0 = await reqJson(env, "GET", "/api/health");
    assert.equal(health0.status, 200);
    assert.deepEqual(health0.json, { ok: true, nodes: 0, active_claims: 0, lanes: [] });

    // Validation failure on missing required fields
    const bad = await reqJson(env, "POST", "/api/nodes", { lane: "A1" });
    assert.equal(bad.status, 400);
    assert.equal(bad.json.error, "missing_required_fields");

    // Post baseline node (explicit id)
    const n1 = await reqJson(env, "POST", "/api/nodes", {
      id: "A1:run1:T0001",
      lane: "A1",
      run_id: "run1",
      trial_id: "T0001",
      agent_model: "GLM-5.3-Flash",
      title: "T0001: baseline",
      hypothesis: "Diagonal baseline",
      decision: "keep_baseline",
      soft_min_eff: 0.0005,
      ratio_vs_incumbent: 1.0,
      dims: [6],
      codex_refs: ["fisher-low-rank"],
      code_sha: "aaa111",
      created_at: "2026-11-21T10:00:00Z",
    });
    assert.equal(n1.status, 201);
    assert.equal(n1.json.node.id, "A1:run1:T0001");

    // Post kept child with real-Stan verification
    const n2 = await reqJson(env, "POST", "/api/nodes", {
      id: "A1:run1:T0002",
      parent_id: "A1:run1:T0001",
      lane: "A1",
      run_id: "run1",
      trial_id: "T0002",
      agent_model: "GLM-5.3-Flash",
      title: "T0002: Stein-damped Fisher",
      hypothesis: "Low-rank Fisher improves ill-conditioned targets",
      decision: "keep",
      soft_min_eff: 0.0012,
      ratio_vs_incumbent: 2.4,
      real_stan_passed: 4,
      real_stan_total: 5,
      real_ess_per_cost: 0.024,
      overhead_ratio: 4.5,
      verified: true,
      created_at: "2026-11-21T10:05:00Z",
    });
    assert.equal(n2.status, 201);
    assert.equal(n2.json.node.verified, 1);

    // Post discarded high-eff node (must NOT displace frontier!) + auto-generated ID
    const n3 = await reqJson(env, "POST", "/api/nodes", {
      parent_id: "A1:run1:T0002",
      lane: "A1",
      run_id: "run1",
      trial_id: "T0003",
      decision: "discard_tier0",
      soft_min_eff: 0.99,
      created_at: "2026-11-21T10:10:00Z",
    });
    assert.equal(n3.status, 201);
    assert.ok(n3.json.node.id.length > 8);

    // Post A3 lane node
    const n4 = await reqJson(env, "POST", "/api/nodes", {
      id: "A3:pdmp:T0001",
      lane: "A3",
      run_id: "pdmp",
      trial_id: "T0001",
      decision: "keep_baseline",
      soft_min_eff: 0.0003,
      ratio_vs_incumbent: 1.0,
      short_horizon_eff: 0.0002,
      long_horizon_eff: 0.0004,
      dims: [6, 20],
      created_at: "2026-11-21T11:00:00Z",
    });
    assert.equal(n4.status, 201);

    // GET /api/health
    const health = await reqJson(env, "GET", "/api/health");
    assert.deepEqual(health.json, {
      ok: true,
      nodes: 4,
      active_claims: 0,
      lanes: ["A1", "A3"],
    });

    // GET /api/frontier -> discarded 0.99 node excluded; A1 frontier is T0002
    const frontierRes = await reqJson(env, "GET", "/api/frontier");
    assert.equal(frontierRes.status, 200);
    assert.equal(frontierRes.json.frontier.A1.id, "A1:run1:T0002");
    assert.equal(frontierRes.json.frontier.A3.id, "A3:pdmp:T0001");

    // GET /api/dag
    const dagRes = await reqJson(env, "GET", "/api/dag");
    assert.equal(dagRes.status, 200);
    assert.equal(dagRes.json.nodes.length, 4);
    assert.equal(dagRes.json.edges.length, 2);
    assert.equal(dagRes.json.alignment.length, 1);
    assert.equal(dagRes.json.alignment[0].real_stan_pass_rate, 0.8);

    // GET /api/nodes filtering
    const filteredVerified = await reqJson(env, "GET", "/api/nodes?lane=A1&verified=1&limit=10");
    assert.equal(filteredVerified.status, 200);
    assert.equal(filteredVerified.json.count, 1);
    assert.equal(filteredVerified.json.nodes[0].id, "A1:run1:T0002");

    const filteredDecision = await reqJson(env, "GET", "/api/nodes?decision=keep_baseline");
    assert.equal(filteredDecision.json.count, 2);

    // GET /api/nodes/:id
    const detail = await reqJson(
      env,
      "GET",
      `/api/nodes/${encodeURIComponent("A1:run1:T0002")}`
    );
    assert.equal(detail.status, 200);
    assert.equal(detail.json.parent.id, "A1:run1:T0001");
    assert.equal(detail.json.children.length, 1);

    const missing = await reqJson(env, "GET", "/api/nodes/nonexistent");
    assert.equal(missing.status, 404);
  });

  it("enforces claim conflict 409, force override, PATCH transitions, and TTL expiry", async () => {
    const { env, sqlite } = makeEnv();

    // Create initial claim
    const c1 = await reqJson(env, "POST", "/api/claims", {
      task_id: "A3:boomerang-v1",
      lane: "A3",
      agent: "GLM-5.3-Flash",
      summary: "Testing Boomerang PDMP",
      ttl_minutes: 60,
    });
    assert.equal(c1.status, 201);
    assert.equal(c1.json.claim.status, "active");

    // Duplicate active claim returns 409
    const dup = await reqJson(env, "POST", "/api/claims", {
      task_id: "A3:boomerang-v1",
      lane: "A3",
      agent: "DeepSeek-4.1",
    });
    assert.equal(dup.status, 409);
    assert.equal(dup.json.error, "already_claimed");
    assert.equal(dup.json.existing_claim.id, c1.json.claim.id);

    // Force claim succeeds and replaces active claim
    const forced = await reqJson(env, "POST", "/api/claims", {
      task_id: "A3:boomerang-v1",
      lane: "A3",
      agent: "DeepSeek-4.1",
      force: true,
    });
    assert.equal(forced.status, 201);
    assert.equal(forced.json.claim.agent, "DeepSeek-4.1");

    const activeList = await reqJson(env, "GET", "/api/claims");
    assert.equal(activeList.json.count, 1);
    assert.equal(activeList.json.claims[0].id, forced.json.claim.id);

    // PATCH invalid status -> 400
    const badPatch = await reqJson(env, "PATCH", `/api/claims/${forced.json.claim.id}`, {
      status: "active",
    });
    assert.equal(badPatch.status, 400);

    // PATCH done -> disappears from active claims
    const donePatch = await reqJson(env, "PATCH", `/api/claims/${forced.json.claim.id}`, {
      status: "done",
      summary: "Finished",
    });
    assert.equal(donePatch.status, 200);
    assert.equal(donePatch.json.claim.status, "done");

    const afterDone = await reqJson(env, "GET", "/api/claims");
    assert.equal(afterDone.json.count, 0);

    // Expired claim auto-filtered and does not block re-claiming same task_id
    sqlite
      .prepare(
        `INSERT INTO claims (id, task_id, lane, agent, summary, status, expires_at, created_at)
         VALUES ('expired-1', 'A2:expired-task', 'A2', 'old-agent', 'stale', 'active',
                 '2020-01-01T00:00:00.000Z', '2019-12-31T23:00:00.000Z')`
      )
      .run();

    const claimsList = await reqJson(env, "GET", "/api/claims");
    assert.equal(claimsList.json.count, 0);

    const reclaim = await reqJson(env, "POST", "/api/claims", {
      task_id: "A2:expired-task",
      lane: "A2",
      agent: "Muse-1.3",
    });
    assert.equal(reclaim.status, 201);
    assert.equal(reclaim.json.claim.agent, "Muse-1.3");
  });

  it("enforces Bearer auth on write routes and MCP write tools when FLYWHEEL_API_KEY is set", async () => {
    const { env } = makeEnv({ apiKey: "super-secret-token" });
    const authHeader = { Authorization: "Bearer super-secret-token" };

    // Public reads remain accessible without auth
    assert.equal((await reqJson(env, "GET", "/api/health")).status, 200);
    assert.equal((await reqJson(env, "GET", "/api/dag")).status, 200);
    assert.equal((await reqJson(env, "GET", "/api/frontier")).status, 200);

    // Unauthenticated writes rejected with 401
    assert.equal(
      (await reqJson(env, "POST", "/api/nodes", { lane: "A1", run_id: "r1", decision: "keep" }))
        .status,
      401
    );
    assert.equal(
      (
        await reqJson(
          env,
          "POST",
          "/api/claims",
          { task_id: "t1", lane: "A1", agent: "bot" },
          { Authorization: "Bearer wrong-token" }
        )
      ).status,
      401
    );

    // Authenticated writes succeed
    const createdClaim = await reqJson(
      env,
      "POST",
      "/api/claims",
      { task_id: "t1", lane: "A1", agent: "bot" },
      authHeader
    );
    assert.equal(createdClaim.status, 201);

    // PATCH without auth fails, with auth succeeds
    assert.equal(
      (
        await reqJson(env, "PATCH", `/api/claims/${createdClaim.json.claim.id}`, {
          status: "abandoned",
        })
      ).status,
      401
    );
    assert.equal(
      (
        await reqJson(
          env,
          "PATCH",
          `/api/claims/${createdClaim.json.claim.id}`,
          { status: "abandoned" },
          authHeader
        )
      ).status,
      200
    );

    // MCP JSON-RPC: initialize & tools/list are public
    const initResp = await reqJson(env, "POST", "/api/mcp", {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {},
    });
    assert.equal(initResp.status, 200);
    assert.equal(initResp.json.result.serverInfo.name, "apin-flywheel");

    const listResp = await reqJson(env, "POST", "/api/mcp", {
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
    });
    assert.equal(listResp.status, 200);
    const toolNames = listResp.json.result.tools.map((t) => t.name).sort();
    assert.deepEqual(toolNames, [
      "flywheel_add_note",
      "flywheel_claim_task",
      "flywheel_frontier",
      "flywheel_get_node",
      "flywheel_get_source",
      "flywheel_list_nodes",
      "flywheel_list_notes",
      "flywheel_post_node",
      "flywheel_register",
      "flywheel_search",
      "flywheel_set_node_state",
      "flywheel_update_note",
      "flywheel_verify_claim",
      "flywheel_verify_queue",
      "flywheel_verify_result",
    ]);

    // MCP write tool without bearer token -> 401
    const mcpWriteDenied = await reqJson(env, "POST", "/api/mcp", {
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: {
        name: "flywheel_post_node",
        arguments: { lane: "A0", run_id: "mcp_run", decision: "keep", soft_min_eff: 0.002 },
      },
    });
    assert.equal(mcpWriteDenied.status, 401);
    assert.equal(mcpWriteDenied.json.error.code, -32001);

    // MCP write tool WITH bearer token -> succeeds
    const mcpWriteOk = await reqJson(
      env,
      "POST",
      "/api/mcp",
      {
        jsonrpc: "2.0",
        id: 4,
        method: "tools/call",
        params: {
          name: "flywheel_post_node",
          arguments: {
            id: "A0:mcp_run:T0001",
            lane: "A0",
            run_id: "mcp_run",
            trial_id: "T0001",
            decision: "keep",
            soft_min_eff: 0.002,
          },
        },
      },
      authHeader
    );
    assert.equal(mcpWriteOk.status, 200);
    assert.equal(mcpWriteOk.json.result.isError, false);
    assert.equal(mcpWriteOk.json.result.structuredContent.node.id, "A0:mcp_run:T0001");

    // MCP claim tool WITH bearer token -> succeeds, second call reports isError=true (conflict)
    const mcpClaim1 = await reqJson(
      env,
      "POST",
      "/api/mcp",
      {
        jsonrpc: "2.0",
        id: 5,
        method: "tools/call",
        params: {
          name: "flywheel_claim_task",
          arguments: { task_id: "A0:mcp_task", lane: "A0", agent: "mcp-agent" },
        },
      },
      authHeader
    );
    assert.equal(mcpClaim1.status, 200);
    assert.equal(mcpClaim1.json.result.isError, false);

    const mcpClaimConflict = await reqJson(
      env,
      "POST",
      "/api/mcp",
      {
        jsonrpc: "2.0",
        id: 6,
        method: "tools/call",
        params: {
          name: "flywheel_claim_task",
          arguments: { task_id: "A0:mcp_task", lane: "A0", agent: "other-agent" },
        },
      },
      authHeader
    );
    assert.equal(mcpClaimConflict.status, 200);
    assert.equal(mcpClaimConflict.json.result.isError, true);
    assert.equal(mcpClaimConflict.json.result.structuredContent.error, "already_claimed");

    // MCP read tools work without auth
    const mcpFrontier = await reqJson(env, "POST", "/api/mcp", {
      jsonrpc: "2.0",
      id: 7,
      method: "tools/call",
      params: { name: "flywheel_frontier", arguments: { lane: "A0" } },
    });
    assert.equal(mcpFrontier.status, 200);
    assert.equal(
      mcpFrontier.json.result.structuredContent.frontier.A0.id,
      "A0:mcp_run:T0001"
    );

    const mcpList = await reqJson(env, "POST", "/api/mcp", {
      jsonrpc: "2.0",
      id: 8,
      method: "tools/call",
      params: { name: "flywheel_list_nodes", arguments: { lane: "A0" } },
    });
    assert.equal(mcpList.status, 200);
    assert.equal(mcpList.json.result.structuredContent.count, 1);
  });

  it("enforces multi-contributor api_keys, ownership isolation, verified=0, rate limits, admin verify/revoke, and invite registration", async () => {
    const { env } = makeEnv({ apiKey: "root-admin-secret" });
    env.FLYWHEEL_INVITE_CODE = "apin-community-2026";
    const adminHdr = { Authorization: "Bearer root-admin-secret" };

    // 1. Admin mints a contributor key for alice with tight rate limit = 3/hour
    const mintAlice = await reqJson(
      env,
      "POST",
      "/api/admin/keys",
      { handle: "alice", role: "contributor", hourly_limit: 3 },
      adminHdr
    );
    assert.equal(mintAlice.status, 201);
    assert.ok(mintAlice.json.token.startsWith("apfw_"));
    const aliceHdr = { Authorization: `Bearer ${mintAlice.json.token}` };

    // 2. Self-serve registration via invite code for bob
    const badInvite = await reqJson(env, "POST", "/api/register", {
      handle: "bob",
      invite_code: "wrong",
    });
    assert.equal(badInvite.status, 403);

    const regBob = await reqJson(env, "POST", "/api/register", {
      handle: "bob",
      invite_code: "apin-community-2026",
    });
    assert.equal(regBob.status, 201);
    assert.equal(regBob.json.role, "contributor");
    const bobHdr = { Authorization: `Bearer ${regBob.json.token}` };

    // Duplicate handle rejected with 409
    const dupBob = await reqJson(env, "POST", "/api/register", {
      handle: "Bob",
      invite_code: "apin-community-2026",
    });
    assert.equal(dupBob.status, 409);

    // 3. Contributor cannot access admin endpoints
    const aliceKeys = await reqJson(env, "GET", "/api/admin/keys", undefined, aliceHdr);
    assert.equal(aliceKeys.status, 403);

    // 4. Contributor posts node trying to spoof verified=true and author="admin" -> forced verified=0 & author="alice"
    const nAlice = await reqJson(
      env,
      "POST",
      "/api/nodes",
      {
        id: "A1:alice_run:T0001",
        lane: "A1",
        run_id: "alice_run",
        trial_id: "T0001",
        author: "admin",
        decision: "keep",
        soft_min_eff: 0.002,
        verified: true,
        real_stan_passed: 5,
        real_stan_total: 5,
      },
      aliceHdr
    );
    assert.equal(nAlice.status, 201);
    assert.equal(nAlice.json.node.verified, 0);
    assert.equal(nAlice.json.node.author, "alice");
    assert.equal(nAlice.json.node.real_stan_passed, null);

    // 5. Bob cannot overwrite Alice's node -> 403
    const bobOverwrite = await reqJson(
      env,
      "POST",
      "/api/nodes",
      {
        id: "A1:alice_run:T0001",
        lane: "A1",
        run_id: "alice_run",
        decision: "discard_tier0",
      },
      bobHdr
    );
    assert.equal(bobOverwrite.status, 403);
    assert.equal(bobOverwrite.json.error, "forbidden_node_owner");

    // 6. Alice claims a task -> agent stamped with "alice:..."; Bob cannot force or PATCH Alice's claim
    const cAlice = await reqJson(
      env,
      "POST",
      "/api/claims",
      { task_id: "A4:warp", lane: "A4", agent: "glm53" },
      aliceHdr
    );
    assert.equal(cAlice.status, 201);
    assert.equal(cAlice.json.claim.agent, "alice:glm53");

    const bobForce = await reqJson(
      env,
      "POST",
      "/api/claims",
      { task_id: "A4:warp", lane: "A4", agent: "deepseek", force: true },
      bobHdr
    );
    assert.equal(bobForce.status, 403);

    const bobPatch = await reqJson(
      env,
      "PATCH",
      `/api/claims/${cAlice.json.claim.id}`,
      { status: "abandoned" },
      bobHdr
    );
    assert.equal(bobPatch.status, 403);

    // 7. Alice closes her own claim (3rd write -> hits hourly_limit=3), 4th write -> 429!
    const alicePatch = await reqJson(
      env,
      "PATCH",
      `/api/claims/${cAlice.json.claim.id}`,
      { status: "done" },
      aliceHdr
    );
    assert.equal(alicePatch.status, 200);

    const aliceRateLimited = await reqJson(
      env,
      "POST",
      "/api/nodes",
      { lane: "A1", run_id: "alice_run", decision: "keep" },
      aliceHdr
    );
    assert.equal(aliceRateLimited.status, 429);
    assert.equal(aliceRateLimited.json.error, "rate_limit_exceeded");

    // 8. Admin verifies Alice's node with reproduced real-Stan score, and revokes Bob's key
    const verifiedRes = await reqJson(
      env,
      "POST",
      `/api/admin/nodes/${encodeURIComponent("A1:alice_run:T0001")}/verify`,
      { verified: true, real_stan_passed: 4, real_stan_total: 5, real_ess_per_cost: 0.025 },
      adminHdr
    );
    assert.equal(verifiedRes.status, 200);
    assert.equal(verifiedRes.json.node.verified, 1);
    assert.equal(verifiedRes.json.node.real_stan_passed, 4);

    const revokeBob = await reqJson(
      env,
      "DELETE",
      `/api/admin/keys/${regBob.json.key_id}`,
      undefined,
      adminHdr
    );
    assert.equal(revokeBob.status, 200);

    // Revoked key is rejected immediately with 401
    const bobAfterRevoke = await reqJson(
      env,
      "POST",
      "/api/nodes",
      { lane: "A1", run_id: "bob_run", decision: "keep" },
      bobHdr
    );
    assert.equal(bobAfterRevoke.status, 401);
  });

  it("caches public GET reads with canonical anti-cache-bust keys and 30s Cache-Control", async () => {
    const { env } = makeEnv();
    const store = new Map();
    let dbPrepares = 0;
    const origPrepare = env.DB.prepare.bind(env.DB);
    env.DB.prepare = (q) => {
      dbPrepares += 1;
      return origPrepare(q);
    };
    env.CACHE = {
      async match(req) {
        const key = typeof req === "string" ? req : req.url;
        const hit = store.get(key);
        return hit ? hit.clone() : undefined;
      },
      async put(req, resp) {
        const key = typeof req === "string" ? req : req.url;
        store.set(key, resp.clone());
      },
    };

    const r1 = await reqJson(env, "GET", "/api/dag?random_bust=1");
    assert.equal(r1.status, 200);
    assert.equal(r1.headers.get("Cache-Control"), "public, max-age=30, s-maxage=30");
    const preparesAfterFirst = dbPrepares;
    assert.ok(preparesAfterFirst > 0);

    // Different unknown query param must still hit the exact same canonical cache entry (0 extra D1 queries!)
    const r2 = await reqJson(env, "GET", "/api/dag?random_bust=99999&foo=bar");
    assert.equal(r2.status, 200);
    assert.equal(dbPrepares, preparesAfterFirst);
  });
});

// ---------------------------------------------------------------------------
// Research objects: open registration, sources, parents, notes, state, search,
// verification queue, abuse limits.
// ---------------------------------------------------------------------------

const ADMIN = { Authorization: "Bearer root-admin-secret" };
const bearer = (token) => ({ Authorization: `Bearer ${token}` });
const fromIp = (ip) => ({ "CF-Connecting-IP": ip });

async function register(env, handle, extra = {}, ip = "198.51.100.7") {
  return reqJson(env, "POST", "/api/register", { handle, ...extra }, fromIp(ip));
}

async function registerToken(env, handle, ip) {
  const res = await register(env, handle, {}, ip ?? `203.0.113.${Math.floor(Math.random() * 250) + 1}`);
  assert.equal(res.status, 201, JSON.stringify(res.json));
  return res.json.token;
}

function trial(id, extra = {}) {
  const [lane, run_id, trial_id] = id.split(":");
  return { id, lane, run_id, trial_id, decision: "keep", soft_min_eff: 0.001, ...extra };
}

async function rawPost(env, pathname, bodyText, headers = {}) {
  const resp = await handleRequest(
    new Request(`https://flywheel.test${pathname}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: bodyText,
    }),
    env
  );
  return { status: resp.status, json: await resp.json() };
}

async function streamedPost(env, pathname, bodyText, headers = {}) {
  const bytes = new TextEncoder().encode(bodyText);
  const stream = new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += 4096) controller.enqueue(bytes.slice(i, i + 4096));
      controller.close();
    },
  });
  const resp = await handleRequest(
    new Request(`https://flywheel.test${pathname}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: stream,
      duplex: "half",
    }),
    env
  );
  return { status: resp.status, json: await resp.json() };
}

describe("open registration and abuse limits", () => {
  it("registers provisional tokens without an invite, stores only hashes, and enforces handle rules", async () => {
    const { env, sqlite } = makeEnv({ apiKey: "root-admin-secret" });

    const reg = await register(env, "agent-ada", { agent: "claude-code", model: "claude-opus-5-5" });
    assert.equal(reg.status, 201);
    assert.equal(reg.json.role, "provisional");
    assert.equal(reg.json.tier, "provisional");
    assert.equal(reg.json.daily_write_limit, 300);
    assert.ok(reg.json.token.startsWith("apfw_"));

    // Stored hashed: the raw token appears in no column; the hash matches sha256(token).
    const row = sqlite.prepare("SELECT * FROM api_keys WHERE handle = 'agent-ada'").get();
    assert.ok(!Object.values(row).includes(reg.json.token));
    const crypto = await import("node:crypto");
    assert.equal(row.key_hash, crypto.createHash("sha256").update(reg.json.token).digest("hex"));
    assert.equal(row.agent, "claude-code");
    assert.equal(row.model, "claude-opus-5-5");

    assert.equal((await register(env, "x")).status, 400); // too short
    assert.equal((await register(env, "bad handle!")).status, 400);
    assert.equal((await register(env, "Admin")).json.error, "reserved_handle");
    // Case-insensitive uniqueness.
    const dup = await register(env, "AGENT-ADA", {}, "198.51.100.8");
    assert.equal(dup.status, 409);
    assert.equal(dup.json.error, "handle_already_registered");

    // A handle that already authors trials cannot be claimed by a newcomer.
    await reqJson(env, "POST", "/api/nodes", trial("A1:seed:T0001", { author: "GLM-5.3-Flash" }), ADMIN);
    const squat = await register(env, "glm-5.3-flash", {}, "198.51.100.9");
    assert.equal(squat.status, 409);

    // A revoked handle stays reserved.
    await reqJson(env, "POST", "/api/admin/handles/agent-ada/revoke", undefined, ADMIN);
    assert.equal((await register(env, "agent-ada", {}, "198.51.100.10")).status, 409);

    // A valid invite code still gives the contributor tier at once; a wrong one is refused.
    env.FLYWHEEL_INVITE_CODE = "apin-community-2026";
    assert.equal((await register(env, "invited", { invite_code: "nope" }, "198.51.100.11")).status, 403);
    const invited = await register(env, "invited", { invite_code: "apin-community-2026" }, "198.51.100.11");
    assert.equal(invited.status, 201);
    assert.equal(invited.json.role, "contributor");
  });

  it("refuses registrations over the per-IP and global daily caps", async () => {
    const { env, sqlite } = makeEnv({ apiKey: "root-admin-secret" });
    env.FLYWHEEL_REGISTRATIONS_PER_IP_PER_DAY = "2";
    env.FLYWHEEL_REGISTRATIONS_PER_DAY = "3";

    assert.equal((await register(env, "ip-one", {}, "192.0.2.1")).status, 201);
    assert.equal((await register(env, "ip-two", {}, "192.0.2.1")).status, 201);
    const third = await register(env, "ip-three", {}, "192.0.2.1");
    assert.equal(third.status, 429);
    assert.equal(third.json.error, "registration_ip_cap");
    // The IP is never stored raw.
    const keys = sqlite.prepare("SELECT key_id FROM rate_limits").all().map((r) => r.key_id);
    assert.ok(keys.every((k) => !k.includes("192.0.2.1")));

    assert.equal((await register(env, "other-ip", {}, "192.0.2.2")).status, 201);
    const overGlobal = await register(env, "late-ip", {}, "192.0.2.3");
    assert.equal(overGlobal.status, 429);
    assert.equal(overGlobal.json.error, "registration_daily_cap");

    // Global cap 0 closes registration.
    env.FLYWHEEL_REGISTRATIONS_PER_DAY = "0";
    assert.equal((await register(env, "closed", {}, "192.0.2.4")).status, 429);
  });

  it("refuses oversized bodies on every write endpoint, with or without Content-Length", async () => {
    const { env } = makeEnv({ apiKey: "root-admin-secret" });
    const token = await registerToken(env, "big-body");
    const big = (n) => JSON.stringify({ handle: "h", pad: "x".repeat(n) });

    assert.equal((await rawPost(env, "/api/register", big(4096))).status, 413);
    assert.equal((await streamedPost(env, "/api/register", big(4096))).status, 413);
    for (const [path, size] of [
      ["/api/nodes", 400 * 1024],
      ["/api/notes", 40 * 1024],
      ["/api/claims", 9 * 1024],
      ["/api/verify/claim", 9 * 1024],
      ["/api/nodes/x/state", 9 * 1024],
      ["/api/verify/x/result", 40 * 1024],
      ["/api/mcp", 400 * 1024],
    ]) {
      const viaLength = await rawPost(env, path, big(size), bearer(token));
      assert.equal(viaLength.status, 413, `${path} with Content-Length`);
      const viaStream = await streamedPost(env, path, big(size), bearer(token));
      assert.equal(viaStream.status, 413, `${path} streamed`);
    }

    // Trial fields outside `source` are capped separately; the source itself at 256 KB.
    const fat = await reqJson(env, "POST", "/api/nodes", trial("A1:r:T0001", { hypothesis: "h".repeat(40 * 1024) }), bearer(token));
    assert.equal(fat.status, 413);
    const hugeSource = await reqJson(env, "POST", "/api/nodes", trial("A1:r:T0001", { source: "#".repeat(257 * 1024) }), bearer(token));
    assert.equal(hugeSource.status, 413);
    assert.equal(hugeSource.json.error, "source_too_large");
  });

  it("caps provisional writes per day and provisional trials per token and globally", async () => {
    const { env } = makeEnv({ apiKey: "root-admin-secret" });
    env.FLYWHEEL_PROVISIONAL_DAILY_WRITES = "4";
    env.FLYWHEEL_PROVISIONAL_TRIALS_PER_DAY = "2";
    env.FLYWHEEL_PROVISIONAL_TRIALS_PER_DAY_GLOBAL = "3";
    const a = await registerToken(env, "prov-a");
    const b = await registerToken(env, "prov-b");

    assert.equal((await reqJson(env, "POST", "/api/nodes", trial("A1:a:T0001"), bearer(a))).status, 201);
    assert.equal((await reqJson(env, "POST", "/api/nodes", trial("A1:a:T0002"), bearer(a))).status, 201);
    const third = await reqJson(env, "POST", "/api/nodes", trial("A1:a:T0003"), bearer(a));
    assert.equal(third.status, 429);
    assert.equal(third.json.scope, "provisional_trials_daily");
    // Re-posting an existing trial is not a new trial.
    assert.equal((await reqJson(env, "POST", "/api/nodes", trial("A1:a:T0001"), bearer(a))).status, 200);
    // Fifth write of the day hits the daily write cap.
    const fifth = await reqJson(env, "POST", "/api/notes", { kind: "insight", title: "t" }, bearer(a));
    assert.equal(fifth.status, 429);
    assert.equal(fifth.json.scope, "daily");

    assert.equal((await reqJson(env, "POST", "/api/nodes", trial("A1:b:T0001"), bearer(b))).status, 201);
    const globalCap = await reqJson(env, "POST", "/api/nodes", trial("A1:b:T0002"), bearer(b));
    assert.equal(globalCap.status, 429);
    assert.equal(globalCap.json.scope, "provisional_trials_global_daily");

    // Provisional trials stay visible in the DAG and frontier, marked with their tier.
    const dag = await reqJson(env, "GET", "/api/dag");
    assert.equal(dag.json.nodes.length, 3);
    assert.ok(dag.json.nodes.every((n) => n.author_tier === "provisional" && n.state === "unverified"));
    assert.equal(dag.json.frontier.A1.author_tier, "provisional");
  });

  it("lets maintainers promote a handle, revoke it, and purge everything it posted in one call", async () => {
    const { env, sqlite } = makeEnv({ apiKey: "root-admin-secret" });
    const spam = await registerToken(env, "spammer");
    const good = await registerToken(env, "good-agent");

    await reqJson(env, "POST", "/api/nodes", trial("A1:good:T0001", { source: "print('good')\n" }), bearer(good));
    await reqJson(env, "POST", "/api/nodes", trial("A1:spam:T0001", { source: "print('spam')\n" }), bearer(spam));
    await reqJson(
      env,
      "POST",
      "/api/nodes",
      trial("A1:spam:T0002", { parent_ids: [{ id: "A1:good:T0001", relation: "combines" }] }),
      bearer(spam)
    );
    await reqJson(env, "POST", "/api/notes", { kind: "insight", lane: "A1", title: "buy pills" }, bearer(spam));
    await reqJson(env, "POST", "/api/claims", { task_id: "t", lane: "A1", agent: "bot" }, bearer(spam));

    // Promote good-agent: role and existing trials move to contributor.
    const promoted = await reqJson(env, "POST", "/api/admin/handles/good-agent/promote", undefined, ADMIN);
    assert.equal(promoted.status, 200);
    assert.equal(promoted.json.promoted, 1);
    assert.equal(sqlite.prepare("SELECT role FROM api_keys WHERE handle = 'good-agent'").get().role, "contributor");
    assert.equal(sqlite.prepare("SELECT author_tier FROM nodes WHERE id = 'A1:good:T0001'").get().author_tier, "contributor");
    // Non-admins cannot use maintainer routes.
    assert.equal((await reqJson(env, "POST", "/api/admin/handles/spammer/purge", undefined, bearer(good))).status, 403);

    const purge = await reqJson(env, "POST", "/api/admin/handles/spammer/purge", undefined, ADMIN);
    assert.equal(purge.status, 200);
    assert.deepEqual(purge.json.deleted, { nodes: 2, notes: 1, claims: 1, sources: 1 });
    assert.equal(purge.json.revoked, 1);
    assert.equal(sqlite.prepare("SELECT COUNT(*) AS c FROM node_parents").get().c, 0);
    assert.equal(sqlite.prepare("SELECT COUNT(*) AS c FROM sources").get().c, 1); // good-agent's source kept
    assert.equal((await reqJson(env, "POST", "/api/nodes", trial("A1:spam:T0003"), bearer(spam))).status, 401);
    const dag = await reqJson(env, "GET", "/api/dag");
    assert.deepEqual(dag.json.nodes.map((n) => n.id), ["A1:good:T0001"]);

    // Admin key listing shows tier and agent metadata, never hashes.
    const keys = await reqJson(env, "GET", "/api/admin/keys?role=contributor", undefined, ADMIN);
    assert.deepEqual(keys.json.keys.map((k) => k.handle), ["good-agent"]);
    assert.ok(!("key_hash" in keys.json.keys[0]));
  });
});

describe("trials: idempotent writes, sources, parents and research state", () => {
  it("creates idempotently, never overwrites another author's trial, and keeps state on re-post", async () => {
    const { env } = makeEnv({ apiKey: "root-admin-secret" });
    const alice = await registerToken(env, "alice-p");
    const bob = await registerToken(env, "bob-p");

    const first = await reqJson(env, "POST", "/api/nodes", trial("A2:run:T0001", { hypothesis: "v1" }), bearer(alice));
    assert.equal(first.status, 201);
    assert.equal(first.json.created, true);
    const again = await reqJson(env, "POST", "/api/nodes", trial("A2:run:T0001", { hypothesis: "v1" }), bearer(alice));
    assert.equal(again.status, 200);
    assert.equal(again.json.created, false);
    assert.equal(again.json.node.created_at, first.json.node.created_at);

    const steal = await reqJson(env, "POST", "/api/nodes", trial("A2:run:T0001", { hypothesis: "mine now" }), bearer(bob));
    assert.equal(steal.status, 403);
    assert.equal(steal.json.error, "forbidden_node_owner");

    // Admin attaches reproduced real-Stan metrics; the author's re-post does not wipe them or the state.
    await reqJson(env, "POST", `/api/admin/nodes/${encodeURIComponent("A2:run:T0001")}/verify`, { real_stan_passed: 3, real_stan_total: 5 }, ADMIN);
    const repost = await reqJson(env, "POST", "/api/nodes", trial("A2:run:T0001", { hypothesis: "v2" }), bearer(alice));
    assert.equal(repost.json.node.hypothesis, "v2");
    assert.equal(repost.json.node.author, "alice-p");
    assert.equal(repost.json.node.real_stan_passed, 3);
    assert.equal(repost.json.node.state, "verified");
    assert.equal(repost.json.node.verified, 1);
  });

  it("stores sources out of line, content-addressed, and keeps bodies out of lists and the DAG", async () => {
    const { env, sqlite } = makeEnv({ apiKey: "root-admin-secret" });
    const token = await registerToken(env, "src-agent");
    const source = "import numpy as np\n# UNIQUE-SOURCE-MARKER\ndef adapt(x):\n    return x\n";
    const crypto = await import("node:crypto");
    const sha = crypto.createHash("sha256").update(source).digest("hex");

    const bad = await reqJson(env, "POST", "/api/nodes", trial("A1:s:T0001", { source, code_sha: "deadbeefdeadbeef" }), bearer(token));
    assert.equal(bad.status, 400);
    assert.equal(bad.json.error, "code_sha_mismatch");

    const posted = await reqJson(env, "POST", "/api/nodes", trial("A1:s:T0001", { source, code_sha: sha.slice(0, 16) }), bearer(token));
    assert.equal(posted.status, 201);
    assert.equal(posted.json.node.has_source, true);
    await reqJson(env, "POST", "/api/nodes", trial("A1:s:T0002", { source }), bearer(token));
    assert.equal(sqlite.prepare("SELECT COUNT(*) AS c FROM sources").get().c, 1);
    await reqJson(env, "POST", "/api/nodes", trial("A1:s:T0003"), bearer(token));

    const fetched = await reqJson(env, "GET", `/api/nodes/${encodeURIComponent("A1:s:T0002")}/source`);
    assert.equal(fetched.status, 200);
    assert.equal(fetched.json.source, source);
    assert.equal(fetched.json.sha256, sha);
    assert.equal(fetched.json.size, Buffer.byteLength(source));
    const none = await reqJson(env, "GET", `/api/nodes/${encodeURIComponent("A1:s:T0003")}/source`);
    assert.equal(none.status, 404);
    assert.equal(none.json.error, "no_source");

    for (const path of ["/api/dag", "/api/nodes", "/api/nodes?lane=A1", `/api/nodes/${encodeURIComponent("A1:s:T0001")}`, "/api/search?q=adapt"]) {
      const res = await reqJson(env, "GET", path);
      assert.ok(!JSON.stringify(res.json).includes("UNIQUE-SOURCE-MARKER"), `${path} leaked a source body`);
    }
    const listed = await reqJson(env, "GET", "/api/nodes?lane=A1");
    assert.deepEqual(listed.json.nodes.map((n) => n.has_source).sort(), [false, true, true]);
  });

  it("records several parents with relations, backfills legacy parent_id, and returns all DAG edges", async () => {
    const legacy = `INSERT INTO nodes (id, parent_id, lane, run_id, decision, verified, created_at) VALUES
      ('A3:old:T0001', NULL, 'A3', 'old', 'keep_baseline', 1, '2026-11-01T00:00:00Z'),
      ('A3:old:T0002', 'A3:old:T0001', 'A3', 'old', 'keep', 0, '2026-11-01T00:01:00Z');`;
    const { env, sqlite } = makeEnv({ apiKey: "root-admin-secret", preMigration: legacy });
    // Migration backfill: legacy parent_id became an incumbent edge; verified became state.
    assert.deepEqual(
      { ...sqlite.prepare("SELECT child, parent, relation FROM node_parents").get() },
      { child: "A3:old:T0002", parent: "A3:old:T0001", relation: "incumbent" }
    );
    assert.equal(sqlite.prepare("SELECT state FROM nodes WHERE id = 'A3:old:T0001'").get().state, "verified");

    const token = await registerToken(env, "combiner");
    const bad = await reqJson(env, "POST", "/api/nodes", trial("A3:new:T0001", { parent_ids: [{ id: "A3:old:T0001", relation: "stole" }] }), bearer(token));
    assert.equal(bad.status, 400);
    assert.equal(bad.json.error, "invalid_relation");

    const child = await reqJson(
      env,
      "POST",
      "/api/nodes",
      trial("A3:new:T0001", {
        parent_ids: [
          { id: "A3:old:T0002", relation: "incumbent" },
          { id: "A3:old:T0001", relation: "combines" },
        ],
      }),
      bearer(token)
    );
    assert.equal(child.status, 201);
    assert.equal(child.json.node.parent_id, "A3:old:T0002"); // kept for old clients

    const dag = await reqJson(env, "GET", "/api/dag");
    const edges = dag.json.edges.map((e) => `${e.source}>${e.target}:${e.relation}`).sort();
    assert.deepEqual(edges, [
      "A3:old:T0001>A3:new:T0001:combines",
      "A3:old:T0001>A3:old:T0002:incumbent",
      "A3:old:T0002>A3:new:T0001:incumbent",
    ]);
    assert.ok(dag.json.edges.every((e) => e.lane === "A3" && typeof e.decision === "string"));

    const detail = await reqJson(env, "GET", `/api/nodes/${encodeURIComponent("A3:new:T0001")}`);
    assert.deepEqual(detail.json.parents.map((p) => `${p.id}:${p.relation}`).sort(), [
      "A3:old:T0001:combines",
      "A3:old:T0002:incumbent",
    ]);
    const root = await reqJson(env, "GET", `/api/nodes/${encodeURIComponent("A3:old:T0001")}`);
    assert.deepEqual(root.json.children.map((c) => c.id).sort(), ["A3:new:T0001", "A3:old:T0002"]);
  });

  it("enforces research-state rules and keeps the verified column consistent", async () => {
    const { env, sqlite } = makeEnv({ apiKey: "root-admin-secret" });
    env.FLYWHEEL_INVITE_CODE = "inv";
    const author = await registerToken(env, "state-author");
    const prov = await registerToken(env, "state-prov");
    const contrib = (await register(env, "state-contrib", { invite_code: "inv" }, "192.0.2.50")).json.token;
    const id = "A4:st:T0001";
    const path = `/api/nodes/${encodeURIComponent(id)}/state`;
    await reqJson(env, "POST", "/api/nodes", trial(id, { soft_min_eff: 0.9 }), bearer(author));
    await reqJson(env, "POST", "/api/nodes", trial("A4:st:T0002", { soft_min_eff: 0.1 }), bearer(author));

    assert.equal((await reqJson(env, "POST", path, { state: "bogus" }, bearer(contrib))).status, 400);
    // Only maintainers set verified, even the author cannot.
    assert.equal((await reqJson(env, "POST", path, { state: "verified" }, bearer(author))).status, 403);
    assert.equal((await reqJson(env, "POST", path, { state: "verified" }, bearer(contrib))).status, 403);
    // Provisional tokens cannot flag other authors' trials.
    assert.equal((await reqJson(env, "POST", path, { state: "disputed", reason: "x" }, bearer(prov))).json.error, "forbidden_tier");
    // Contributors flag with a reason.
    assert.equal((await reqJson(env, "POST", path, { state: "disputed" }, bearer(contrib))).json.error, "reason_required");
    const flagged = await reqJson(env, "POST", path, { state: "disputed", reason: "fails on dim 20" }, bearer(contrib));
    assert.equal(flagged.status, 200);
    assert.equal(flagged.json.node.state, "disputed");
    assert.equal(flagged.json.node.state_by, "state-contrib");
    // Disputed trials leave the frontier.
    assert.equal((await reqJson(env, "GET", "/api/frontier")).json.frontier.A4.id, "A4:st:T0002");

    const verified = await reqJson(env, "POST", path, { state: "verified", reason: "reproduced" }, ADMIN);
    assert.equal(verified.json.node.verified, 1);
    assert.equal(sqlite.prepare("SELECT verified FROM nodes WHERE id = ?").get(id).verified, 1);
    // Verified is locked against contributor flags.
    assert.equal((await reqJson(env, "POST", path, { state: "needs-replication", reason: "again" }, bearer(contrib))).status, 409);
    // Authors may retract their own trial at any time; verified drops back to 0.
    const retracted = await reqJson(env, "POST", path, { state: "retracted", reason: "bug in my eval" }, bearer(author));
    assert.equal(retracted.status, 200);
    assert.equal(retracted.json.node.state, "retracted");
    assert.equal(retracted.json.node.verified, 0);
    // ...but not someone else's.
    const other = `/api/nodes/${encodeURIComponent("A4:st:T0002")}/state`;
    assert.equal((await reqJson(env, "POST", other, { state: "retracted" }, bearer(contrib))).status, 403);
  });
});

describe("notes, search and MCP tools", () => {
  it("creates, lists, reads and updates hypotheses and insights with size caps and permissions", async () => {
    const { env } = makeEnv({ apiKey: "root-admin-secret" });
    env.FLYWHEEL_INVITE_CODE = "inv";
    const ada = await registerToken(env, "note-ada");
    const prov = await registerToken(env, "note-prov");
    const contrib = (await register(env, "note-contrib", { invite_code: "inv" }, "192.0.2.60")).json.token;

    assert.equal((await reqJson(env, "POST", "/api/notes", { kind: "rumour", title: "t" }, bearer(ada))).status, 400);
    assert.equal((await reqJson(env, "POST", "/api/notes", { kind: "insight", title: "t".repeat(201) }, bearer(ada))).status, 400);
    assert.equal((await reqJson(env, "POST", "/api/notes", { kind: "insight", title: "t", body: "b".repeat(16001) }, bearer(ada))).status, 400);
    assert.equal((await reqJson(env, "POST", "/api/notes", { kind: "insight", title: "t", state: "supported" }, bearer(ada))).status, 400);

    const hyp = await reqJson(
      env,
      "POST",
      "/api/notes",
      { kind: "hypothesis", lane: "A1", title: "Robust Fisher beats diagonal on funnels", body: "MAD scale estimate", node_ids: ["A1:r:T0002"] },
      bearer(ada)
    );
    assert.equal(hyp.status, 201);
    assert.equal(hyp.json.note.state, "open");
    assert.equal(hyp.json.note.author, "note-ada");
    assert.deepEqual(hyp.json.note.node_ids, ["A1:r:T0002"]);
    const id = hyp.json.note.id;
    await reqJson(env, "POST", "/api/notes", { kind: "insight", lane: "A1", title: "Warmup starvation games Tier 1" }, bearer(ada));
    await reqJson(env, "POST", "/api/notes", { kind: "insight", lane: "A3", title: "Zigzag needs bounds" }, bearer(ada));

    const list = await reqJson(env, "GET", "/api/notes?lane=A1&kind=hypothesis&state=open");
    assert.equal(list.json.count, 1);
    assert.equal(list.json.notes[0].id, id);
    assert.equal((await reqJson(env, "GET", "/api/notes?lane=A1")).json.count, 2);
    const paged = await reqJson(env, "GET", "/api/notes?limit=2");
    assert.equal(paged.json.next_offset, 2);
    assert.equal((await reqJson(env, "GET", "/api/notes?limit=2&offset=2")).json.count, 1);
    assert.equal((await reqJson(env, "GET", `/api/notes/${id}`)).json.note.title, "Robust Fisher beats diagonal on funnels");

    const path = `/api/notes/${id}`;
    assert.equal((await reqJson(env, "PATCH", path, { state: "verified" }, bearer(ada))).status, 400);
    assert.equal((await reqJson(env, "PATCH", path, { state: "refuted" }, bearer(prov))).json.error, "forbidden_tier");
    assert.equal((await reqJson(env, "PATCH", path, { title: "hijack" }, bearer(contrib))).json.error, "forbidden_note_owner");
    const refuted = await reqJson(env, "PATCH", path, { state: "refuted", reason: "A1:r:T0007 shows no gain" }, bearer(contrib));
    assert.equal(refuted.status, 200);
    assert.equal(refuted.json.note.state, "refuted");
    assert.equal(refuted.json.note.state_by, "note-contrib");
    const edited = await reqJson(env, "PATCH", path, { body: "MAD scale, revised", state: "needs-replication" }, bearer(ada));
    assert.equal(edited.json.note.body, "MAD scale, revised");
    assert.equal(edited.json.note.state, "needs-replication");
  });

  it("searches trials and notes with AND terms, lane and kind filters, and pagination", async () => {
    const { env } = makeEnv({ apiKey: "root-admin-secret" });
    const token = await registerToken(env, "searcher");
    await reqJson(env, "POST", "/api/nodes", trial("A3:pdmp:T0001", { hypothesis: "Zigzag PDMP with adaptive refresh", codex_refs: ["pdmp-zigzag"] }), bearer(token));
    await reqJson(env, "POST", "/api/nodes", trial("A3:pdmp:T0002", { title: "Boomerang sampler", hypothesis: "elliptical refresh" }), bearer(token));
    await reqJson(env, "POST", "/api/nodes", trial("A1:fisher:T0001", { hypothesis: "Low-rank Fisher 100% better" }), bearer(token));
    await reqJson(env, "POST", "/api/notes", { kind: "insight", lane: "A3", title: "Refresh rate matters", body: "zigzag needs a refresh rate near 1" }, bearer(token));

    const both = await reqJson(env, "GET", "/api/search?q=zigzag");
    assert.equal(both.status, 200);
    assert.deepEqual(both.json.results.map((r) => r.kind).sort(), ["node", "note"]);
    const node = both.json.results.find((r) => r.kind === "node");
    assert.equal(node.id, "A3:pdmp:T0001");
    assert.match(node.snippet, /\[Zigzag\]/);
    assert.equal(node.author, "searcher");
    assert.equal(node.state, "unverified");

    assert.equal((await reqJson(env, "GET", "/api/search?q=zigzag%20refresh&kind=node")).json.count, 1);
    assert.equal((await reqJson(env, "GET", "/api/search?q=refresh&lane=A1")).json.count, 0);
    assert.equal((await reqJson(env, "GET", "/api/search?q=searcher")).json.count, 4); // author
    assert.equal((await reqJson(env, "GET", "/api/search?q=pdmp-zigzag")).json.count, 1); // technique ref
    // LIKE wildcards are literal.
    assert.equal((await reqJson(env, "GET", "/api/search?q=100%25")).json.count, 1);
    assert.equal((await reqJson(env, "GET", "/api/search?q=%25%25")).status, 400);

    const page1 = await reqJson(env, "GET", "/api/search?q=refresh&limit=2");
    assert.equal(page1.json.count, 2);
    assert.equal(page1.json.next_offset, 2);
    // Title hits rank first.
    assert.equal(page1.json.results[0].title, "Refresh rate matters");
    const page2 = await reqJson(env, "GET", "/api/search?q=refresh&limit=2&offset=2");
    assert.equal(page2.json.count, 1);
    assert.equal(page2.json.next_offset, null);
  });

  it("exposes the new operations as MCP tools with the same auth and limits", async () => {
    const { env } = makeEnv({ apiKey: "root-admin-secret" });
    const call = (name, args, headers = {}) =>
      reqJson(env, "POST", "/api/mcp", { jsonrpc: "2.0", id: 1, method: "tools/call", params: { name, arguments: args } }, headers);

    const reg = await call("flywheel_register", { handle: "mcp-agent" }, fromIp("192.0.2.70"));
    assert.equal(reg.json.result.isError, false);
    const token = reg.json.result.structuredContent.token;

    assert.equal((await call("flywheel_add_note", { kind: "insight", title: "x" })).status, 401);
    const note = await call("flywheel_add_note", { kind: "hypothesis", lane: "A2", title: "Funnel reparam helps" }, bearer(token));
    assert.equal(note.json.result.isError, false);
    const posted = await call(
      "flywheel_post_node",
      { ...trial("A2:mcp:T0001"), source: "print(1)\n", parent_ids: [{ id: "A2:x:T0009", relation: "combines" }] },
      bearer(token)
    );
    assert.equal(posted.json.result.structuredContent.node.has_source, true);

    const src = await call("flywheel_get_source", { node_id: "A2:mcp:T0001" });
    assert.equal(src.json.result.structuredContent.source, "print(1)\n");
    const found = await call("flywheel_search", { q: "funnel" });
    assert.equal(found.json.result.structuredContent.results[0].kind, "note");
    const notes = await call("flywheel_list_notes", { lane: "A2", kind: "hypothesis" });
    assert.equal(notes.json.result.structuredContent.count, 1);
    const queue = await call("flywheel_verify_queue", {});
    assert.equal(queue.json.result.structuredContent.items[0].node_id, "A2:mcp:T0001");
    const own = await call("flywheel_verify_claim", {}, bearer(token));
    assert.equal(own.json.result.isError, true);
    const retract = await call("flywheel_set_node_state", { id: "A2:mcp:T0001", state: "retracted" }, bearer(token));
    assert.equal(retract.json.result.structuredContent.node.state, "retracted");
  });
});

describe("verification queue", () => {
  async function setup() {
    const { env, sqlite } = makeEnv({ apiKey: "root-admin-secret" });
    env.FLYWHEEL_INVITE_CODE = "inv";
    // Verifying needs the contributor tier, so every party here registers with the invite.
    const author = (await register(env, "q-author", { invite_code: "inv" }, "192.0.2.81")).json.token;
    const v1 = (await register(env, "q-verifier-1", { invite_code: "inv" }, "192.0.2.82")).json.token;
    const v2 = (await register(env, "q-verifier-2", { invite_code: "inv" }, "192.0.2.80")).json.token;
    const post = (id, extra) => reqJson(env, "POST", "/api/nodes", trial(id, { source: `# ${id}\n`, ...extra }), bearer(author));
    return { env, sqlite, author, v1, v2, post };
  }

  it("queues kept trials with source that beat the lane's best reviewed trial, and only those", async () => {
    const { env, sqlite, post } = await setup();
    // No reviewed trial yet: every kept trial with source is queued.
    assert.equal((await post("A1:q:T0001", { decision: "keep_baseline", soft_min_eff: 0.001 })).json.verify_queued, true);
    assert.equal((await post("A1:q:T0002", { decision: "keep", soft_min_eff: 0.0005 })).json.verify_queued, true);
    assert.equal((await post("A1:q:T0003", { decision: "discard_no_gain", soft_min_eff: 0.5 })).json.verify_queued, false); // not kept
    assert.equal((await post("A1:q:T0004", { decision: "keep", soft_min_eff: 0.002, source: null })).json.verify_queued, false); // no source
    assert.equal((await post("A1:q:T0005", { decision: "keep", soft_min_eff: 0.003 })).json.verify_queued, true);
    // Idempotent: a re-post does not queue twice.
    assert.equal((await post("A1:q:T0005", { decision: "keep", soft_min_eff: 0.003 })).json.verify_queued, false);
    // Once T0005 is replicated, only trials beating it are queued.
    sqlite.prepare("UPDATE nodes SET state = 'replicated' WHERE id = 'A1:q:T0005'").run();
    assert.equal((await post("A1:q:T0006", { decision: "keep", soft_min_eff: 0.0029 })).json.verify_queued, false);
    assert.equal((await post("A1:q:T0007", { decision: "keep", soft_min_eff: 0.004 })).json.verify_queued, true);

    const queue = await reqJson(env, "GET", "/api/verify/queue?lane=A1");
    assert.deepEqual(queue.json.items.map((i) => i.node_id), ["A1:q:T0001", "A1:q:T0002", "A1:q:T0005", "A1:q:T0007"]);
    const item = queue.json.items[2];
    assert.equal(item.status, "queued");
    assert.equal(item.available, true);
    assert.equal(item.reason, "beats_frontier");
    assert.equal(item.node.author, "q-author");
    assert.equal(item.node.has_source, true);
    assert.ok(!("lease_token_hash" in item));
  });

  it("a kept trial without source cannot block the queue with an inflated score", async () => {
    const { env, post } = await setup();
    // Sourceless (and even maintainer-verified sourceless) claims never set the bar.
    assert.equal((await post("A2:p:T0001", { decision: "keep", soft_min_eff: 999, source: null })).json.verify_queued, false);
    await reqJson(env, "POST", `/api/admin/nodes/${encodeURIComponent("A2:p:T0001")}/verify`, {}, ADMIN);
    assert.equal((await post("A2:p:T0002", { decision: "keep", soft_min_eff: 0.002 })).json.verify_queued, true);
  });

  it("only contributors and maintainers may lease or post verification results (REST and MCP)", async () => {
    const { env, post } = await setup();
    const prov = await registerToken(env, "q-provisional");
    await post("A1:q:T0001", { decision: "keep_baseline", soft_min_eff: 0.001 });
    const lease = await reqJson(env, "POST", "/api/verify/claim", { lane: "A1" }, bearer(prov));
    assert.equal(lease.status, 403);
    assert.equal(lease.json.error, "forbidden_tier");
    const result = await reqJson(env, "POST", `/api/verify/${encodeURIComponent("A1:q:T0001")}/result`, { lease_token: "vlease_x", passed: false }, bearer(prov));
    assert.equal(result.status, 403);
    assert.equal(result.json.error, "forbidden_tier");
    const mcp = await reqJson(env, "POST", "/api/mcp", { jsonrpc: "2.0", id: 1, method: "tools/call", params: { name: "flywheel_verify_claim", arguments: {} } }, bearer(prov));
    assert.equal(mcp.json.result.structuredContent.error, "forbidden_tier");
    const flag = await reqJson(env, "POST", `/api/nodes/${encodeURIComponent("A1:q:T0001")}/state`, { state: "disputed", reason: "x" }, bearer(prov));
    assert.equal(flag.json.error, "forbidden_tier");
    // Provisional accounts still retract their own trials.
    await reqJson(env, "POST", "/api/nodes", trial("A1:own:T0001"), bearer(prov));
    assert.equal((await reqJson(env, "POST", `/api/nodes/${encodeURIComponent("A1:own:T0001")}/state`, { state: "retracted" }, bearer(prov))).status, 200);
  });

  it("leases atomically, refuses self-verification and stale or expired leases, and sets replicated/disputed", async () => {
    const { env, sqlite, author, v1, v2, post } = await setup();
    await post("A1:q:T0001", { decision: "keep_baseline", soft_min_eff: 0.001 });
    await post("A1:q:T0002", { decision: "keep", soft_min_eff: 0.002 });

    // An author cannot lease or verify its own trial.
    const self = await reqJson(env, "POST", "/api/verify/claim", { lane: "A1" }, bearer(author));
    assert.equal(self.status, 404);
    assert.equal(self.json.error, "queue_empty");
    const selfNamed = await reqJson(env, "POST", "/api/verify/claim", { node_id: "A1:q:T0001" }, bearer(author));
    assert.equal(selfNamed.json.error, "forbidden_self_verification");
    const selfResult = await reqJson(env, "POST", `/api/verify/${encodeURIComponent("A1:q:T0001")}/result`, { lease_token: "vlease_x", passed: true }, bearer(author));
    assert.equal(selfResult.status, 403);
    assert.equal(selfResult.json.error, "forbidden_self_verification");

    // Two verifiers get two different items; a third claim finds nothing.
    const l1 = await reqJson(env, "POST", "/api/verify/claim", { lane: "A1" }, bearer(v1));
    const l2 = await reqJson(env, "POST", "/api/verify/claim", {}, bearer(v2));
    assert.equal(l1.status, 200);
    assert.equal(l2.status, 200);
    assert.notEqual(l1.json.lease.node_id, l2.json.lease.node_id);
    assert.ok(l1.json.lease.lease_token.startsWith("vlease_"));
    assert.equal(l1.json.node.has_source, true);
    assert.equal((await reqJson(env, "POST", "/api/verify/claim", {}, bearer(v2))).json.error, "queue_empty");
    // The lease token is stored hashed.
    assert.ok(!JSON.stringify(sqlite.prepare("SELECT * FROM verify_queue").all()).includes(l1.json.lease.lease_token));

    const r1 = `/api/verify/${encodeURIComponent(l1.json.lease.node_id)}/result`;
    const stale = await reqJson(env, "POST", r1, { lease_token: l2.json.lease.lease_token, passed: true }, bearer(v1));
    assert.equal(stale.status, 409);
    assert.equal(stale.json.error, "stale_lease");
    // Someone else's correct token does not work for a different holder either.
    assert.equal((await reqJson(env, "POST", r1, { lease_token: l1.json.lease.lease_token, passed: true }, bearer(v2))).status, 409);

    const pass = await reqJson(env, "POST", r1, { lease_token: l1.json.lease.lease_token, passed: true, metrics: { soft_min_eff: 0.0011 } }, bearer(v1));
    assert.equal(pass.status, 200);
    assert.equal(pass.json.node.state, "replicated");
    assert.equal(pass.json.node.verified, 0);
    // Replaying the same token after completion is refused.
    assert.equal((await reqJson(env, "POST", r1, { lease_token: l1.json.lease.lease_token, passed: false }, bearer(v1))).json.error, "not_leased");

    // Expired lease: the item becomes available again and the old token is refused.
    const r2 = `/api/verify/${encodeURIComponent(l2.json.lease.node_id)}/result`;
    sqlite.prepare("UPDATE verify_queue SET lease_expires_at = '2020-01-01T00:00:00.000Z' WHERE node_id = ?").run(l2.json.lease.node_id);
    assert.equal((await reqJson(env, "POST", r2, { lease_token: l2.json.lease.lease_token, passed: true }, bearer(v2))).json.error, "lease_expired");
    const re = await reqJson(env, "POST", "/api/verify/claim", {}, bearer(v1));
    assert.equal(re.json.lease.node_id, l2.json.lease.node_id);
    assert.equal(re.json.lease.attempts, 2);
    assert.equal((await reqJson(env, "POST", r2, { lease_token: l2.json.lease.lease_token, passed: true }, bearer(v2))).status, 409);
    const fail = await reqJson(env, "POST", r2, { lease_token: re.json.lease.lease_token, passed: false, note: "tier1 failed on seed 7" }, bearer(v1));
    assert.equal(fail.json.node.state, "disputed");
    assert.match(fail.json.node.state_reason, /tier1 failed on seed 7/);

    const done = await reqJson(env, "GET", "/api/verify/queue?status=done");
    assert.deepEqual(done.json.items.map((i) => i.result).sort(), ["fail", "pass"]);
    assert.equal(done.json.items.find((i) => i.result === "pass").result_metrics.soft_min_eff, 0.0011);
    // A needs-replication flag puts a done item back in the queue.
    const flag = await reqJson(env, "POST", `/api/nodes/${encodeURIComponent(l1.json.lease.node_id)}/state`, { state: "needs-replication", reason: "rerun on new judge" }, bearer(v2));
    assert.equal(flag.status, 200);
    assert.equal((await reqJson(env, "GET", "/api/verify/queue")).json.items[0].reason, "needs-replication");
  });
});

// ---------------------------------------------------------------------------
// Bounded reads: /api/overview, /api/runs, /api/nodes filters, node detail, bounded DAG.
// ---------------------------------------------------------------------------

const TRIAL_COLUMNS = [
  "id", "parent_id", "lane", "run_id", "trial_id", "agent_model", "author", "author_tier", "title", "hypothesis",
  "decision", "soft_min_eff", "real_stan_passed", "real_stan_total", "verified", "state", "codex_refs",
  "metrics_json", "created_at",
];

/** Inserts trials straight into SQLite (triggers keep run_stats current). */
function seedTrials(sqlite, rows) {
  const stmt = sqlite.prepare(
    `INSERT INTO nodes (${TRIAL_COLUMNS.join(", ")}) VALUES (${TRIAL_COLUMNS.map(() => "?").join(", ")})`
  );
  sqlite.exec("BEGIN");
  for (const r of rows) {
    const [lane, run_id, trial_id] = r.id.split(":");
    const row = {
      parent_id: null, lane, run_id, trial_id, agent_model: "model-x", author: `${run_id}-author`, author_tier: null,
      title: `${trial_id}: ${r.hypothesis ?? "idea"}`, hypothesis: "idea", decision: "discard_no_gain",
      soft_min_eff: null, real_stan_passed: null, real_stan_total: null, verified: 0, state: "unverified",
      codex_refs: "[]", metrics_json: '{"loc":10}', ...r,
    };
    stmt.run(...TRIAL_COLUMNS.map((c) => row[c] ?? null));
  }
  sqlite.exec("COMMIT");
}

const ts = (i) => new Date(Date.UTC(2026, 0, 1) + i * 60_000).toISOString();

describe("bounded reads", () => {
  it("/api/overview: totals, lanes with frontier, run steps, Stan pick, claims, alignment and models", async () => {
    const { env, sqlite } = makeEnv({ apiKey: "root-admin-secret" });
    seedTrials(sqlite, [
      // Lane A1, run r1: first kept scored trial at T2 (0.1), then improvements; T6 is the frontier.
      { id: "A1:r1:T1", decision: "discard_tier0", created_at: ts(1) },
      { id: "A1:r1:T2", decision: "keep_baseline", soft_min_eff: 0.1, created_at: ts(2), agent_model: "glm" },
      { id: "A1:r1:T3", decision: "discard_no_gain", soft_min_eff: 0.05, created_at: ts(3) },
      { id: "A1:r1:T4", decision: "keep", soft_min_eff: 0.2, created_at: ts(4), real_stan_passed: 3, real_stan_total: 5 },
      { id: "A1:r1:T5", decision: "archive_only", soft_min_eff: 0.9, created_at: ts(5) },
      { id: "A1:r1:T6", decision: "keep_incumbent_failed", soft_min_eff: 0.3, created_at: ts(6) },
      { id: "A1:r1:T7", decision: "keep", soft_min_eff: 0.4, created_at: ts(7), real_stan_passed: 4, real_stan_total: 5 },
      // Another A1 run, never kept; a disputed high score that must not be the frontier.
      { id: "A1:r2:T1", decision: "keep", soft_min_eff: 0.95, state: "disputed", created_at: ts(8), author: "bob", agent_model: "deepseek" },
      { id: "A1:r2:T2", decision: "discard_no_gain", soft_min_eff: 0.01, created_at: ts(9), author: "bob",
        real_stan_passed: 4, real_stan_total: 5 },
      // Lane A3 with only discards: frontier falls back to the best non-excluded trial.
      { id: "A3:z:T1", decision: "discard_tier1", soft_min_eff: 0.002, created_at: ts(10), agent_model: null },
    ]);
    await reqJson(env, "POST", "/api/claims", { task_id: "t", lane: "A1", agent: "x" }, ADMIN);

    const res = await reqJson(env, "GET", "/api/overview");
    assert.equal(res.status, 200);
    assert.equal(res.headers.get("Cache-Control"), "public, max-age=30, s-maxage=30");
    const o = res.json;
    assert.deepEqual(Object.keys(o).sort(), ["alignment", "claims", "lanes", "models", "totals"]);
    assert.deepEqual(
      { ...o.totals, latest_at: undefined },
      { trials: 10, kept: 5, runs: 3, contributors: 3, claims: 1, latest_at: undefined }
    );
    assert.equal(o.totals.latest_at, ts(10));
    assert.deepEqual(o.lanes.map((l) => l.lane), ["A1", "A3"]);

    const a1 = o.lanes[0];
    assert.deepEqual({ trials: a1.trials, kept: a1.kept, runs: a1.runs, latest_at: a1.latest_at }, { trials: 9, kept: 5, runs: 2, latest_at: ts(9) });
    const frontier = await reqJson(env, "GET", "/api/frontier");
    assert.deepEqual(a1.frontier, frontier.json.frontier.A1); // same row as /api/frontier
    assert.equal(a1.frontier.id, "A1:r1:T7");
    // Running best kept efficiency from T2 on: T5 is archive_only (not kept), T6 keep_incumbent_failed counts.
    assert.deepEqual(a1.run, { run_id: "r1", first_eff: 0.1, gain: 4, steps: [0.1, 0.1, 0.2, 0.2, 0.3, 0.4], steps_truncated: false });
    // Highest pass share wins, newest on ties (T7 and r2:T2 both 4/5; r2:T2 is newer).
    assert.deepEqual(a1.stan, { id: "A1:r2:T2", run_id: "r2", trial_id: "T2", real_stan_passed: 4, real_stan_total: 5 });

    const a3 = o.lanes[1];
    assert.equal(a3.frontier.id, "A3:z:T1");
    assert.deepEqual(a3.run, { run_id: "z", first_eff: null, gain: null, steps: [], steps_truncated: false });
    assert.equal(a3.stan, null);

    assert.equal(o.claims.length, 1);
    assert.deepEqual(o.alignment.map((p) => p.id), ["A1:r1:T4", "A1:r1:T7", "A1:r2:T2"]);
    assert.equal(o.alignment[0].real_stan_pass_rate, 0.6);
    assert.deepEqual(o.models, ["deepseek", "glm", "model-x"]);
  });

  it("/api/overview: long runs keep the change points plus the last one, at most 120 steps", async () => {
    const { env, sqlite } = makeEnv();
    const rows = [];
    for (let i = 1; i <= 500; i += 1) {
      // A new best every 10 trials: 50 change points over 500 trials.
      const kept = i % 10 === 1;
      rows.push({ id: `A2:long:T${i}`, decision: kept ? "keep" : "discard_no_gain", soft_min_eff: kept ? i / 1000 : 0.0001, created_at: ts(i) });
    }
    seedTrials(sqlite, rows);
    const run = (await reqJson(env, "GET", "/api/overview")).json.lanes[0].run;
    assert.equal(run.steps.length, 51); // 50 changes + the last point
    assert.equal(run.steps[0], 0.001);
    assert.equal(run.steps.at(-1), 0.491);
    assert.equal(run.steps.at(-2), 0.491);
    assert.equal(run.first_eff, 0.001);

    // Short runs send every point.
    const { env: env2, sqlite: sq2 } = makeEnv();
    seedTrials(sq2, rows.slice(0, 30));
    assert.equal((await reqJson(env2, "GET", "/api/overview")).json.lanes[0].run.steps.length, 30);

    // More than 120 change points: the first plus the most recent 119.
    const { env: env3, sqlite: sq3 } = makeEnv();
    seedTrials(sq3, Array.from({ length: 300 }, (_, k) => ({ id: `A2:up:T${k + 1}`, decision: "keep", soft_min_eff: (k + 1) / 1000, created_at: ts(k) })));
    const up = (await reqJson(env3, "GET", "/api/overview")).json.lanes[0].run;
    assert.equal(up.steps.length, 120);
    assert.equal(up.steps[0], 0.001);
    assert.equal(up.steps[1], 0.182);
    assert.equal(up.steps.at(-1), 0.3);
  });

  it("/api/runs: newest activity first, paging, slim nodes, truncation to the latest 150, base_eff over the whole run", async () => {
    const { env, sqlite } = makeEnv();
    const rows = [];
    for (let i = 1; i <= 200; i += 1) {
      rows.push({
        id: `A1:big:T${String(i).padStart(4, "0")}`,
        decision: i === 3 ? "keep_baseline" : i % 50 === 0 ? "keep" : "discard_no_gain",
        soft_min_eff: i === 3 ? 0.07 : 0.01,
        real_stan_total: i === 199 ? 5 : null,
        real_stan_passed: i === 199 ? 2 : null,
        created_at: ts(i),
        author: "ada",
        author_tier: "provisional",
        agent_model: "claude",
      });
    }
    for (let r = 0; r < 30; r += 1) {
      rows.push({ id: `A2:small${r}:T1`, decision: "keep", soft_min_eff: 0.2, created_at: ts(1000 + r) });
    }
    seedTrials(sqlite, rows);

    const page1 = await reqJson(env, "GET", "/api/runs?limit=24");
    assert.equal(page1.status, 200);
    assert.equal(page1.json.count, 31);
    assert.equal(page1.json.runs.length, 24);
    assert.equal(page1.json.next_offset, 24);
    assert.equal(page1.json.runs[0].run_id, "small29"); // newest activity first
    const page2 = await reqJson(env, "GET", "/api/runs?limit=24&offset=24");
    assert.equal(page2.json.runs.length, 7);
    assert.equal(page2.json.next_offset, null);
    assert.equal(page2.json.runs.at(-1).run_id, "big");

    const big = (await reqJson(env, "GET", "/api/runs?lane=A1")).json;
    assert.equal(big.count, 1);
    const run = big.runs[0];
    assert.deepEqual(
      { ...run, nodes: undefined },
      {
        lane: "A1", run_id: "big", author: "ada", author_tier: "provisional", agent_model: "claude",
        trials: 200, kept: 5, first_at: ts(1), last_at: ts(200), base_eff: 0.07, truncated: true, nodes: undefined,
      }
    );
    assert.equal(run.nodes.length, 150);
    assert.equal(run.nodes[0].id, "A1:big:T0051"); // the latest 150, in trial order
    assert.equal(run.nodes.at(-1).id, "A1:big:T0200");
    assert.deepEqual(Object.keys(run.nodes[0]).sort(), ["created_at", "decision", "id", "soft_min_eff", "stan", "state", "trial_id"]);
    assert.equal(run.nodes.find((n) => n.id === "A1:big:T0199").stan, true);
    assert.equal(run.nodes.find((n) => n.id === "A1:big:T0198").stan, false);
    const small = page1.json.runs[0];
    assert.equal(small.truncated, false);
    assert.equal(small.nodes.length, 1);
    assert.equal((await reqJson(env, "GET", "/api/runs?limit=500")).json.limit, 60);
  });

  it("/api/nodes: offset, total count, outcome/stan/q/model filters, no metrics_json", async () => {
    const { env, sqlite } = makeEnv();
    const rows = [];
    const decisions = ["keep", "keep_baseline", "keep_simpler", "keep_incumbent_failed", "archive_only", "discard_no_gain", "discard_tier0"];
    for (let i = 0; i < 140; i += 1) {
      rows.push({
        id: `A${i % 2}:run${i % 7}:T${i}`,
        decision: decisions[i % 7],
        soft_min_eff: i / 1000,
        agent_model: i % 3 === 0 ? "glm-5" : "opus",
        hypothesis: i === 77 ? "Zigzag PDMP with 100% refresh" : "plain idea",
        real_stan_total: i % 10 === 0 ? 5 : null,
        real_stan_passed: i % 10 === 0 ? 3 : null,
        created_at: ts(i),
      });
    }
    seedTrials(sqlite, rows);

    const first = await reqJson(env, "GET", "/api/nodes");
    assert.equal(first.json.count, 140);
    assert.equal(first.json.limit, 50);
    assert.equal(first.json.nodes.length, 50);
    assert.equal(first.json.next_offset, 50);
    assert.equal(first.json.nodes[0].id, "A1:run6:T139"); // newest first
    assert.ok(!("metrics_json" in first.json.nodes[0]));
    assert.deepEqual(first.json.nodes[0].metrics, { loc: 10 });
    const last = await reqJson(env, "GET", "/api/nodes?offset=100");
    assert.equal(last.json.nodes.length, 40);
    assert.equal(last.json.next_offset, null);
    assert.equal((await reqJson(env, "GET", "/api/nodes?limit=999")).json.limit, 200);

    const count = async (qs) => (await reqJson(env, "GET", `/api/nodes?${qs}`)).json.count;
    assert.equal(await count("outcome=kept"), 80);
    assert.equal(await count("outcome=archived"), 20);
    assert.equal(await count("outcome=discarded"), 40);
    assert.equal(await count("stan=1"), 14);
    assert.equal(await count("lane=A0&outcome=kept"), 40);
    assert.equal(await count("lane=A0&run_id=run0&stan=1"), 2);
    assert.equal(await count("model=glm-5"), 47);
    assert.equal(await count("model=glm-5&outcome=kept&stan=1"), 3);
    const kept = await reqJson(env, "GET", "/api/nodes?outcome=kept&limit=200");
    assert.ok(kept.json.nodes.every((n) => decisions.slice(0, 4).includes(n.decision)));

    const q = await reqJson(env, "GET", "/api/nodes?q=zigzag%20100%25");
    assert.equal(q.json.count, 1);
    assert.equal(q.json.nodes[0].id, "A1:run0:T77");
    assert.equal(await count("q=run6&lane=A0"), 10); // run id is matched too
    assert.equal((await reqJson(env, "GET", "/api/nodes?outcome=lost")).status, 400);
    assert.equal((await reqJson(env, "GET", "/api/nodes?q=%25")).status, 400);
  });

  it("/api/nodes/:id: parents and children with lane/trial/decision, and the notes that cite it", async () => {
    const { env, sqlite } = makeEnv({ apiKey: "root-admin-secret" });
    seedTrials(sqlite, [
      { id: "A1:r:T1", decision: "keep_baseline", created_at: ts(1) },
      { id: "A3:o:T9", decision: "keep", created_at: ts(2) },
    ]);
    const token = await registerToken(env, "linker");
    await reqJson(env, "POST", "/api/nodes", trial("A1:r:T2", { parent_id: "A1:r:T1", parent_ids: [{ id: "A3:o:T9", relation: "combines" }, { id: "A9:gone:T1", relation: "fork" }] }), bearer(token));
    await reqJson(env, "POST", "/api/nodes", trial("A1:r:T3", { parent_id: "A1:r:T2" }), bearer(token));
    const n1 = await reqJson(env, "POST", "/api/notes", { kind: "insight", title: "T2 combines well", node_ids: ["A1:r:T2", "A1:r:T1"] }, bearer(token));
    await reqJson(env, "POST", "/api/notes", { kind: "hypothesis", title: "unrelated", node_ids: ["A1:r:T1"] }, bearer(token));

    const d = (await reqJson(env, "GET", `/api/nodes/${encodeURIComponent("A1:r:T2")}`)).json;
    assert.deepEqual(d.parents, [
      { id: "A3:o:T9", relation: "combines", lane: "A3", trial_id: "T9", decision: "keep" },
      { id: "A9:gone:T1", relation: "fork", lane: null, trial_id: null, decision: null },
      { id: "A1:r:T1", relation: "incumbent", lane: "A1", trial_id: "T1", decision: "keep_baseline" },
    ]);
    assert.deepEqual(d.children, [{ id: "A1:r:T3", relation: "incumbent", lane: "A1", trial_id: "T3", decision: "keep" }]);
    assert.deepEqual(d.notes, [{ id: n1.json.note.id, kind: "insight", title: "T2 combines well", state: "open" }]);
    assert.equal(d.parent.id, "A1:r:T1");
    assert.equal(d.node.id, "A1:r:T2");
    // Re-linking a note moves its citation.
    await reqJson(env, "PATCH", `/api/notes/${n1.json.note.id}`, { node_ids: ["A1:r:T3"] }, bearer(token));
    assert.equal((await reqJson(env, "GET", `/api/nodes/${encodeURIComponent("A1:r:T2")}`)).json.notes.length, 0);
    assert.equal((await reqJson(env, "GET", `/api/nodes/${encodeURIComponent("A1:r:T1")}`)).json.notes.length, 1);
  });

  it("/api/dag: the newest 300 trials with their edges, plus total and truncated", async () => {
    const { env, sqlite } = makeEnv();
    seedTrials(
      sqlite,
      Array.from({ length: 400 }, (_, i) => ({
        id: `A1:chain:T${String(i).padStart(4, "0")}`,
        parent_id: i > 0 ? `A1:chain:T${String(i - 1).padStart(4, "0")}` : null,
        decision: "keep",
        soft_min_eff: i / 1e4,
        created_at: ts(i),
      }))
    );
    const dag = (await reqJson(env, "GET", "/api/dag")).json;
    assert.equal(dag.nodes.length, 300);
    assert.equal(dag.total, 400);
    assert.equal(dag.truncated, true);
    assert.equal(dag.nodes[0].id, "A1:chain:T0100"); // oldest of the newest 300 first
    assert.equal(dag.edges.length, 299); // only edges whose both ends are present
    assert.ok(!("metrics_json" in dag.nodes[0]));
    assert.equal(dag.frontier.A1.id, "A1:chain:T0399");
    const small = (await reqJson(env, "GET", "/api/dag?lane=A2")).json;
    assert.deepEqual({ n: small.nodes.length, total: small.total, truncated: small.truncated }, { n: 0, total: 0, truncated: false });
  });

  it("keeps run_stats equal to a full recount through inserts, re-posts, state changes, moves and deletes", async () => {
    const { env, sqlite } = makeEnv({ apiKey: "root-admin-secret" });
    const token = await registerToken(env, "counter");
    for (let i = 0; i < 30; i += 1) {
      const d = ["keep", "archive_only", "discard_no_gain", "keep_incumbent_failed"][i % 4];
      await reqJson(env, "POST", "/api/nodes", trial(`A${i % 3}:run${i % 4}:T${i}`, { decision: d, soft_min_eff: (i % 5) / 10 }), bearer(token));
    }
    // Idempotent re-posts, a decision change, a run move, Stan scores, deletes, a purge.
    for (let i = 0; i < 30; i += 3) {
      await reqJson(env, "POST", "/api/nodes", trial(`A${i % 3}:run${i % 4}:T${i}`, { decision: ["keep", "archive_only", "discard_no_gain", "keep_incumbent_failed"][i % 4], soft_min_eff: (i % 5) / 10 }), bearer(token));
    }
    await reqJson(env, "POST", "/api/nodes", trial("A0:run0:T0", { decision: "discard_tier0", soft_min_eff: 0 }), bearer(token));
    await reqJson(env, "POST", "/api/nodes", { ...trial("A1:run1:T1"), run_id: "moved" }, bearer(token));
    await reqJson(env, "POST", `/api/admin/nodes/${encodeURIComponent("A2:run2:T2")}/verify`, { real_stan_passed: 2, real_stan_total: 4 }, ADMIN);
    await reqJson(env, "DELETE", `/api/admin/nodes/${encodeURIComponent("A0:run3:T3")}`, undefined, ADMIN);
    const recount = () =>
      sqlite
        .prepare(
          `SELECT lane, run_id, COUNT(*) AS trials,
             SUM(decision IN ('keep','keep_baseline','keep_simpler','keep_incumbent_failed')) AS kept,
             SUM(decision = 'archive_only') AS archived, SUM(coalesce(real_stan_total, 0) > 0) AS stan,
             MIN(created_at) AS first_at, MAX(created_at) AS last_at
           FROM nodes GROUP BY lane, run_id ORDER BY lane, run_id`
        )
        .all()
        .map((r) => ({ ...r }));
    const stats = () =>
      sqlite
        .prepare("SELECT lane, run_id, trials, kept, archived, stan, first_at, last_at FROM run_stats ORDER BY lane, run_id")
        .all()
        .map((r) => ({ ...r }));
    assert.deepEqual(stats(), recount());
    assert.equal((await reqJson(env, "GET", "/api/health")).json.nodes, 29);
    await reqJson(env, "POST", "/api/admin/handles/counter/purge", undefined, ADMIN);
    assert.deepEqual(stats(), []);
    assert.deepEqual((await reqJson(env, "GET", "/api/overview")).json.lanes, []);
  });
});

// ---------------------------------------------------------------------------
// Review fixes: one focused test per finding.
// ---------------------------------------------------------------------------

describe("review fixes", () => {
  it("freezes a reviewed or leased trial: re-posts that change what was scored get 409, identical ones succeed", async () => {
    const { env, sqlite } = makeEnv({ apiKey: "root-admin-secret" });
    env.FLYWHEEL_INVITE_CODE = "inv";
    const author = await registerToken(env, "freeze-author");
    const verifier = (await register(env, "freeze-verifier", { invite_code: "inv" }, "192.0.2.90")).json.token;
    const base = trial("A1:f:T0001", { soft_min_eff: 0.002, source: "print('v1')\n", metrics_json: '{"a": 1, "b": 2}', dims: [6, 20] });
    const post = (extra = {}) => reqJson(env, "POST", "/api/nodes", { ...base, ...extra }, bearer(author));
    assert.equal((await post()).status, 201);

    // Unverified and unleased: the author may still change it.
    assert.equal((await post({ soft_min_eff: 0.003 })).status, 200);
    assert.equal((await post({ soft_min_eff: 0.002 })).status, 200);

    // Leased: changes are refused, identical re-posts (even with reordered JSON) are fine.
    const lease = await reqJson(env, "POST", "/api/verify/claim", { node_id: "A1:f:T0001" }, bearer(verifier));
    assert.equal(lease.status, 200);
    const leased = await post({ soft_min_eff: 0.9 });
    assert.equal(leased.status, 409);
    assert.equal(leased.json.error, "trial_locked");
    assert.equal(leased.json.leased, true);
    assert.deepEqual(leased.json.fields, ["soft_min_eff"]);
    assert.equal((await post({ metrics_json: '{"b":2,"a":1}', hypothesis: "reworded" })).status, 200);

    // Replicated: source, lane and decision are frozen too; Stan results stay.
    await reqJson(env, "POST", `/api/verify/${encodeURIComponent("A1:f:T0001")}/result`, { lease_token: lease.json.lease.lease_token, passed: true }, bearer(verifier));
    sqlite.prepare("UPDATE nodes SET real_stan_passed = 4, real_stan_total = 5 WHERE id = 'A1:f:T0001'").run();
    for (const change of [{ source: "print('v2')\n" }, { lane: "A2" }, { decision: "archive_only" }, { b2_max: 0.5 }]) {
      const res = await post(change);
      assert.equal(res.status, 409, JSON.stringify(change));
      assert.equal(res.json.state, "replicated");
    }
    const same = await post();
    assert.equal(same.status, 200);
    assert.equal(same.json.node.state, "replicated");
    assert.equal(same.json.node.soft_min_eff, 0.002);
    assert.equal(same.json.node.real_stan_passed, 4);
    assert.equal(sqlite.prepare("SELECT COUNT(*) AS c FROM sources").get().c, 1); // the refused source was not stored
  });

  it("fails closed: no FLYWHEEL_API_KEY means no maintainer, unless FLYWHEEL_DEV_OPEN_ADMIN opts in", async () => {
    for (const blank of [undefined, "", "   "]) {
      const { env } = makeEnv({ closed: true });
      env.FLYWHEEL_API_KEY = blank;
      assert.equal((await reqJson(env, "GET", "/api/admin/keys")).status, 401);
      assert.equal((await reqJson(env, "POST", "/api/admin/keys", { handle: "evil", role: "admin" })).status, 401);
      assert.equal((await reqJson(env, "POST", "/api/admin/handles/someone/purge")).status, 401);
      assert.equal((await reqJson(env, "POST", "/api/nodes", trial("A1:x:T0001"))).status, 401);
      // Registered tokens still work.
      const token = await registerToken(env, "closed-user");
      assert.equal((await reqJson(env, "POST", "/api/nodes", trial("A1:x:T0001"), bearer(token))).status, 201);
      assert.equal((await reqJson(env, "GET", "/api/admin/keys", undefined, bearer(token))).status, 403);
    }
    const { env } = makeEnv({ closed: true });
    env.FLYWHEEL_DEV_OPEN_ADMIN = "1";
    assert.equal((await reqJson(env, "GET", "/api/admin/keys")).status, 200);
  });

  it("uses one parse of the query for dispatch and the cache key", async () => {
    const { env, sqlite } = makeEnv();
    seedTrials(sqlite, [
      { id: "A1:r:T1", decision: "keep", soft_min_eff: 0.1, created_at: ts(1) },
      { id: "A2:r:T1", decision: "keep", soft_min_eff: 0.2, created_at: ts(2) },
    ]);
    const store = new Map();
    env.CACHE = {
      async match(req) {
        const hit = store.get(typeof req === "string" ? req : req.url);
        return hit ? hit.clone() : undefined;
      },
      async put(req, resp) {
        store.set(typeof req === "string" ? req : req.url, resp.clone());
      },
    };
    const repeated = await reqJson(env, "GET", "/api/nodes?lane=A1&lane=A2");
    assert.deepEqual(repeated.json.nodes.map((n) => n.lane), ["A2"]); // last occurrence wins
    assert.deepEqual((await reqJson(env, "GET", "/api/nodes?lane=A1")).json.nodes.map((n) => n.lane), ["A1"]);
    assert.deepEqual((await reqJson(env, "GET", "/api/nodes?lane=A2")).json.nodes.map((n) => n.lane), ["A2"]);
    // /api/frontier keeps `lane` in its key.
    assert.deepEqual(Object.keys((await reqJson(env, "GET", "/api/frontier?lane=A1")).json.frontier), ["A1"]);
    assert.deepEqual(Object.keys((await reqJson(env, "GET", "/api/frontier")).json.frontier), ["A1", "A2"]);
  });

  it("returns next_offset null at the 10,000 offset cap (trials, runs, notes)", async () => {
    const { env, sqlite } = makeEnv();
    seedTrials(sqlite, Array.from({ length: 10_100 }, (_, i) => ({ id: `A1:r${i}:T1`, created_at: ts(i) })));
    const ins = sqlite.prepare("INSERT INTO notes (id, kind, title, state, author, node_ids, created_at, updated_at) VALUES (?, 'insight', 't', 'open', 'a', '[]', ?, ?)");
    sqlite.exec("BEGIN");
    for (let i = 0; i < 10_100; i += 1) ins.run(`note_${i}`, ts(i), ts(i));
    sqlite.exec("COMMIT");
    for (const path of ["/api/nodes", "/api/runs", "/api/notes"]) {
      const before = (await reqJson(env, "GET", `${path}?limit=50&offset=9900`)).json;
      assert.equal(before.next_offset, 9950, path);
      const atCap = (await reqJson(env, "GET", `${path}?limit=50&offset=9960`)).json;
      assert.equal(atCap.next_offset, null, path);
      assert.equal(atCap.offset, 9960, path);
    }
  });

  it("takes task claims in one statement and renews only the caller's own claims", async () => {
    const { env, sqlite } = makeEnv({ apiKey: "root-admin-secret" });
    const a = await registerToken(env, "claim-a");
    const b = await registerToken(env, "claim-b");
    // Two agents racing for one task: exactly one gets it.
    const race = await Promise.all([
      reqJson(env, "POST", "/api/claims", { task_id: "race", lane: "A1", agent: "x" }, bearer(a)),
      reqJson(env, "POST", "/api/claims", { task_id: "race", lane: "A1", agent: "y" }, bearer(b)),
    ]);
    assert.deepEqual(race.map((r) => r.status).sort(), [201, 409]);
    assert.equal(sqlite.prepare("SELECT COUNT(*) AS c FROM claims WHERE task_id = 'race' AND status = 'active'").get().c, 1);

    // Two active claims (left by the old race): b's forced renewal must not abandon a's.
    const later = new Date(Date.now() + 3_600_000).toISOString();
    sqlite.prepare("INSERT INTO claims (id, task_id, lane, agent, status, expires_at, created_at) VALUES ('ca', 'dup', 'A1', 'claim-a:x', 'active', ?, '2026-01-01T00:00:00Z')").run(later);
    sqlite.prepare("INSERT INTO claims (id, task_id, lane, agent, status, expires_at, created_at) VALUES ('cb', 'dup', 'A1', 'claim-b:y', 'active', ?, '2026-01-02T00:00:00Z')").run(later);
    const forced = await reqJson(env, "POST", "/api/claims", { task_id: "dup", lane: "A1", agent: "y", force: true }, bearer(b));
    assert.equal(forced.status, 403);
    assert.equal(sqlite.prepare("SELECT status FROM claims WHERE id = 'ca'").get().status, "active");
    // A handle that is a prefix of another does not own its claims.
    const c = await registerToken(env, "claim");
    assert.equal((await reqJson(env, "PATCH", "/api/claims/ca", { status: "done" }, bearer(c))).status, 403);
    assert.equal((await reqJson(env, "PATCH", "/api/claims/ca", { status: "done" }, bearer(a))).status, 200);
  });

  it("keeps run_stats.author on the earliest surviving trial, and lane and contributor counts exact", async () => {
    const { env, sqlite } = makeEnv({ apiKey: "root-admin-secret" });
    const first = await registerToken(env, "first-author");
    const second = await registerToken(env, "second-author");
    await reqJson(env, "POST", "/api/nodes", trial("A1:shared:T1", { created_at: ts(1) }), bearer(first));
    await reqJson(env, "POST", "/api/nodes", trial("A1:shared:T2", { created_at: ts(2) }), bearer(second));
    const runAuthor = () => sqlite.prepare("SELECT author, author_tier FROM run_stats WHERE run_id = 'shared'").get();
    assert.equal(runAuthor().author, "first-author");
    await reqJson(env, "POST", "/api/admin/handles/first-author/purge", undefined, ADMIN);
    assert.deepEqual({ ...runAuthor() }, { author: "second-author", author_tier: "provisional" });
    await reqJson(env, "POST", "/api/admin/handles/second-author/promote", undefined, ADMIN);
    assert.equal(runAuthor().author_tier, "contributor");

    const lanes = () => sqlite.prepare("SELECT lane, trials, kept, archived, stan, runs FROM lane_stats ORDER BY lane").all().map((r) => ({ ...r }));
    const recount = () =>
      sqlite
        .prepare(
          `SELECT lane, SUM(trials) AS trials, SUM(kept) AS kept, SUM(archived) AS archived, SUM(stan) AS stan, COUNT(*) AS runs
           FROM run_stats GROUP BY lane ORDER BY lane`
        )
        .all()
        .map((r) => ({ ...r }));
    assert.deepEqual(lanes(), recount());
    assert.equal((await reqJson(env, "GET", "/api/overview")).json.totals.contributors, 1);
  });

  it("bounds D1 reads: capped filtered counts and a windowed q= and search, both labelled", async () => {
    const { env, sqlite } = makeEnv();
    seedTrials(sqlite, Array.from({ length: 2600 }, (_, i) => ({
      id: `A1:r${i % 40}:T${i}`, decision: "discard_tier0", hypothesis: i < 100 ? "ancient zigzag" : "plain", created_at: ts(i),
    })));
    const counted = (await reqJson(env, "GET", "/api/nodes?decision=discard_tier0")).json;
    assert.deepEqual({ count: counted.count, capped: counted.count_capped, next: counted.next_offset }, { count: 1000, capped: true, next: 50 });
    const exact = (await reqJson(env, "GET", "/api/nodes?lane=A1")).json;
    assert.deepEqual({ count: exact.count, capped: exact.count_capped }, { count: 2600, capped: false });
    // The 100 oldest trials are outside the 2,000-trial window, so q= and search do not see them.
    const q = (await reqJson(env, "GET", "/api/nodes?q=zigzag")).json;
    assert.deepEqual({ count: q.count, window: q.search_window, truncated: q.window_truncated }, { count: 0, window: 2000, truncated: true });
    const search = (await reqJson(env, "GET", "/api/search?q=zigzag")).json;
    assert.deepEqual({ count: search.count, window: search.search_window, truncated: search.window_truncated }, { count: 0, window: 2000, truncated: true });
    assert.equal((await reqJson(env, "GET", "/api/search?q=plain&kind=node")).json.count, 20);
  });

  it("rejects unsafe identifiers on write", async () => {
    const { env } = makeEnv({ apiKey: "root-admin-secret" });
    const token = await registerToken(env, "safe-ids");
    // Real run ids use underscores (glm53_a1_dual_1h); the rule must not shut them out.
    const ok = await reqJson(env, "POST", "/api/nodes", { ...trial("A1:glm53_a1_dual_1h:T0001"), run_id: "glm53_a1_dual_1h" }, bearer(token));
    assert.equal(ok.status, 201, JSON.stringify(ok.body));
    const bad = [
      { id: "A1:r:T1;rm -rf ~" },
      { id: "../../etc/passwd" },
      { run_id: "run$(whoami)" },
      { trial_id: "T 1" },
      { lane: "A1-x" },
      { lane: "LONGLANE9" },
      { parent_ids: [{ id: "A1:r:`id`", relation: "combines" }] },
    ];
    for (const extra of bad) {
      const res = await reqJson(env, "POST", "/api/nodes", { ...trial("A1:r:T1"), ...extra }, bearer(token));
      assert.equal(res.status, 400, JSON.stringify(extra));
      assert.equal(res.json.error, "invalid_identifier");
    }
    assert.equal((await reqJson(env, "POST", "/api/nodes", trial("A1:r_1.x-y:T0001"), bearer(token))).status, 201);
    assert.equal((await reqJson(env, "POST", "/api/claims", { task_id: "t/../x", lane: "A1", agent: "a" }, bearer(token))).status, 400);
    assert.equal((await reqJson(env, "POST", "/api/claims", { task_id: "t", lane: "A 1", agent: "a" }, bearer(token))).status, 400);
    assert.equal((await reqJson(env, "POST", "/api/notes", { kind: "insight", title: "t", node_ids: ["a b"] }, bearer(token))).status, 400);
    assert.equal((await reqJson(env, "POST", "/api/notes", { kind: "insight", title: "t", lane: "A_1" }, bearer(token))).status, 400);
    assert.equal((await register(env, "bad:handle")).status, 400);
    assert.equal((await register(env, "bad/handle")).status, 400);
  });
});
