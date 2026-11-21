import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import { handleRequest } from "../src/index.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCHEMA_SQL = fs.readFileSync(path.resolve(__dirname, "../schema.sql"), "utf8");

/**
 * Minimal in-memory Cloudflare D1Database shim backed by node:sqlite.
 */
function createD1Shim() {
  const sqlite = new DatabaseSync(":memory:");
  sqlite.exec(SCHEMA_SQL);

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
    },
  };
}

function makeEnv(options = {}) {
  const { sqlite, db } = createD1Shim();
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
      "flywheel_claim_task",
      "flywheel_frontier",
      "flywheel_list_nodes",
      "flywheel_post_node",
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
});
