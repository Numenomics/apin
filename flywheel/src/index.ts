export interface D1Result<T = Record<string, unknown>> {
  results: T[];
  success: boolean;
  meta?: Record<string, unknown>;
}

export interface D1PreparedStatement {
  bind(...values: unknown[]): D1PreparedStatement;
  first<T = Record<string, unknown>>(colName?: string): Promise<T | null>;
  all<T = Record<string, unknown>>(): Promise<D1Result<T>>;
  run(): Promise<D1Result>;
}

export interface D1DatabaseLike {
  prepare(query: string): D1PreparedStatement;
}

export interface AssetsBindingLike {
  fetch(request: Request): Promise<Response>;
}

export interface Env {
  DB: D1DatabaseLike;
  ASSETS?: AssetsBindingLike;
  FLYWHEEL_API_KEY?: string;
}

export interface NodeRow {
  id: string;
  parent_id: string | null;
  lane: string;
  run_id: string;
  trial_id: string | null;
  agent_model: string | null;
  author: string | null;
  title: string | null;
  hypothesis: string | null;
  decision: string;
  soft_min_eff: number | null;
  ratio_vs_incumbent: number | null;
  short_horizon_eff: number | null;
  long_horizon_eff: number | null;
  real_stan_passed: number | null;
  real_stan_total: number | null;
  real_ess_per_cost: number | null;
  overhead_ratio: number | null;
  b2_max: number | null;
  dims: string | null;
  codex_refs: string | null;
  code_sha: string | null;
  metrics_json: string | null;
  verified: number;
  created_at: string;
}

export interface NodeRecord extends Omit<NodeRow, "dims" | "codex_refs" | "verified"> {
  dims: number[] | null;
  codex_refs: string[];
  verified: number;
  metrics?: Record<string, unknown> | null;
}

export interface ClaimRow {
  id: string;
  task_id: string;
  lane: string;
  agent: string;
  summary: string | null;
  status: string;
  expires_at: string;
  created_at: string;
}

export interface DagEdge {
  source: string;
  target: string;
  lane: string;
  decision: string;
}

export interface AlignmentPoint {
  id: string;
  lane: string;
  run_id: string;
  trial_id: string | null;
  agent_model: string | null;
  title: string | null;
  decision: string;
  soft_min_eff: number;
  ratio_vs_incumbent: number | null;
  real_stan_passed: number;
  real_stan_total: number;
  real_stan_pass_rate: number;
  real_ess_per_cost: number | null;
  overhead_ratio: number | null;
  verified: number;
}

const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PATCH, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

const KEPT_DECISIONS = new Set(["keep", "keep_baseline", "keep_simpler"]);
const VALID_CLAIM_PATCH_STATUSES = new Set(["done", "abandoned"]);

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...CORS_HEADERS,
    },
  });
}

function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i += 1) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}

function isAuthorized(request: Request, env: Env): boolean {
  const configuredKey = env.FLYWHEEL_API_KEY?.trim();
  if (!configuredKey) return true;
  const authHeader = request.headers.get("Authorization") ?? "";
  if (!authHeader.startsWith("Bearer ")) return false;
  const token = authHeader.slice(7).trim();
  return token.length > 0 && timingSafeEqual(token, configuredKey);
}

function parseJsonSafe<T>(raw: string | null | undefined, fallback: T): T {
  if (raw == null || raw === "") return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function toFiniteNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function toFiniteInt(value: unknown): number | null {
  const num = toFiniteNumber(value);
  return num === null ? null : Math.trunc(num);
}

function toNullableString(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  const s = String(value).trim();
  return s.length > 0 ? s : null;
}

function serializeJsonField(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }
  return JSON.stringify(value);
}

function deserializeNode(row: NodeRow): NodeRecord {
  const parsedDims = parseJsonSafe<unknown>(row.dims, null);
  const dims = Array.isArray(parsedDims)
    ? parsedDims.map((x) => Number(x)).filter((x) => Number.isFinite(x))
    : null;

  const parsedRefs = parseJsonSafe<unknown>(row.codex_refs, []);
  const codex_refs = Array.isArray(parsedRefs)
    ? parsedRefs.map((x) => String(x)).filter((x) => x.length > 0)
    : [];

  const parsedMetrics = parseJsonSafe<Record<string, unknown> | null>(row.metrics_json, null);

  return {
    id: row.id,
    parent_id: row.parent_id ?? null,
    lane: row.lane,
    run_id: row.run_id,
    trial_id: row.trial_id ?? null,
    agent_model: row.agent_model ?? null,
    author: row.author ?? null,
    title: row.title ?? null,
    hypothesis: row.hypothesis ?? null,
    decision: row.decision,
    soft_min_eff: toFiniteNumber(row.soft_min_eff),
    ratio_vs_incumbent: toFiniteNumber(row.ratio_vs_incumbent),
    short_horizon_eff: toFiniteNumber(row.short_horizon_eff),
    long_horizon_eff: toFiniteNumber(row.long_horizon_eff),
    real_stan_passed: toFiniteInt(row.real_stan_passed),
    real_stan_total: toFiniteInt(row.real_stan_total),
    real_ess_per_cost: toFiniteNumber(row.real_ess_per_cost),
    overhead_ratio: toFiniteNumber(row.overhead_ratio),
    b2_max: toFiniteNumber(row.b2_max),
    dims,
    codex_refs,
    code_sha: row.code_sha ?? null,
    metrics_json: row.metrics_json ?? null,
    metrics: parsedMetrics,
    verified: row.verified ? 1 : 0,
    created_at: row.created_at,
  };
}

function isKeptOrVerified(node: NodeRecord): boolean {
  return node.verified === 1 || KEPT_DECISIONS.has(node.decision);
}

function compareFrontierCandidates(a: NodeRecord, b: NodeRecord): number {
  const aEff = a.soft_min_eff ?? -Infinity;
  const bEff = b.soft_min_eff ?? -Infinity;
  if (aEff !== bEff) return bEff - aEff;

  if (a.verified !== b.verified) return b.verified - a.verified;

  const aStan = a.real_stan_passed ?? -1;
  const bStan = b.real_stan_passed ?? -1;
  if (aStan !== bStan) return bStan - aStan;

  if (a.created_at !== b.created_at) return b.created_at.localeCompare(a.created_at);
  return b.id.localeCompare(a.id);
}

function computeFrontier(nodes: NodeRecord[]): Record<string, NodeRecord> {
  const byLane = new Map<string, NodeRecord[]>();
  for (const node of nodes) {
    const list = byLane.get(node.lane) ?? [];
    list.push(node);
    byLane.set(node.lane, list);
  }

  const frontier: Record<string, NodeRecord> = {};
  for (const lane of [...byLane.keys()].sort()) {
    const laneNodes = byLane.get(lane) ?? [];
    const eligible = laneNodes.filter(isKeptOrVerified);
    const pool = eligible.length > 0 ? eligible : laneNodes;
    if (pool.length > 0) {
      pool.sort(compareFrontierCandidates);
      frontier[lane] = pool[0];
    }
  }
  return frontier;
}

function computeAlignment(nodes: NodeRecord[]): AlignmentPoint[] {
  const points: AlignmentPoint[] = [];
  for (const node of nodes) {
    if (
      node.soft_min_eff !== null &&
      node.real_stan_passed !== null &&
      node.real_stan_total !== null &&
      node.real_stan_total > 0
    ) {
      points.push({
        id: node.id,
        lane: node.lane,
        run_id: node.run_id,
        trial_id: node.trial_id,
        agent_model: node.agent_model,
        title: node.title,
        decision: node.decision,
        soft_min_eff: node.soft_min_eff,
        ratio_vs_incumbent: node.ratio_vs_incumbent,
        real_stan_passed: node.real_stan_passed,
        real_stan_total: node.real_stan_total,
        real_stan_pass_rate: node.real_stan_passed / node.real_stan_total,
        real_ess_per_cost: node.real_ess_per_cost,
        overhead_ratio: node.overhead_ratio,
        verified: node.verified,
      });
    }
  }
  return points;
}

async function queryAllNodes(db: D1DatabaseLike): Promise<NodeRecord[]> {
  const res = await db
    .prepare("SELECT * FROM nodes ORDER BY created_at ASC, id ASC")
    .all<NodeRow>();
  return (res.results ?? []).map(deserializeNode);
}

async function queryActiveClaims(db: D1DatabaseLike, lane?: string | null): Promise<ClaimRow[]> {
  const nowIso = new Date().toISOString();
  if (lane) {
    const res = await db
      .prepare(
        "SELECT * FROM claims WHERE status = 'active' AND expires_at > ? AND lane = ? ORDER BY created_at DESC"
      )
      .bind(nowIso, lane)
      .all<ClaimRow>();
    return res.results ?? [];
  }
  const res = await db
    .prepare(
      "SELECT * FROM claims WHERE status = 'active' AND expires_at > ? ORDER BY created_at DESC"
    )
    .bind(nowIso)
    .all<ClaimRow>();
  return res.results ?? [];
}

async function queryFilteredNodes(
  db: D1DatabaseLike,
  filters: {
    lane?: string | null;
    run_id?: string | null;
    decision?: string | null;
    verified?: string | boolean | number | null;
    limit?: number | null;
  }
): Promise<NodeRecord[]> {
  const clauses: string[] = [];
  const params: unknown[] = [];

  if (filters.lane) {
    clauses.push("lane = ?");
    params.push(filters.lane);
  }
  if (filters.run_id) {
    clauses.push("run_id = ?");
    params.push(filters.run_id);
  }
  if (filters.decision) {
    clauses.push("decision = ?");
    params.push(filters.decision);
  }
  if (filters.verified !== undefined && filters.verified !== null && filters.verified !== "") {
    const vStr = String(filters.verified).toLowerCase();
    const vInt = vStr === "1" || vStr === "true" ? 1 : 0;
    clauses.push("verified = ?");
    params.push(vInt);
  }

  let limit = 200;
  if (filters.limit !== undefined && filters.limit !== null) {
    const parsed = Math.trunc(Number(filters.limit));
    if (Number.isFinite(parsed) && parsed > 0) {
      limit = Math.min(parsed, 1000);
    }
  }

  const where = clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "";
  const sql = `SELECT * FROM nodes ${where} ORDER BY created_at DESC, id DESC LIMIT ?`;
  params.push(limit);

  const res = await db.prepare(sql).bind(...params).all<NodeRow>();
  return (res.results ?? []).map(deserializeNode);
}

async function insertNode(
  db: D1DatabaseLike,
  payload: Record<string, unknown>
): Promise<{ status: number; body: Record<string, unknown> }> {
  const lane = toNullableString(payload.lane);
  const run_id = toNullableString(payload.run_id);
  const decision = toNullableString(payload.decision);

  if (!lane || !run_id || !decision) {
    return {
      status: 400,
      body: { error: "missing_required_fields", required: ["lane", "run_id", "decision"] },
    };
  }

  const id = toNullableString(payload.id) ?? crypto.randomUUID();
  const created_at = toNullableString(payload.created_at) ?? new Date().toISOString();
  const verifiedRaw = payload.verified;
  const verified =
    verifiedRaw === true || verifiedRaw === 1 || verifiedRaw === "1" || verifiedRaw === "true"
      ? 1
      : 0;

  const metricsJson =
    payload.metrics_json !== undefined
      ? serializeJsonField(payload.metrics_json)
      : payload.metrics !== undefined
        ? serializeJsonField(payload.metrics)
        : null;

  await db
    .prepare(
      `INSERT OR REPLACE INTO nodes (
        id, parent_id, lane, run_id, trial_id, agent_model, author, title, hypothesis,
        decision, soft_min_eff, ratio_vs_incumbent, short_horizon_eff, long_horizon_eff,
        real_stan_passed, real_stan_total, real_ess_per_cost, overhead_ratio, b2_max,
        dims, codex_refs, code_sha, metrics_json, verified, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .bind(
      id,
      toNullableString(payload.parent_id),
      lane,
      run_id,
      toNullableString(payload.trial_id),
      toNullableString(payload.agent_model),
      toNullableString(payload.author),
      toNullableString(payload.title),
      toNullableString(payload.hypothesis),
      decision,
      toFiniteNumber(payload.soft_min_eff),
      toFiniteNumber(payload.ratio_vs_incumbent),
      toFiniteNumber(payload.short_horizon_eff),
      toFiniteNumber(payload.long_horizon_eff),
      toFiniteInt(payload.real_stan_passed),
      toFiniteInt(payload.real_stan_total),
      toFiniteNumber(payload.real_ess_per_cost),
      toFiniteNumber(payload.overhead_ratio),
      toFiniteNumber(payload.b2_max),
      serializeJsonField(payload.dims),
      serializeJsonField(payload.codex_refs ?? []),
      toNullableString(payload.code_sha),
      metricsJson,
      verified,
      created_at
    )
    .run();

  const row = await db.prepare("SELECT * FROM nodes WHERE id = ?").bind(id).first<NodeRow>();
  return {
    status: 201,
    body: { ok: true, node: row ? deserializeNode(row) : null },
  };
}

async function createClaim(
  db: D1DatabaseLike,
  payload: Record<string, unknown>
): Promise<{ status: number; body: Record<string, unknown> }> {
  const task_id = toNullableString(payload.task_id);
  const lane = toNullableString(payload.lane);
  const agent = toNullableString(payload.agent);

  if (!task_id || !lane || !agent) {
    return {
      status: 400,
      body: { error: "missing_required_fields", required: ["task_id", "lane", "agent"] },
    };
  }

  const nowMs = Date.now();
  const nowIso = new Date(nowMs).toISOString();
  const ttlRaw = payload.ttl_minutes === undefined ? 60 : Number(payload.ttl_minutes);
  if (!Number.isFinite(ttlRaw) || ttlRaw <= 0) {
    return { status: 400, body: { error: "invalid_ttl_minutes" } };
  }
  const ttlMinutes = Math.min(ttlRaw, 1440);
  const force = payload.force === true;

  const existing = await db
    .prepare(
      "SELECT * FROM claims WHERE task_id = ? AND status = 'active' AND expires_at > ? ORDER BY created_at DESC LIMIT 1"
    )
    .bind(task_id, nowIso)
    .first<ClaimRow>();

  if (existing && !force) {
    return {
      status: 409,
      body: { error: "already_claimed", existing_claim: existing },
    };
  }

  if (existing && force) {
    await db
      .prepare(
        "UPDATE claims SET status = 'abandoned' WHERE task_id = ? AND status = 'active' AND expires_at > ?"
      )
      .bind(task_id, nowIso)
      .run();
  }

  const id = toNullableString(payload.id) ?? crypto.randomUUID();
  const expiresAt = new Date(nowMs + ttlMinutes * 60_000).toISOString();
  const summary = toNullableString(payload.summary);

  await db
    .prepare(
      `INSERT INTO claims (id, task_id, lane, agent, summary, status, expires_at, created_at)
       VALUES (?, ?, ?, ?, ?, 'active', ?, ?)`
    )
    .bind(id, task_id, lane, agent, summary, expiresAt, nowIso)
    .run();

  const claim = await db.prepare("SELECT * FROM claims WHERE id = ?").bind(id).first<ClaimRow>();
  return { status: 201, body: { ok: true, claim } };
}

async function patchClaim(
  db: D1DatabaseLike,
  claimId: string,
  payload: Record<string, unknown>
): Promise<{ status: number; body: Record<string, unknown> }> {
  const statusVal = toNullableString(payload.status);
  if (!statusVal || !VALID_CLAIM_PATCH_STATUSES.has(statusVal)) {
    return {
      status: 400,
      body: { error: "invalid_status", allowed: ["done", "abandoned"] },
    };
  }

  const existing = await db.prepare("SELECT * FROM claims WHERE id = ?").bind(claimId).first<ClaimRow>();
  if (!existing) {
    return { status: 404, body: { error: "not_found" } };
  }

  const summary =
    payload.summary !== undefined ? toNullableString(payload.summary) : existing.summary;

  await db
    .prepare("UPDATE claims SET status = ?, summary = ? WHERE id = ?")
    .bind(statusVal, summary, claimId)
    .run();

  const updated = await db.prepare("SELECT * FROM claims WHERE id = ?").bind(claimId).first<ClaimRow>();
  return { status: 200, body: { ok: true, claim: updated } };
}

const MCP_TOOLS = [
  {
    name: "flywheel_frontier",
    description: "Return the current best kept/verified apin research frontier node per lane.",
    inputSchema: {
      type: "object",
      properties: {
        lane: { type: "string", description: "Optional lane filter (e.g. A0, A1, A2, A3)." },
      },
      additionalProperties: false,
    },
  },
  {
    name: "flywheel_list_nodes",
    description: "List apin research DAG nodes with optional lane, run_id, decision, verified, and limit filters.",
    inputSchema: {
      type: "object",
      properties: {
        lane: { type: "string" },
        run_id: { type: "string" },
        decision: { type: "string" },
        verified: { type: "boolean" },
        limit: { type: "integer", minimum: 1, maximum: 1000 },
      },
      additionalProperties: false,
    },
  },
  {
    name: "flywheel_claim_task",
    description: "Claim an apin research task with a TTL lease to prevent duplicate agent work.",
    inputSchema: {
      type: "object",
      required: ["task_id", "lane", "agent"],
      properties: {
        task_id: { type: "string" },
        lane: { type: "string" },
        agent: { type: "string" },
        summary: { type: "string" },
        ttl_minutes: { type: "number", default: 60 },
        force: { type: "boolean", default: false },
      },
      additionalProperties: false,
    },
  },
  {
    name: "flywheel_post_node",
    description: "Record a new apin autoresearch trial node into the Flywheel DAG.",
    inputSchema: {
      type: "object",
      required: ["lane", "run_id", "decision"],
      properties: {
        id: { type: "string" },
        parent_id: { type: "string" },
        lane: { type: "string" },
        run_id: { type: "string" },
        trial_id: { type: "string" },
        agent_model: { type: "string" },
        author: { type: "string" },
        title: { type: "string" },
        hypothesis: { type: "string" },
        decision: { type: "string" },
        soft_min_eff: { type: "number" },
        ratio_vs_incumbent: { type: "number" },
        short_horizon_eff: { type: "number" },
        long_horizon_eff: { type: "number" },
        real_stan_passed: { type: "integer" },
        real_stan_total: { type: "integer" },
        real_ess_per_cost: { type: "number" },
        overhead_ratio: { type: "number" },
        b2_max: { type: "number" },
        dims: { type: "array", items: { type: "integer" } },
        codex_refs: { type: "array", items: { type: "string" } },
        code_sha: { type: "string" },
        metrics_json: { type: "string" },
        verified: { type: "boolean" },
      },
      additionalProperties: true,
    },
  },
];

const MCP_WRITE_TOOLS = new Set(["flywheel_claim_task", "flywheel_post_node"]);

async function handleMcp(request: Request, env: Env): Promise<Response> {
  let body: Record<string, unknown>;
  try {
    body = (await request.json()) as Record<string, unknown>;
  } catch {
    return jsonResponse(
      { jsonrpc: "2.0", id: null, error: { code: -32700, message: "Parse error" } },
      400
    );
  }

  const rpcId = body.id ?? null;
  const method = typeof body.method === "string" ? body.method : "";
  const params = (body.params && typeof body.params === "object"
    ? body.params
    : {}) as Record<string, unknown>;

  if (body.jsonrpc !== "2.0" || !method) {
    return jsonResponse(
      { jsonrpc: "2.0", id: rpcId, error: { code: -32600, message: "Invalid Request" } },
      400
    );
  }

  if (method === "initialize") {
    return jsonResponse({
      jsonrpc: "2.0",
      id: rpcId,
      result: {
        protocolVersion: "2024-11-05",
        capabilities: { tools: {} },
        serverInfo: { name: "apin-flywheel", version: "0.1.0" },
      },
    });
  }

  if (method === "notifications/initialized" || method === "ping") {
    return jsonResponse({ jsonrpc: "2.0", id: rpcId, result: {} });
  }

  if (method === "tools/list") {
    return jsonResponse({
      jsonrpc: "2.0",
      id: rpcId,
      result: { tools: MCP_TOOLS },
    });
  }

  if (method === "tools/call") {
    const toolName = typeof params.name === "string" ? params.name : "";
    const args = (params.arguments && typeof params.arguments === "object"
      ? params.arguments
      : {}) as Record<string, unknown>;

    if (!MCP_TOOLS.some((t) => t.name === toolName)) {
      return jsonResponse(
        {
          jsonrpc: "2.0",
          id: rpcId,
          error: { code: -32602, message: `Unknown tool: ${toolName}` },
        },
        400
      );
    }

    if (MCP_WRITE_TOOLS.has(toolName) && !isAuthorized(request, env)) {
      return jsonResponse(
        {
          jsonrpc: "2.0",
          id: rpcId,
          error: { code: -32001, message: "Unauthorized: valid Bearer token required" },
        },
        401
      );
    }

    if (toolName === "flywheel_frontier") {
      const nodes = await queryAllNodes(env.DB);
      const frontier = computeFrontier(nodes);
      const laneFilter = toNullableString(args.lane);
      const data = laneFilter
        ? { frontier: frontier[laneFilter] ? { [laneFilter]: frontier[laneFilter] } : {} }
        : { frontier };
      return jsonResponse({
        jsonrpc: "2.0",
        id: rpcId,
        result: {
          content: [{ type: "text", text: JSON.stringify(data, null, 2) }],
          structuredContent: data,
          isError: false,
        },
      });
    }

    if (toolName === "flywheel_list_nodes") {
      const nodes = await queryFilteredNodes(env.DB, {
        lane: toNullableString(args.lane),
        run_id: toNullableString(args.run_id),
        decision: toNullableString(args.decision),
        verified: args.verified as string | boolean | number | null | undefined,
        limit: args.limit as number | null | undefined,
      });
      const data = { nodes, count: nodes.length };
      return jsonResponse({
        jsonrpc: "2.0",
        id: rpcId,
        result: {
          content: [{ type: "text", text: JSON.stringify(data, null, 2) }],
          structuredContent: data,
          isError: false,
        },
      });
    }

    if (toolName === "flywheel_claim_task") {
      const outcome = await createClaim(env.DB, args);
      const isError = outcome.status >= 400;
      return jsonResponse({
        jsonrpc: "2.0",
        id: rpcId,
        result: {
          content: [{ type: "text", text: JSON.stringify(outcome.body, null, 2) }],
          structuredContent: outcome.body,
          isError,
        },
      });
    }

    if (toolName === "flywheel_post_node") {
      const outcome = await insertNode(env.DB, args);
      const isError = outcome.status >= 400;
      return jsonResponse({
        jsonrpc: "2.0",
        id: rpcId,
        result: {
          content: [{ type: "text", text: JSON.stringify(outcome.body, null, 2) }],
          structuredContent: outcome.body,
          isError,
        },
      });
    }
  }

  return jsonResponse(
    {
      jsonrpc: "2.0",
      id: rpcId,
      error: { code: -32601, message: `Method not found: ${method}` },
    },
    404
  );
}

async function parseJsonBody(request: Request): Promise<Record<string, unknown> | null> {
  try {
    const data = await request.json();
    return data && typeof data === "object" && !Array.isArray(data)
      ? (data as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

export async function handleRequest(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  const { pathname, searchParams } = url;
  const method = request.method.toUpperCase();

  if (pathname.startsWith("/api/")) {
    if (method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    try {
      if (pathname === "/api/health" && method === "GET") {
        const nowIso = new Date().toISOString();
        const nodeCountRow = await env.DB.prepare("SELECT COUNT(*) AS count FROM nodes").first<{
          count: number;
        }>();
        const claimCountRow = await env.DB.prepare(
          "SELECT COUNT(*) AS count FROM claims WHERE status = 'active' AND expires_at > ?"
        )
          .bind(nowIso)
          .first<{ count: number }>();
        const lanesRes = await env.DB.prepare(
          "SELECT DISTINCT lane FROM nodes ORDER BY lane ASC"
        ).all<{ lane: string }>();

        return jsonResponse({
          ok: true,
          nodes: Number(nodeCountRow?.count ?? 0),
          active_claims: Number(claimCountRow?.count ?? 0),
          lanes: (lanesRes.results ?? []).map((r) => r.lane),
        });
      }

      if (pathname === "/api/dag" && method === "GET") {
        const nodes = await queryAllNodes(env.DB);
        const nodeIds = new Set(nodes.map((n) => n.id));
        const edges: DagEdge[] = nodes
          .filter((n) => n.parent_id !== null && nodeIds.has(n.parent_id))
          .map((n) => ({
            source: n.parent_id as string,
            target: n.id,
            lane: n.lane,
            decision: n.decision,
          }));
        const frontier = computeFrontier(nodes);
        const claims = await queryActiveClaims(env.DB);
        const alignment = computeAlignment(nodes);

        return jsonResponse({
          nodes,
          edges,
          frontier,
          claims,
          alignment,
        });
      }

      if (pathname === "/api/frontier" && method === "GET") {
        const nodes = await queryAllNodes(env.DB);
        const frontier = computeFrontier(nodes);
        return jsonResponse({
          frontier,
          count: Object.keys(frontier).length,
        });
      }

      if (pathname === "/api/nodes" && method === "GET") {
        const limitParam = searchParams.get("limit");
        const nodes = await queryFilteredNodes(env.DB, {
          lane: searchParams.get("lane"),
          run_id: searchParams.get("run_id"),
          decision: searchParams.get("decision"),
          verified: searchParams.get("verified"),
          limit: limitParam !== null ? Number(limitParam) : null,
        });
        return jsonResponse({ nodes, count: nodes.length });
      }

      if (pathname.startsWith("/api/nodes/") && method === "GET") {
        const rawId = pathname.slice("/api/nodes/".length);
        const nodeId = decodeURIComponent(rawId);
        const row = await env.DB.prepare("SELECT * FROM nodes WHERE id = ?")
          .bind(nodeId)
          .first<NodeRow>();
        if (!row) {
          return jsonResponse({ error: "not_found" }, 404);
        }
        const node = deserializeNode(row);
        const childrenRes = await env.DB.prepare(
          "SELECT * FROM nodes WHERE parent_id = ? ORDER BY created_at ASC, id ASC"
        )
          .bind(nodeId)
          .all<NodeRow>();
        const parentRow = node.parent_id
          ? await env.DB.prepare("SELECT * FROM nodes WHERE id = ?")
              .bind(node.parent_id)
              .first<NodeRow>()
          : null;

        return jsonResponse({
          node,
          parent: parentRow ? deserializeNode(parentRow) : null,
          children: (childrenRes.results ?? []).map(deserializeNode),
        });
      }

      if (pathname === "/api/nodes" && method === "POST") {
        if (!isAuthorized(request, env)) {
          return jsonResponse({ error: "unauthorized" }, 401);
        }
        const payload = await parseJsonBody(request);
        if (!payload) {
          return jsonResponse({ error: "invalid_json" }, 400);
        }
        const outcome = await insertNode(env.DB, payload);
        return jsonResponse(outcome.body, outcome.status);
      }

      if (pathname === "/api/claims" && method === "GET") {
        const claims = await queryActiveClaims(env.DB, searchParams.get("lane"));
        return jsonResponse({ claims, count: claims.length });
      }

      if (pathname === "/api/claims" && method === "POST") {
        if (!isAuthorized(request, env)) {
          return jsonResponse({ error: "unauthorized" }, 401);
        }
        const payload = await parseJsonBody(request);
        if (!payload) {
          return jsonResponse({ error: "invalid_json" }, 400);
        }
        const outcome = await createClaim(env.DB, payload);
        return jsonResponse(outcome.body, outcome.status);
      }

      if (pathname.startsWith("/api/claims/") && method === "PATCH") {
        if (!isAuthorized(request, env)) {
          return jsonResponse({ error: "unauthorized" }, 401);
        }
        const claimId = decodeURIComponent(pathname.slice("/api/claims/".length));
        const payload = await parseJsonBody(request);
        if (!payload) {
          return jsonResponse({ error: "invalid_json" }, 400);
        }
        const outcome = await patchClaim(env.DB, claimId, payload);
        return jsonResponse(outcome.body, outcome.status);
      }

      if (pathname === "/api/mcp" && method === "POST") {
        return await handleMcp(request, env);
      }

      return jsonResponse({ error: "not_found", path: pathname }, 404);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return jsonResponse({ error: "internal_error", message }, 500);
    }
  }

  if (env.ASSETS) {
    return env.ASSETS.fetch(request);
  }
  return jsonResponse({ error: "not_found", path: pathname }, 404);
}

export default {
  fetch(request: Request, env: Env): Promise<Response> {
    return handleRequest(request, env);
  },
};
