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
  FLYWHEEL_INVITE_CODE?: string;
}

export interface ApiKeyRow {
  id: string;
  key_hash: string;
  handle: string;
  role: string;
  hourly_limit: number;
  revoked_at: string | null;
  created_at: string;
}

export interface AuthPrincipal {
  role: "admin" | "contributor";
  keyId: string;
  handle: string;
  hourlyLimit: number;
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
  "Access-Control-Allow-Methods": "GET, POST, PATCH, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

const KEPT_DECISIONS = new Set(["keep", "keep_baseline", "keep_simpler"]);
const VALID_CLAIM_PATCH_STATUSES = new Set(["done", "abandoned"]);
const VALID_ROLES = new Set(["contributor", "admin"]);

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

function bytesToHex(bytes: Uint8Array): string {
  let out = "";
  for (let i = 0; i < bytes.length; i += 1) {
    out += bytes[i].toString(16).padStart(2, "0");
  }
  return out;
}

async function sha256Hex(input: string): Promise<string> {
  const data = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return bytesToHex(new Uint8Array(digest));
}

function generateApiToken(): string {
  const bytes = new Uint8Array(30);
  crypto.getRandomValues(bytes);
  return `apfw_${bytesToHex(bytes)}`;
}

function sanitizeHandle(raw: unknown): string | null {
  const s = toNullableString(raw);
  if (!s) return null;
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_.-]{1,39}$/.test(s)) return null;
  return s;
}

async function authenticate(request: Request, env: Env): Promise<AuthPrincipal | null> {
  const configuredKey = env.FLYWHEEL_API_KEY?.trim();
  const authHeader = request.headers.get("Authorization") ?? "";
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7).trim() : "";

  if (configuredKey && token.length > 0 && timingSafeEqual(token, configuredKey)) {
    return { role: "admin", keyId: "root-admin", handle: "admin", hourlyLimit: Number.POSITIVE_INFINITY };
  }

  if (token.length > 0) {
    const keyHash = await sha256Hex(token);
    const row = await env.DB.prepare(
      "SELECT id, key_hash, handle, role, hourly_limit, revoked_at, created_at FROM api_keys WHERE key_hash = ? AND revoked_at IS NULL LIMIT 1"
    )
      .bind(keyHash)
      .first<ApiKeyRow>();
    if (row) {
      const role = row.role === "admin" ? "admin" : "contributor";
      const limit = Number.isFinite(Number(row.hourly_limit)) && Number(row.hourly_limit) > 0
        ? Math.trunc(Number(row.hourly_limit))
        : 120;
      return {
        role,
        keyId: row.id,
        handle: row.handle,
        hourlyLimit: role === "admin" ? Number.POSITIVE_INFINITY : limit,
      };
    }
  }

  if (!configuredKey) {
    return { role: "admin", keyId: "local-dev", handle: "admin", hourlyLimit: Number.POSITIVE_INFINITY };
  }
  return null;
}

async function checkAndConsumeRateLimit(
  db: D1DatabaseLike,
  principal: AuthPrincipal
): Promise<{ allowed: boolean; count: number; limit: number }> {
  if (principal.role === "admin" || !Number.isFinite(principal.hourlyLimit)) {
    return { allowed: true, count: 0, limit: Number.POSITIVE_INFINITY };
  }
  const bucket = new Date().toISOString().slice(0, 13); // UTC hour YYYY-MM-DDTHH
  await db
    .prepare(
      "INSERT INTO rate_limits (key_id, bucket, count) VALUES (?, ?, 1) ON CONFLICT(key_id, bucket) DO UPDATE SET count = count + 1"
    )
    .bind(principal.keyId, bucket)
    .run();
  const row = await db
    .prepare("SELECT count FROM rate_limits WHERE key_id = ? AND bucket = ?")
    .bind(principal.keyId, bucket)
    .first<{ count: number }>();
  const count = Number(row?.count ?? 1);
  return { allowed: count <= principal.hourlyLimit, count, limit: principal.hourlyLimit };
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
  payload: Record<string, unknown>,
  principal: AuthPrincipal
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

  // Contributors cannot overwrite existing nodes owned by another author.
  if (principal.role !== "admin") {
    const existing = await db
      .prepare("SELECT id, author FROM nodes WHERE id = ?")
      .bind(id)
      .first<{ id: string; author: string | null }>();
    if (existing && existing.author !== principal.handle) {
      return {
        status: 403,
        body: { error: "forbidden_node_owner", id, owner: existing.author },
      };
    }
  }

  const created_at = toNullableString(payload.created_at) ?? new Date().toISOString();
  const verifiedRaw = payload.verified;
  const requestedVerified =
    verifiedRaw === true || verifiedRaw === 1 || verifiedRaw === "1" || verifiedRaw === "true"
      ? 1
      : 0;
  // Contributors are always forced to verified = 0 and stamped with their key handle.
  const verified = principal.role === "admin" ? requestedVerified : 0;
  const author = principal.role === "admin" ? toNullableString(payload.author) : principal.handle;

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
      author,
      toNullableString(payload.title),
      toNullableString(payload.hypothesis),
      decision,
      toFiniteNumber(payload.soft_min_eff),
      toFiniteNumber(payload.ratio_vs_incumbent),
      toFiniteNumber(payload.short_horizon_eff),
      toFiniteNumber(payload.long_horizon_eff),
      principal.role === "admin" ? toFiniteInt(payload.real_stan_passed) : null,
      principal.role === "admin" ? toFiniteInt(payload.real_stan_total) : null,
      principal.role === "admin" ? toFiniteNumber(payload.real_ess_per_cost) : null,
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

function claimOwnedBy(agentField: string, handle: string): boolean {
  return agentField === handle || agentField.startsWith(`${handle}:`);
}

async function createClaim(
  db: D1DatabaseLike,
  payload: Record<string, unknown>,
  principal: AuthPrincipal
): Promise<{ status: number; body: Record<string, unknown> }> {
  const task_id = toNullableString(payload.task_id);
  const lane = toNullableString(payload.lane);
  const rawAgent = toNullableString(payload.agent);

  if (!task_id || !lane || !rawAgent) {
    return {
      status: 400,
      body: { error: "missing_required_fields", required: ["task_id", "lane", "agent"] },
    };
  }

  const agent =
    principal.role === "admin"
      ? rawAgent
      : claimOwnedBy(rawAgent, principal.handle)
        ? rawAgent
        : `${principal.handle}:${rawAgent}`;

  const nowMs = Date.now();
  const nowIso = new Date(nowMs).toISOString();
  const ttlRaw = payload.ttl_minutes === undefined ? 60 : Number(payload.ttl_minutes);
  if (!Number.isFinite(ttlRaw) || ttlRaw <= 0) {
    return { status: 400, body: { error: "invalid_ttl_minutes" } };
  }
  const ttlMinutes = Math.min(ttlRaw, principal.role === "admin" ? 1440 : 240);
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
    if (principal.role !== "admin" && !claimOwnedBy(existing.agent, principal.handle)) {
      return {
        status: 403,
        body: { error: "forbidden_force_claim", existing_claim: existing },
      };
    }
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
  payload: Record<string, unknown>,
  principal: AuthPrincipal
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

  if (principal.role !== "admin" && !claimOwnedBy(existing.agent, principal.handle)) {
    return { status: 403, body: { error: "forbidden_claim_owner", owner: existing.agent } };
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

    let principal: AuthPrincipal | null = null;
    if (MCP_WRITE_TOOLS.has(toolName)) {
      principal = await authenticate(request, env);
      if (!principal) {
        return jsonResponse(
          {
            jsonrpc: "2.0",
            id: rpcId,
            error: { code: -32001, message: "Unauthorized: valid Bearer token required" },
          },
          401
        );
      }
      const rl = await checkAndConsumeRateLimit(env.DB, principal);
      if (!rl.allowed) {
        return jsonResponse(
          {
            jsonrpc: "2.0",
            id: rpcId,
            error: { code: -32029, message: `Rate limit exceeded (${rl.limit}/hour)` },
          },
          429
        );
      }
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
      const outcome = await createClaim(env.DB, args, principal as AuthPrincipal);
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
      const outcome = await insertNode(env.DB, args, principal as AuthPrincipal);
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

      if (pathname === "/api/register" && method === "POST") {
        const expectedInvite = env.FLYWHEEL_INVITE_CODE?.trim();
        if (!expectedInvite) {
          return jsonResponse({ error: "registration_closed" }, 403);
        }
        const payload = await parseJsonBody(request);
        if (!payload) {
          return jsonResponse({ error: "invalid_json" }, 400);
        }
        const invite = toNullableString(payload.invite_code);
        if (!invite || !timingSafeEqual(invite, expectedInvite)) {
          return jsonResponse({ error: "invalid_invite_code" }, 403);
        }
        const handle = sanitizeHandle(payload.handle);
        if (!handle) {
          return jsonResponse(
            { error: "invalid_handle", detail: "2-40 chars matching ^[a-zA-Z0-9][a-zA-Z0-9_.-]+$" },
            400
          );
        }
        const dup = await env.DB.prepare(
          "SELECT id FROM api_keys WHERE lower(handle) = lower(?) AND revoked_at IS NULL LIMIT 1"
        )
          .bind(handle)
          .first<{ id: string }>();
        if (dup) {
          return jsonResponse({ error: "handle_already_registered", handle }, 409);
        }
        const id = crypto.randomUUID();
        const token = generateApiToken();
        const keyHash = await sha256Hex(token);
        const nowIso = new Date().toISOString();
        const hourlyLimit = 60;
        await env.DB.prepare(
          "INSERT INTO api_keys (id, key_hash, handle, role, hourly_limit, revoked_at, created_at) VALUES (?, ?, ?, 'contributor', ?, NULL, ?)"
        )
          .bind(id, keyHash, handle, hourlyLimit, nowIso)
          .run();
        return jsonResponse(
          {
            ok: true,
            key_id: id,
            handle,
            role: "contributor",
            hourly_limit: hourlyLimit,
            token,
            note: "Save this token now; it is stored hashed and cannot be shown again.",
          },
          201
        );
      }

      if (pathname === "/api/admin/keys" && method === "GET") {
        const principal = await authenticate(request, env);
        if (!principal) return jsonResponse({ error: "unauthorized" }, 401);
        if (principal.role !== "admin") return jsonResponse({ error: "forbidden_admin_only" }, 403);
        const res = await env.DB.prepare(
          "SELECT id, handle, role, hourly_limit, revoked_at, created_at FROM api_keys ORDER BY created_at DESC"
        ).all<Omit<ApiKeyRow, "key_hash">>();
        return jsonResponse({ keys: res.results ?? [], count: (res.results ?? []).length });
      }

      if (pathname === "/api/admin/keys" && method === "POST") {
        const principal = await authenticate(request, env);
        if (!principal) return jsonResponse({ error: "unauthorized" }, 401);
        if (principal.role !== "admin") return jsonResponse({ error: "forbidden_admin_only" }, 403);
        const payload = await parseJsonBody(request);
        if (!payload) return jsonResponse({ error: "invalid_json" }, 400);
        const handle = sanitizeHandle(payload.handle);
        if (!handle) {
          return jsonResponse(
            { error: "invalid_handle", detail: "2-40 chars matching ^[a-zA-Z0-9][a-zA-Z0-9_.-]+$" },
            400
          );
        }
        const role = toNullableString(payload.role) ?? "contributor";
        if (!VALID_ROLES.has(role)) {
          return jsonResponse({ error: "invalid_role", allowed: ["contributor", "admin"] }, 400);
        }
        const limitRaw = payload.hourly_limit === undefined ? 120 : Number(payload.hourly_limit);
        if (!Number.isFinite(limitRaw) || limitRaw < 1 || limitRaw > 10000) {
          return jsonResponse({ error: "invalid_hourly_limit" }, 400);
        }
        const hourlyLimit = Math.trunc(limitRaw);
        const id = crypto.randomUUID();
        const token = generateApiToken();
        const keyHash = await sha256Hex(token);
        const nowIso = new Date().toISOString();
        await env.DB.prepare(
          "INSERT INTO api_keys (id, key_hash, handle, role, hourly_limit, revoked_at, created_at) VALUES (?, ?, ?, ?, ?, NULL, ?)"
        )
          .bind(id, keyHash, handle, role, hourlyLimit, nowIso)
          .run();
        return jsonResponse(
          { ok: true, key_id: id, handle, role, hourly_limit: hourlyLimit, token, created_at: nowIso },
          201
        );
      }

      if (pathname.startsWith("/api/admin/keys/") && method === "DELETE") {
        const principal = await authenticate(request, env);
        if (!principal) return jsonResponse({ error: "unauthorized" }, 401);
        if (principal.role !== "admin") return jsonResponse({ error: "forbidden_admin_only" }, 403);
        const keyId = decodeURIComponent(pathname.slice("/api/admin/keys/".length));
        const existing = await env.DB.prepare("SELECT id FROM api_keys WHERE id = ?")
          .bind(keyId)
          .first<{ id: string }>();
        if (!existing) return jsonResponse({ error: "not_found" }, 404);
        const nowIso = new Date().toISOString();
        await env.DB.prepare("UPDATE api_keys SET revoked_at = ? WHERE id = ?")
          .bind(nowIso, keyId)
          .run();
        return jsonResponse({ ok: true, key_id: keyId, revoked_at: nowIso });
      }

      if (pathname.startsWith("/api/admin/nodes/") && pathname.endsWith("/verify") && method === "POST") {
        const principal = await authenticate(request, env);
        if (!principal) return jsonResponse({ error: "unauthorized" }, 401);
        if (principal.role !== "admin") return jsonResponse({ error: "forbidden_admin_only" }, 403);
        const rawId = pathname.slice("/api/admin/nodes/".length, -"/verify".length);
        const nodeId = decodeURIComponent(rawId);
        const existing = await env.DB.prepare("SELECT * FROM nodes WHERE id = ?")
          .bind(nodeId)
          .first<NodeRow>();
        if (!existing) return jsonResponse({ error: "not_found" }, 404);
        const payload = (await parseJsonBody(request)) ?? {};
        const verified = payload.verified === false || payload.verified === 0 ? 0 : 1;
        const stanPassed =
          payload.real_stan_passed !== undefined
            ? toFiniteInt(payload.real_stan_passed)
            : existing.real_stan_passed;
        const stanTotal =
          payload.real_stan_total !== undefined
            ? toFiniteInt(payload.real_stan_total)
            : existing.real_stan_total;
        const stanEss =
          payload.real_ess_per_cost !== undefined
            ? toFiniteNumber(payload.real_ess_per_cost)
            : existing.real_ess_per_cost;
        await env.DB.prepare(
          "UPDATE nodes SET verified = ?, real_stan_passed = ?, real_stan_total = ?, real_ess_per_cost = ? WHERE id = ?"
        )
          .bind(verified, stanPassed, stanTotal, stanEss, nodeId)
          .run();
        const updated = await env.DB.prepare("SELECT * FROM nodes WHERE id = ?")
          .bind(nodeId)
          .first<NodeRow>();
        return jsonResponse({ ok: true, node: updated ? deserializeNode(updated) : null });
      }

      if (pathname.startsWith("/api/admin/nodes/") && method === "DELETE") {
        const principal = await authenticate(request, env);
        if (!principal) return jsonResponse({ error: "unauthorized" }, 401);
        if (principal.role !== "admin") return jsonResponse({ error: "forbidden_admin_only" }, 403);
        const nodeId = decodeURIComponent(pathname.slice("/api/admin/nodes/".length));
        const existing = await env.DB.prepare("SELECT id FROM nodes WHERE id = ?")
          .bind(nodeId)
          .first<{ id: string }>();
        if (!existing) return jsonResponse({ error: "not_found" }, 404);
        await env.DB.prepare("DELETE FROM nodes WHERE id = ?").bind(nodeId).run();
        return jsonResponse({ ok: true, deleted: nodeId });
      }

      if (pathname === "/api/nodes" && method === "POST") {
        const principal = await authenticate(request, env);
        if (!principal) {
          return jsonResponse({ error: "unauthorized" }, 401);
        }
        const rl = await checkAndConsumeRateLimit(env.DB, principal);
        if (!rl.allowed) {
          return jsonResponse({ error: "rate_limit_exceeded", limit: rl.limit, count: rl.count }, 429);
        }
        const payload = await parseJsonBody(request);
        if (!payload) {
          return jsonResponse({ error: "invalid_json" }, 400);
        }
        const outcome = await insertNode(env.DB, payload, principal);
        return jsonResponse(outcome.body, outcome.status);
      }

      if (pathname === "/api/claims" && method === "GET") {
        const claims = await queryActiveClaims(env.DB, searchParams.get("lane"));
        return jsonResponse({ claims, count: claims.length });
      }

      if (pathname === "/api/claims" && method === "POST") {
        const principal = await authenticate(request, env);
        if (!principal) {
          return jsonResponse({ error: "unauthorized" }, 401);
        }
        const rl = await checkAndConsumeRateLimit(env.DB, principal);
        if (!rl.allowed) {
          return jsonResponse({ error: "rate_limit_exceeded", limit: rl.limit, count: rl.count }, 429);
        }
        const payload = await parseJsonBody(request);
        if (!payload) {
          return jsonResponse({ error: "invalid_json" }, 400);
        }
        const outcome = await createClaim(env.DB, payload, principal);
        return jsonResponse(outcome.body, outcome.status);
      }

      if (pathname.startsWith("/api/claims/") && method === "PATCH") {
        const principal = await authenticate(request, env);
        if (!principal) {
          return jsonResponse({ error: "unauthorized" }, 401);
        }
        const rl = await checkAndConsumeRateLimit(env.DB, principal);
        if (!rl.allowed) {
          return jsonResponse({ error: "rate_limit_exceeded", limit: rl.limit, count: rl.count }, 429);
        }
        const claimId = decodeURIComponent(pathname.slice("/api/claims/".length));
        const payload = await parseJsonBody(request);
        if (!payload) {
          return jsonResponse({ error: "invalid_json" }, 400);
        }
        const outcome = await patchClaim(env.DB, claimId, payload, principal);
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
