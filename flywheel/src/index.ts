// apin Flywheel Worker: research DAG, sources, notes, search, verification queue,
// claim ledger and MCP endpoint, on Cloudflare Workers Free + D1.
//
// Every operation is one `Op` function. The REST router and the MCP adapter both
// call the same ops, so auth, rate limits, body caps and validation live in one place.
//
// This file is loaded directly by `node --test` (type stripping), so it uses only
// erasable TypeScript: no enums, namespaces or constructor parameter properties.

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
  batch?<T = Record<string, unknown>>(statements: D1PreparedStatement[]): Promise<D1Result<T>[]>;
}

export interface AssetsBindingLike {
  fetch(request: Request): Promise<Response>;
}

export interface ExecutionContextLike {
  waitUntil(promise: Promise<unknown>): void;
}

export interface CacheLike {
  match(request: Request | string): Promise<Response | undefined>;
  put(request: Request | string, response: Response): Promise<void>;
}

type EnvNumber = string | number | undefined;

export interface Env {
  DB: D1DatabaseLike;
  ASSETS?: AssetsBindingLike;
  FLYWHEEL_API_KEY?: string;
  FLYWHEEL_INVITE_CODE?: string;
  CACHE?: CacheLike;
  /** Salt for hashing client IPs in the registration cap. Optional. */
  FLYWHEEL_IP_SALT?: string;
  FLYWHEEL_REGISTRATIONS_PER_IP_PER_DAY?: EnvNumber;
  /** Set to 0 to close open registration (invite-code registration is capped by it too). */
  FLYWHEEL_REGISTRATIONS_PER_DAY?: EnvNumber;
  FLYWHEEL_PROVISIONAL_HOURLY_WRITES?: EnvNumber;
  FLYWHEEL_PROVISIONAL_DAILY_WRITES?: EnvNumber;
  /** New trials one provisional token may create per UTC day. */
  FLYWHEEL_PROVISIONAL_TRIALS_PER_DAY?: EnvNumber;
  /** New trials all provisional tokens together may create per UTC day (bounds /api/dag growth). */
  FLYWHEEL_PROVISIONAL_TRIALS_PER_DAY_GLOBAL?: EnvNumber;
}

/**
 * Abuse and size limits. Counts can be overridden per deployment with the
 * `FLYWHEEL_*` variables in `Env` (wrangler `[vars]` or `wrangler secret put`).
 */
export const LIMITS = {
  registrationsPerIpPerDay: 5,
  registrationsPerDay: 200,
  provisionalHourlyWrites: 60,
  provisionalDailyWrites: 300,
  invitedHourlyWrites: 60,
  promotedHourlyWrites: 120,
  registerBodyBytes: 2 * 1024,
  smallBodyBytes: 8 * 1024,
  noteBodyBytes: 32 * 1024,
  verifyResultBodyBytes: 32 * 1024,
  nodeBodyBytes: 384 * 1024,
  nodeMetaBytes: 32 * 1024,
  sourceBytes: 256 * 1024,
  titleChars: 500,
  hypothesisChars: 8000,
  shortFieldChars: 200,
  noteTitleChars: 200,
  noteBodyChars: 16000,
  reasonChars: 1000,
  maxParents: 16,
  maxNoteLinks: 32,
  leaseMinutesDefault: 120,
  leaseMinutesMax: 480,
  searchLimitMax: 50,
  notesLimitMax: 100,
  queueLimitMax: 100,
  queryChars: 200,
  // Bounded reads (see "Bounded reads" below).
  nodesLimitDefault: 50,
  nodesLimitMax: 200,
  runsLimitDefault: 24,
  runsLimitMax: 60,
  runNodes: 150,
  runSteps: 120,
  dagNodes: 300,
  alignmentPoints: 300,
  overviewLanes: 32,
  overviewModels: 100,
  overviewClaims: 100,
  claimsList: 200,
  nodeChildren: 200,
  nodeNotes: 20,
  searchOffsetMax: 500,
  provisionalTrialsPerDay: 100,
  provisionalTrialsPerDayGlobal: 1000,
} as const;

type Role = "admin" | "contributor" | "provisional";

export interface ApiKeyRow {
  id: string;
  key_hash: string;
  handle: string;
  role: string;
  hourly_limit: number;
  revoked_at: string | null;
  created_at: string;
  agent?: string | null;
  model?: string | null;
}

export interface AuthPrincipal {
  role: Role;
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
  state?: string | null;
  state_reason?: string | null;
  state_by?: string | null;
  state_at?: string | null;
  source_sha?: string | null;
  author_tier?: string | null;
  updated_at?: string | null;
}

export interface NodeRecord {
  id: string;
  parent_id: string | null;
  lane: string;
  run_id: string;
  trial_id: string | null;
  agent_model: string | null;
  author: string | null;
  author_tier: string | null;
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
  dims: number[] | null;
  codex_refs: string[];
  code_sha: string | null;
  metrics_json: string | null;
  metrics: Record<string, unknown> | null;
  verified: number;
  state: string;
  state_reason: string | null;
  state_by: string | null;
  state_at: string | null;
  has_source: boolean;
  created_at: string;
  updated_at: string | null;
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
  relation: string;
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

interface NoteRow {
  id: string;
  kind: string;
  lane: string | null;
  title: string;
  body: string | null;
  state: string;
  state_reason: string | null;
  state_by: string | null;
  author: string;
  node_ids: string | null;
  created_at: string;
  updated_at: string;
}

interface QueueRow {
  node_id: string;
  lane: string;
  status: string;
  reason: string | null;
  lease_token_hash: string | null;
  leased_by: string | null;
  lease_expires_at: string | null;
  attempts: number;
  result: string | null;
  result_json: string | null;
  verifier: string | null;
  created_at: string;
  updated_at: string;
}

type Args = Record<string, unknown>;

interface Outcome {
  status: number;
  body: Record<string, unknown>;
}

interface OpContext {
  env: Env;
  request: Request;
  principal: AuthPrincipal | null;
}

type Op = (ctx: OpContext, args: Args) => Promise<Outcome>;

const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PATCH, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

const KEPT_DECISIONS = new Set(["keep", "keep_baseline", "keep_simpler"]);
const VALID_CLAIM_PATCH_STATUSES = new Set(["done", "abandoned"]);
const VALID_ROLES = new Set(["contributor", "admin", "provisional"]);
const NODE_STATES = new Set([
  "unverified",
  "needs-replication",
  "replicated",
  "verified",
  "disputed",
  "retracted",
]);
/** States a contributor may set on someone else's trial (with a reason). */
const FLAG_STATES = new Set(["needs-replication", "disputed"]);
/** States only a maintainer may move a trial out of. */
const LOCKED_STATES = new Set(["verified", "retracted"]);
/** Trials in these states never define a lane's frontier. */
const FRONTIER_EXCLUDED_STATES = ["retracted", "disputed"];
/** Node states a verification result may overwrite. */
const RESULT_WRITABLE_STATES = ["unverified", "needs-replication", "replicated", "disputed"];
const PARENT_RELATIONS = new Set(["incumbent", "combines", "fork"]);
const NOTE_STATES: Record<string, string[]> = {
  hypothesis: ["open", "supported", "refuted", "needs-replication"],
  insight: ["open", "superseded", "retracted"],
};
const RESERVED_HANDLES = new Set([
  "admin",
  "root",
  "root-admin",
  "local-dev",
  "maintainer",
  "maintainers",
  "system",
  "flywheel",
  "apin",
  "anonymous",
  "null",
  "undefined",
]);
const HANDLE_RULE = "2-40 chars matching ^[a-zA-Z0-9][a-zA-Z0-9_.-]+$";
const LANE_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$/;

const PUBLIC_READ_CACHE_SECONDS = 30;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function jsonResponse(
  body: unknown,
  status = 200,
  extraHeaders: Record<string, string> = {}
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...CORS_HEADERS,
      ...extraHeaders,
    },
  });
}

function out(status: number, body: Record<string, unknown>): Outcome {
  return { status, body };
}

function envLimit(value: EnvNumber, fallback: number): number {
  if (value === undefined || value === null || value === "") return fallback;
  const num = Number(value);
  return Number.isFinite(num) && num >= 0 ? Math.trunc(num) : fallback;
}

function nowIso(): string {
  return new Date().toISOString();
}

function utcDay(offsetDays = 0): string {
  return new Date(Date.now() + offsetDays * 86_400_000).toISOString().slice(0, 10);
}

function getEdgeCache(env: Env): CacheLike | null {
  if (env.CACHE) return env.CACHE;
  const globalCaches = (globalThis as unknown as { caches?: { default?: CacheLike } }).caches;
  return globalCaches?.default ?? null;
}

function canonicalQuery(url: URL, allowed: readonly string[]): string {
  const params = new URLSearchParams();
  for (const key of allowed) {
    const val = url.searchParams.get(key);
    if (val !== null && val !== "") params.set(key, val);
  }
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

function canonicalPublicCacheKey(url: URL): Request | null {
  const { origin, pathname } = url;
  const key = (qs: string) => new Request(`${origin}${pathname}${qs}`, { method: "GET" });
  if (pathname === "/api/health" || pathname === "/api/frontier") return key("");
  if (pathname === "/api/dag") return key(canonicalQuery(url, ["lane"]));
  if (pathname === "/api/claims") return key(canonicalQuery(url, ["lane"]));
  if (pathname === "/api/overview") return key("");
  if (pathname === "/api/runs") return key(canonicalQuery(url, ["lane", "limit", "offset"]));
  if (pathname === "/api/nodes") {
    return key(canonicalQuery(url, [
      "lane", "run_id", "decision", "verified", "state", "model", "outcome", "stan", "q", "limit", "offset",
    ]));
  }
  if (pathname.startsWith("/api/nodes/") && pathname.length > "/api/nodes/".length) return key("");
  if (pathname === "/api/search") {
    return key(canonicalQuery(url, ["q", "lane", "kind", "limit", "offset"]));
  }
  if (pathname === "/api/notes") {
    return key(canonicalQuery(url, ["lane", "kind", "state", "author", "limit", "offset"]));
  }
  if (pathname.startsWith("/api/notes/") && pathname.length > "/api/notes/".length) return key("");
  return null;
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
  let s = "";
  for (let i = 0; i < bytes.length; i += 1) {
    s += bytes[i].toString(16).padStart(2, "0");
  }
  return s;
}

async function sha256Hex(input: string | Uint8Array<ArrayBuffer>): Promise<string> {
  const data = typeof input === "string" ? new TextEncoder().encode(input) : input;
  const digest = await crypto.subtle.digest("SHA-256", data);
  return bytesToHex(new Uint8Array(digest));
}

function randomToken(prefix: string, bytes = 30): string {
  const buf = new Uint8Array(bytes);
  crypto.getRandomValues(buf);
  return `${prefix}${bytesToHex(buf)}`;
}

function utf8Length(s: string): number {
  return new TextEncoder().encode(s).length;
}

function sanitizeHandle(raw: unknown): string | null {
  const s = toNullableString(raw);
  if (!s) return null;
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_.-]{1,39}$/.test(s)) return null;
  return s;
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

function toBool(value: unknown): boolean {
  return value === true || value === 1 || value === "1" || value === "true";
}

function serializeJsonField(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }
  return JSON.stringify(value);
}

function clampLimit(raw: unknown, fallback: number, max: number): number {
  const parsed = Math.trunc(Number(raw));
  if (raw === undefined || raw === null || raw === "" || !Number.isFinite(parsed) || parsed <= 0) {
    return fallback;
  }
  return Math.min(parsed, max);
}

function clampOffset(raw: unknown): number {
  const parsed = Math.trunc(Number(raw));
  return Number.isFinite(parsed) && parsed > 0 ? Math.min(parsed, 10_000) : 0;
}

/** Returns the first field whose string value exceeds its cap, or null. */
function tooLongField(args: Args, caps: Record<string, number>): { field: string; max: number } | null {
  for (const [field, max] of Object.entries(caps)) {
    const val = args[field];
    if (typeof val === "string" && val.length > max) return { field, max };
  }
  return null;
}

function placeholders(n: number): string {
  return new Array(n).fill("?").join(", ");
}

function changesOf(res: D1Result): number {
  return Number((res.meta as { changes?: number } | undefined)?.changes ?? 0);
}

// ---------------------------------------------------------------------------
// Auth, rate limits and request bodies
// ---------------------------------------------------------------------------

function roleOf(raw: string): Role {
  if (raw === "admin") return "admin";
  if (raw === "provisional") return "provisional";
  return "contributor";
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
      "SELECT id, handle, role, hourly_limit FROM api_keys WHERE key_hash = ? AND revoked_at IS NULL LIMIT 1"
    )
      .bind(keyHash)
      .first<ApiKeyRow>();
    if (row) {
      const role = roleOf(row.role);
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

async function bumpCounter(db: D1DatabaseLike, key: string, bucket: string): Promise<number> {
  const row = await db
    .prepare(
      "INSERT INTO rate_limits (key_id, bucket, count) VALUES (?, ?, 1) ON CONFLICT(key_id, bucket) DO UPDATE SET count = count + 1 RETURNING count"
    )
    .bind(key, bucket)
    .first<{ count: number }>();
  return Number(row?.count ?? 1);
}

/** Consumes one write from the token's hourly bucket and, for provisional tokens, its daily bucket. */
async function consumeWrite(env: Env, principal: AuthPrincipal): Promise<Outcome | null> {
  if (principal.role === "admin" || !Number.isFinite(principal.hourlyLimit)) return null;
  const hourBucket = new Date().toISOString().slice(0, 13); // UTC hour YYYY-MM-DDTHH
  const hourly = await bumpCounter(env.DB, principal.keyId, hourBucket);
  if (hourly > principal.hourlyLimit) {
    return out(429, { error: "rate_limit_exceeded", scope: "hourly", limit: principal.hourlyLimit, count: hourly });
  }
  if (principal.role === "provisional") {
    const dailyLimit = envLimit(env.FLYWHEEL_PROVISIONAL_DAILY_WRITES, LIMITS.provisionalDailyWrites);
    const daily = await bumpCounter(env.DB, `day:${principal.keyId}`, utcDay());
    if (daily > dailyLimit) {
      return out(429, { error: "rate_limit_exceeded", scope: "daily", limit: dailyLimit, count: daily });
    }
  }
  return null;
}

/**
 * New-trial budget for provisional tokens: per token and across all provisional tokens.
 * Every trial is read by /api/dag, so this bounds how fast open registration can grow it.
 */
async function consumeProvisionalTrial(env: Env, principal: AuthPrincipal): Promise<Outcome | null> {
  const day = utcDay();
  const perToken = envLimit(env.FLYWHEEL_PROVISIONAL_TRIALS_PER_DAY, LIMITS.provisionalTrialsPerDay);
  const mine = await bumpCounter(env.DB, `trials:${principal.keyId}`, day);
  if (mine > perToken) {
    return out(429, { error: "rate_limit_exceeded", scope: "provisional_trials_daily", limit: perToken, count: mine });
  }
  const globalCap = envLimit(env.FLYWHEEL_PROVISIONAL_TRIALS_PER_DAY_GLOBAL, LIMITS.provisionalTrialsPerDayGlobal);
  const all = await bumpCounter(env.DB, "trials:provisional", day);
  if (all > globalCap) {
    return out(429, { error: "rate_limit_exceeded", scope: "provisional_trials_global_daily", limit: globalCap });
  }
  return null;
}

type BodyResult = { ok: true; value: Args } | { ok: false; outcome: Outcome };

async function readJsonBody(request: Request, maxBytes: number, allowEmpty = false): Promise<BodyResult> {
  const tooLarge: BodyResult = {
    ok: false,
    outcome: out(413, { error: "payload_too_large", max_bytes: maxBytes }),
  };
  const lengthHeader = request.headers.get("Content-Length");
  if (lengthHeader !== null && Number(lengthHeader) > maxBytes) return tooLarge;

  const chunks: Uint8Array[] = [];
  let total = 0;
  if (request.body) {
    const reader = request.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maxBytes) {
        await reader.cancel().catch(() => {});
        return tooLarge;
      }
      chunks.push(value);
    }
  }
  if (total === 0 && allowEmpty) return { ok: true, value: {} };
  const buf = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    buf.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    const parsed: unknown = JSON.parse(new TextDecoder().decode(buf));
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return { ok: true, value: parsed as Args };
    }
  } catch {
    // fall through
  }
  return { ok: false, outcome: out(400, { error: "invalid_json" }) };
}

// ---------------------------------------------------------------------------
// Node (trial) records
// ---------------------------------------------------------------------------

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
  const verified = row.verified ? 1 : 0;

  return {
    id: row.id,
    parent_id: row.parent_id ?? null,
    lane: row.lane,
    run_id: row.run_id,
    trial_id: row.trial_id ?? null,
    agent_model: row.agent_model ?? null,
    author: row.author ?? null,
    author_tier: row.author_tier ?? null,
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
    verified,
    state: row.state ?? (verified ? "verified" : "unverified"),
    state_reason: row.state_reason ?? null,
    state_by: row.state_by ?? null,
    state_at: row.state_at ?? null,
    has_source: Boolean(row.source_sha),
    created_at: row.created_at,
    updated_at: row.updated_at ?? null,
  };
}

// ---------------------------------------------------------------------------
// Bounded reads
//
// Every helper below hands the Worker a bounded number of rows, whatever the number of
// trials: counts come from run_stats (kept current by triggers, migration 0002) and lists
// use indexed ORDER BY ... LIMIT. Nothing loads the whole nodes table into JavaScript.
// ---------------------------------------------------------------------------

/** Decisions counted as "kept" in counts, filters and run charts (run_stats.kept). */
const OUTCOME_KEPT = ["keep", "keep_baseline", "keep_simpler", "keep_incumbent_failed"];
const sqlList = (xs: readonly string[]) => xs.map((x) => `'${x}'`).join(", ");
const SQL_OUTCOME_KEPT = sqlList(OUTCOME_KEPT);
/** Frontier eligibility: not retracted or disputed, and kept or verified. */
const SQL_FRONTIER_OK = `state NOT IN (${sqlList(FRONTIER_EXCLUDED_STATES)})`;
const SQL_FRONTIER_ELIGIBLE = `(verified = 1 OR decision IN (${sqlList([...KEPT_DECISIONS])}))`;
const SQL_FRONTIER_ORDER =
  "ORDER BY soft_min_eff DESC, verified DESC, coalesce(real_stan_passed, -1) DESC, created_at DESC, id DESC";

type NodeWithoutRaw = Omit<NodeRecord, "metrics_json">;

function stripRaw(n: NodeRecord): NodeWithoutRaw {
  const { metrics_json: _raw, ...rest } = n;
  return rest;
}

/** Runs read statements in one D1 round trip when the binding supports batch(). */
async function batchAll<T>(db: D1DatabaseLike, stmts: D1PreparedStatement[]): Promise<T[][]> {
  if (stmts.length === 0) return [];
  if (db.batch) return (await db.batch<T>(stmts)).map((r) => r.results ?? []);
  const rows: T[][] = [];
  for (const s of stmts) rows.push((await s.all<T>()).results ?? []);
  return rows;
}

async function getNodeRow(db: D1DatabaseLike, id: string): Promise<NodeRow | null> {
  return db.prepare("SELECT * FROM nodes WHERE id = ?").bind(id).first<NodeRow>();
}

interface LaneStat {
  lane: string;
  trials: number;
  kept: number;
  runs: number;
  latest_at: string | null;
}

/** Lanes that have trials, from run_stats (at most LIMITS.overviewLanes, busiest first, then sorted by name). */
async function laneStats(db: D1DatabaseLike, lane?: string | null): Promise<LaneStat[]> {
  const res = await db
    .prepare(
      `SELECT lane, SUM(trials) AS trials, SUM(kept) AS kept, COUNT(*) AS runs, MAX(last_at) AS latest_at
       FROM run_stats ${lane ? "WHERE lane = ?" : ""} GROUP BY lane HAVING SUM(trials) > 0
       ORDER BY SUM(trials) DESC, lane LIMIT ?`
    )
    .bind(...(lane ? [lane] : []), LIMITS.overviewLanes)
    .all<LaneStat>();
  return (res.results ?? [])
    .map((r) => ({
      lane: r.lane,
      trials: Number(r.trials),
      kept: Number(r.kept),
      runs: Number(r.runs),
      latest_at: r.latest_at ?? null,
    }))
    .sort((a, b) => a.lane.localeCompare(b.lane));
}

/** Best kept-or-verified trial per lane (falling back to any non-excluded trial), one indexed LIMIT 1 query per lane. */
async function laneFrontiers(db: D1DatabaseLike, lanes: string[]): Promise<Record<string, NodeRecord>> {
  const pick = (eligible: boolean) =>
    `SELECT * FROM nodes INDEXED BY idx_nodes_lane_eff WHERE lane = ? AND ${SQL_FRONTIER_OK}${eligible ? ` AND ${SQL_FRONTIER_ELIGIBLE}` : ""} ${SQL_FRONTIER_ORDER} LIMIT 1`;
  const frontier: Record<string, NodeRecord> = {};
  const first = await batchAll<NodeRow>(db, lanes.map((l) => db.prepare(pick(true)).bind(l)));
  const missing: string[] = [];
  lanes.forEach((l, i) => {
    if (first[i]?.[0]) frontier[l] = deserializeNode(first[i][0]);
    else missing.push(l);
  });
  const second = await batchAll<NodeRow>(db, missing.map((l) => db.prepare(pick(false)).bind(l)));
  missing.forEach((l, i) => {
    if (second[i]?.[0]) frontier[l] = deserializeNode(second[i][0]);
  });
  const sorted: Record<string, NodeRecord> = {};
  for (const l of Object.keys(frontier).sort()) sorted[l] = frontier[l];
  return sorted;
}

/**
 * Running best kept efficiency over one run, starting at its first kept scored trial.
 * The window functions run in D1; the Worker receives at most LIMITS.runSteps rows: every
 * point when the run is short enough, otherwise the points where the value changes plus
 * the last one (and, if even those are too many, the first plus the most recent ones).
 */
function runStepsStmt(db: D1DatabaseLike, lane: string, runId: string): D1PreparedStatement {
  return db
    .prepare(
      `WITH r AS (
         SELECT created_at, id,
           MAX(CASE WHEN decision IN (${SQL_OUTCOME_KEPT}) AND soft_min_eff > 0 THEN soft_min_eff END)
             OVER (ORDER BY created_at, id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS best
         FROM nodes WHERE lane = ? AND run_id = ?
       ), s AS (
         SELECT best, ROW_NUMBER() OVER (ORDER BY created_at, id) AS rn,
           LAG(best) OVER (ORDER BY created_at, id) AS prev, COUNT(*) OVER () AS n
         FROM r WHERE best IS NOT NULL
       )
       SELECT best, rn, n FROM s WHERE n <= ? OR rn = 1 OR rn = n OR best <> prev
       ORDER BY rn = 1 DESC, rn DESC LIMIT ?`
    )
    .bind(lane, runId, LIMITS.runSteps, LIMITS.runSteps);
}

function runSummary(runId: string, frontierEff: number | null, rows: { best: number; rn: number }[]) {
  const steps = [...rows].sort((a, b) => Number(a.rn) - Number(b.rn)).map((r) => Number(r.best));
  const firstEff = steps.length > 0 ? steps[0] : null;
  const gain = firstEff && frontierEff !== null ? frontierEff / firstEff : null;
  return { run_id: runId, first_eff: firstEff, gain, steps };
}

interface StanRow {
  id: string;
  run_id: string;
  trial_id: string | null;
  real_stan_passed: number | null;
  real_stan_total: number;
}

function laneStanStmt(db: D1DatabaseLike, lane: string): D1PreparedStatement {
  return db
    .prepare(
      `SELECT id, run_id, trial_id, real_stan_passed, real_stan_total FROM nodes INDEXED BY idx_nodes_stan_lane
       WHERE lane = ? AND real_stan_total > 0
       ORDER BY CAST(coalesce(real_stan_passed, 0) AS REAL) / real_stan_total DESC, created_at DESC, id DESC LIMIT 1`
    )
    .bind(lane);
}

const ALIGNMENT_COLUMNS =
  "id, lane, run_id, trial_id, agent_model, title, decision, soft_min_eff, ratio_vs_incumbent, " +
  "real_stan_passed, real_stan_total, real_ess_per_cost, overhead_ratio, verified";

/** Newest trials with both a proxy score and a real-Stan score, oldest first. */
async function recentAlignment(db: D1DatabaseLike, lane?: string | null): Promise<AlignmentPoint[]> {
  const res = await db
    .prepare(
      `SELECT ${ALIGNMENT_COLUMNS} FROM nodes
       WHERE real_stan_total > 0 AND real_stan_passed IS NOT NULL AND soft_min_eff IS NOT NULL${lane ? " AND lane = ?" : ""}
       ORDER BY created_at DESC, id DESC LIMIT ?`
    )
    .bind(...(lane ? [lane] : []), LIMITS.alignmentPoints)
    .all<NodeRow>();
  return (res.results ?? []).reverse().map((n) => ({
    id: n.id,
    lane: n.lane,
    run_id: n.run_id,
    trial_id: n.trial_id ?? null,
    agent_model: n.agent_model ?? null,
    title: n.title ?? null,
    decision: n.decision,
    soft_min_eff: Number(n.soft_min_eff),
    ratio_vs_incumbent: toFiniteNumber(n.ratio_vs_incumbent),
    real_stan_passed: Number(n.real_stan_passed),
    real_stan_total: Number(n.real_stan_total),
    real_stan_pass_rate: Number(n.real_stan_passed) / Number(n.real_stan_total),
    real_ess_per_cost: toFiniteNumber(n.real_ess_per_cost),
    overhead_ratio: toFiniteNumber(n.overhead_ratio),
    verified: n.verified ? 1 : 0,
  }));
}

/** Alignment points among already-loaded trials (the DAG's newest trials). */
function alignmentOf(nodes: NodeRecord[]): AlignmentPoint[] {
  return nodes
    .filter((n) => n.soft_min_eff !== null && n.real_stan_passed !== null && (n.real_stan_total ?? 0) > 0)
    .map((n) => ({
      id: n.id,
      lane: n.lane,
      run_id: n.run_id,
      trial_id: n.trial_id,
      agent_model: n.agent_model,
      title: n.title,
      decision: n.decision,
      soft_min_eff: n.soft_min_eff as number,
      ratio_vs_incumbent: n.ratio_vs_incumbent,
      real_stan_passed: n.real_stan_passed as number,
      real_stan_total: n.real_stan_total as number,
      real_stan_pass_rate: (n.real_stan_passed as number) / (n.real_stan_total as number),
      real_ess_per_cost: n.real_ess_per_cost,
      overhead_ratio: n.overhead_ratio,
      verified: n.verified,
    }));
}

async function queryActiveClaims(db: D1DatabaseLike, lane?: string | null, limit: number = LIMITS.claimsList): Promise<ClaimRow[]> {
  const res = await db
    .prepare(
      `SELECT * FROM claims WHERE status = 'active' AND expires_at > ?${lane ? " AND lane = ?" : ""}
       ORDER BY created_at DESC LIMIT ?`
    )
    .bind(nowIso(), ...(lane ? [lane] : []), limit)
    .all<ClaimRow>();
  return res.results ?? [];
}

async function countActiveClaims(db: D1DatabaseLike): Promise<number> {
  const row = await db
    .prepare("SELECT COUNT(*) AS count FROM claims WHERE status = 'active' AND expires_at > ?")
    .bind(nowIso())
    .first<{ count: number }>();
  return Number(row?.count ?? 0);
}

async function totalTrials(db: D1DatabaseLike, lane?: string | null): Promise<number> {
  const row = await db
    .prepare(`SELECT COALESCE(SUM(trials), 0) AS n FROM run_stats${lane ? " WHERE lane = ?" : ""}`)
    .bind(...(lane ? [lane] : []))
    .first<{ n: number }>();
  return Number(row?.n ?? 0);
}

/** Lineage edges among a set of trials: node_parents rows plus legacy nodes.parent_id. */
async function edgesAmong(db: D1DatabaseLike, nodes: NodeRecord[]): Promise<DagEdge[]> {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const ids = [...byId.keys()];
  const rows: { child: string; parent: string; relation: string }[] = [];
  // D1 allows 100 bound parameters per statement.
  const stmts: D1PreparedStatement[] = [];
  for (let i = 0; i < ids.length; i += 90) {
    const chunk = ids.slice(i, i + 90);
    stmts.push(
      db.prepare(`SELECT child, parent, relation FROM node_parents WHERE child IN (${placeholders(chunk.length)})`).bind(...chunk)
    );
  }
  for (const part of await batchAll<{ child: string; parent: string; relation: string }>(db, stmts)) rows.push(...part);
  const seen = new Set<string>();
  const edges: DagEdge[] = [];
  const push = (parent: string, child: string, relation: string) => {
    const target = byId.get(child);
    if (!target || !byId.has(parent)) return;
    const key = `${parent}\u0000${child}\u0000${relation}`;
    if (seen.has(key)) return;
    seen.add(key);
    edges.push({ source: parent, target: child, lane: target.lane, decision: target.decision, relation });
  };
  for (const n of nodes) if (n.parent_id) push(n.parent_id, n.id, "incumbent");
  rows.sort((a, b) => (a.child + a.parent + a.relation).localeCompare(b.child + b.parent + b.relation));
  for (const r of rows) push(r.parent, r.child, r.relation);
  return edges;
}

const NODE_OUTCOMES = new Set(["kept", "archived", "discarded"]);

/** WHERE clause for /api/nodes filters, plus a run_stats column when the count can come from there. */
function nodeFilters(args: Args): { where: string; params: unknown[]; statCount: string | null } | Outcome {
  const clauses: string[] = [];
  const params: unknown[] = [];
  const used = new Set<string>();
  const eq = (key: string, col: string) => {
    const val = toNullableString(args[key]);
    if (val) {
      clauses.push(`${col} = ?`);
      params.push(val);
      used.add(key);
    }
  };
  eq("lane", "lane");
  eq("run_id", "run_id");
  eq("decision", "decision");
  eq("state", "state");
  eq("model", "agent_model");
  const verified = args.verified;
  if (verified !== undefined && verified !== null && verified !== "") {
    clauses.push("verified = ?");
    params.push(toBool(String(verified).toLowerCase()) ? 1 : 0);
    used.add("verified");
  }
  const outcome = toNullableString(args.outcome);
  if (outcome) {
    if (!NODE_OUTCOMES.has(outcome)) return out(400, { error: "invalid_outcome", allowed: [...NODE_OUTCOMES] });
    clauses.push(
      outcome === "kept"
        ? `decision IN (${SQL_OUTCOME_KEPT})`
        : outcome === "archived"
          ? "decision = 'archive_only'"
          : `decision NOT IN (${SQL_OUTCOME_KEPT}, 'archive_only')`
    );
    used.add("outcome");
  }
  if (toBool(args.stan)) {
    clauses.push("real_stan_total > 0");
    used.add("stan");
  }
  const q = toNullableString(args.q);
  if (q) {
    if (q.length > LIMITS.queryChars) return out(400, { error: "field_too_long", field: "q", max: LIMITS.queryChars });
    const terms = searchTerms(q);
    if (terms.length === 0) return out(400, { error: "empty_query" });
    for (const t of terms) {
      clauses.push(`${NODE_SEARCH_TEXT} LIKE ? ESCAPE '\\'`);
      params.push(likePattern(t));
    }
    used.add("q");
  }
  // Counts restricted to lane / run_id and at most one of outcome / stan come from run_stats.
  let statCount: string | null = null;
  const rest = [...used].filter((k) => k !== "lane" && k !== "run_id");
  if (rest.length === 0) statCount = "trials";
  else if (rest.length === 1 && rest[0] === "stan") statCount = "stan";
  else if (rest.length === 1 && rest[0] === "outcome") {
    statCount = outcome === "kept" ? "kept" : outcome === "archived" ? "archived" : "trials - kept - archived";
  }
  return { where: clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "", params, statCount };
}

interface ParentEdge {
  parent: string;
  relation: string;
}

/** Legacy `parent_id` (relation incumbent) plus `parent_ids: [id | {id, relation}]`. */
function parseParents(payload: Args, childId: string): { edges: ParentEdge[]; given: boolean } | Outcome {
  const edges: ParentEdge[] = [];
  const seen = new Set<string>();
  const add = (parentRaw: unknown, relationRaw: unknown): Outcome | null => {
    const parent = toNullableString(parentRaw);
    if (!parent) return out(400, { error: "invalid_parent", detail: "parent id must be a non-empty string" });
    if (parent.length > LIMITS.shortFieldChars) return out(400, { error: "field_too_long", field: "parent_ids" });
    const relation = toNullableString(relationRaw) ?? "incumbent";
    if (!PARENT_RELATIONS.has(relation)) {
      return out(400, { error: "invalid_relation", allowed: [...PARENT_RELATIONS] });
    }
    if (parent === childId) return out(400, { error: "self_parent" });
    const key = `${parent}\u0000${relation}`;
    if (!seen.has(key)) {
      seen.add(key);
      edges.push({ parent, relation });
    }
    return null;
  };

  const hasLegacy = payload.parent_id !== undefined && payload.parent_id !== null && payload.parent_id !== "";
  if (hasLegacy) {
    const err = add(payload.parent_id, "incumbent");
    if (err) return err;
  }
  const list = payload.parent_ids;
  if (list !== undefined && list !== null) {
    if (!Array.isArray(list)) return out(400, { error: "invalid_parent_ids", detail: "expected an array" });
    if (list.length > LIMITS.maxParents) {
      return out(400, { error: "too_many_parents", max: LIMITS.maxParents });
    }
    for (const item of list) {
      const err = item && typeof item === "object"
        ? add((item as Args).id, (item as Args).relation)
        : add(item, "incumbent");
      if (err) return err;
    }
  }
  return { edges, given: hasLegacy || (Array.isArray(list) && list.length > 0) || "parent_id" in payload };
}

const NODE_DATA_COLUMNS = [
  "parent_id", "lane", "run_id", "trial_id", "agent_model", "title", "hypothesis", "decision",
  "soft_min_eff", "ratio_vs_incumbent", "short_horizon_eff", "long_horizon_eff", "overhead_ratio",
  "b2_max", "dims", "codex_refs", "code_sha", "metrics_json",
] as const;
const NODE_ADMIN_COLUMNS = ["real_stan_passed", "real_stan_total", "real_ess_per_cost"] as const;

async function frontierBest(db: D1DatabaseLike, lane: string, excludeId: string): Promise<number | null> {
  const row = await db
    .prepare(
      `SELECT soft_min_eff AS best FROM nodes INDEXED BY idx_nodes_lane_eff
       WHERE lane = ? AND id <> ? AND soft_min_eff IS NOT NULL AND ${SQL_FRONTIER_OK} AND ${SQL_FRONTIER_ELIGIBLE}
       ORDER BY soft_min_eff DESC LIMIT 1`
    )
    .bind(lane, excludeId)
    .first<{ best: number | null }>();
  return toFiniteNumber(row?.best);
}

/** A kept trial with source that beats its lane's frontier enters the verification queue. */
async function maybeEnqueue(db: D1DatabaseLike, node: NodeRecord): Promise<boolean> {
  if (!KEPT_DECISIONS.has(node.decision) || node.soft_min_eff === null || !node.has_source) return false;
  if (LOCKED_STATES.has(node.state) || FRONTIER_EXCLUDED_STATES.includes(node.state)) return false;
  const best = await frontierBest(db, node.lane, node.id);
  if (best !== null && node.soft_min_eff <= best) return false;
  const now = nowIso();
  const res = await db
    .prepare(
      `INSERT OR IGNORE INTO verify_queue (node_id, lane, status, reason, attempts, created_at, updated_at)
       VALUES (?, ?, 'queued', 'beats_frontier', 0, ?, ?)`
    )
    .bind(node.id, node.lane, now, now)
    .run();
  return changesOf(res) > 0;
}

const opPostNode: Op = async (ctx, payload) => {
  const db = ctx.env.DB;
  const principal = ctx.principal as AuthPrincipal;
  const isAdmin = principal.role === "admin";

  const lane = toNullableString(payload.lane);
  const run_id = toNullableString(payload.run_id);
  const decision = toNullableString(payload.decision);
  if (!lane || !run_id || !decision) {
    return out(400, { error: "missing_required_fields", required: ["lane", "run_id", "decision"] });
  }

  const source = payload.source;
  if (source !== undefined && source !== null && typeof source !== "string") {
    return out(400, { error: "invalid_source", detail: "source must be a string" });
  }
  const metaBytes = utf8Length(JSON.stringify({ ...payload, source: null }));
  if (metaBytes > LIMITS.nodeMetaBytes) {
    return out(413, { error: "payload_too_large", detail: "trial fields excluding source", max_bytes: LIMITS.nodeMetaBytes });
  }
  const long = tooLongField(payload, {
    id: LIMITS.shortFieldChars,
    lane: 32,
    run_id: LIMITS.shortFieldChars,
    trial_id: LIMITS.shortFieldChars,
    agent_model: LIMITS.shortFieldChars,
    author: LIMITS.shortFieldChars,
    decision: 64,
    code_sha: 128,
    title: LIMITS.titleChars,
    hypothesis: LIMITS.hypothesisChars,
  });
  if (long) return out(400, { error: "field_too_long", ...long });

  const id = toNullableString(payload.id) ?? crypto.randomUUID();
  const parents = parseParents(payload, id);
  if ("status" in parents) return parents;

  let sourceSha: string | null = null;
  let sourceBytes: Uint8Array<ArrayBuffer> | null = null;
  if (typeof source === "string" && source.length > 0) {
    sourceBytes = new TextEncoder().encode(source);
    if (sourceBytes.length > LIMITS.sourceBytes) {
      return out(413, { error: "source_too_large", max_bytes: LIMITS.sourceBytes, bytes: sourceBytes.length });
    }
    sourceSha = await sha256Hex(sourceBytes);
    const codeSha = toNullableString(payload.code_sha)?.toLowerCase();
    if (codeSha && /^[0-9a-f]{8,64}$/.test(codeSha) && !sourceSha.startsWith(codeSha)) {
      return out(400, { error: "code_sha_mismatch", code_sha: codeSha, source_sha256: sourceSha });
    }
  }

  const existing = await db
    .prepare("SELECT id, author FROM nodes WHERE id = ?")
    .bind(id)
    .first<{ id: string; author: string | null }>();
  if (existing && !isAdmin && existing.author !== principal.handle) {
    return out(403, { error: "forbidden_node_owner", id, owner: existing.author });
  }
  if (!existing && principal.role === "provisional") {
    const capped = await consumeProvisionalTrial(ctx.env, principal);
    if (capped) return capped;
  }

  const now = nowIso();
  if (sourceSha && sourceBytes) {
    await db
      .prepare("INSERT OR IGNORE INTO sources (sha256, body, size, uploaded_by, created_at) VALUES (?, ?, ?, ?, ?)")
      .bind(sourceSha, source, sourceBytes.length, principal.handle, now)
      .run();
  }

  const incumbent = parents.edges.find((e) => e.relation === "incumbent")?.parent ?? null;
  const explicitVerified = isAdmin && payload.verified !== undefined;
  const verified = isAdmin && toBool(payload.verified) ? 1 : 0;
  const values: Record<string, unknown> = {
    parent_id: toNullableString(payload.parent_id) ?? incumbent,
    lane,
    run_id,
    trial_id: toNullableString(payload.trial_id),
    agent_model: toNullableString(payload.agent_model),
    title: toNullableString(payload.title),
    hypothesis: toNullableString(payload.hypothesis),
    decision,
    soft_min_eff: toFiniteNumber(payload.soft_min_eff),
    ratio_vs_incumbent: toFiniteNumber(payload.ratio_vs_incumbent),
    short_horizon_eff: toFiniteNumber(payload.short_horizon_eff),
    long_horizon_eff: toFiniteNumber(payload.long_horizon_eff),
    overhead_ratio: toFiniteNumber(payload.overhead_ratio),
    b2_max: toFiniteNumber(payload.b2_max),
    dims: serializeJsonField(payload.dims),
    codex_refs: serializeJsonField(payload.codex_refs ?? []),
    code_sha: toNullableString(payload.code_sha),
    metrics_json:
      payload.metrics_json !== undefined
        ? serializeJsonField(payload.metrics_json)
        : payload.metrics !== undefined
          ? serializeJsonField(payload.metrics)
          : null,
    real_stan_passed: isAdmin ? toFiniteInt(payload.real_stan_passed) : null,
    real_stan_total: isAdmin ? toFiniteInt(payload.real_stan_total) : null,
    real_ess_per_cost: isAdmin ? toFiniteNumber(payload.real_ess_per_cost) : null,
    id,
    author: isAdmin ? toNullableString(payload.author) : principal.handle,
    author_tier: isAdmin ? null : principal.role,
    verified,
    state: verified ? "verified" : "unverified",
    source_sha: sourceSha,
    created_at: toNullableString(payload.created_at) ?? now,
    updated_at: now,
  };

  // Idempotent upsert. A re-post by the same author updates the trial's data but never
  // its research state, author or created_at; another author's row is never touched.
  const columns = Object.keys(values);
  const updates: string[] = NODE_DATA_COLUMNS
    .filter((c) => c !== "parent_id" || parents.given)
    .map((c) => `${c} = excluded.${c}`);
  updates.push("source_sha = COALESCE(excluded.source_sha, nodes.source_sha)", "updated_at = excluded.updated_at");
  if (isAdmin) {
    for (const c of NODE_ADMIN_COLUMNS) updates.push(`${c} = COALESCE(excluded.${c}, nodes.${c})`);
    updates.push("author = COALESCE(excluded.author, nodes.author)");
    if (explicitVerified) {
      updates.push(
        "verified = excluded.verified",
        "state = CASE WHEN excluded.verified = 1 THEN 'verified' WHEN nodes.state = 'verified' THEN 'unverified' ELSE nodes.state END"
      );
    }
  }
  const guard = isAdmin ? "" : " WHERE nodes.author IS excluded.author";
  const res = await db
    .prepare(
      `INSERT INTO nodes (${columns.join(", ")}) VALUES (${placeholders(columns.length)})
       ON CONFLICT(id) DO UPDATE SET ${updates.join(", ")}${guard}`
    )
    .bind(...columns.map((c) => values[c]))
    .run();
  if (changesOf(res) === 0) {
    return out(403, { error: "forbidden_node_owner", id });
  }

  if (parents.given) {
    if (existing) await db.prepare("DELETE FROM node_parents WHERE child = ?").bind(id).run();
    if (parents.edges.length > 0) {
      const binds: unknown[] = [];
      for (const e of parents.edges) binds.push(id, e.parent, e.relation, now);
      await db
        .prepare(
          `INSERT OR IGNORE INTO node_parents (child, parent, relation, created_at) VALUES ${parents.edges
            .map(() => "(?, ?, ?, ?)")
            .join(", ")}`
        )
        .bind(...binds)
        .run();
    }
  }

  const row = await getNodeRow(db, id);
  const node = row ? deserializeNode(row) : null;
  const queued = node ? await maybeEnqueue(db, node) : false;
  return out(existing ? 200 : 201, { ok: true, created: !existing, node, verify_queued: queued });
};

const opListNodes: Op = async (ctx, args) => {
  const db = ctx.env.DB;
  const filters = nodeFilters(args);
  if ("status" in filters) return filters;
  const limit = clampLimit(args.limit, LIMITS.nodesLimitDefault, LIMITS.nodesLimitMax);
  const offset = clampOffset(args.offset);
  const res = await db
    .prepare(`SELECT * FROM nodes ${filters.where} ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?`)
    .bind(...filters.params, limit, offset)
    .all<NodeRow>();
  const nodes = (res.results ?? []).map((r) => stripRaw(deserializeNode(r)));
  let count: number;
  if (filters.statCount) {
    const lane = toNullableString(args.lane);
    const runId = toNullableString(args.run_id);
    const where = [lane ? "lane = ?" : "", runId ? "run_id = ?" : ""].filter(Boolean).join(" AND ");
    const row = await db
      .prepare(`SELECT COALESCE(SUM(${filters.statCount}), 0) AS n FROM run_stats${where ? ` WHERE ${where}` : ""}`)
      .bind(...[lane, runId].filter((x): x is string => Boolean(x)))
      .first<{ n: number }>();
    count = Number(row?.n ?? 0);
  } else {
    const row = await db
      .prepare(`SELECT COUNT(*) AS n FROM nodes ${filters.where}`)
      .bind(...filters.params)
      .first<{ n: number }>();
    count = Number(row?.n ?? 0);
  }
  return out(200, {
    nodes,
    count,
    limit,
    offset,
    next_offset: offset + nodes.length < count && nodes.length === limit ? offset + limit : null,
  });
};

const opFrontier: Op = async (ctx, args) => {
  const lane = toNullableString(args.lane);
  const lanes = lane ? [lane] : (await laneStats(ctx.env.DB)).map((l) => l.lane);
  const frontier = await laneFrontiers(ctx.env.DB, lanes);
  if (lane) return out(200, { frontier });
  return out(200, { frontier, count: Object.keys(frontier).length });
};

const opGetNode: Op = async (ctx, args) => {
  const db = ctx.env.DB;
  const id = toNullableString(args.id);
  if (!id) return out(400, { error: "missing_required_fields", required: ["id"] });
  const row = await getNodeRow(db, id);
  if (!row) return out(404, { error: "not_found" });
  const node = deserializeNode(row);

  type Link = { id: string; relation: string; lane: string | null; trial_id: string | null; decision: string | null };
  const [parentRows, childRows, noteRows, queueRows, legacyParent] = await batchAll<Record<string, unknown>>(db, [
    db
      .prepare(
        `SELECT p.parent AS id, p.relation, n.lane, n.trial_id, n.decision
         FROM node_parents p LEFT JOIN nodes n ON n.id = p.parent WHERE p.child = ? ORDER BY p.relation, p.parent`
      )
      .bind(id),
    db
      .prepare(
        `SELECT c.id, c.relation, n.lane, n.trial_id, n.decision, n.created_at FROM (
           SELECT child AS id, relation FROM node_parents WHERE parent = ?
           UNION SELECT id, 'incumbent' AS relation FROM nodes WHERE parent_id = ?
         ) c JOIN nodes n ON n.id = c.id ORDER BY n.created_at DESC, n.id DESC LIMIT ?`
      )
      .bind(id, id, LIMITS.nodeChildren),
    db
      .prepare(
        `SELECT n.id, n.kind, n.title, n.state FROM note_links l JOIN notes n ON n.id = l.note_id
         WHERE l.node_id = ? ORDER BY n.updated_at DESC, n.id DESC LIMIT ?`
      )
      .bind(id, LIMITS.nodeNotes),
    db.prepare("SELECT * FROM verify_queue WHERE node_id = ?").bind(id),
    db.prepare("SELECT * FROM nodes WHERE id = ?").bind(node.parent_id ?? ""),
  ]);
  const link = (r: Record<string, unknown>): Link => ({
    id: String(r.id),
    relation: String(r.relation),
    lane: (r.lane as string | null) ?? null,
    trial_id: (r.trial_id as string | null) ?? null,
    decision: (r.decision as string | null) ?? null,
  });
  const parents = parentRows.map(link);
  const parentNode = legacyParent[0] ? deserializeNode(legacyParent[0] as unknown as NodeRow) : null;
  if (node.parent_id && !parents.some((p) => p.id === node.parent_id && p.relation === "incumbent")) {
    parents.unshift({
      id: node.parent_id,
      relation: "incumbent",
      lane: parentNode?.lane ?? null,
      trial_id: parentNode?.trial_id ?? null,
      decision: parentNode?.decision ?? null,
    });
  }
  const queue = queueRows[0] as unknown as QueueRow | undefined;
  return out(200, {
    node,
    parent: parentNode,
    parents,
    children: childRows.reverse().map(link),
    notes: noteRows.map((r) => ({ id: r.id, kind: r.kind, title: r.title, state: r.state })),
    verification: queue ? publicQueueFields(queue) : null,
  });
};

/** Home page and frontier header: totals, per-lane frontier/run/Stan, claims, alignment, models. */
const opOverview: Op = async (ctx) => {
  const db = ctx.env.DB;
  const lanes = await laneStats(db);
  const frontier = await laneFrontiers(db, lanes.map((l) => l.lane));
  const runLanes = lanes.filter((l) => frontier[l.lane]);
  const [totalsRows, modelRows, ...perLane] = await batchAll<Record<string, unknown>>(db, [
    // Distinct authors and models by skip scan: one index seek per distinct value, so the
    // cost follows the number of contributors and models, not the number of trials.
    db.prepare(
      `WITH RECURSIVE a(v) AS (
         SELECT MIN(lower(author)) FROM nodes WHERE lower(author) IS NOT NULL
         UNION ALL SELECT (SELECT MIN(lower(author)) FROM nodes WHERE lower(author) > a.v) FROM a WHERE a.v IS NOT NULL
       )
       SELECT (SELECT COALESCE(SUM(trials), 0) FROM run_stats) AS trials,
              (SELECT COALESCE(SUM(kept), 0) FROM run_stats) AS kept,
              (SELECT COUNT(*) FROM run_stats) AS runs,
              (SELECT MAX(last_at) FROM run_stats) AS latest_at,
              (SELECT COUNT(*) FROM a WHERE v IS NOT NULL) AS contributors`
    ),
    db
      .prepare(
        `WITH RECURSIVE m(v) AS (
           SELECT MIN(agent_model) FROM nodes WHERE agent_model IS NOT NULL
           UNION ALL SELECT (SELECT MIN(agent_model) FROM nodes WHERE agent_model > m.v) FROM m WHERE m.v IS NOT NULL
         )
         SELECT v AS agent_model FROM m WHERE v IS NOT NULL LIMIT ?`
      )
      .bind(LIMITS.overviewModels),
    ...runLanes.map((l) => runStepsStmt(db, l.lane, frontier[l.lane].run_id)),
    ...lanes.map((l) => laneStanStmt(db, l.lane)),
  ]);
  const stepsByLane = new Map(runLanes.map((l, i) => [l.lane, perLane[i] as unknown as { best: number; rn: number }[]]));
  const stanByLane = new Map(lanes.map((l, i) => [l.lane, perLane[runLanes.length + i]?.[0] as unknown as StanRow | undefined]));
  const t = totalsRows[0] ?? {};
  return out(200, {
    totals: {
      trials: Number(t.trials ?? 0),
      kept: Number(t.kept ?? 0),
      runs: Number(t.runs ?? 0),
      contributors: Number(t.contributors ?? 0),
      claims: await countActiveClaims(db),
      latest_at: (t.latest_at as string | null) ?? null,
    },
    lanes: lanes.map((l) => {
      const f = frontier[l.lane] ?? null;
      const stan = stanByLane.get(l.lane);
      return {
        ...l,
        frontier: f,
        run: f ? runSummary(f.run_id, f.soft_min_eff, stepsByLane.get(l.lane) ?? []) : null,
        stan: stan
          ? {
              id: stan.id,
              run_id: stan.run_id,
              trial_id: stan.trial_id ?? null,
              real_stan_passed: toFiniteInt(stan.real_stan_passed),
              real_stan_total: Number(stan.real_stan_total),
            }
          : null,
      };
    }),
    claims: await queryActiveClaims(db, null, LIMITS.overviewClaims),
    alignment: await recentAlignment(db),
    models: modelRows.map((r) => String(r.agent_model)),
  });
};

interface RunStatRow {
  lane: string;
  run_id: string;
  trials: number;
  kept: number;
  first_at: string | null;
  last_at: string | null;
  author: string | null;
  author_tier: string | null;
  agent_model: string | null;
  base_eff: number | null;
}

/** Runs by latest activity, each with its newest LIMITS.runNodes trials as slim rows. */
const opRuns: Op = async (ctx, args) => {
  const db = ctx.env.DB;
  const lane = toNullableString(args.lane);
  const limit = clampLimit(args.limit, LIMITS.runsLimitDefault, LIMITS.runsLimitMax);
  const offset = clampOffset(args.offset);
  const laneWhere = lane ? "WHERE lane = ?" : "";
  const laneParams = lane ? [lane] : [];
  const [runRows, countRows] = await batchAll<Record<string, unknown>>(db, [
    db
      .prepare(`SELECT * FROM run_stats ${laneWhere} ORDER BY last_at DESC, lane DESC, run_id DESC LIMIT ? OFFSET ?`)
      .bind(...laneParams, limit, offset),
    db.prepare(`SELECT COUNT(*) AS n FROM run_stats ${laneWhere}`).bind(...laneParams),
  ]);
  const runs = runRows as unknown as RunStatRow[];
  const slices = await batchAll<Record<string, unknown>>(
    db,
    runs.map((r) =>
      db
        .prepare(
          `SELECT id, trial_id, decision, soft_min_eff, state, verified, real_stan_total, created_at FROM nodes
           WHERE lane = ? AND run_id = ? ORDER BY created_at DESC, id DESC LIMIT ?`
        )
        .bind(r.lane, r.run_id, LIMITS.runNodes)
    )
  );
  const count = Number(countRows[0]?.n ?? 0);
  return out(200, {
    runs: runs.map((r, i) => ({
      lane: r.lane,
      run_id: r.run_id,
      author: r.author ?? null,
      author_tier: r.author_tier ?? null,
      agent_model: r.agent_model ?? null,
      trials: Number(r.trials),
      kept: Number(r.kept),
      first_at: r.first_at ?? null,
      last_at: r.last_at ?? null,
      base_eff: toFiniteNumber(r.base_eff),
      truncated: Number(r.trials) > LIMITS.runNodes,
      nodes: (slices[i] ?? []).reverse().map((n) => ({
        id: n.id as string,
        trial_id: (n.trial_id as string | null) ?? null,
        decision: n.decision as string,
        soft_min_eff: toFiniteNumber(n.soft_min_eff),
        state: (n.state as string | null) ?? (n.verified ? "verified" : "unverified"),
        stan: Number(n.real_stan_total ?? 0) > 0,
        created_at: n.created_at as string,
      })),
    })),
    count,
    limit,
    offset,
    next_offset: offset + runs.length < count && runs.length === limit ? offset + limit : null,
  });
};

const opGetSource: Op = async (ctx, args) => {
  const id = toNullableString(args.node_id ?? args.id);
  if (!id) return out(400, { error: "missing_required_fields", required: ["node_id"] });
  const row = await ctx.env.DB
    .prepare("SELECT id, code_sha, source_sha FROM nodes WHERE id = ?")
    .bind(id)
    .first<{ id: string; code_sha: string | null; source_sha: string | null }>();
  if (!row) return out(404, { error: "not_found" });
  if (!row.source_sha) return out(404, { error: "no_source", node_id: id });
  const src = await ctx.env.DB
    .prepare("SELECT sha256, body, size FROM sources WHERE sha256 = ?")
    .bind(row.source_sha)
    .first<{ sha256: string; body: string; size: number }>();
  if (!src) return out(404, { error: "no_source", node_id: id });
  return out(200, {
    node_id: id,
    sha256: src.sha256,
    size: Number(src.size),
    code_sha: row.code_sha ?? null,
    source: src.body,
  });
};

const opSetNodeState: Op = async (ctx, args) => {
  const db = ctx.env.DB;
  const principal = ctx.principal as AuthPrincipal;
  const id = toNullableString(args.id ?? args.node_id);
  const state = toNullableString(args.state);
  const reason = toNullableString(args.reason);
  if (!id || !state) return out(400, { error: "missing_required_fields", required: ["id", "state"] });
  if (!NODE_STATES.has(state)) return out(400, { error: "invalid_state", allowed: [...NODE_STATES] });
  if (reason && reason.length > LIMITS.reasonChars) {
    return out(400, { error: "field_too_long", field: "reason", max: LIMITS.reasonChars });
  }
  const row = await getNodeRow(db, id);
  if (!row) return out(404, { error: "not_found" });
  const node = deserializeNode(row);

  if (principal.role !== "admin") {
    const own = node.author === principal.handle;
    if (own) {
      if (state !== "retracted") {
        return out(403, { error: "forbidden_state", detail: "authors may only retract their own trial", allowed: ["retracted"] });
      }
    } else {
      if (principal.role === "provisional") {
        return out(403, { error: "forbidden_tier", detail: "flagging another author's trial needs a contributor token" });
      }
      if (!FLAG_STATES.has(state)) {
        return out(403, { error: "forbidden_state", allowed: [...FLAG_STATES] });
      }
      if (!reason) return out(400, { error: "reason_required" });
      if (LOCKED_STATES.has(node.state)) {
        return out(409, { error: "state_locked", state: node.state });
      }
    }
  }

  const now = nowIso();
  await db
    .prepare("UPDATE nodes SET state = ?, verified = ?, state_reason = ?, state_by = ?, state_at = ? WHERE id = ?")
    .bind(state, state === "verified" ? 1 : 0, reason, principal.handle, now, id)
    .run();

  if (state === "needs-replication" && node.has_source) {
    await db
      .prepare(
        `INSERT INTO verify_queue (node_id, lane, status, reason, attempts, created_at, updated_at)
         VALUES (?, ?, 'queued', 'needs-replication', 0, ?, ?)
         ON CONFLICT(node_id) DO UPDATE SET status = 'queued', reason = excluded.reason,
           lease_token_hash = NULL, leased_by = NULL, lease_expires_at = NULL, updated_at = excluded.updated_at
         WHERE verify_queue.status = 'done'`
      )
      .bind(id, node.lane, now, now)
      .run();
  } else if (LOCKED_STATES.has(state)) {
    await db
      .prepare(
        "UPDATE verify_queue SET status = 'done', result = ?, lease_token_hash = NULL, updated_at = ? WHERE node_id = ? AND status <> 'done'"
      )
      .bind(state, now, id)
      .run();
  }

  const updated = await getNodeRow(db, id);
  return out(200, { ok: true, node: updated ? deserializeNode(updated) : null });
};

// ---------------------------------------------------------------------------
// Claims (task leases)
// ---------------------------------------------------------------------------

function claimOwnedBy(agentField: string, handle: string): boolean {
  return agentField === handle || agentField.startsWith(`${handle}:`);
}

const opCreateClaim: Op = async (ctx, payload) => {
  const db = ctx.env.DB;
  const principal = ctx.principal as AuthPrincipal;
  const task_id = toNullableString(payload.task_id);
  const lane = toNullableString(payload.lane);
  const rawAgent = toNullableString(payload.agent);

  if (!task_id || !lane || !rawAgent) {
    return out(400, { error: "missing_required_fields", required: ["task_id", "lane", "agent"] });
  }

  const agent =
    principal.role === "admin"
      ? rawAgent
      : claimOwnedBy(rawAgent, principal.handle)
        ? rawAgent
        : `${principal.handle}:${rawAgent}`;

  const nowMs = Date.now();
  const now = new Date(nowMs).toISOString();
  const ttlRaw = payload.ttl_minutes === undefined ? 60 : Number(payload.ttl_minutes);
  if (!Number.isFinite(ttlRaw) || ttlRaw <= 0) {
    return out(400, { error: "invalid_ttl_minutes" });
  }
  const ttlMinutes = Math.min(ttlRaw, principal.role === "admin" ? 1440 : 240);
  const force = payload.force === true;

  const existing = await db
    .prepare(
      "SELECT * FROM claims WHERE task_id = ? AND status = 'active' AND expires_at > ? ORDER BY created_at DESC LIMIT 1"
    )
    .bind(task_id, now)
    .first<ClaimRow>();

  if (existing && !force) {
    return out(409, { error: "already_claimed", existing_claim: existing });
  }

  if (existing && force) {
    if (principal.role !== "admin" && !claimOwnedBy(existing.agent, principal.handle)) {
      return out(403, { error: "forbidden_force_claim", existing_claim: existing });
    }
    await db
      .prepare("UPDATE claims SET status = 'abandoned' WHERE task_id = ? AND status = 'active' AND expires_at > ?")
      .bind(task_id, now)
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
    .bind(id, task_id, lane, agent, summary, expiresAt, now)
    .run();

  const claim = await db.prepare("SELECT * FROM claims WHERE id = ?").bind(id).first<ClaimRow>();
  return out(201, { ok: true, claim });
};

const opPatchClaim: Op = async (ctx, payload) => {
  const db = ctx.env.DB;
  const principal = ctx.principal as AuthPrincipal;
  const claimId = toNullableString(payload.id) ?? "";
  const statusVal = toNullableString(payload.status);
  if (!statusVal || !VALID_CLAIM_PATCH_STATUSES.has(statusVal)) {
    return out(400, { error: "invalid_status", allowed: ["done", "abandoned"] });
  }

  const existing = await db.prepare("SELECT * FROM claims WHERE id = ?").bind(claimId).first<ClaimRow>();
  if (!existing) return out(404, { error: "not_found" });

  if (principal.role !== "admin" && !claimOwnedBy(existing.agent, principal.handle)) {
    return out(403, { error: "forbidden_claim_owner", owner: existing.agent });
  }

  const summary = payload.summary !== undefined ? toNullableString(payload.summary) : existing.summary;
  await db.prepare("UPDATE claims SET status = ?, summary = ? WHERE id = ?").bind(statusVal, summary, claimId).run();

  const updated = await db.prepare("SELECT * FROM claims WHERE id = ?").bind(claimId).first<ClaimRow>();
  return out(200, { ok: true, claim: updated });
};

const opListClaims: Op = async (ctx, args) => {
  const claims = await queryActiveClaims(ctx.env.DB, toNullableString(args.lane));
  return out(200, { claims, count: claims.length });
};

// ---------------------------------------------------------------------------
// Notes: hypotheses and insights
// ---------------------------------------------------------------------------

function deserializeNote(row: NoteRow) {
  const ids = parseJsonSafe<unknown>(row.node_ids, []);
  return {
    id: row.id,
    kind: row.kind,
    lane: row.lane ?? null,
    title: row.title,
    body: row.body ?? null,
    state: row.state,
    state_reason: row.state_reason ?? null,
    state_by: row.state_by ?? null,
    author: row.author,
    node_ids: Array.isArray(ids) ? ids.map((x) => String(x)) : [],
    created_at: row.created_at,
    updated_at: row.updated_at,
  };
}

function parseNodeIds(raw: unknown): string[] | Outcome {
  if (raw === undefined || raw === null) return [];
  if (!Array.isArray(raw)) return out(400, { error: "invalid_node_ids", detail: "expected an array of node ids" });
  if (raw.length > LIMITS.maxNoteLinks) return out(400, { error: "too_many_node_ids", max: LIMITS.maxNoteLinks });
  const ids: string[] = [];
  for (const item of raw) {
    const s = toNullableString(item);
    if (!s || s.length > LIMITS.shortFieldChars) return out(400, { error: "invalid_node_ids" });
    if (!ids.includes(s)) ids.push(s);
  }
  return ids;
}

function validateNoteText(args: Args): Outcome | null {
  const long = tooLongField(args, {
    title: LIMITS.noteTitleChars,
    body: LIMITS.noteBodyChars,
    reason: LIMITS.reasonChars,
  });
  return long ? out(400, { error: "field_too_long", ...long }) : null;
}

const opCreateNote: Op = async (ctx, args) => {
  const principal = ctx.principal as AuthPrincipal;
  const kind = toNullableString(args.kind);
  const title = toNullableString(args.title);
  if (!kind || !title) return out(400, { error: "missing_required_fields", required: ["kind", "title"] });
  const states = NOTE_STATES[kind];
  if (!states) return out(400, { error: "invalid_kind", allowed: Object.keys(NOTE_STATES) });
  const tooLong = validateNoteText(args);
  if (tooLong) return tooLong;
  const lane = toNullableString(args.lane);
  if (lane && !LANE_RE.test(lane)) return out(400, { error: "invalid_lane" });
  const state = toNullableString(args.state) ?? "open";
  if (!states.includes(state)) return out(400, { error: "invalid_state", allowed: states });
  const nodeIds = parseNodeIds(args.node_ids);
  if (!Array.isArray(nodeIds)) return nodeIds;

  const id = randomToken("note_", 9);
  const now = nowIso();
  await ctx.env.DB
    .prepare(
      `INSERT INTO notes (id, kind, lane, title, body, state, state_reason, state_by, author, node_ids, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)`
    )
    .bind(id, kind, lane, title, toNullableString(args.body), state, principal.handle, JSON.stringify(nodeIds), now, now)
    .run();
  const row = await ctx.env.DB.prepare("SELECT * FROM notes WHERE id = ?").bind(id).first<NoteRow>();
  return out(201, { ok: true, note: row ? deserializeNote(row) : null });
};

const opListNotes: Op = async (ctx, args) => {
  const clauses: string[] = [];
  const params: unknown[] = [];
  for (const col of ["lane", "kind", "state", "author"]) {
    const val = toNullableString(args[col]);
    if (val) {
      clauses.push(`${col} = ?`);
      params.push(val);
    }
  }
  const limit = clampLimit(args.limit, 50, LIMITS.notesLimitMax);
  const offset = clampOffset(args.offset);
  const where = clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "";
  const res = await ctx.env.DB
    .prepare(`SELECT * FROM notes ${where} ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?`)
    .bind(...params, limit, offset)
    .all<NoteRow>();
  const notes = (res.results ?? []).map(deserializeNote);
  return out(200, {
    notes,
    count: notes.length,
    limit,
    offset,
    next_offset: notes.length === limit ? offset + limit : null,
  });
};

const opGetNote: Op = async (ctx, args) => {
  const id = toNullableString(args.id);
  if (!id) return out(400, { error: "missing_required_fields", required: ["id"] });
  const row = await ctx.env.DB.prepare("SELECT * FROM notes WHERE id = ?").bind(id).first<NoteRow>();
  if (!row) return out(404, { error: "not_found" });
  return out(200, { note: deserializeNote(row) });
};

const opUpdateNote: Op = async (ctx, args) => {
  const db = ctx.env.DB;
  const principal = ctx.principal as AuthPrincipal;
  const id = toNullableString(args.id);
  if (!id) return out(400, { error: "missing_required_fields", required: ["id"] });
  const row = await db.prepare("SELECT * FROM notes WHERE id = ?").bind(id).first<NoteRow>();
  if (!row) return out(404, { error: "not_found" });
  const tooLong = validateNoteText(args);
  if (tooLong) return tooLong;

  const isAdmin = principal.role === "admin";
  const own = row.author === principal.handle;
  const sets: string[] = [];
  const params: unknown[] = [];

  const contentFields = ["title", "body", "node_ids"].filter((f) => args[f] !== undefined);
  if (contentFields.length > 0) {
    if (!own && !isAdmin) return out(403, { error: "forbidden_note_owner", owner: row.author });
    if (args.title !== undefined) {
      const title = toNullableString(args.title);
      if (!title) return out(400, { error: "invalid_title" });
      sets.push("title = ?");
      params.push(title);
    }
    if (args.body !== undefined) {
      sets.push("body = ?");
      params.push(toNullableString(args.body));
    }
    if (args.node_ids !== undefined) {
      const ids = parseNodeIds(args.node_ids);
      if (!Array.isArray(ids)) return ids;
      sets.push("node_ids = ?");
      params.push(JSON.stringify(ids));
    }
  }
  if (args.state !== undefined) {
    const state = toNullableString(args.state);
    const allowed = NOTE_STATES[row.kind] ?? [];
    if (!state || !allowed.includes(state)) return out(400, { error: "invalid_state", allowed });
    if (!own && !isAdmin && principal.role === "provisional") {
      return out(403, { error: "forbidden_tier", detail: "changing another author's note needs a contributor token" });
    }
    sets.push("state = ?", "state_reason = ?", "state_by = ?");
    params.push(state, toNullableString(args.reason), principal.handle);
  }
  if (sets.length === 0) return out(400, { error: "nothing_to_update", fields: ["state", "reason", "title", "body", "node_ids"] });

  sets.push("updated_at = ?");
  params.push(nowIso(), id);
  await db.prepare(`UPDATE notes SET ${sets.join(", ")} WHERE id = ?`).bind(...params).run();
  const updated = await db.prepare("SELECT * FROM notes WHERE id = ?").bind(id).first<NoteRow>();
  return out(200, { ok: true, note: updated ? deserializeNote(updated) : null });
};

// ---------------------------------------------------------------------------
// Search (LIKE over trials and notes)
//
// Not FTS5: D1 cannot export a database that contains virtual tables, and export is
// the backup path. Every term must appear (AND); title hits rank first, then recency.
// One search reads every node and note row, so results are capped and edge-cached.
// ---------------------------------------------------------------------------

const NODE_SEARCH_TEXT =
  "(coalesce(title, '') || ' ' || coalesce(hypothesis, '') || ' ' || id || ' ' || run_id || ' ' || " +
  "coalesce(author, '') || ' ' || coalesce(agent_model, '') || ' ' || coalesce(codex_refs, ''))";
const NOTE_SEARCH_TEXT = "(title || ' ' || coalesce(body, '') || ' ' || author || ' ' || id)";

function searchTerms(q: string): string[] {
  return (q.match(/[\p{L}\p{N}_.:-]+/gu) ?? [])
    .map((t) => t.slice(0, 64))
    .filter((t, i, all) => all.indexOf(t) === i)
    .slice(0, 6);
}

function likePattern(term: string): string {
  return `%${term.replace(/[\\%_]/g, (c) => `\\${c}`)}%`;
}

function makeSnippet(text: string, terms: string[], width = 80): string {
  const lower = text.toLowerCase();
  let at = -1;
  let term = "";
  for (const t of terms) {
    const i = lower.indexOf(t.toLowerCase());
    if (i >= 0 && (at < 0 || i < at)) {
      at = i;
      term = t;
    }
  }
  if (at < 0) return text.slice(0, width * 2);
  const start = Math.max(0, at - width);
  const end = Math.min(text.length, at + term.length + width);
  const hit = text.slice(at, at + term.length);
  return `${start > 0 ? "…" : ""}${text.slice(start, at)}[${hit}]${text.slice(at + term.length, end)}${end < text.length ? "…" : ""}`;
}

const opSearch: Op = async (ctx, args) => {
  const q = toNullableString(args.q);
  if (!q) return out(400, { error: "missing_required_fields", required: ["q"] });
  if (q.length > LIMITS.queryChars) return out(400, { error: "field_too_long", field: "q", max: LIMITS.queryChars });
  const terms = searchTerms(q);
  if (terms.length === 0) return out(400, { error: "empty_query" });
  const lane = toNullableString(args.lane);
  const kind = toNullableString(args.kind);
  if (kind && kind !== "node" && kind !== "note") return out(400, { error: "invalid_kind", allowed: ["node", "note"] });
  const limit = clampLimit(args.limit, 20, LIMITS.searchLimitMax);
  const offset = Math.min(clampOffset(args.offset), LIMITS.searchOffsetMax);
  const patterns = terms.map(likePattern);

  const arm = (table: "nodes" | "notes"): { sql: string; params: unknown[] } => {
    const text = table === "nodes" ? NODE_SEARCH_TEXT : NOTE_SEARCH_TEXT;
    const where = patterns.map(() => `${text} LIKE ? ESCAPE '\\'`);
    const params: unknown[] = [patterns[0], ...patterns];
    if (lane) {
      where.push("lane = ?");
      params.push(lane);
    }
    const cols = table === "nodes"
      ? "'node' AS kind, id, lane, title, author, decision, state, verified, soft_min_eff, source_sha, NULL AS note_kind, created_at, created_at AS updated_at, coalesce(hypothesis, '') AS text"
      : "'note' AS kind, id, lane, title, author, NULL AS decision, state, 0 AS verified, NULL AS soft_min_eff, NULL AS source_sha, kind AS note_kind, created_at, updated_at, coalesce(body, '') AS text";
    return {
      sql: `SELECT ${cols}, (coalesce(title, '') LIKE ? ESCAPE '\\') AS title_hit FROM ${table} WHERE ${where.join(" AND ")}`,
      params,
    };
  };
  const arms = (kind ? [kind === "node" ? "nodes" : "notes"] : ["nodes", "notes"]).map((t) => arm(t as "nodes" | "notes"));
  const res = await ctx.env.DB
    .prepare(
      `SELECT * FROM (${arms.map((a) => a.sql).join(" UNION ALL ")})
       ORDER BY title_hit DESC, updated_at DESC, id DESC LIMIT ? OFFSET ?`
    )
    .bind(...arms.flatMap((a) => a.params), limit, offset)
    .all<Record<string, unknown>>();
  const rows = res.results ?? [];

  const results = rows.map((r) => {
    const title = (r.title as string | null) ?? null;
    const text = String(r.text ?? "");
    const inText = terms.some((t) => text.toLowerCase().includes(t.toLowerCase()));
    const snippet = makeSnippet(inText ? text : title ?? text, terms);
    const base = { kind: r.kind as string, id: r.id as string, lane: (r.lane as string | null) ?? null, title, author: (r.author as string | null) ?? null, snippet };
    if (r.kind === "node") {
      return {
        ...base,
        decision: r.decision as string,
        state: (r.state as string | null) ?? (r.verified ? "verified" : "unverified"),
        soft_min_eff: toFiniteNumber(r.soft_min_eff),
        has_source: Boolean(r.source_sha),
        created_at: r.created_at as string,
      };
    }
    return {
      ...base,
      note_kind: r.note_kind as string,
      state: r.state as string,
      created_at: r.created_at as string,
      updated_at: r.updated_at as string,
    };
  });
  return out(200, {
    q,
    terms,
    lane,
    kind,
    results,
    count: results.length,
    limit,
    offset,
    next_offset: rows.length === limit && offset + limit <= LIMITS.searchOffsetMax ? offset + limit : null,
  });
};

// ---------------------------------------------------------------------------
// Verification queue
// ---------------------------------------------------------------------------

function publicQueueFields(row: QueueRow) {
  const now = nowIso();
  return {
    node_id: row.node_id,
    lane: row.lane,
    status: row.status,
    available: row.status === "queued" || (row.status === "leased" && (row.lease_expires_at ?? "") <= now),
    reason: row.reason ?? null,
    attempts: Number(row.attempts ?? 0),
    leased_by: row.leased_by ?? null,
    lease_expires_at: row.lease_expires_at ?? null,
    result: row.result ?? null,
    result_metrics: parseJsonSafe<Record<string, unknown> | null>(row.result_json, null),
    verifier: row.verifier ?? null,
    created_at: row.created_at,
    updated_at: row.updated_at,
  };
}

function queueNodeSummary(n: NodeRecord) {
  return {
    id: n.id,
    lane: n.lane,
    run_id: n.run_id,
    trial_id: n.trial_id,
    title: n.title,
    author: n.author,
    author_tier: n.author_tier,
    agent_model: n.agent_model,
    decision: n.decision,
    soft_min_eff: n.soft_min_eff,
    dims: n.dims,
    state: n.state,
    has_source: n.has_source,
    code_sha: n.code_sha,
  };
}

const opVerifyQueue: Op = async (ctx, args) => {
  const status = toNullableString(args.status) ?? "open";
  const statuses = status === "open" ? ["queued", "leased"] : status === "all" ? ["queued", "leased", "done"] : [status];
  if (!statuses.every((s) => ["queued", "leased", "done"].includes(s))) {
    return out(400, { error: "invalid_status", allowed: ["open", "queued", "leased", "done", "all"] });
  }
  const lane = toNullableString(args.lane);
  const limit = clampLimit(args.limit, 50, LIMITS.queueLimitMax);
  const params: unknown[] = [...statuses];
  let laneClause = "";
  if (lane) {
    laneClause = " AND q.lane = ?";
    params.push(lane);
  }
  const res = await ctx.env.DB
    .prepare(
      `SELECT q.*, n.* , q.lane AS q_lane, q.created_at AS q_created_at, q.updated_at AS q_updated_at
       FROM verify_queue q JOIN nodes n ON n.id = q.node_id
       WHERE q.status IN (${placeholders(statuses.length)})${laneClause}
       ORDER BY q.created_at ASC, q.node_id ASC LIMIT ?`
    )
    .bind(...params, limit)
    .all<NodeRow & QueueRow & { q_lane: string; q_created_at: string; q_updated_at: string }>();
  const items = (res.results ?? []).map((r) => ({
    ...publicQueueFields({ ...r, lane: r.q_lane, created_at: r.q_created_at, updated_at: r.q_updated_at }),
    node: queueNodeSummary(deserializeNode(r)),
  }));
  return out(200, { items, count: items.length });
};

const opVerifyClaim: Op = async (ctx, args) => {
  const db = ctx.env.DB;
  const principal = ctx.principal as AuthPrincipal;
  const lane = toNullableString(args.lane);
  const nodeId = toNullableString(args.node_id);
  const ttlRaw = args.ttl_minutes === undefined ? LIMITS.leaseMinutesDefault : Number(args.ttl_minutes);
  if (!Number.isFinite(ttlRaw) || ttlRaw <= 0) return out(400, { error: "invalid_ttl_minutes" });
  const ttl = Math.min(ttlRaw, LIMITS.leaseMinutesMax);

  const nowMs = Date.now();
  const now = new Date(nowMs).toISOString();
  const expires = new Date(nowMs + ttl * 60_000).toISOString();
  const leaseToken = randomToken("vlease_", 24);
  const leaseHash = await sha256Hex(leaseToken);

  const pick: string[] = [
    "(q.status = 'queued' OR (q.status = 'leased' AND q.lease_expires_at <= ?))",
    "n.state NOT IN ('verified', 'retracted')",
    "n.source_sha IS NOT NULL",
  ];
  const pickParams: unknown[] = [now];
  if (principal.role !== "admin") {
    pick.push("n.author IS NOT ?");
    pickParams.push(principal.handle);
  }
  if (lane) {
    pick.push("q.lane = ?");
    pickParams.push(lane);
  }
  if (nodeId) {
    pick.push("q.node_id = ?");
    pickParams.push(nodeId);
  }
  // One statement: pick and lease atomically, so two verifiers never hold the same item.
  const leased = await db
    .prepare(
      `UPDATE verify_queue SET status = 'leased', lease_token_hash = ?, leased_by = ?, lease_expires_at = ?,
         attempts = attempts + 1, updated_at = ?
       WHERE node_id = (
         SELECT q.node_id FROM verify_queue q JOIN nodes n ON n.id = q.node_id
         WHERE ${pick.join(" AND ")}
         ORDER BY q.created_at ASC, q.node_id ASC LIMIT 1)
       AND (status = 'queued' OR (status = 'leased' AND lease_expires_at <= ?))
       RETURNING node_id, lane, lease_expires_at, attempts`
    )
    .bind(leaseHash, principal.handle, expires, now, ...pickParams, now)
    .first<{ node_id: string; lane: string; lease_expires_at: string; attempts: number }>();

  if (!leased) {
    if (nodeId) {
      const item = await db.prepare("SELECT * FROM verify_queue WHERE node_id = ?").bind(nodeId).first<QueueRow>();
      if (!item) return out(404, { error: "not_queued", node_id: nodeId });
      const node = await getNodeRow(db, nodeId);
      if (node && principal.role !== "admin" && node.author === principal.handle) {
        return out(403, { error: "forbidden_self_verification" });
      }
      return out(409, { error: "not_available", status: item.status, lease_expires_at: item.lease_expires_at });
    }
    return out(404, { error: "queue_empty", lane });
  }
  const row = await getNodeRow(db, leased.node_id);
  return out(200, {
    ok: true,
    lease: {
      node_id: leased.node_id,
      lane: leased.lane,
      lease_token: leaseToken,
      lease_expires_at: leased.lease_expires_at,
      attempts: Number(leased.attempts),
    },
    node: row ? deserializeNode(row) : null,
  });
};

const opVerifyResult: Op = async (ctx, args) => {
  const db = ctx.env.DB;
  const principal = ctx.principal as AuthPrincipal;
  const nodeId = toNullableString(args.node_id);
  const leaseToken = toNullableString(args.lease_token);
  if (!nodeId || !leaseToken || typeof args.passed !== "boolean") {
    return out(400, { error: "missing_required_fields", required: ["node_id", "lease_token", "passed (boolean)"] });
  }
  const note = toNullableString(args.note);
  if (note && note.length > LIMITS.reasonChars) return out(400, { error: "field_too_long", field: "note", max: LIMITS.reasonChars });
  const metricsJson = args.metrics === undefined || args.metrics === null ? null : JSON.stringify(args.metrics);
  if (metricsJson && metricsJson.length > 8 * 1024) return out(413, { error: "payload_too_large", field: "metrics", max_bytes: 8192 });

  const nodeRow = await getNodeRow(db, nodeId);
  if (!nodeRow) return out(404, { error: "not_found" });
  if (principal.role !== "admin" && nodeRow.author === principal.handle) {
    return out(403, { error: "forbidden_self_verification" });
  }

  const passed = args.passed === true;
  const now = nowIso();
  const leaseHash = await sha256Hex(leaseToken);
  const holderClause = principal.role === "admin" ? "" : " AND leased_by = ?";
  const holderParams = principal.role === "admin" ? [] : [principal.handle];
  const done = await db
    .prepare(
      `UPDATE verify_queue SET status = 'done', result = ?, result_json = ?, verifier = ?,
         lease_token_hash = NULL, updated_at = ?
       WHERE node_id = ? AND status = 'leased' AND lease_token_hash = ? AND lease_expires_at > ?${holderClause}
       RETURNING node_id`
    )
    .bind(passed ? "pass" : "fail", metricsJson, principal.handle, now, nodeId, leaseHash, now, ...holderParams)
    .first<{ node_id: string }>();

  if (!done) {
    const item = await db.prepare("SELECT * FROM verify_queue WHERE node_id = ?").bind(nodeId).first<QueueRow>();
    if (!item) return out(404, { error: "not_queued", node_id: nodeId });
    if (item.status !== "leased") return out(409, { error: "not_leased", status: item.status });
    if ((item.lease_expires_at ?? "") <= now) return out(409, { error: "lease_expired", lease_expires_at: item.lease_expires_at });
    return out(409, { error: "stale_lease", detail: "lease token does not match the current lease" });
  }

  const newState = passed ? "replicated" : "disputed";
  const reason = `independent rerun by ${principal.handle}: ${passed ? "pass" : "fail"}${note ? ` (${note})` : ""}`;
  await db
    .prepare(
      `UPDATE nodes SET state = ?, verified = 0, state_reason = ?, state_by = ?, state_at = ?
       WHERE id = ? AND state IN (${placeholders(RESULT_WRITABLE_STATES.length)})`
    )
    .bind(newState, reason.slice(0, LIMITS.reasonChars), principal.handle, now, nodeId, ...RESULT_WRITABLE_STATES)
    .run();
  const updated = await getNodeRow(db, nodeId);
  return out(200, { ok: true, node_id: nodeId, result: passed ? "pass" : "fail", node: updated ? deserializeNode(updated) : null });
};

// ---------------------------------------------------------------------------
// Registration and maintainer operations
// ---------------------------------------------------------------------------

const opRegister: Op = async (ctx, payload) => {
  const { env, request } = ctx;
  const handle = sanitizeHandle(payload.handle);
  if (!handle) return out(400, { error: "invalid_handle", detail: HANDLE_RULE });
  if (RESERVED_HANDLES.has(handle.toLowerCase())) return out(400, { error: "reserved_handle", handle });
  const long = tooLongField(payload, { agent: 100, model: 100, invite_code: 200 });
  if (long) return out(400, { error: "field_too_long", ...long });

  let role: Role = "provisional";
  let hourlyLimit = envLimit(env.FLYWHEEL_PROVISIONAL_HOURLY_WRITES, LIMITS.provisionalHourlyWrites);
  const invite = toNullableString(payload.invite_code);
  if (invite) {
    const expected = env.FLYWHEEL_INVITE_CODE?.trim();
    if (!expected || !timingSafeEqual(invite, expected)) return out(403, { error: "invalid_invite_code" });
    role = "contributor";
    hourlyLimit = LIMITS.invitedHourlyWrites;
  }

  // Caps before any write that could be abused: per hashed IP first, then global.
  const day = utcDay();
  const ip = request.headers.get("CF-Connecting-IP") ?? "unknown";
  const salt = env.FLYWHEEL_IP_SALT ?? "apin-flywheel";
  const ipHash = (await sha256Hex(`${salt}|${day}|${ip}`)).slice(0, 32);
  const perIp = envLimit(env.FLYWHEEL_REGISTRATIONS_PER_IP_PER_DAY, LIMITS.registrationsPerIpPerDay);
  const ipCount = await bumpCounter(env.DB, `reg-ip:${ipHash}`, day);
  if (ipCount > perIp) return out(429, { error: "registration_ip_cap", limit: perIp, retry_after: "tomorrow (UTC)" });
  const global = envLimit(env.FLYWHEEL_REGISTRATIONS_PER_DAY, LIMITS.registrationsPerDay);
  const globalCount = await bumpCounter(env.DB, "reg-global", day);
  if (globalCount > global) return out(429, { error: "registration_daily_cap", limit: global });

  const id = crypto.randomUUID();
  const token = randomToken("apfw_");
  const keyHash = await sha256Hex(token);
  const now = nowIso();
  // Atomic: a handle is taken if any key (even revoked) or any trial author already uses it.
  const res = await env.DB
    .prepare(
      `INSERT INTO api_keys (id, key_hash, handle, role, hourly_limit, revoked_at, created_at, agent, model)
       SELECT ?, ?, ?, ?, ?, NULL, ?, ?, ?
       WHERE NOT EXISTS (SELECT 1 FROM api_keys WHERE lower(handle) = lower(?))
         AND NOT EXISTS (SELECT 1 FROM nodes WHERE lower(author) = lower(?))`
    )
    .bind(
      id, keyHash, handle, role, hourlyLimit, now,
      toNullableString(payload.agent), toNullableString(payload.model), handle, handle
    )
    .run();
  if (changesOf(res) === 0) return out(409, { error: "handle_already_registered", handle });

  // Old counter buckets are no longer needed (hourly and daily buckets share this table).
  await env.DB.prepare("DELETE FROM rate_limits WHERE bucket < ?").bind(utcDay(-2)).run();

  return out(201, {
    ok: true,
    key_id: id,
    handle,
    role,
    tier: role,
    hourly_limit: hourlyLimit,
    daily_write_limit:
      role === "provisional"
        ? envLimit(env.FLYWHEEL_PROVISIONAL_DAILY_WRITES, LIMITS.provisionalDailyWrites)
        : null,
    token,
    note: "Save this token now; it is stored hashed and cannot be shown again.",
  });
};

const opAdminListKeys: Op = async (ctx, args) => {
  const role = toNullableString(args.role);
  const sql = `SELECT id, handle, role, hourly_limit, revoked_at, created_at, agent, model FROM api_keys${
    role ? " WHERE role = ?" : ""
  } ORDER BY created_at DESC`;
  const stmt = ctx.env.DB.prepare(sql);
  const res = await (role ? stmt.bind(role) : stmt).all<Omit<ApiKeyRow, "key_hash">>();
  return out(200, { keys: res.results ?? [], count: (res.results ?? []).length });
};

const opAdminMintKey: Op = async (ctx, payload) => {
  const handle = sanitizeHandle(payload.handle);
  if (!handle) return out(400, { error: "invalid_handle", detail: HANDLE_RULE });
  const role = toNullableString(payload.role) ?? "contributor";
  if (!VALID_ROLES.has(role)) return out(400, { error: "invalid_role", allowed: [...VALID_ROLES] });
  const limitRaw = payload.hourly_limit === undefined ? 120 : Number(payload.hourly_limit);
  if (!Number.isFinite(limitRaw) || limitRaw < 1 || limitRaw > 10000) {
    return out(400, { error: "invalid_hourly_limit" });
  }
  const hourlyLimit = Math.trunc(limitRaw);
  const id = crypto.randomUUID();
  const token = randomToken("apfw_");
  const keyHash = await sha256Hex(token);
  const now = nowIso();
  await ctx.env.DB
    .prepare(
      "INSERT INTO api_keys (id, key_hash, handle, role, hourly_limit, revoked_at, created_at) VALUES (?, ?, ?, ?, ?, NULL, ?)"
    )
    .bind(id, keyHash, handle, role, hourlyLimit, now)
    .run();
  return out(201, { ok: true, key_id: id, handle, role, hourly_limit: hourlyLimit, token, created_at: now });
};

const opAdminRevokeKey: Op = async (ctx, args) => {
  const keyId = toNullableString(args.id) ?? "";
  const existing = await ctx.env.DB.prepare("SELECT id FROM api_keys WHERE id = ?").bind(keyId).first<{ id: string }>();
  if (!existing) return out(404, { error: "not_found" });
  const now = nowIso();
  await ctx.env.DB.prepare("UPDATE api_keys SET revoked_at = ? WHERE id = ?").bind(now, keyId).run();
  return out(200, { ok: true, key_id: keyId, revoked_at: now });
};

const opAdminHandle: Op = async (ctx, args) => {
  const db = ctx.env.DB;
  const handle = toNullableString(args.handle) ?? "";
  const action = args.action;
  const keys = await db
    .prepare("SELECT id FROM api_keys WHERE lower(handle) = lower(?)")
    .bind(handle)
    .all<{ id: string }>();
  if ((keys.results ?? []).length === 0) return out(404, { error: "not_found", handle });
  const now = nowIso();
  if (action === "purge") {
    // Revoke every key of the handle and delete everything it posted, in one call.
    const revoked = await db
      .prepare("UPDATE api_keys SET revoked_at = ? WHERE lower(handle) = lower(?) AND revoked_at IS NULL")
      .bind(now, handle)
      .run();
    const owned = "SELECT id FROM nodes WHERE lower(author) = lower(?)";
    await db.prepare(`DELETE FROM node_parents WHERE child IN (${owned}) OR parent IN (${owned})`).bind(handle, handle).run();
    await db.prepare(`DELETE FROM verify_queue WHERE node_id IN (${owned})`).bind(handle).run();
    const nodes = await db.prepare("DELETE FROM nodes WHERE lower(author) = lower(?)").bind(handle).run();
    const notes = await db.prepare("DELETE FROM notes WHERE lower(author) = lower(?)").bind(handle).run();
    const claims = await db
      .prepare("DELETE FROM claims WHERE lower(agent) = lower(?) OR lower(agent) LIKE lower(?) ESCAPE '\\'")
      .bind(handle, `${handle.replace(/[\\%_]/g, (c) => `\\${c}`)}:%`)
      .run();
    const sources = await db
      .prepare(
        "DELETE FROM sources WHERE lower(uploaded_by) = lower(?) AND sha256 NOT IN (SELECT source_sha FROM nodes WHERE source_sha IS NOT NULL)"
      )
      .bind(handle)
      .run();
    // Lease results this handle posted on others' trials are reopened, not trusted.
    await db
      .prepare(
        `UPDATE nodes SET state = 'unverified', state_reason = NULL, state_by = NULL, state_at = NULL
         WHERE state IN ('replicated', 'disputed', 'needs-replication') AND lower(state_by) = lower(?)`
      )
      .bind(handle)
      .run();
    await db
      .prepare(
        `UPDATE verify_queue SET status = 'queued', result = NULL, result_json = NULL, verifier = NULL,
           lease_token_hash = NULL, leased_by = NULL, lease_expires_at = NULL, updated_at = ?
         WHERE lower(verifier) = lower(?) OR lower(leased_by) = lower(?)`
      )
      .bind(now, handle, handle)
      .run();
    return out(200, {
      ok: true,
      handle,
      revoked: changesOf(revoked),
      deleted: {
        nodes: changesOf(nodes),
        notes: changesOf(notes),
        claims: changesOf(claims),
        sources: changesOf(sources),
      },
    });
  }
  if (action === "revoke") {
    const res = await db
      .prepare("UPDATE api_keys SET revoked_at = ? WHERE lower(handle) = lower(?) AND revoked_at IS NULL")
      .bind(now, handle)
      .run();
    return out(200, { ok: true, handle, revoked: changesOf(res), revoked_at: now });
  }
  const res = await db
    .prepare(
      `UPDATE api_keys SET role = 'contributor', hourly_limit = MAX(hourly_limit, ?)
       WHERE lower(handle) = lower(?) AND role = 'provisional' AND revoked_at IS NULL`
    )
    .bind(LIMITS.promotedHourlyWrites, handle)
    .run();
  await db
    .prepare("UPDATE nodes SET author_tier = 'contributor' WHERE lower(author) = lower(?) AND author_tier = 'provisional'")
    .bind(handle)
    .run();
  return out(200, { ok: true, handle, role: "contributor", promoted: changesOf(res) });
};

const opAdminVerifyNode: Op = async (ctx, payload) => {
  const db = ctx.env.DB;
  const nodeId = toNullableString(payload.id) ?? "";
  const existing = await getNodeRow(db, nodeId);
  if (!existing) return out(404, { error: "not_found" });
  const verified = payload.verified === false || payload.verified === 0 ? 0 : 1;
  const pick = <T>(key: string, conv: (v: unknown) => T, fallback: T): T =>
    payload[key] !== undefined ? conv(payload[key]) : fallback;
  const stanPassed = pick("real_stan_passed", toFiniteInt, existing.real_stan_passed);
  const stanTotal = pick("real_stan_total", toFiniteInt, existing.real_stan_total);
  const stanEss = pick("real_ess_per_cost", toFiniteNumber, existing.real_ess_per_cost);
  const state = verified ? "verified" : (existing.state === "verified" ? "unverified" : existing.state ?? "unverified");
  const now = nowIso();
  await db
    .prepare(
      `UPDATE nodes SET verified = ?, state = ?, state_by = ?, state_at = ?,
         real_stan_passed = ?, real_stan_total = ?, real_ess_per_cost = ? WHERE id = ?`
    )
    .bind(verified, state, (ctx.principal as AuthPrincipal).handle, now, stanPassed, stanTotal, stanEss, nodeId)
    .run();
  if (verified) {
    await db
      .prepare("UPDATE verify_queue SET status = 'done', result = 'verified', lease_token_hash = NULL, updated_at = ? WHERE node_id = ? AND status <> 'done'")
      .bind(now, nodeId)
      .run();
  }
  const updated = await getNodeRow(db, nodeId);
  return out(200, { ok: true, node: updated ? deserializeNode(updated) : null });
};

const opAdminDeleteNode: Op = async (ctx, args) => {
  const db = ctx.env.DB;
  const nodeId = toNullableString(args.id) ?? "";
  const existing = await db.prepare("SELECT id FROM nodes WHERE id = ?").bind(nodeId).first<{ id: string }>();
  if (!existing) return out(404, { error: "not_found" });
  await db.prepare("DELETE FROM nodes WHERE id = ?").bind(nodeId).run();
  await db.prepare("DELETE FROM node_parents WHERE child = ? OR parent = ?").bind(nodeId, nodeId).run();
  await db.prepare("DELETE FROM verify_queue WHERE node_id = ?").bind(nodeId).run();
  return out(200, { ok: true, deleted: nodeId });
};

const opHealth: Op = async (ctx) => {
  const db = ctx.env.DB;
  const [totals, lanes] = await batchAll<Record<string, unknown>>(db, [
    db.prepare("SELECT COALESCE(SUM(trials), 0) AS n FROM run_stats"),
    db.prepare("SELECT lane FROM run_stats GROUP BY lane HAVING SUM(trials) > 0 ORDER BY lane LIMIT ?").bind(LIMITS.overviewLanes),
  ]);
  return out(200, {
    ok: true,
    nodes: Number(totals[0]?.n ?? 0),
    active_claims: await countActiveClaims(db),
    lanes: lanes.map((r) => String(r.lane)),
  });
};

/**
 * Kept for old clients, bounded: the newest LIMITS.dagNodes trials (oldest first) with the
 * edges among them, the frontier, active claims and the alignment points among those trials. `total` is
 * the number of trials (in the lane, with ?lane=) and `truncated` says whether some were left out.
 * Nodes carry the parsed `metrics` but not the raw `metrics_json`.
 */
const opDag: Op = async (ctx, args) => {
  const db = ctx.env.DB;
  const lane = toNullableString(args.lane);
  const res = await db
    .prepare(`SELECT * FROM nodes${lane ? " WHERE lane = ?" : ""} ORDER BY created_at DESC, id DESC LIMIT ?`)
    .bind(...(lane ? [lane] : []), LIMITS.dagNodes)
    .all<NodeRow>();
  const recent = (res.results ?? []).reverse().map(deserializeNode);
  const edges = await edgesAmong(db, recent);
  const lanes = lane ? [lane] : (await laneStats(db)).map((l) => l.lane);
  const total = await totalTrials(db, lane);
  return out(200, {
    nodes: recent.map(stripRaw),
    edges,
    frontier: await laneFrontiers(db, lanes),
    claims: await queryActiveClaims(db, lane),
    alignment: alignmentOf(recent),
    total,
    truncated: total > recent.length,
  });
};

// ---------------------------------------------------------------------------
// MCP
// ---------------------------------------------------------------------------

interface McpTool {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  write: boolean;
  op: Op;
}

const str = { type: "string" };
const LANE_PROP = { type: "string", description: "Lane id, e.g. A0, A1, A2, A3, A4." };

const MCP_TOOLS: McpTool[] = [
  {
    name: "flywheel_frontier",
    description: "Return the current best kept/verified apin research frontier node per lane.",
    inputSchema: {
      type: "object",
      properties: { lane: { type: "string", description: "Optional lane filter (e.g. A0, A1, A2, A3)." } },
      additionalProperties: false,
    },
    write: false,
    op: opFrontier,
  },
  {
    name: "flywheel_list_nodes",
    description:
      "List trials, newest first, with filters (lane, run_id, decision, state, verified, model, outcome kept|archived|discarded, stan, q) and paging (limit, offset).",
    inputSchema: {
      type: "object",
      properties: {
        lane: str,
        run_id: str,
        decision: str,
        state: str,
        verified: { type: "boolean" },
        model: str,
        outcome: { type: "string", enum: ["kept", "archived", "discarded"] },
        stan: { type: "boolean" },
        q: str,
        limit: { type: "integer", minimum: 1, maximum: LIMITS.nodesLimitMax },
        offset: { type: "integer", minimum: 0 },
      },
      additionalProperties: false,
    },
    write: false,
    op: opListNodes,
  },
  {
    name: "flywheel_get_node",
    description: "Read one trial with all its parents (incumbent/combines/fork), children and verification status.",
    inputSchema: { type: "object", required: ["id"], properties: { id: str }, additionalProperties: false },
    write: false,
    op: opGetNode,
  },
  {
    name: "flywheel_get_source",
    description: "Fetch the submission source a trial was scored with, to adopt or rerun it.",
    inputSchema: { type: "object", required: ["node_id"], properties: { node_id: str }, additionalProperties: false },
    write: false,
    op: opGetSource,
  },
  {
    name: "flywheel_search",
    description: "Full-text search over trials (hypothesis, title, run, author, technique refs) and notes.",
    inputSchema: {
      type: "object",
      required: ["q"],
      properties: {
        q: str,
        lane: LANE_PROP,
        kind: { type: "string", enum: ["node", "note"] },
        limit: { type: "integer", minimum: 1, maximum: LIMITS.searchLimitMax },
        offset: { type: "integer", minimum: 0 },
      },
      additionalProperties: false,
    },
    write: false,
    op: opSearch,
  },
  {
    name: "flywheel_list_notes",
    description: "List hypotheses and insights, newest first, filtered by lane, kind, state or author.",
    inputSchema: {
      type: "object",
      properties: {
        lane: LANE_PROP,
        kind: { type: "string", enum: ["hypothesis", "insight"] },
        state: str,
        author: str,
        limit: { type: "integer", minimum: 1, maximum: LIMITS.notesLimitMax },
        offset: { type: "integer", minimum: 0 },
      },
      additionalProperties: false,
    },
    write: false,
    op: opListNotes,
  },
  {
    name: "flywheel_verify_queue",
    description: "List kept trials waiting for an independent rerun.",
    inputSchema: {
      type: "object",
      properties: {
        lane: LANE_PROP,
        status: { type: "string", enum: ["open", "queued", "leased", "done", "all"] },
        limit: { type: "integer", minimum: 1, maximum: LIMITS.queueLimitMax },
      },
      additionalProperties: false,
    },
    write: false,
    op: opVerifyQueue,
  },
  {
    name: "flywheel_register",
    description: "Register a handle and receive a personal provisional token (shown once). No token needed.",
    inputSchema: {
      type: "object",
      required: ["handle"],
      properties: { handle: str, agent: str, model: str, invite_code: str },
      additionalProperties: false,
    },
    write: false,
    op: opRegister,
  },
  {
    name: "flywheel_claim_task",
    description: "Claim an apin research task with a TTL lease to prevent duplicate agent work.",
    inputSchema: {
      type: "object",
      required: ["task_id", "lane", "agent"],
      properties: {
        task_id: str,
        lane: str,
        agent: str,
        summary: str,
        ttl_minutes: { type: "number", default: 60 },
        force: { type: "boolean", default: false },
      },
      additionalProperties: false,
    },
    write: true,
    op: opCreateClaim,
  },
  {
    name: "flywheel_post_node",
    description:
      "Record an apin autoresearch trial. Optional `source` (the submission text, <= 256 KB) and `parent_ids` ([{id, relation: incumbent|combines|fork}]).",
    inputSchema: {
      type: "object",
      required: ["lane", "run_id", "decision"],
      properties: {
        id: str,
        parent_id: str,
        parent_ids: {
          type: "array",
          items: {
            type: "object",
            required: ["id"],
            properties: { id: str, relation: { type: "string", enum: [...PARENT_RELATIONS] } },
          },
        },
        lane: str,
        run_id: str,
        trial_id: str,
        agent_model: str,
        author: str,
        title: str,
        hypothesis: str,
        decision: str,
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
        codex_refs: { type: "array", items: str },
        code_sha: str,
        metrics_json: str,
        source: str,
        verified: { type: "boolean" },
      },
      additionalProperties: true,
    },
    write: true,
    op: opPostNode,
  },
  {
    name: "flywheel_set_node_state",
    description:
      "Change a trial's research state: authors may retract their own; contributors may flag another's as needs-replication or disputed (reason required); maintainers set anything.",
    inputSchema: {
      type: "object",
      required: ["id", "state"],
      properties: { id: str, state: { type: "string", enum: [...NODE_STATES] }, reason: str },
      additionalProperties: false,
    },
    write: true,
    op: opSetNodeState,
  },
  {
    name: "flywheel_add_note",
    description: "Record a hypothesis or an insight, optionally linked to trial ids.",
    inputSchema: {
      type: "object",
      required: ["kind", "title"],
      properties: {
        kind: { type: "string", enum: ["hypothesis", "insight"] },
        lane: LANE_PROP,
        title: str,
        body: str,
        state: str,
        node_ids: { type: "array", items: str },
      },
      additionalProperties: false,
    },
    write: true,
    op: opCreateNote,
  },
  {
    name: "flywheel_update_note",
    description:
      "Update a note. State (hypothesis: open|supported|refuted|needs-replication; insight: open|superseded|retracted) by its author or any contributor; title/body/node_ids by its author.",
    inputSchema: {
      type: "object",
      required: ["id"],
      properties: { id: str, state: str, reason: str, title: str, body: str, node_ids: { type: "array", items: str } },
      additionalProperties: false,
    },
    write: true,
    op: opUpdateNote,
  },
  {
    name: "flywheel_verify_claim",
    description: "Lease the next queued trial to rerun (never your own). Returns a lease_token needed to post the result.",
    inputSchema: {
      type: "object",
      properties: { lane: LANE_PROP, node_id: str, ttl_minutes: { type: "number", default: LIMITS.leaseMinutesDefault } },
      additionalProperties: false,
    },
    write: true,
    op: opVerifyClaim,
  },
  {
    name: "flywheel_verify_result",
    description: "Post the result of an independent rerun: passed=true sets the trial replicated, false sets it disputed.",
    inputSchema: {
      type: "object",
      required: ["node_id", "lease_token", "passed"],
      properties: { node_id: str, lease_token: str, passed: { type: "boolean" }, metrics: { type: "object" }, note: str },
      additionalProperties: false,
    },
    write: true,
    op: opVerifyResult,
  },
];

async function handleMcp(request: Request, env: Env): Promise<Response> {
  const parsed = await readJsonBody(request, LIMITS.nodeBodyBytes);
  if (!parsed.ok) {
    const code = parsed.outcome.status === 413 ? -32600 : -32700;
    const message = parsed.outcome.status === 413 ? "Request too large" : "Parse error";
    return jsonResponse({ jsonrpc: "2.0", id: null, error: { code, message } }, parsed.outcome.status);
  }
  const body = parsed.value;
  const rpcId = body.id ?? null;
  const method = typeof body.method === "string" ? body.method : "";
  const params = (body.params && typeof body.params === "object" ? body.params : {}) as Args;
  const rpcError = (code: number, message: string, status: number) =>
    jsonResponse({ jsonrpc: "2.0", id: rpcId, error: { code, message } }, status);

  if (body.jsonrpc !== "2.0" || !method) return rpcError(-32600, "Invalid Request", 400);

  if (method === "initialize") {
    return jsonResponse({
      jsonrpc: "2.0",
      id: rpcId,
      result: {
        protocolVersion: "2024-11-05",
        capabilities: { tools: {} },
        serverInfo: { name: "apin-flywheel", version: "0.2.0" },
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
      result: { tools: MCP_TOOLS.map(({ name, description, inputSchema }) => ({ name, description, inputSchema })) },
    });
  }
  if (method === "tools/call") {
    const toolName = typeof params.name === "string" ? params.name : "";
    const args = (params.arguments && typeof params.arguments === "object" ? params.arguments : {}) as Args;
    const tool = MCP_TOOLS.find((t) => t.name === toolName);
    if (!tool) return rpcError(-32602, `Unknown tool: ${toolName}`, 400);

    let principal: AuthPrincipal | null = null;
    if (tool.write) {
      principal = await authenticate(request, env);
      if (!principal) return rpcError(-32001, "Unauthorized: valid Bearer token required", 401);
      const limited = await consumeWrite(env, principal);
      if (limited) return rpcError(-32029, `Rate limit exceeded (${String(limited.body.limit)}/${String(limited.body.scope)})`, 429);
    }
    const outcome = await tool.op({ env, request, principal }, args);
    return jsonResponse({
      jsonrpc: "2.0",
      id: rpcId,
      result: {
        content: [{ type: "text", text: JSON.stringify(outcome.body, null, 2) }],
        structuredContent: outcome.body,
        isError: outcome.status >= 400,
      },
    });
  }
  return rpcError(-32601, `Method not found: ${method}`, 404);
}

// ---------------------------------------------------------------------------
// REST router
// ---------------------------------------------------------------------------

type Access = "public" | "write" | "admin";

interface Route {
  method: string;
  pattern: RegExp;
  /** Names for the pattern's capture groups, merged into the op's args. */
  params?: string[];
  access: Access;
  /** Max request body in bytes; absent means the route takes no body (args come from the query). */
  body?: number;
  allowEmptyBody?: boolean;
  op: Op;
}

const ROUTES: Route[] = [
  { method: "GET", pattern: /^\/api\/health$/, access: "public", op: opHealth },
  { method: "GET", pattern: /^\/api\/dag$/, access: "public", op: opDag },
  { method: "GET", pattern: /^\/api\/frontier$/, access: "public", op: opFrontier },
  { method: "GET", pattern: /^\/api\/overview$/, access: "public", op: opOverview },
  { method: "GET", pattern: /^\/api\/runs$/, access: "public", op: opRuns },
  { method: "GET", pattern: /^\/api\/nodes$/, access: "public", op: opListNodes },
  { method: "GET", pattern: /^\/api\/nodes\/([^/]+)\/source$/, params: ["node_id"], access: "public", op: opGetSource },
  { method: "GET", pattern: /^\/api\/nodes\/([^/]+)$/, params: ["id"], access: "public", op: opGetNode },
  { method: "GET", pattern: /^\/api\/claims$/, access: "public", op: opListClaims },
  { method: "GET", pattern: /^\/api\/search$/, access: "public", op: opSearch },
  { method: "GET", pattern: /^\/api\/notes$/, access: "public", op: opListNotes },
  { method: "GET", pattern: /^\/api\/notes\/([^/]+)$/, params: ["id"], access: "public", op: opGetNote },
  { method: "GET", pattern: /^\/api\/verify\/queue$/, access: "public", op: opVerifyQueue },
  { method: "POST", pattern: /^\/api\/register$/, access: "public", body: LIMITS.registerBodyBytes, op: opRegister },

  { method: "POST", pattern: /^\/api\/nodes$/, access: "write", body: LIMITS.nodeBodyBytes, op: opPostNode },
  {
    method: "POST",
    pattern: /^\/api\/nodes\/([^/]+)\/state$/,
    params: ["id"],
    access: "write",
    body: LIMITS.smallBodyBytes,
    op: opSetNodeState,
  },
  { method: "POST", pattern: /^\/api\/claims$/, access: "write", body: LIMITS.smallBodyBytes, op: opCreateClaim },
  {
    method: "PATCH",
    pattern: /^\/api\/claims\/([^/]+)$/,
    params: ["id"],
    access: "write",
    body: LIMITS.smallBodyBytes,
    op: opPatchClaim,
  },
  { method: "POST", pattern: /^\/api\/notes$/, access: "write", body: LIMITS.noteBodyBytes, op: opCreateNote },
  {
    method: "PATCH",
    pattern: /^\/api\/notes\/([^/]+)$/,
    params: ["id"],
    access: "write",
    body: LIMITS.noteBodyBytes,
    op: opUpdateNote,
  },
  {
    method: "POST",
    pattern: /^\/api\/verify\/claim$/,
    access: "write",
    body: LIMITS.smallBodyBytes,
    allowEmptyBody: true,
    op: opVerifyClaim,
  },
  {
    method: "POST",
    pattern: /^\/api\/verify\/([^/]+)\/result$/,
    params: ["node_id"],
    access: "write",
    body: LIMITS.verifyResultBodyBytes,
    op: opVerifyResult,
  },

  { method: "GET", pattern: /^\/api\/admin\/keys$/, access: "admin", op: opAdminListKeys },
  { method: "POST", pattern: /^\/api\/admin\/keys$/, access: "admin", body: LIMITS.smallBodyBytes, op: opAdminMintKey },
  { method: "DELETE", pattern: /^\/api\/admin\/keys\/([^/]+)$/, params: ["id"], access: "admin", op: opAdminRevokeKey },
  {
    method: "POST",
    pattern: /^\/api\/admin\/handles\/([^/]+)\/(promote|revoke|purge)$/,
    params: ["handle", "action"],
    access: "admin",
    body: LIMITS.smallBodyBytes,
    allowEmptyBody: true,
    op: opAdminHandle,
  },
  {
    method: "POST",
    pattern: /^\/api\/admin\/nodes\/([^/]+)\/verify$/,
    params: ["id"],
    access: "admin",
    body: LIMITS.smallBodyBytes,
    allowEmptyBody: true,
    op: opAdminVerifyNode,
  },
  { method: "DELETE", pattern: /^\/api\/admin\/nodes\/([^/]+)$/, params: ["id"], access: "admin", op: opAdminDeleteNode },
];

function safeDecode(raw: string): string | null {
  try {
    return decodeURIComponent(raw);
  } catch {
    return null;
  }
}

async function dispatch(request: Request, env: Env, url: URL, method: string): Promise<Response | null> {
  for (const route of ROUTES) {
    if (route.method !== method) continue;
    const m = route.pattern.exec(url.pathname);
    if (!m) continue;

    const pathArgs: Args = {};
    for (let i = 0; i < (route.params ?? []).length; i += 1) {
      const decoded = safeDecode(m[i + 1]);
      if (decoded === null) return jsonResponse({ error: "invalid_path" }, 400);
      pathArgs[(route.params as string[])[i]] = decoded;
    }

    // Cheap checks first: declared body size, then auth, then the write budget, then the body.
    if (route.body !== undefined) {
      const declared = request.headers.get("Content-Length");
      if (declared !== null && Number(declared) > route.body) {
        return jsonResponse({ error: "payload_too_large", max_bytes: route.body }, 413);
      }
    }
    let principal: AuthPrincipal | null = null;
    if (route.access !== "public") {
      principal = await authenticate(request, env);
      if (!principal) return jsonResponse({ error: "unauthorized" }, 401);
      if (route.access === "admin" && principal.role !== "admin") {
        return jsonResponse({ error: "forbidden_admin_only" }, 403);
      }
      if (route.access === "write") {
        const limited = await consumeWrite(env, principal);
        if (limited) return jsonResponse(limited.body, limited.status);
      }
    }

    let args: Args;
    if (route.body !== undefined) {
      const parsed = await readJsonBody(request, route.body, route.allowEmptyBody ?? false);
      if (!parsed.ok) return jsonResponse(parsed.outcome.body, parsed.outcome.status);
      args = parsed.value;
    } else {
      args = Object.fromEntries(url.searchParams.entries());
    }
    const outcome = await route.op({ env, request, principal }, { ...args, ...pathArgs });
    return jsonResponse(outcome.body, outcome.status);
  }
  return null;
}

export async function handleRequest(
  request: Request,
  env: Env,
  ctx?: ExecutionContextLike
): Promise<Response> {
  const url = new URL(request.url);
  const { pathname } = url;
  const method = request.method.toUpperCase();

  if (!pathname.startsWith("/api/")) {
    if (env.ASSETS) return env.ASSETS.fetch(request);
    return jsonResponse({ error: "not_found", path: pathname }, 404);
  }
  if (method === "OPTIONS") {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }

  const cacheKey = method === "GET" ? canonicalPublicCacheKey(url) : null;
  const edgeCache = cacheKey ? getEdgeCache(env) : null;
  if (cacheKey && edgeCache) {
    const cached = await edgeCache.match(cacheKey);
    if (cached) return cached;
  }

  const finalizePublicRead = (resp: Response): Response => {
    if (!cacheKey || resp.status !== 200) return resp;
    const headers = new Headers(resp.headers);
    headers.set(
      "Cache-Control",
      `public, max-age=${PUBLIC_READ_CACHE_SECONDS}, s-maxage=${PUBLIC_READ_CACHE_SECONDS}`
    );
    const cachedResp = new Response(resp.body, { status: resp.status, headers });
    if (edgeCache) {
      const putPromise = edgeCache.put(cacheKey, cachedResp.clone());
      if (ctx) {
        ctx.waitUntil(putPromise);
      } else {
        putPromise.catch(() => {});
      }
    }
    return cachedResp;
  };

  try {
    if (pathname === "/api/mcp" && method === "POST") {
      return await handleMcp(request, env);
    }
    const resp = await dispatch(request, env, url, method);
    if (resp) return finalizePublicRead(resp);
    return jsonResponse({ error: "not_found", path: pathname }, 404);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return jsonResponse({ error: "internal_error", message }, 500);
  }
}

export default {
  fetch(request: Request, env: Env, ctx?: ExecutionContextLike): Promise<Response> {
    return handleRequest(request, env, ctx);
  },
};
