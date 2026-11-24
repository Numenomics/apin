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
  /** Local development only: "1" makes token-less requests maintainer when FLYWHEEL_API_KEY is unset. */
  FLYWHEEL_DEV_OPEN_ADMIN?: string;
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
  dagNodes: 300,
  alignmentPoints: 300,
  overviewLanes: 32,
  overviewModels: 100,
  overviewClaims: 100,
  claimsList: 200,
  nodeChildren: 200,
  nodeNotes: 20,
  // D1 rows read per request (the free plan's daily read quota).
  countCap: 1000,
  searchWindow: 2000,
  searchOffsetMax: 500,
  provisionalTrialsPerDay: 100,
  provisionalTrialsPerDayGlobal: 1000,
  chartPoints: 1000,
  chartTitleChars: 140,
  yardstickChars: 120,
} as const;

/**
 * A later trial moves the frontier when its speedup is at least this many times the latest
 * step's: a difference under 5% is noise (an A/A calibration of the grader). Not exported:
 * workerd refuses a module that exports a bare number.
 */
const STEP_MARGIN = 1.05;
/** The factor the SQL compares with: STEP_MARGIN less a rounding allowance, so 1.26 over 1.2 counts. */
const STEP_FACTOR = STEP_MARGIN - 1e-12;

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
  speedup?: number | null;
  yardstick?: string | null;
  frontier_step?: number | null;
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
  /** Efficiency over the lane's starting code (1 is the starting code). Null when failed or exploratory. */
  speedup: number | null;
  /** The grading setup's id. Null for an exploratory trial. */
  yardstick: string | null;
  /** Graded: yardstick is not null. Only standard trials are on the chart and can move a frontier. */
  standard: boolean;
  /** Moved its lane's frontier (see "Frontier steps"). */
  frontier_step: boolean;
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
/** Identifiers reach shells and file paths on clients: letters, digits and `_.:-` only. */
const ID_RE = /^(?=.*[A-Za-z0-9])[A-Za-z0-9_.:-]{1,200}$/;
const LANE_RE = /^[A-Za-z0-9]{1,8}$/;
const ID_RULE = "letters, digits and _.:- only, at least one letter or digit, at most 200 characters";
const LANE_RULE = "letters and digits only, at most 8 characters";

/** 400 for the first present field that is not a safe identifier (lane fields use the lane rule). */
function badIdentifier(args: Args, idFields: string[], laneFields: string[] = []): Outcome | null {
  for (const f of laneFields) {
    const v = toNullableString(args[f]);
    if (v !== null && !LANE_RE.test(v)) return out(400, { error: "invalid_identifier", field: f, rule: LANE_RULE });
  }
  for (const f of idFields) {
    const v = toNullableString(args[f]);
    if (v !== null && !ID_RE.test(v)) return out(400, { error: "invalid_identifier", field: f, rule: ID_RULE });
  }
  return null;
}

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

/**
 * The effective query arguments: one value per name, the last occurrence winning. The
 * router and the cache key both use this, so a repeated parameter cannot make them disagree.
 */
function queryArgs(url: URL): Record<string, string> {
  return Object.fromEntries(url.searchParams.entries());
}

function canonicalQuery(query: Record<string, string>, allowed: readonly string[]): string {
  const params = new URLSearchParams();
  for (const key of allowed) {
    const val = query[key];
    if (val !== undefined && val !== "") params.set(key, val);
  }
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

function canonicalPublicCacheKey(url: URL, query: Record<string, string>): Request | null {
  const { origin, pathname } = url;
  const key = (qs: string) => new Request(`${origin}${pathname}${qs}`, { method: "GET" });
  if (pathname === "/api/health") return key("");
  if (pathname === "/api/frontier") return key(canonicalQuery(query, ["lane"]));
  if (pathname === "/api/dag") return key(canonicalQuery(query, ["lane"]));
  if (pathname === "/api/claims") return key(canonicalQuery(query, ["lane"]));
  if (pathname === "/api/overview") return key("");
  if (pathname === "/api/runs") return key(canonicalQuery(query, ["lane", "limit", "offset"]));
  if (pathname === "/api/nodes") {
    return key(canonicalQuery(query, [
      "lane", "run_id", "decision", "verified", "state", "model", "outcome", "stan", "step", "q", "limit", "offset",
    ]));
  }
  if (/^\/api\/lanes\/[^/]+\/chart$/.test(pathname)) return key(canonicalQuery(query, ["after", "limit"]));
  if (pathname.startsWith("/api/nodes/") && pathname.length > "/api/nodes/".length) return key("");
  if (pathname === "/api/search") {
    return key(canonicalQuery(query, ["q", "lane", "kind", "limit", "offset"]));
  }
  if (pathname === "/api/notes") {
    return key(canonicalQuery(query, ["lane", "kind", "state", "author", "limit", "offset"]));
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

const OFFSET_MAX = 10_000;

function clampOffset(raw: unknown): number {
  const parsed = Math.trunc(Number(raw));
  return Number.isFinite(parsed) && parsed > 0 ? Math.min(parsed, OFFSET_MAX) : 0;
}

/** The next page's offset, or null when this is the last page or the next one is past the cap. */
function nextOffset(offset: number, limit: number, returned: number, total: number | null, cap = OFFSET_MAX): number | null {
  const next = offset + limit;
  if (returned < limit || next > cap) return null;
  return total !== null && next >= total ? null : next;
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

  // Fail closed: without FLYWHEEL_API_KEY nobody is maintainer, unless local development
  // opts in explicitly with FLYWHEEL_DEV_OPEN_ADMIN=1 (never set it on a deployed Worker).
  if (!configuredKey && toBool(String(env.FLYWHEEL_DEV_OPEN_ADMIN ?? "").trim().toLowerCase())) {
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
    speedup: toFiniteNumber(row.speedup),
    yardstick: row.yardstick ?? null,
    standard: row.yardstick !== null && row.yardstick !== undefined,
    frontier_step: Number(row.frontier_step) === 1,
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
/**
 * A trial that takes part in the step walk: graded, with a speedup, not retracted or disputed,
 * and kept or verified. `t` is a table alias with its dot, or "" for the bare table. Must match
 * the triggers, the BACKFILL block and the condition of idx_nodes_eligible in migration 0005.
 */
const sqlStepEligible = (t = "") =>
  `(${t}yardstick IS NOT NULL AND ${t}speedup > 0 AND ${t}state NOT IN (${sqlList(FRONTIER_EXCLUDED_STATES)})
    AND (${t}verified = 1 OR ${t}decision IN (${sqlList([...KEPT_DECISIONS])})))`;

type NodeWithoutRaw = Omit<NodeRecord, "metrics_json">;

function stripRaw(n: NodeRecord): NodeWithoutRaw {
  const { metrics_json: _raw, ...rest } = n;
  return rest;
}

/** Runs statements as one D1 batch (one transaction); sequentially where batch() is missing. */
async function runBatch(db: D1DatabaseLike, stmts: D1PreparedStatement[]): Promise<D1Result[]> {
  if (db.batch) return db.batch(stmts);
  const results: D1Result[] = [];
  for (const s of stmts) results.push(await s.run());
  return results;
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
  /** Frontier steps in the lane, the first one included (lane_stats.steps, migration 0004). */
  steps: number;
}

/** Lanes that have trials, from lane_stats (at most LIMITS.overviewLanes, busiest first, then sorted by name). */
async function laneStats(db: D1DatabaseLike, lane?: string | null): Promise<LaneStat[]> {
  const res = await db
    .prepare(
      `SELECT lane, trials, kept, runs, latest_at, steps FROM lane_stats
       WHERE trials > 0${lane ? " AND lane = ?" : ""} ORDER BY trials DESC, lane LIMIT ?`
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
      steps: Number(r.steps ?? 0),
    }))
    .sort((a, b) => a.lane.localeCompare(b.lane));
}

/**
 * Each lane's frontier, its latest step: one row of idx_nodes_steps per lane. Only trials on the
 * lane's current yardstick are steps, so this is the best on it. No step, no frontier.
 */
async function laneFrontiers(db: D1DatabaseLike, lanes: string[]): Promise<Record<string, NodeRecord>> {
  const rows = await batchAll<NodeRow>(
    db,
    lanes.map((l) =>
      db
        .prepare(
          `SELECT * FROM nodes INDEXED BY idx_nodes_steps WHERE lane = ? AND frontier_step = 1
           ORDER BY created_at DESC, id DESC LIMIT 1`
        )
        .bind(l)
    )
  );
  const frontier: Record<string, NodeRecord> = {};
  lanes.forEach((l, i) => {
    if (rows[i]?.[0]) frontier[l] = deserializeNode(rows[i][0]);
  });
  const sorted: Record<string, NodeRecord> = {};
  for (const l of Object.keys(frontier).sort()) sorted[l] = frontier[l];
  return sorted;
}

// ---------------------------------------------------------------------------
// Frontier steps
//
// Speed-ups are comparable only within one yardstick. Each lane has one current yardstick
// (lane_stats.yardstick): the one whose first eligible trial (sqlStepEligible) is the newest.
// Walk the lane's eligible trials on it in (created_at, id) order. The first is a step. A later
// one is a step when its speedup is at least STEP_MARGIN times the latest step's. A trial under
// that margin is not a step and does not raise the bar. A trial on another yardstick is never a
// step. nodes.frontier_step stores the result, and the lane frontier is the latest step.
//
// Every flag a write changes commits in the write's own batch, computed in SQL from the rows
// as they are inside that transaction, so overlapping requests cannot leave stale flags.
//   * A new trial: stepSelfStmt sets its own flag from the latest step before it. When it is on
//     the current yardstick, is not that yardstick's first eligible trial and no step follows it
//     (the newest trial, the usual post), nothing else can change: a step raises the bar, so a
//     later trial that was not a step still is not. Otherwise the insert trigger of migration
//     0005 marks the lane dirty. A trial posted from a stale checkout, on an older yardstick
//     that already has an eligible trial, changes nothing.
//   * Any other change: the triggers of migration 0005 mark the lane dirty when other flags or
//     the current yardstick may move (the trial was a step, is eligible now and changed its
//     position, speedup or yardstick, or was the first eligible trial of its yardstick).
//   * settleStepsStmts, last in every batch that writes trials, sets each dirty lane's current
//     yardstick, walks the lane again from its first trial and clears the mark. That reads the
//     lane's eligible trials on the yardstick about once: the rare path.
// ---------------------------------------------------------------------------

/**
 * Sets one trial's own flag from the latest step before it. Earlier flags never depend on
 * later trials. Correct for a new trial on a lane that is not dirty, and a no-op for an
 * unchanged one. A dirty lane is walked again by settleStepsStmts in the same batch.
 */
function stepSelfStmt(db: D1DatabaseLike, id: string): D1PreparedStatement {
  return db
    .prepare(
      `UPDATE nodes SET frontier_step = CASE WHEN ${sqlStepEligible()}
         AND yardstick = (SELECT l.yardstick FROM lane_stats AS l WHERE l.lane = nodes.lane)
         AND coalesce(speedup >= ${STEP_FACTOR} * (
           SELECT p.speedup FROM nodes AS p INDEXED BY idx_nodes_steps
           WHERE p.lane = nodes.lane AND p.frontier_step = 1 AND (p.created_at, p.id) < (nodes.created_at, nodes.id)
           ORDER BY p.created_at DESC, p.id DESC LIMIT 1), 1)
         THEN 1 ELSE 0 END
       WHERE id = ?`
    )
    .bind(id);
}

/**
 * Sets each dirty lane's current yardstick. Lists the lane's yardsticks that have an eligible
 * trial (one seek of idx_nodes_eligible each), takes the first eligible trial of each (one more
 * seek) and keeps the yardstick whose first trial is the newest. When the yardstick changes it
 * counts the lane's graded trials on the new one again (lane_stats.standard). Reads no row when
 * no lane is dirty. Must equal the first recompute statement of migration 0005's BACKFILL block.
 */
const SQL_SETTLE_YARDSTICK = `WITH RECURSIVE ys(lane, yardstick) AS (
    SELECT lane, '' FROM lane_stats INDEXED BY idx_lane_stats_dirty WHERE steps_dirty = 1
    UNION ALL
    SELECT ys.lane, (
      SELECT e.yardstick FROM nodes AS e INDEXED BY idx_nodes_eligible
      WHERE e.lane = ys.lane AND e.yardstick > ys.yardstick
        AND ${sqlStepEligible("e.")}
      ORDER BY e.yardstick LIMIT 1)
    FROM ys WHERE ys.yardstick IS NOT NULL
  ),
  cur(lane, yardstick) AS MATERIALIZED (
    SELECT d.lane, (
      SELECT ys.yardstick FROM ys JOIN nodes AS f ON f.id = (
        SELECT e.id FROM nodes AS e INDEXED BY idx_nodes_eligible
        WHERE e.lane = ys.lane AND e.yardstick = ys.yardstick
          AND ${sqlStepEligible("e.")}
        ORDER BY e.created_at, e.id LIMIT 1)
      WHERE ys.lane = d.lane ORDER BY f.created_at DESC, f.id DESC LIMIT 1)
    FROM lane_stats AS d INDEXED BY idx_lane_stats_dirty WHERE d.steps_dirty = 1
  )
UPDATE lane_stats SET yardstick = cur.yardstick,
  standard = (SELECT COUNT(*) FROM nodes AS g INDEXED BY idx_nodes_lane_graded WHERE g.lane = cur.lane AND g.yardstick = cur.yardstick)
FROM cur WHERE lane_stats.lane = cur.lane AND lane_stats.yardstick IS NOT cur.yardstick`;

/**
 * Walks every dirty lane from step to step on its current yardstick and toggles each flag that
 * differs from the walk. The walk starts before the lane's first trial with a bar of 0. From
 * each step it seeks the first eligible trial on the yardstick after it whose speedup clears the
 * bar, so the searches together read each eligible trial of the yardstick about once. A marked
 * trial the walk skipped, on any yardstick, is cleared. Reads no row when no lane is dirty. Must
 * equal the second recompute statement of migration 0005's BACKFILL block.
 */
const SQL_SETTLE_STEPS = `UPDATE nodes SET frontier_step = 1 - frontier_step WHERE id IN (
  WITH RECURSIVE walk(lane, yardstick, created_at, id, speedup) AS (
    SELECT lane, yardstick, '', '', 0 FROM lane_stats INDEXED BY idx_lane_stats_dirty WHERE steps_dirty = 1
    UNION ALL
    SELECT n.lane, w.yardstick, n.created_at, n.id, n.speedup FROM walk AS w JOIN nodes AS n ON n.id = (
      SELECT p.id FROM nodes AS p INDEXED BY idx_nodes_eligible
      WHERE p.lane = w.lane AND p.yardstick = w.yardstick AND (p.created_at, p.id) > (w.created_at, w.id)
        AND p.speedup >= w.speedup * ${STEP_FACTOR}
        AND ${sqlStepEligible("p.")}
      ORDER BY p.created_at, p.id LIMIT 1)
  )
  SELECT n.id FROM walk AS w JOIN nodes AS n ON n.id = w.id WHERE n.frontier_step = 0
  UNION ALL
  SELECT n.id FROM lane_stats AS d INDEXED BY idx_lane_stats_dirty CROSS JOIN nodes AS n INDEXED BY idx_nodes_steps
  WHERE d.steps_dirty = 1 AND n.lane = d.lane AND n.frontier_step = 1 AND n.id NOT IN (SELECT id FROM walk))`;

function settleStepsStmts(db: D1DatabaseLike): D1PreparedStatement[] {
  return [
    db.prepare(SQL_SETTLE_YARDSTICK),
    db.prepare(SQL_SETTLE_STEPS),
    db.prepare("UPDATE lane_stats INDEXED BY idx_lane_stats_dirty SET steps_dirty = 0 WHERE steps_dirty = 1"),
  ];
}

interface StanRow {
  id: string;
  run_id: string;
  trial_id: string | null;
  real_stan_passed: number | null;
  real_stan_total: number;
}

/** The lane's best real-Stan pass share, newest on ties: one row of idx_nodes_stan_share (exact, no window). */
function laneStanStmt(db: D1DatabaseLike, lane: string): D1PreparedStatement {
  return db
    .prepare(
      `SELECT id, run_id, trial_id, real_stan_passed, real_stan_total FROM nodes INDEXED BY idx_nodes_stan_share
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
    .prepare(`SELECT COALESCE(SUM(trials), 0) AS n FROM lane_stats${lane ? " WHERE lane = ?" : ""}`)
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

/** SQL for an outcome; the same predicates define the partial outcome indexes (migration 0003). */
const OUTCOME_SQL: Record<string, string> = {
  kept: `decision IN (${SQL_OUTCOME_KEPT})`,
  archived: "decision = 'archive_only'",
  discarded: `decision NOT IN (${SQL_OUTCOME_KEPT}, 'archive_only')`,
};

/**
 * Index-served filter groups for /api/nodes, tried in order: the first group whose filters
 * are all present drives the read through its index; every other filter is "remaining".
 */
const DRIVING_INDEXES: { keys: string[]; index: (outcome: string | null) => string }[] = [
  // Steps are few, so any other filter (q included) runs over all of them, not over a window of trials.
  { keys: ["lane", "step"], index: () => "idx_nodes_steps" },
  { keys: ["step"], index: () => "idx_nodes_steps_created" },
  { keys: ["lane", "run_id"], index: () => "idx_nodes_run_created" },
  { keys: ["run_id"], index: () => "idx_nodes_runid_created" },
  { keys: ["lane", "outcome"], index: (o) => `idx_nodes_lane_${o}` },
  { keys: ["lane", "stan"], index: () => "idx_nodes_stan_lane" },
  { keys: ["lane"], index: () => "idx_nodes_lane_newest" },
  { keys: ["outcome"], index: (o) => `idx_nodes_${o}_created` },
  { keys: ["stan"], index: () => "idx_nodes_stan_created" },
  { keys: ["model"], index: () => "idx_nodes_model_newest" },
  { keys: [], index: () => "idx_nodes_created" },
];

interface NodeQueryPlan {
  /** FROM clause: the driving index, wrapped in a newest-first window when filters remain. */
  from: string;
  fromParams: unknown[];
  where: string;
  whereParams: unknown[];
  windowed: boolean;
  /** lane_stats / run_stats column when the count is exact from the counters. */
  statCount: string | null;
  lane: string | null;
  runId: string | null;
}

/**
 * Plans an /api/nodes read whose D1 cost is bounded: one filter group is served by its index;
 * any other filter (decision, state, verified, q, or a second group) is applied inside the
 * newest LIMITS.searchWindow trials of that index, so a sparse or empty filter never scans
 * the table.
 */
function planNodeQuery(args: Args): NodeQueryPlan | Outcome {
  const clauses = new Map<string, { sql: string; params: unknown[] }>();
  for (const [key, col] of [["lane", "lane"], ["run_id", "run_id"], ["decision", "decision"], ["state", "state"], ["model", "agent_model"]]) {
    const val = toNullableString(args[key]);
    if (val) clauses.set(key, { sql: `${col} = ?`, params: [val] });
  }
  const verified = args.verified;
  if (verified !== undefined && verified !== null && verified !== "") {
    clauses.set("verified", { sql: "verified = ?", params: [toBool(String(verified).toLowerCase()) ? 1 : 0] });
  }
  const outcome = toNullableString(args.outcome);
  if (outcome) {
    if (!NODE_OUTCOMES.has(outcome)) return out(400, { error: "invalid_outcome", allowed: [...NODE_OUTCOMES] });
    clauses.set("outcome", { sql: OUTCOME_SQL[outcome], params: [] });
  }
  if (toBool(args.stan)) clauses.set("stan", { sql: "real_stan_total > 0", params: [] });
  if (toBool(args.step)) clauses.set("step", { sql: "frontier_step = 1", params: [] });
  const q = toNullableString(args.q);
  if (q) {
    if (q.length > LIMITS.queryChars) return out(400, { error: "field_too_long", field: "q", max: LIMITS.queryChars });
    const terms = searchTerms(q);
    if (terms.length === 0) return out(400, { error: "empty_query" });
    clauses.set("q", {
      sql: terms.map(() => `${NODE_SEARCH_TEXT} LIKE ? ESCAPE '\\'`).join(" AND "),
      params: terms.map(likePattern),
    });
  }

  const driving = DRIVING_INDEXES.find((d) => d.keys.every((k) => clauses.has(k))) as (typeof DRIVING_INDEXES)[number];
  const join = (keys: string[]) => {
    const parts = keys.map((k) => clauses.get(k) as { sql: string; params: unknown[] });
    return { sql: parts.map((p) => p.sql).join(" AND "), params: parts.flatMap((p) => p.params) };
  };
  const drive = join(driving.keys);
  const remainingKeys = [...clauses.keys()].filter((k) => !driving.keys.includes(k));
  const remaining = join(remainingKeys);
  const indexed = `nodes INDEXED BY ${driving.index(outcome)}${drive.sql ? ` WHERE ${drive.sql}` : ""}`;
  const windowed = remainingKeys.length > 0;

  let statCount: string | null = null;
  const used = [...clauses.keys()];
  if (!windowed && clauses.has("step")) {
    statCount = "steps"; // only lane and step are in use: lane_stats.steps
  } else if (!windowed && used.every((k) => ["lane", "run_id", "outcome", "stan"].includes(k)) && !(clauses.has("outcome") && clauses.has("stan"))) {
    statCount = outcome === "kept" ? "kept" : outcome === "archived" ? "archived" : outcome === "discarded"
      ? "trials - kept - archived" : clauses.has("stan") ? "stan" : "trials";
  }
  return {
    // The window takes the newest trials in the page's own order; every driving index ends
    // in (created_at, id), so neither needs a sort.
    from: windowed ? `(SELECT * FROM ${indexed} ORDER BY created_at DESC, id DESC LIMIT ?) AS nodes` : indexed,
    fromParams: windowed ? [...drive.params, LIMITS.searchWindow] : drive.params,
    where: windowed ? `WHERE ${remaining.sql}` : "",
    whereParams: windowed ? remaining.params : [],
    windowed,
    statCount,
    lane: toNullableString(args.lane),
    runId: toNullableString(args.run_id),
  };
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
    if (!ID_RE.test(parent)) return out(400, { error: "invalid_identifier", field: "parent_ids", rule: ID_RULE });
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
  "b2_max", "dims", "codex_refs", "code_sha", "metrics_json", "speedup", "yardstick",
] as const;
const NODE_ADMIN_COLUMNS = ["real_stan_passed", "real_stan_total", "real_ess_per_cost"] as const;

/**
 * A queue item worth verifying now: its trial has a speedup on its lane's current yardstick. A
 * trial re-posted ungraded, without a speedup or on another yardstick keeps its queue row, so
 * the queue checks the trial as it is now. `n` is the trial.
 */
const SQL_QUEUE_WORTH = "n.speedup > 0 AND n.yardstick = (SELECT l.yardstick FROM lane_stats AS l WHERE l.lane = n.lane)";

/**
 * The bar for the verification queue: the best speedup among the lane's replicated or verified
 * trials with source on the given yardstick. Unreviewed or sourceless scores never set it, so
 * nobody can block the queue by posting an inflated claim. One row from a partial index.
 */
async function reviewedBest(db: D1DatabaseLike, lane: string, yardstick: string, excludeId: string): Promise<number | null> {
  const row = await db
    .prepare(
      `SELECT speedup AS best FROM nodes INDEXED BY idx_nodes_reviewed_best
       WHERE lane = ? AND yardstick = ? AND state IN ('replicated', 'verified') AND source_sha IS NOT NULL
         AND speedup IS NOT NULL AND id <> ?
       ORDER BY speedup DESC LIMIT 1`
    )
    .bind(lane, yardstick, excludeId)
    .first<{ best: number | null }>();
  return toFiniteNumber(row?.best);
}

/**
 * A kept trial with source and a speedup on its lane's current yardstick is queued when the lane
 * has no reviewed trial on that yardstick yet, or its speedup is at least STEP_MARGIN times the
 * best reviewed one (the step rule).
 */
async function maybeEnqueue(db: D1DatabaseLike, node: NodeRecord): Promise<boolean> {
  if (!KEPT_DECISIONS.has(node.decision) || node.speedup === null || !node.has_source || node.yardstick === null) return false;
  if (LOCKED_STATES.has(node.state) || FRONTIER_EXCLUDED_STATES.includes(node.state)) return false;
  const best = await reviewedBest(db, node.lane, node.yardstick, node.id);
  if (best !== null && node.speedup < STEP_FACTOR * best) return false;
  const now = nowIso();
  const res = await db
    .prepare(
      `INSERT OR IGNORE INTO verify_queue (node_id, lane, status, reason, attempts, created_at, updated_at)
       SELECT id, lane, 'queued', 'beats_frontier', 0, ?, ? FROM nodes AS n WHERE id = ? AND ${SQL_QUEUE_WORTH}`
    )
    .bind(now, now, node.id)
    .run();
  return changesOf(res) > 0;
}

/** Fields whose change would alter what a reviewer scored. */
const SCORED_COLUMNS = [
  "lane", "decision", "soft_min_eff", "ratio_vs_incumbent", "short_horizon_eff", "long_horizon_eff",
  "overhead_ratio", "b2_max", "code_sha", "speedup", "yardstick",
] as const;
const SCORED_JSON_COLUMNS = ["dims", "metrics_json"] as const;

/** JSON text compared by value (key order and spacing ignored). */
function canonicalJson(raw: unknown): string | null {
  if (raw === null || raw === undefined) return null;
  const sortKeys = (v: unknown): unknown =>
    Array.isArray(v)
      ? v.map(sortKeys)
      : v && typeof v === "object"
        ? Object.fromEntries(Object.keys(v as Args).sort().map((k) => [k, sortKeys((v as Args)[k])]))
        : v;
  try {
    return JSON.stringify(sortKeys(JSON.parse(String(raw))));
  } catch {
    return String(raw);
  }
}

function scoredFieldsChanged(existing: NodeRow, values: Record<string, unknown>, sourceSha: string | null): string[] {
  const row = existing as unknown as Record<string, unknown>;
  const changed: string[] = SCORED_COLUMNS.filter((c) => (row[c] ?? null) !== (values[c] ?? null));
  for (const c of SCORED_JSON_COLUMNS) if (canonicalJson(row[c]) !== canonicalJson(values[c])) changed.push(c);
  if (sourceSha && sourceSha !== existing.source_sha) changed.push("source");
  return changed;
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
    yardstick: LIMITS.yardstickChars,
    title: LIMITS.titleChars,
    hypothesis: LIMITS.hypothesisChars,
  });
  if (long) return out(400, { error: "field_too_long", ...long });
  const badId = badIdentifier(payload, ["id", "run_id", "trial_id", "author", "yardstick"], ["lane"]);
  if (badId) return badId;
  const speedup = payload.speedup ?? null;
  if (speedup !== null && !(typeof speedup === "number" && Number.isFinite(speedup) && speedup > 0)) {
    return out(400, { error: "invalid_speedup", detail: "speedup must be a number above 0, or null" });
  }
  // created_at is the order of the frontier walk, so only a maintainer (seeding, backfills)
  // may set it. Everyone else gets the server time, and nobody can backdate a trial.
  const givenAt = isAdmin ? toNullableString(payload.created_at) : null;
  const givenMs = givenAt === null ? null : Date.parse(givenAt);
  if (givenMs !== null && !Number.isFinite(givenMs)) {
    return out(400, { error: "invalid_created_at", detail: "created_at must be a date, for example 2026-11-24T10:00:00Z" });
  }

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

  const existing = await getNodeRow(db, id);
  if (existing && !isAdmin && existing.author !== principal.handle) {
    return out(403, { error: "forbidden_node_owner", id, owner: existing.author });
  }
  if (!existing && principal.role === "provisional") {
    const capped = await consumeProvisionalTrial(ctx.env, principal);
    if (capped) return capped;
  }

  const now = nowIso();

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
    speedup,
    yardstick: toNullableString(payload.yardstick),
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
    created_at: givenMs === null ? now : new Date(givenMs).toISOString(),
    updated_at: now,
  };

  // A reviewed or leased trial is frozen: a re-post that changes what was scored (source,
  // lane, decision, scores) is refused unless the trial is still unverified and unleased.
  // An identical re-post stays a no-op success. Maintainers are exempt.
  if (existing && !isAdmin && existing.lane !== lane) {
    return out(409, { error: "lane_immutable", detail: "a trial's lane cannot change; post a new trial", lane: existing.lane });
  }
  const changed = existing && !isAdmin ? scoredFieldsChanged(existing, values, sourceSha) : [];
  if (changed.length > 0) {
    const lease = await db
      .prepare("SELECT 1 AS leased FROM verify_queue WHERE node_id = ? AND status = 'leased' AND lease_expires_at > ?")
      .bind(id, now)
      .first<{ leased: number }>();
    if ((existing?.state ?? "unverified") !== "unverified" || lease) {
      return out(409, {
        error: "trial_locked",
        detail: "a reviewed or leased trial's source, lane, decision and scores cannot change; post a new trial",
        state: existing?.state ?? null,
        leased: Boolean(lease),
        fields: changed,
      });
    }
  }

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
  // The lock lives in the write itself, for every non-maintainer update: the row is updated
  // only if it is still unverified and unleased, or if neither this post (checked above) nor
  // anyone since the read (checked here, against the row as read) changed what was scored.
  const asRead = existing as unknown as Record<string, unknown> | null;
  const lockParams: unknown[] = [
    now,
    existing && changed.length === 0 ? 1 : 0,
    ...SCORED_COLUMNS.map((c) => asRead?.[c] ?? null),
    ...SCORED_JSON_COLUMNS.map((c) => asRead?.[c] ?? null),
    existing?.source_sha ?? null,
    existing?.state ?? null,
  ];
  const lock = [
    "nodes.author IS excluded.author",
    "nodes.lane IS excluded.lane",
    `((nodes.state = 'unverified' AND NOT EXISTS (SELECT 1 FROM verify_queue q WHERE q.node_id = nodes.id AND q.status = 'leased' AND q.lease_expires_at > ?))
      OR (? = 1 AND ${[...SCORED_COLUMNS, ...SCORED_JSON_COLUMNS, "source_sha", "state"].map((c) => `nodes.${c} IS ?`).join(" AND ")}))`,
  ].join(" AND ");
  const guard = isAdmin ? "" : ` WHERE ${lock}`;
  const upsert = db
    .prepare(
      `INSERT INTO nodes (${columns.join(", ")}) VALUES (${placeholders(columns.length)})
       ON CONFLICT(id) DO UPDATE SET ${updates.join(", ")}${guard}`
    )
    .bind(...columns.map((c) => values[c]), ...(isAdmin ? [] : lockParams));
  // One batch (one transaction in D1): the trial row and the source body it points to are
  // stored together, a maintainer's lane change moves the queue entry with it, and every
  // step flag the write changes is set from the rows as stored (see "Frontier steps").
  const writes: D1PreparedStatement[] = [upsert, stepSelfStmt(db, id)];
  if (sourceSha && sourceBytes) {
    writes.push(
      db
        .prepare(
          `INSERT OR IGNORE INTO sources (sha256, body, size, uploaded_by, created_at)
           SELECT ?, ?, ?, ?, ? WHERE EXISTS (SELECT 1 FROM nodes WHERE id = ? AND source_sha = ?)`
        )
        .bind(sourceSha, source, sourceBytes.length, principal.handle, now, id, sourceSha)
    );
  }
  if (isAdmin) {
    // Inside the batch, from the stored row: a concurrent lane move cannot split the trial
    // and its queue entry across lanes.
    writes.push(
      db
        .prepare(
          `UPDATE verify_queue SET lane = (SELECT lane FROM nodes WHERE id = ?), updated_at = ?
           WHERE node_id = ? AND lane IS NOT (SELECT lane FROM nodes WHERE id = ?)`
        )
        .bind(id, now, id, id)
    );
  }
  writes.push(...settleStepsStmts(db));
  const [res] = await runBatch(db, writes);
  if (changesOf(res) === 0) {
    return existing && existing.author === principal.handle
      ? out(409, { error: "trial_locked", detail: "the trial changed, was leased or was reviewed since it was read", fields: changed })
      : out(403, { error: "forbidden_node_owner", id });
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
  const plan = planNodeQuery(args);
  if ("status" in plan) return plan;
  const limit = clampLimit(args.limit, LIMITS.nodesLimitDefault, LIMITS.nodesLimitMax);
  const offset = clampOffset(args.offset);
  const res = await db
    .prepare(`SELECT * FROM ${plan.from} ${plan.where} ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?`)
    .bind(...plan.fromParams, ...plan.whereParams, limit, offset)
    .all<NodeRow>();
  const nodes = (res.results ?? []).map((r) => stripRaw(deserializeNode(r)));

  // Exact from the counters, exact within the window, or counted up to LIMITS.countCap.
  let count: number;
  let capped = false;
  let windowFull = false;
  if (plan.statCount) {
    const where = [plan.lane ? "lane = ?" : "", plan.runId ? "run_id = ?" : ""].filter(Boolean).join(" AND ");
    const row = await db
      .prepare(`SELECT COALESCE(SUM(${plan.statCount}), 0) AS n FROM ${plan.runId ? "run_stats" : "lane_stats"}${where ? ` WHERE ${where}` : ""}`)
      .bind(...[plan.lane, plan.runId].filter((x): x is string => Boolean(x)))
      .first<{ n: number }>();
    count = Number(row?.n ?? 0);
  } else if (plan.windowed) {
    const row = await db
      .prepare(`SELECT COUNT(*) AS seen, COALESCE(SUM(CASE WHEN ${plan.where.slice("WHERE ".length)} THEN 1 ELSE 0 END), 0) AS n FROM ${plan.from}`)
      .bind(...plan.whereParams, ...plan.fromParams) // the CASE comes before FROM in the statement
      .first<{ seen: number; n: number }>();
    count = Number(row?.n ?? 0);
    windowFull = Number(row?.seen ?? 0) >= LIMITS.searchWindow;
  } else {
    const row = await db
      .prepare(`SELECT COUNT(*) AS n FROM (SELECT 1 FROM ${plan.from} LIMIT ?)`)
      .bind(...plan.fromParams, LIMITS.countCap + 1)
      .first<{ n: number }>();
    count = Math.min(Number(row?.n ?? 0), LIMITS.countCap);
    capped = Number(row?.n ?? 0) > LIMITS.countCap;
  }
  const body: Record<string, unknown> = {
    nodes,
    count,
    count_capped: capped,
    limit,
    offset,
    next_offset: nextOffset(offset, limit, nodes.length, capped ? null : count),
  };
  if (plan.windowed) {
    // Filters without an index ran over the newest LIMITS.searchWindow trials of the indexed part.
    body.filter_window = LIMITS.searchWindow;
    body.window_truncated = windowFull;
    if (args.q) body.search_window = LIMITS.searchWindow;
  }
  return out(200, body);
};

const opFrontier: Op = async (ctx, args) => {
  const lane = toNullableString(args.lane);
  const lanes = lane ? [lane] : (await laneStats(ctx.env.DB)).map((l) => l.lane);
  const frontier = await laneFrontiers(ctx.env.DB, lanes);
  if (lane) return out(200, { frontier });
  return out(200, { frontier, count: Object.keys(frontier).length });
};

/**
 * One lane's chart: its graded trials on the lane's current yardstick oldest first, slim, with
 * their step flags. One range of idx_nodes_lane_graded plus the lane's counter row. Trials on
 * other yardsticks are not comparable, so they are left out.
 */
const opLaneChart: Op = async (ctx, args) => {
  const db = ctx.env.DB;
  const bad = badIdentifier(args, [], ["lane"]);
  if (bad) return bad;
  const lane = toNullableString(args.lane);
  if (!lane) return out(400, { error: "missing_required_fields", required: ["lane"] });
  const limit = clampLimit(args.limit, LIMITS.chartPoints, LIMITS.chartPoints);
  // `after` is the `next_after` of the previous page: "<created_at>|<id>" of its last point.
  // A cursor, not an offset, so every page costs the same and no lane is too long to read.
  // Split on the last "|": ids cannot hold one, but old created_at values are free text. Both
  // came in one trial payload, so a cursor the server issued is never longer than that.
  const after = toNullableString(args.after);
  const cut = after ? after.lastIndexOf("|") : -1;
  if (after && (cut <= 0 || cut === after.length - 1 || after.length > LIMITS.nodeMetaBytes + 1)) {
    return out(400, { error: "invalid_after" });
  }
  const cursor = after ? [after.slice(0, cut), after.slice(cut + 1)] : [];
  const [rows, stats] = await batchAll<Record<string, unknown>>(db, [
    db
      .prepare(
        `SELECT id, trial_id, run_id, created_at, speedup, soft_min_eff, decision, state, verified, frontier_step,
                author, author_tier, agent_model, substr(coalesce(hypothesis, title), 1, ?) AS title
         FROM nodes INDEXED BY idx_nodes_lane_graded
         WHERE lane = ? AND yardstick = (SELECT l.yardstick FROM lane_stats AS l WHERE l.lane = ?)
           ${after ? "AND (created_at, id) > (?, ?)" : ""}
         ORDER BY created_at, id LIMIT ?`
      )
      .bind(LIMITS.chartTitleChars, lane, lane, ...cursor, limit),
    db.prepare("SELECT standard, steps, yardstick FROM lane_stats WHERE lane = ?").bind(lane),
  ]);
  const total = Number(stats[0]?.standard ?? 0);
  const last = rows.length === limit ? rows[rows.length - 1] : null;
  return out(200, {
    lane,
    yardstick: (stats[0]?.yardstick as string | null | undefined) ?? null,
    points: rows.map((r) => ({
      id: r.id as string,
      trial_id: (r.trial_id as string | null) ?? null,
      run_id: r.run_id as string,
      created_at: r.created_at as string,
      speedup: toFiniteNumber(r.speedup),
      soft_min_eff: toFiniteNumber(r.soft_min_eff),
      decision: r.decision as string,
      state: (r.state as string | null) ?? (r.verified ? "verified" : "unverified"),
      step: Number(r.frontier_step) === 1,
      author: (r.author as string | null) ?? null,
      author_tier: (r.author_tier as string | null) ?? null,
      agent_model: (r.agent_model as string | null) ?? null,
      title: (r.title as string | null) ?? null,
    })),
    total,
    steps: Number(stats[0]?.steps ?? 0),
    limit,
    next_after: last ? `${last.created_at as string}|${last.id as string}` : null,
  });
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
  const [totalsRows, modelRows, ...stanRows] = await batchAll<Record<string, unknown>>(db, [
    // One row per lane plus a counter row; distinct models by skip scan (one index seek per
    // model, at most LIMITS.overviewModels).
    db.prepare(
      `SELECT COALESCE(SUM(trials), 0) AS trials, COALESCE(SUM(kept), 0) AS kept, COALESCE(SUM(runs), 0) AS runs,
              MAX(latest_at) AS latest_at, COALESCE(SUM(MAX(steps - 1, 0)), 0) AS improvements,
              (SELECT value FROM flywheel_counters WHERE name = 'contributors') AS contributors
       FROM lane_stats`
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
    ...lanes.map((l) => laneStanStmt(db, l.lane)),
  ]);
  const stanByLane = new Map(lanes.map((l, i) => [l.lane, stanRows[i]?.[0] as unknown as StanRow | undefined]));
  const t = totalsRows[0] ?? {};
  return out(200, {
    totals: {
      trials: Number(t.trials ?? 0),
      kept: Number(t.kept ?? 0),
      runs: Number(t.runs ?? 0),
      contributors: Number(t.contributors ?? 0),
      improvements: Number(t.improvements ?? 0),
      claims: await countActiveClaims(db),
      latest_at: (t.latest_at as string | null) ?? null,
    },
    lanes: lanes.map((l) => {
      const f = frontier[l.lane] ?? null;
      const stan = stanByLane.get(l.lane);
      return {
        ...l,
        frontier: f,
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
    db.prepare(`SELECT COALESCE(SUM(runs), 0) AS n FROM lane_stats ${laneWhere}`).bind(...laneParams),
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
    next_offset: nextOffset(offset, limit, runs.length, count),
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
  await runBatch(db, [
    db
      .prepare("UPDATE nodes SET state = ?, verified = ?, state_reason = ?, state_by = ?, state_at = ? WHERE id = ?")
      .bind(state, state === "verified" ? 1 : 0, reason, principal.handle, now, id),
    ...settleStepsStmts(db),
  ]);

  if (state === "needs-replication" && node.has_source) {
    await db
      .prepare(
        `INSERT INTO verify_queue (node_id, lane, status, reason, attempts, created_at, updated_at)
         SELECT id, lane, 'queued', 'needs-replication', 0, ?, ? FROM nodes WHERE id = ?
         ON CONFLICT(node_id) DO UPDATE SET status = 'queued', reason = excluded.reason, lane = excluded.lane,
           lease_token_hash = NULL, leased_by = NULL, lease_expires_at = NULL, updated_at = excluded.updated_at
         WHERE verify_queue.status = 'done'`
      )
      .bind(now, now, id)
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

/** SQL condition: the claim row belongs to the principal (always true for maintainers). */
function claimOwnerSql(principal: AuthPrincipal): { sql: string; params: unknown[] } {
  if (principal.role === "admin") return { sql: "1 = 1", params: [] };
  const prefix = `${principal.handle}:`;
  return { sql: "(agent = ? OR substr(agent, 1, ?) = ?)", params: [principal.handle, prefix.length, prefix] };
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
  const badId = badIdentifier(payload, ["task_id"], ["lane"]);
  if (badId) return badId;

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

  // A forced renewal releases only the caller's own active claims (maintainers: all of them).
  if (force) {
    const own = claimOwnerSql(principal);
    await db
      .prepare(`UPDATE claims SET status = 'abandoned' WHERE task_id = ? AND status = 'active' AND expires_at > ? AND ${own.sql}`)
      .bind(task_id, now, ...own.params)
      .run();
  }

  const id = toNullableString(payload.id) ?? crypto.randomUUID();
  const expiresAt = new Date(nowMs + ttlMinutes * 60_000).toISOString();
  const summary = toNullableString(payload.summary);

  // One statement: the lease is taken only if no active claim exists, so two agents
  // racing for the same task cannot both get it.
  const res = await db
    .prepare(
      `INSERT INTO claims (id, task_id, lane, agent, summary, status, expires_at, created_at)
       SELECT ?, ?, ?, ?, ?, 'active', ?, ?
       WHERE NOT EXISTS (SELECT 1 FROM claims WHERE task_id = ? AND status = 'active' AND expires_at > ?)`
    )
    .bind(id, task_id, lane, agent, summary, expiresAt, now, task_id, now)
    .run();
  if (changesOf(res) === 0) {
    const existing = await db
      .prepare(
        "SELECT * FROM claims WHERE task_id = ? AND status = 'active' AND expires_at > ? ORDER BY created_at DESC LIMIT 1"
      )
      .bind(task_id, now)
      .first<ClaimRow>();
    return force
      ? out(403, { error: "forbidden_force_claim", existing_claim: existing })
      : out(409, { error: "already_claimed", existing_claim: existing });
  }

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

  // Guarded by owner in the statement itself.
  const own = claimOwnerSql(principal);
  const hasSummary = payload.summary !== undefined;
  const res = await db
    .prepare(`UPDATE claims SET status = ?, summary = CASE WHEN ? THEN ? ELSE summary END WHERE id = ? AND ${own.sql}`)
    .bind(statusVal, hasSummary ? 1 : 0, hasSummary ? toNullableString(payload.summary) : null, claimId, ...own.params)
    .run();
  if (changesOf(res) === 0) {
    const existing = await db.prepare("SELECT agent FROM claims WHERE id = ?").bind(claimId).first<{ agent: string }>();
    if (!existing) return out(404, { error: "not_found" });
    return out(403, { error: "forbidden_claim_owner", owner: existing.agent });
  }

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
    if (!s || !ID_RE.test(s)) return out(400, { error: "invalid_identifier", field: "node_ids", rule: ID_RULE });
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
  if (lane && !LANE_RE.test(lane)) return out(400, { error: "invalid_identifier", field: "lane", rule: LANE_RULE });
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
    next_offset: nextOffset(offset, limit, notes.length, null),
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

/** True when older trials or notes exist beyond the search window. */
async function searchWindowTruncated(db: D1DatabaseLike, lane: string | null, kind: string | null): Promise<boolean> {
  if (kind !== "note" && (await totalTrials(db, lane)) > LIMITS.searchWindow) return true;
  if (kind === "node") return false;
  const older = await db
    .prepare(`SELECT 1 AS x FROM notes${lane ? " WHERE lane = ?" : ""} ORDER BY updated_at DESC, id DESC LIMIT 1 OFFSET ?`)
    .bind(...(lane ? [lane] : []), LIMITS.searchWindow)
    .first<{ x: number }>();
  return Boolean(older);
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

  // Each arm scans only the newest LIMITS.searchWindow trials / notes (of the lane, when given).
  const arm = (table: "nodes" | "notes"): { sql: string; params: unknown[] } => {
    const text = table === "nodes" ? NODE_SEARCH_TEXT : NOTE_SEARCH_TEXT;
    const where = patterns.map(() => `${text} LIKE ? ESCAPE '\\'`);
    const newest = table === "nodes" ? "created_at DESC, id DESC" : "updated_at DESC, id DESC";
    const windowed = `(SELECT * FROM ${table}${lane ? " WHERE lane = ?" : ""} ORDER BY ${newest} LIMIT ?) AS ${table}`;
    const params: unknown[] = [patterns[0], ...(lane ? [lane] : []), LIMITS.searchWindow, ...patterns];
    const cols = table === "nodes"
      ? "'node' AS kind, id, lane, title, author, decision, state, verified, soft_min_eff, speedup, yardstick, source_sha, NULL AS note_kind, created_at, created_at AS updated_at, coalesce(hypothesis, '') AS text"
      : "'note' AS kind, id, lane, title, author, NULL AS decision, state, 0 AS verified, NULL AS soft_min_eff, NULL AS speedup, NULL AS yardstick, NULL AS source_sha, kind AS note_kind, created_at, updated_at, coalesce(body, '') AS text";
    return {
      sql: `SELECT ${cols}, (coalesce(title, '') LIKE ? ESCAPE '\\') AS title_hit FROM ${windowed} WHERE ${where.join(" AND ")}`,
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
        speedup: toFiniteNumber(r.speedup),
        yardstick: (r.yardstick as string | null) ?? null,
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
    next_offset: nextOffset(offset, limit, rows.length, null, LIMITS.searchOffsetMax),
    search_window: LIMITS.searchWindow,
    window_truncated: await searchWindowTruncated(ctx.env.DB, lane, kind),
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
    speedup: n.speedup,
    yardstick: n.yardstick,
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
       WHERE q.status IN (${placeholders(statuses.length)}) AND ${SQL_QUEUE_WORTH}${laneClause}
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

/** Verification needs a contributor (invited or promoted) or maintainer token. */
function verifierTierRefusal(principal: AuthPrincipal): Outcome | null {
  return principal.role === "provisional"
    ? out(403, { error: "forbidden_tier", detail: "verification needs a contributor token (invite code or maintainer promotion)" })
    : null;
}

const opVerifyClaim: Op = async (ctx, args) => {
  const db = ctx.env.DB;
  const principal = ctx.principal as AuthPrincipal;
  const refused = verifierTierRefusal(principal);
  if (refused) return refused;
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
    SQL_QUEUE_WORTH,
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
  const refused = verifierTierRefusal(principal);
  if (refused) return refused;
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
  const leaseOk = `node_id = ? AND status = 'leased' AND lease_token_hash = ? AND lease_expires_at > ?${holderClause}`;
  const leaseParams = [nodeId, leaseHash, now, ...holderParams];
  const newState = passed ? "replicated" : "disputed";
  const reason = `independent rerun by ${principal.handle}: ${passed ? "pass" : "fail"}${note ? ` (${note})` : ""}`;
  // One batch (one transaction in D1): the trial's reviewed state and the queue result commit
  // together, both guarded by the same lease check, so the trial is never unleased and still
  // unverified in between (when its author could change it).
  const [, done] = await runBatch(db, [
    db
      .prepare(
        `UPDATE nodes SET state = ?, verified = 0, state_reason = ?, state_by = ?, state_at = ?
         WHERE id = ? AND state IN (${placeholders(RESULT_WRITABLE_STATES.length)})
           AND EXISTS (SELECT 1 FROM verify_queue WHERE ${leaseOk})`
      )
      .bind(newState, reason.slice(0, LIMITS.reasonChars), principal.handle, now, nodeId, ...RESULT_WRITABLE_STATES, ...leaseParams),
    db
      .prepare(
        `UPDATE verify_queue SET status = 'done', result = ?, result_json = ?, verifier = ?,
           lease_token_hash = NULL, updated_at = ?
         WHERE ${leaseOk}
         RETURNING node_id`
      )
      .bind(passed ? "pass" : "fail", metricsJson, principal.handle, now, ...leaseParams),
    ...settleStepsStmts(db),
  ]);

  if ((done?.results ?? []).length === 0) {
    const item = await db.prepare("SELECT * FROM verify_queue WHERE node_id = ?").bind(nodeId).first<QueueRow>();
    if (!item) return out(404, { error: "not_queued", node_id: nodeId });
    if (item.status !== "leased") return out(409, { error: "not_leased", status: item.status });
    if ((item.lease_expires_at ?? "") <= now) return out(409, { error: "lease_expired", lease_expires_at: item.lease_expires_at });
    return out(409, { error: "stale_lease", detail: "lease token does not match the current lease" });
  }

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
    // Revoke every key of the handle and delete everything it posted, in one batch (one
    // transaction), which ends by recomputing the steps of every lane it changed.
    const owned = "SELECT id FROM nodes WHERE lower(author) = lower(?)";
    const [revoked, , , nodes, notes, claims, sources] = await runBatch(db, [
      db.prepare("UPDATE api_keys SET revoked_at = ? WHERE lower(handle) = lower(?) AND revoked_at IS NULL").bind(now, handle),
      db.prepare(`DELETE FROM node_parents WHERE child IN (${owned}) OR parent IN (${owned})`).bind(handle, handle),
      db.prepare(`DELETE FROM verify_queue WHERE node_id IN (${owned})`).bind(handle),
      db.prepare("DELETE FROM nodes WHERE lower(author) = lower(?)").bind(handle),
      db.prepare("DELETE FROM notes WHERE lower(author) = lower(?)").bind(handle),
      db
        .prepare("DELETE FROM claims WHERE lower(agent) = lower(?) OR lower(agent) LIKE lower(?) ESCAPE '\\'")
        .bind(handle, `${handle.replace(/[\\%_]/g, (c) => `\\${c}`)}:%`),
      db
        .prepare(
          "DELETE FROM sources WHERE lower(uploaded_by) = lower(?) AND sha256 NOT IN (SELECT source_sha FROM nodes WHERE source_sha IS NOT NULL)"
        )
        .bind(handle),
      // Lease results this handle posted on others' trials are reopened, not trusted.
      db
        .prepare(
          `UPDATE nodes SET state = 'unverified', state_reason = NULL, state_by = NULL, state_at = NULL
           WHERE state IN ('replicated', 'disputed', 'needs-replication') AND lower(state_by) = lower(?)`
        )
        .bind(handle),
      db
        .prepare(
          `UPDATE verify_queue SET status = 'queued', result = NULL, result_json = NULL, verifier = NULL,
             lease_token_hash = NULL, leased_by = NULL, lease_expires_at = NULL, updated_at = ?
           WHERE lower(verifier) = lower(?) OR lower(leased_by) = lower(?)`
        )
        .bind(now, handle, handle),
      ...settleStepsStmts(db),
    ]);
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
  await runBatch(db, [
    db
      .prepare(
        `UPDATE nodes SET verified = ?, state = ?, state_by = ?, state_at = ?,
           real_stan_passed = ?, real_stan_total = ?, real_ess_per_cost = ? WHERE id = ?`
      )
      .bind(verified, state, (ctx.principal as AuthPrincipal).handle, now, stanPassed, stanTotal, stanEss, nodeId),
    ...settleStepsStmts(db),
  ]);
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
  const existing = await getNodeRow(db, nodeId);
  if (!existing) return out(404, { error: "not_found" });
  await runBatch(db, [
    db.prepare("DELETE FROM nodes WHERE id = ?").bind(nodeId),
    db.prepare("DELETE FROM node_parents WHERE child = ? OR parent = ?").bind(nodeId, nodeId),
    db.prepare("DELETE FROM verify_queue WHERE node_id = ?").bind(nodeId),
    ...settleStepsStmts(db),
  ]);
  return out(200, { ok: true, deleted: nodeId });
};

const opHealth: Op = async (ctx) => {
  const db = ctx.env.DB;
  const [totals, lanes] = await batchAll<Record<string, unknown>>(db, [
    db.prepare("SELECT COALESCE(SUM(trials), 0) AS n FROM lane_stats"),
    db.prepare("SELECT lane FROM lane_stats WHERE trials > 0 ORDER BY lane LIMIT ?").bind(LIMITS.overviewLanes),
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
    description:
      "Return each lane's frontier: the latest trial that moved it, among kept or verified graded trials, compared by speedup.",
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
      "List trials, newest first, with filters (lane, run_id, decision, state, verified, model, outcome kept|archived|discarded, stan, step: only trials that moved their lane's frontier, q) and paging (limit, offset).",
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
        step: { type: "boolean" },
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
        speedup: { type: ["number", "null"], exclusiveMinimum: 0, description: "Efficiency over the lane's starting code. Null when failed or exploratory." },
        yardstick: { type: ["string", "null"], description: "Id of the grading setup. Null for an exploratory trial." },
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
  { method: "GET", pattern: /^\/api\/lanes\/([^/]+)\/chart$/, params: ["lane"], access: "public", op: opLaneChart },
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

async function dispatch(
  request: Request,
  env: Env,
  url: URL,
  method: string,
  query: Record<string, string>
): Promise<Response | null> {
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
      args = query;
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

  const query = queryArgs(url);
  const cacheKey = method === "GET" ? canonicalPublicCacheKey(url, query) : null;
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
    const resp = await dispatch(request, env, url, method, query);
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
