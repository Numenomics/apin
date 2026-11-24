// Scaling check for the reads the website uses, at 1,000 and 20,000 synthetic trials (plus a fixed tie block).
//
//  1. Worker JavaScript time per request, with the time spent inside SQLite subtracted
//     (that part runs in D1, not under the Worker's 10 ms CPU limit). node:sqlite.
//  2. D1 rows read per request (`meta.rows_read`, what the free plan's daily read quota
//     counts), from a real local D1 through wrangler's getPlatformProxy. Skipped with a
//     message when wrangler is not installed (`npm install` in flywheel/).
//
//   node --test flywheel/tests/scale.test.mjs          (prints both tables; ~1-2 min)

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { it } from "node:test";
import { fileURLToPath } from "node:url";

import { handleRequest, LIMITS } from "../src/index.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
const SCHEMA = fs.readFileSync(path.resolve(here, "../schema.sql"), "utf8");
const MIGRATIONS = fs
  .readdirSync(path.resolve(here, "../migrations"))
  .filter((f) => f.endsWith(".sql"))
  .sort()
  .map((f) => fs.readFileSync(path.resolve(here, "../migrations", f), "utf8"));
/** Trials below are inserted straight into SQLite, as seed.sql does, so they need 0005's flag backfill. */
const BACKFILL = fs
  .readFileSync(path.resolve(here, "../migrations/0005_speedup.sql"), "utf8")
  .split("-- BACKFILL BEGIN")[1]
  .split("-- BACKFILL END")[0];

const LANES = ["A0", "A1", "A2", "A3", "A4", "S1", "S2", "S3"];
const MODELS = ["glm-5.3", "claude-opus-5-5", "deepseek-4.1", "muse-1.3", "codex-5", "qwen-4"];
const HYPOTHESIS = "Shrink the diagonal Fisher toward identity with a James-Stein weight; ".repeat(3);
const METRICS = JSON.stringify({
  loc: 140, magic_constants: 4, wall_s: 81.2, queue_wait_s: 0, archive_cells: ["funnel__low"], tier0_ok: true,
  tier1_ok: true, incumbent_soft_min: 0.0012, median_adapt_cost: 912, judge_digest: "720a17a2cc4a7e74", seed: 7,
  repeats: 2, by_class: { banana: 0.0016, funnel: 0.0065, student_t: 0.0067 },
});
const SOURCE = "import numpy as np\n" + "def adapt(state):\n    return state\n".repeat(250);
const NODE_COLS = [
  "id", "parent_id", "lane", "run_id", "trial_id", "agent_model", "author", "author_tier", "title", "hypothesis",
  "decision", "soft_min_eff", "speedup", "yardstick", "ratio_vs_incumbent", "real_stan_passed", "real_stan_total",
  "overhead_ratio", "b2_max", "dims", "codex_refs", "code_sha", "metrics_json", "verified", "state", "created_at",
];
const INSERT_NODE = `INSERT INTO nodes (${NODE_COLS.join(", ")}) VALUES (${NODE_COLS.map(() => "?").join(", ")})`;
const INSERT_EDGE = "INSERT OR IGNORE INTO node_parents (child, parent, relation, created_at) VALUES (?, ?, ?, ?)";
const INSERT_NOTE =
  "INSERT INTO notes (id, kind, lane, title, body, state, author, node_ids, created_at, updated_at) VALUES (?, 'insight', 'A1', 'Note', 'body', 'open', 'x', ?, ?, ?)";

const TIE_TRIALS = 3000;
const STATE_OF_STEP = "POST /api/nodes/:id/state (first step, lane recompute)";
// A chart cursor two hours into the synthetic timeline: at both table sizes lane A1 still
// holds more than a page of 100 trials after it.
const CHART_CURSOR = "2026-01-01T02:00:00.000Z|0";

/**
 * N trials over 8 lanes (one run of N/10 trials in A1, the rest in runs of about 60), a tie block, 30 notes.
 * Nine in ten trials are graded, with a speedup between 0.8 and 1.4 that drifts up over time
 * (one in twenty of them failed its checks and has none). The rest are exploratory.
 */
function synthesize(n) {
  const longRun = Math.round(n / 10);
  const runCount = Math.max(8, Math.round((n - longRun) / 60));
  const runs = [{ lane: "A1", run_id: "long-run", size: longRun }];
  for (let r = 0; r < runCount; r += 1) runs.push({ lane: LANES[r % LANES.length], run_id: `run${r}`, size: 0 });
  for (let i = 0; i < n - longRun; i += 1) runs[1 + (i % runCount)].size += 1;

  let rnd = 12345;
  const rand = () => ((rnd = (rnd * 1103515245 + 12345) % 2147483648) / 2147483648);
  const slots = [];
  for (const run of runs) for (let k = 0; k < run.size; k += 1) slots.push({ run, k, t: rand() });
  slots.sort((a, b) => a.t - b.t); // runs interleave in time
  const statements = [];
  const incumbent = new Map();
  const last = new Map();
  let firstStepA1 = null; // A1's first kept graded trial with a speedup
  slots.forEach(({ run, k }, i) => {
    const id = `${run.lane}:${run.run_id}:T${String(k + 1).padStart(4, "0")}`;
    const u = rand();
    const decision = u < 0.15 ? "keep" : u < 0.25 ? "archive_only" : u < 0.6 ? "discard_no_gain" : "discard_tier1";
    const stan = rand() < 0.02;
    const parent = incumbent.get(run.run_id) ?? null;
    const created = new Date(Date.UTC(2026, 0, 1) + i * 30_000).toISOString();
    const graded = rand() < 0.9;
    const speedup = graded && rand() < 0.95 ? 0.8 + 0.3 * rand() + (0.3 * i) / slots.length : null;
    statements.push([INSERT_NODE, [
      id, parent, run.lane, run.run_id, `T${k + 1}`, MODELS[(k + run.run_id.length) % MODELS.length],
      `agent-${run.run_id}`, "provisional", `T${k + 1}: ${HYPOTHESIS.slice(0, 70)}`, HYPOTHESIS, decision,
      rand() / 100, speedup, graded ? `${run.lane}:judge-1:r8` : null, 1 + rand(), stan ? Math.floor(rand() * 6) : null,
      stan ? 5 : null, 4 + rand(), 0.1, "[6,20]", '["fisher-low-rank"]', "cbc54bbbd03d8511", METRICS, 0, "unverified", created,
    ]]);
    if (parent) statements.push([INSERT_EDGE, [id, parent, "incumbent", created]]);
    if (rand() < 0.03 && last.size > 0) {
      statements.push([INSERT_EDGE, [id, [...last.values()][Math.floor(rand() * last.size)], "combines", created]]);
    }
    if (decision === "keep") incumbent.set(run.run_id, id);
    if (decision === "keep" && speedup !== null && run.lane === "A1" && !firstStepA1) firstStepA1 = id;
    last.set(run.run_id, id);
  });
  // A block of TIE_TRIALS trials sharing the newest timestamp (lane TIE, model tie-model):
  // newest-first reads must page through it without sorting the whole tie.
  const tieAt = new Date(Date.UTC(2026, 0, 1) + slots.length * 30_000).toISOString();
  for (let k = 0; k < TIE_TRIALS; k += 1) {
    statements.push([INSERT_NODE, [
      `TIE:tie:T${String(k).padStart(4, "0")}`, null, "TIE", "tie", `T${k}`, "tie-model", "agent-tie", "provisional",
      `T${k}: tie`, HYPOTHESIS, "discard_tier1", 0.001, 1, "TIE:judge-1:r8", 1, null, null, 4, 0.1, "[6,20]",
      '["fisher-low-rank"]', "cbc54bbbd03d8511", METRICS, 0, "unverified", tieAt,
    ]]);
  }
  const target = incumbent.get("long-run");
  for (let j = 0; j < 30; j += 1) {
    const at = `2026-02-01T00:00:${String(j).padStart(2, "0")}Z`;
    statements.push([INSERT_NOTE, [`note_${j}`, JSON.stringify([target]), at, at]]);
  }
  return { statements, target, firstStepA1 };
}

/**
 * The requests measured, in order. `site` marks the reads the website uses. `flat: false`
 * skips the flat-JS check: for a page the 1,000-trial table cannot fill (its cost grows up to
 * the page cap, then stops), and for the rare lane recompute, whose cost grows with the lane.
 */
function requests(n, target, firstStepA1) {
  const get = (p) => () => new Request(`https://x${p}`);
  return [
    ["/api/overview", true, get("/api/overview")],
    ["/api/runs", true, get("/api/runs")],
    ["/api/runs?limit=60 (max page)", true, get("/api/runs?limit=60")],
    ["/api/nodes (first page)", true, get("/api/nodes")],
    ["/api/nodes?outcome=kept&stan=1", true, get("/api/nodes?outcome=kept&stan=1")],
    ["/api/nodes?lane=A1&outcome=archived", true, get("/api/nodes?lane=A1&outcome=archived")],
    ["/api/nodes?decision=never_used (zero match)", true, get("/api/nodes?decision=never_used")],
    ["/api/nodes?lane=A1&state=verified (sparse)", true, get("/api/nodes?lane=A1&state=verified")],
    ["/api/nodes?model=qwen-4&decision=keep (sparse)", true, get("/api/nodes?model=qwen-4&decision=keep")],
    ["/api/nodes?lane=TIE (timestamp tie)", true, get("/api/nodes?lane=TIE")],
    ["/api/nodes?model=tie-model (timestamp tie)", true, get("/api/nodes?model=tie-model")],
    ["/api/nodes?lane=TIE&state=verified (tie, windowed)", true, get("/api/nodes?lane=TIE&state=verified")],
    ["/api/nodes?q=james", true, get("/api/nodes?q=james")],
    ["/api/nodes/:id", true, get(`/api/nodes/${encodeURIComponent(target)}`)],
    ["/api/frontier", true, get("/api/frontier")],
    ["/api/lanes/A1/chart?limit=200", true, get("/api/lanes/A1/chart?limit=200")],
    ["/api/lanes/A1/chart?after=<cursor>&limit=100", true, get(`/api/lanes/A1/chart?limit=100&after=${encodeURIComponent(CHART_CURSOR)}`)],
    ["/api/lanes/A1/chart (full page, 1,000 points)", true, get("/api/lanes/A1/chart"), { flat: false }],
    ["/api/nodes?step=1", true, get("/api/nodes?step=1")],
    ["/api/nodes?lane=A1&step=1&q=james", true, get("/api/nodes?lane=A1&step=1&q=james")],
    ["/api/search?q=james", true, get("/api/search?q=james")],
    ["/api/dag", false, get("/api/dag")],
    ["POST /api/nodes", false, (k) =>
      new Request("https://x/api/nodes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: `A1:long-run:P${n}-${k}`, lane: "A1", run_id: "long-run", trial_id: `P${k}`, decision: "keep",
          soft_min_eff: 0.5, speedup: 5, yardstick: "A1:judge-1:r8", hypothesis: HYPOTHESIS, metrics_json: METRICS, dims: [6, 20], codex_refs: ["fisher-low-rank"],
          parent_ids: [{ id: target, relation: "incumbent" }], source: `${SOURCE}# ${k}\n`,
        }),
      })],
    // Disputing (and restoring) A1's first step recomputes the whole lane: the rare path.
    [STATE_OF_STEP, false, (k) =>
      new Request(`https://x/api/nodes/${encodeURIComponent(firstStepA1)}/state`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state: k % 2 === 0 ? "disputed" : "unverified", reason: "rerun" }),
      }), { flat: false }],
  ];
}

// No FLYWHEEL_API_KEY: the local maintainer (explicit opt-in), so writes have no caps.
const envFor = (db) => ({ DB: db, FLYWHEEL_DEV_OPEN_ADMIN: "1" });

async function call(env, req) {
  const resp = await handleRequest(req, env);
  const body = await resp.text();
  assert.ok(resp.status < 300, `${req.method} ${req.url} -> ${resp.status}: ${body.slice(0, 300)}`);
}

// --- 1. JavaScript time (node:sqlite) -------------------------------------------------

function sqliteDb(n) {
  const sqlite = new DatabaseSync(":memory:");
  sqlite.exec(SCHEMA);
  for (const m of MIGRATIONS) sqlite.exec(m);
  const { statements, target, firstStepA1 } = synthesize(n);
  sqlite.exec("BEGIN");
  for (const [sql, params] of statements) sqlite.prepare(sql).run(...params);
  sqlite.exec("COMMIT");
  sqlite.exec(BACKFILL);
  const clock = { sqlMs: 0 };
  const timed = (fn) => {
    const t = performance.now();
    try {
      return fn();
    } finally {
      clock.sqlMs += performance.now() - t;
    }
  };
  const stmt = (query, values = []) => ({
    bind: (...v) => stmt(query, v),
    first: async (col) =>
      timed(() => {
        const row = sqlite.prepare(query).get(...values);
        if (!row) return null;
        return col ? row[col] ?? null : { ...row };
      }),
    all: async () => timed(() => ({ results: sqlite.prepare(query).all(...values).map((r) => ({ ...r })), success: true, meta: {} })),
    run: async () =>
      timed(() => {
        const info = sqlite.prepare(query).run(...values);
        return { results: [], success: true, meta: { changes: Number(info.changes ?? 0) } };
      }),
    execSync: () => {
      const results = sqlite.prepare(query).all(...values).map((r) => ({ ...r }));
      return { results, success: true, meta: { changes: Number(sqlite.prepare("SELECT changes() AS c").get().c) } };
    },
  });
  const db = {
    prepare: (q) => stmt(q),
    // Like D1: one transaction, all or nothing.
    async batch(stmts) {
      return timed(() => {
        sqlite.exec("BEGIN");
        try {
          const res = stmts.map((st) => st.execSync());
          sqlite.exec("COMMIT");
          return res;
        } catch (err) {
          sqlite.exec("ROLLBACK");
          throw err;
        }
      });
    },
  };
  return { db, clock, target, firstStepA1 };
}

async function jsTimes(n) {
  const { db, clock, target, firstStepA1 } = sqliteDb(n);
  const env = envFor(db);
  const outMs = {};
  for (const [name, , make] of requests(n, target, firstStepA1)) {
    const samples = [];
    for (let k = 0; k < 11; k += 1) {
      clock.sqlMs = 0;
      const req = make(k);
      const t = performance.now();
      const resp = await handleRequest(req, env);
      const jsMs = performance.now() - t - clock.sqlMs;
      const body = await resp.text();
      assert.ok(resp.status < 300, `${req.method} ${req.url} -> ${resp.status}: ${body.slice(0, 300)}`);
      if (k >= 2) samples.push(jsMs); // two warm-up calls
    }
    samples.sort((a, b) => a - b);
    outMs[name] = samples[Math.floor(samples.length / 2)];
  }
  return outMs;
}

// --- 2. D1 rows read (local D1) -------------------------------------------------------

/** Splits a migration into statements, keeping CREATE TRIGGER ... END; blocks whole. */
function splitSql(sql) {
  const out = [];
  let buf = "";
  let inTrigger = false;
  for (const raw of sql.split("\n")) {
    const line = raw.replace(/--.*$/, "");
    if (!line.trim()) continue;
    buf += `${line}\n`;
    if (/^\s*CREATE TRIGGER/i.test(line)) inTrigger = true;
    const done = inTrigger ? /^\s*END;\s*$/i.test(line) : /;\s*$/.test(line);
    if (done) {
      out.push(buf.trim().replace(/;$/, ""));
      buf = "";
      inTrigger = false;
    }
  }
  return out;
}

/** Wraps a D1 binding and totals meta.rows_read over everything a request runs. */
function countingD1(d1, counter) {
  const add = (r) => {
    counter.rows += Number(r?.meta?.rows_read ?? 0);
    return r;
  };
  const wrap = (stmt) => ({
    inner: stmt,
    bind: (...v) => wrap(stmt.bind(...v)),
    all: async () => add(await stmt.all()),
    run: async () => add(await stmt.run()),
    // D1 runs the whole statement for first() too; all() exposes its meta.
    first: async (col) => {
      const row = add(await stmt.all()).results?.[0];
      if (!row) return null;
      return col ? row[col] ?? null : row;
    },
  });
  return {
    prepare: (q) => wrap(d1.prepare(q)),
    async batch(stmts) {
      const res = await d1.batch(stmts.map((s) => s.inner));
      for (const r of res) add(r);
      return res;
    },
  };
}

async function rowsRead(getPlatformProxy, n) {
  const persist = fs.mkdtempSync(path.join(os.tmpdir(), "flywheel-rows-"));
  const proxy = await getPlatformProxy({ configPath: path.resolve(here, "../wrangler.toml"), persist: { path: persist } });
  try {
    const d1 = proxy.env.DB;
    for (const sql of [SCHEMA, ...MIGRATIONS]) for (const s of splitSql(sql)) await d1.prepare(s).run();
    const { statements, target, firstStepA1 } = synthesize(n);
    // Batches of 100: the local runtime drops the connection on long single batches.
    for (let i = 0; i < statements.length; i += 100) {
      await d1.batch(statements.slice(i, i + 100).map(([sql, params]) => d1.prepare(sql).bind(...params)));
    }
    for (const s of splitSql(BACKFILL)) await d1.prepare(s).run();
    const counter = { rows: 0 };
    const env = envFor(countingD1(d1, counter));
    const result = {};
    for (const [name, , make] of requests(n, target, firstStepA1)) {
      counter.rows = 0;
      await call(env, make(0));
      result[name] = counter.rows;
    }
    return result;
  } finally {
    await proxy.dispose();
    fs.rmSync(persist, { recursive: true, force: true });
  }
}

/**
 * Worst-case D1 rows a site read may touch, from the caps in LIMITS (independent of the
 * number of trials). Each request must stay under its bound at every table size.
 */
const RUN_SLICE = LIMITS.runsLimitMax * LIMITS.runNodes;
const SITE_BOUNDS = {
  "/api/overview":
    LIMITS.overviewLanes * 45 +
    LIMITS.alignmentPoints * 2 + LIMITS.overviewModels * 2 + LIMITS.overviewClaims + 200,
  "/api/runs": LIMITS.runsLimitDefault * (LIMITS.runNodes + 1) + 200,
  "/api/runs?limit=60 (max page)": RUN_SLICE + LIMITS.runsLimitMax + 200,
  "/api/nodes (first page)": LIMITS.nodesLimitDefault * 2 + 50,
  "/api/nodes?outcome=kept&stan=1": LIMITS.searchWindow * 2 + 50,
  "/api/nodes?lane=A1&outcome=archived": LIMITS.nodesLimitDefault * 2 + 50,
  "/api/nodes?decision=never_used (zero match)": LIMITS.searchWindow * 2 + 50,
  "/api/nodes?lane=A1&state=verified (sparse)": LIMITS.searchWindow * 2 + 50,
  "/api/nodes?model=qwen-4&decision=keep (sparse)": LIMITS.searchWindow * 2 + 50,
  "/api/nodes?lane=TIE (timestamp tie)": LIMITS.nodesLimitDefault * 2 + 50,
  // Counted through the index up to LIMITS.countCap.
  "/api/nodes?model=tie-model (timestamp tie)": LIMITS.nodesLimitDefault * 2 + LIMITS.countCap + 50,
  "/api/nodes?lane=TIE&state=verified (tie, windowed)": LIMITS.searchWindow * 2 + 50,
  "/api/nodes?q=james": LIMITS.searchWindow * 3 + 50,
  "/api/nodes/:id": LIMITS.nodeChildren * 2 + LIMITS.nodeNotes * 2 + 100,
  "/api/frontier": LIMITS.overviewLanes * 40 + 50,
  "/api/lanes/A1/chart?limit=200": 200 + 50,
  // A cursor seeks to its place in the index: the rows before it are never read.
  "/api/lanes/A1/chart?after=<cursor>&limit=100": 100 + 50,
  "/api/lanes/A1/chart (full page, 1,000 points)": LIMITS.chartPoints + 50,
  // Steps are few (each one beats the one before it by 5%), and the count comes from lane_stats.
  "/api/nodes?step=1": LIMITS.nodesLimitDefault * 2 + 50,
  "/api/nodes?lane=A1&step=1&q=james": LIMITS.nodesLimitDefault * 2 + 200,
  "/api/search?q=james": LIMITS.searchWindow * 4 + 50,
};

it("keeps Worker CPU flat and D1 rows read bounded from 1,000 to 20,000 trials", { timeout: 900_000 }, async () => {
  const smallMs = await jsTimes(1000);
  const largeMs = await jsTimes(20000);

  let getPlatformProxy = null;
  try {
    ({ getPlatformProxy } = await import("wrangler"));
  } catch {
    console.log("wrangler is not installed (run `npm install` in flywheel/): skipping the D1 rows-read table");
  }
  const smallRows = getPlatformProxy ? await rowsRead(getPlatformProxy, 1000) : null;
  const largeRows = getPlatformProxy ? await rowsRead(getPlatformProxy, 20000) : null;

  const names = Object.keys(smallMs);
  const fmt = (x) => (x === undefined || x === null ? "-" : String(x));
  console.log(
    [
      "",
      "| Endpoint | JS ms @ 1k | JS ms @ 20k | D1 rows read @ 1k | D1 rows read @ 20k | Bound |",
      "| --- | ---: | ---: | ---: | ---: | ---: |",
      ...names.map(
        (k) =>
          `| ${k} | ${smallMs[k].toFixed(2)} | ${largeMs[k].toFixed(2)} | ${fmt(smallRows?.[k])} | ${fmt(largeRows?.[k])} | ${fmt(SITE_BOUNDS[k])} |`
      ),
      "",
    ].join("\n")
  );

  const notFlat = new Set(requests(1, "x", "x").filter((r) => r[3]?.flat === false).map((r) => r[0]));
  for (const k of names) {
    if (notFlat.has(k)) continue;
    // JavaScript time: flat within timer noise (never more than double plus 1.5 ms).
    assert.ok(largeMs[k] <= 2 * smallMs[k] + 1.5, `${k}: JS grew from ${smallMs[k].toFixed(2)} to ${largeMs[k].toFixed(2)} ms`);
  }
  if (largeRows) {
    for (const [k, site] of requests(1, "x", "x").map(([name, s]) => [name, s])) {
      if (!site) continue;
      assert.ok(largeRows[k] <= SITE_BOUNDS[k], `${k}: ${largeRows[k]} rows read at 20,000 trials exceeds its bound ${SITE_BOUNDS[k]}`);
      assert.ok(smallRows[k] <= SITE_BOUNDS[k], `${k}: ${smallRows[k]} rows read at 1,000 trials exceeds its bound ${SITE_BOUNDS[k]}`);
    }
    // The usual write, a new trial at the end of its lane, reads a fixed number of rows too.
    for (const rows of [smallRows, largeRows]) assert.ok(rows["POST /api/nodes"] <= 100, `POST /api/nodes: ${rows["POST /api/nodes"]} rows read`);
    // The lane recompute reads each graded trial of the lane about once: linear, not worse.
    for (const [rows, n] of [[smallRows, 1000], [largeRows, 20000]]) {
      const lane = synthesize(n).statements.filter(([sql, params]) => sql === INSERT_NODE && params[2] === "A1").length;
      assert.ok(rows[STATE_OF_STEP] <= 2 * lane + 100, `${STATE_OF_STEP}: ${rows[STATE_OF_STEP]} rows read for ${lane} trials in A1`);
      console.log(`lane A1 at ${n} trials: ${lane} trials; state change of its first step read ${rows[STATE_OF_STEP]} rows`);
    }
  }
});
