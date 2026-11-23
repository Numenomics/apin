// Scaling check: the JavaScript cost of every read the website uses must not grow with
// the number of trials. Loads synthetic databases of 1,000 and 20,000 trials and measures,
// per endpoint, the Worker-side time with the time spent inside SQLite subtracted (that
// part runs in D1, not under the Worker's CPU limit).
//
//   node --test flywheel/tests/scale.test.mjs          (prints the table; ~20 s)

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { it } from "node:test";
import { fileURLToPath } from "node:url";

import { handleRequest } from "../src/index.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
const SCHEMA = fs.readFileSync(path.resolve(here, "../schema.sql"), "utf8");
const MIGRATIONS = fs
  .readdirSync(path.resolve(here, "../migrations"))
  .filter((f) => f.endsWith(".sql"))
  .sort()
  .map((f) => fs.readFileSync(path.resolve(here, "../migrations", f), "utf8"));

const LANES = ["A0", "A1", "A2", "A3", "A4", "S1", "S2", "S3"];
const MODELS = ["glm-5.3", "claude-opus-5-5", "deepseek-4.1", "muse-1.3", "codex-5", "qwen-4"];
const HYPOTHESIS = "Shrink the diagonal Fisher toward identity with a James-Stein weight; ".repeat(3);
const METRICS = JSON.stringify({
  loc: 140, magic_constants: 4, wall_s: 81.2, queue_wait_s: 0, archive_cells: ["funnel__low"], tier0_ok: true,
  tier1_ok: true, incumbent_soft_min: 0.0012, median_adapt_cost: 912, judge_digest: "720a17a2cc4a7e74", seed: 7,
  repeats: 2, by_class: { banana: 0.0016, funnel: 0.0065, student_t: 0.0067 },
});
const SOURCE = "import numpy as np\n" + "def adapt(state):\n    return state\n".repeat(250);

/** node:sqlite-backed D1 shim that also totals the time spent inside SQLite. */
function makeDb() {
  const sqlite = new DatabaseSync(":memory:");
  sqlite.exec(SCHEMA);
  for (const m of MIGRATIONS) sqlite.exec(m);
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
  });
  const db = {
    prepare: (q) => stmt(q),
    async batch(stmts) {
      const out = [];
      for (const s of stmts) out.push(await s.all());
      return out;
    },
  };
  return { sqlite, db, clock };
}

/** N trials over 8 lanes: one run of N/10 trials in A1, the rest in runs of about 60. */
function load(sqlite, n) {
  const cols = [
    "id", "parent_id", "lane", "run_id", "trial_id", "agent_model", "author", "author_tier", "title", "hypothesis",
    "decision", "soft_min_eff", "ratio_vs_incumbent", "real_stan_passed", "real_stan_total", "overhead_ratio", "b2_max",
    "dims", "codex_refs", "code_sha", "metrics_json", "verified", "state", "created_at",
  ];
  const ins = sqlite.prepare(`INSERT INTO nodes (${cols.join(", ")}) VALUES (${cols.map(() => "?").join(", ")})`);
  const edge = sqlite.prepare("INSERT OR IGNORE INTO node_parents (child, parent, relation, created_at) VALUES (?, ?, ?, ?)");
  const longRun = Math.round(n / 10);
  const runCount = Math.max(8, Math.round((n - longRun) / 60));
  const runs = [{ lane: "A1", run_id: "long-run", size: longRun }];
  for (let r = 0; r < runCount; r += 1) runs.push({ lane: LANES[r % LANES.length], run_id: `run${r}`, size: 0 });
  for (let i = 0; i < n - longRun; i += 1) runs[1 + (i % runCount)].size += 1;

  let rnd = 12345;
  const rand = () => ((rnd = (rnd * 1103515245 + 12345) % 2147483648) / 2147483648);
  const rows = [];
  for (const run of runs) for (let k = 0; k < run.size; k += 1) rows.push({ run, k, t: rand() });
  rows.sort((a, b) => a.t - b.t); // runs interleave in time
  const incumbent = new Map();
  const last = new Map();
  sqlite.exec("BEGIN");
  rows.forEach(({ run, k }, i) => {
    const id = `${run.lane}:${run.run_id}:T${String(k + 1).padStart(4, "0")}`;
    const u = rand();
    const decision = u < 0.15 ? "keep" : u < 0.25 ? "archive_only" : u < 0.6 ? "discard_no_gain" : "discard_tier1";
    const stan = rand() < 0.02;
    const parent = incumbent.get(run.run_id) ?? null;
    const created = new Date(Date.UTC(2026, 0, 1) + i * 30_000).toISOString();
    ins.run(
      id, parent, run.lane, run.run_id, `T${k + 1}`, MODELS[(k + run.run_id.length) % MODELS.length],
      `agent-${run.run_id}`, "provisional", `T${k + 1}: ${HYPOTHESIS.slice(0, 70)}`, HYPOTHESIS, decision,
      rand() / 100, 1 + rand(), stan ? Math.floor(rand() * 6) : null, stan ? 5 : null, 4 + rand(), 0.1, "[6,20]",
      '["fisher-low-rank"]', "cbc54bbbd03d8511", METRICS, 0, "unverified", created
    );
    if (parent) edge.run(id, parent, "incumbent", created);
    if (rand() < 0.03 && last.size > 0) edge.run(id, [...last.values()][Math.floor(rand() * last.size)], "combines", created);
    if (decision === "keep") incumbent.set(run.run_id, id);
    last.set(run.run_id, id);
  });
  const target = incumbent.get("long-run");
  for (let j = 0; j < 30; j += 1) {
    sqlite
      .prepare("INSERT INTO notes (id, kind, lane, title, body, state, author, node_ids, created_at, updated_at) VALUES (?, 'insight', 'A1', 'Note', 'body', 'open', 'x', ?, ?, ?)")
      .run(`note_${j}`, JSON.stringify([target]), `2026-02-01T00:00:${String(j).padStart(2, "0")}Z`, `2026-02-01T00:00:${String(j).padStart(2, "0")}Z`);
  }
  sqlite.exec("COMMIT");
  return { target, total: n };
}

async function measure(env, clock, makeRequest, repeats = 9) {
  const samples = [];
  for (let k = 0; k < repeats + 2; k += 1) {
    clock.sqlMs = 0;
    const req = makeRequest(k);
    const t = performance.now();
    const resp = await handleRequest(req, env);
    const jsMs = performance.now() - t - clock.sqlMs;
    const body = await resp.text();
    assert.ok(resp.status < 300, `${req.method} ${req.url} -> ${resp.status}: ${body.slice(0, 300)}`);
    if (k >= 2) samples.push(jsMs); // two warm-up calls
  }
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)];
}

async function profile(n) {
  const { sqlite, db, clock } = makeDb();
  const { target } = load(sqlite, n);
  const env = { DB: db }; // no FLYWHEEL_API_KEY: local maintainer, no write caps
  const get = (p) => () => new Request(`https://x${p}`);
  const enc = encodeURIComponent(target);
  const results = {
    "/api/overview": await measure(env, clock, get("/api/overview")),
    "/api/runs": await measure(env, clock, get("/api/runs")),
    "/api/runs?limit=60 (max page)": await measure(env, clock, get("/api/runs?limit=60")),
    "/api/nodes (first page)": await measure(env, clock, get("/api/nodes")),
    "/api/nodes/:id": await measure(env, clock, get(`/api/nodes/${enc}`)),
    "/api/frontier": await measure(env, clock, get("/api/frontier")),
    "/api/dag": await measure(env, clock, get("/api/dag")),
    "POST /api/nodes": await measure(env, clock, (k) =>
      new Request("https://x/api/nodes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: `A1:long-run:P${n}-${k}`, lane: "A1", run_id: "long-run", trial_id: `P${k}`, decision: "keep",
          soft_min_eff: 0.5, hypothesis: HYPOTHESIS, metrics_json: METRICS, dims: [6, 20], codex_refs: ["fisher-low-rank"],
          parent_ids: [{ id: target, relation: "incumbent" }], source: `${SOURCE}# ${k}\n`,
        }),
      })
    ),
  };
  return { results, trials: Number(sqlite.prepare("SELECT COUNT(*) AS c FROM nodes").get().c) };
}

it("keeps the Worker's JavaScript cost per read flat from 1,000 to 20,000 trials", { timeout: 300_000 }, async () => {
  const small = await profile(1000);
  const large = await profile(20000);
  assert.ok(large.trials >= 20000);
  const rows = Object.keys(small.results).map((k) => [k, small.results[k], large.results[k]]);
  const table = [
    "| Endpoint | JS ms @ 1,000 trials | JS ms @ 20,000 trials |",
    "| --- | ---: | ---: |",
    ...rows.map(([k, a, b]) => `| ${k} | ${a.toFixed(2)} | ${b.toFixed(2)} |`),
  ].join("\n");
  console.log(`\n${table}\n`);
  for (const [k, a, b] of rows) {
    // Flat within timer noise: never more than double plus 1.5 ms.
    assert.ok(b <= 2 * a + 1.5, `${k} grew from ${a.toFixed(2)} ms to ${b.toFixed(2)} ms`);
  }
});
