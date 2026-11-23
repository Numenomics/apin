/* apin site — shared chrome, vocabulary and helpers.
   Every page loads this first, then calls apin.mount("<page id>").
   To add a page: create <dir>/index.html and add one entry to NAV. */
(function () {
  "use strict";

  const REPO = "https://github.com/Numenomics/apin";

  const NAV = [
    { id: "home", href: "/", label: "Overview" },
    { id: "frontier", href: "/frontier/", label: "Research frontier" },
    { id: "benchmark", href: "/benchmark/", label: "Agent benchmark" },
    { id: "lanes", href: "/lanes/", label: "Lanes and tracks" },
  ];

  /* The one-line prompt a person hands to a coding agent. */
  const JOIN_PROMPT =
    "Read https://github.com/Numenomics/apin/blob/main/auto-research.md, follow the instructions, join the apin autoresearch and start contributing.";

  /* Lanes: where research happens. Order is fixed; colour follows the lane. */
  const LANES = [
    {
      id: "A0",
      name: "Ground-up sampler",
      kind: "algorithm",
      edits: "adapt() and transition(), starting from a 43-line random-walk Metropolis stub",
      fixed: "Nothing is provided: no HMC, NUTS or mass matrix",
      goal: "Build an adjusted sampler from scratch that passes the checks and improves efficiency.",
      start: 'uv run apin autoresearch eval --lane A0 --run-id <handle>_a0 \\\n  --dims 6,20 --repeats 2 \\\n  --hypothesis "baseline: unmodified local incumbent"',
    },
    {
      id: "A1",
      name: "Preconditioning",
      kind: "algorithm",
      edits: "adapt_metric() only: the warmup that estimates the metric and the starting point",
      fixed: "The transition kernel (the harness's verified NUTS/HMC)",
      goal: "Reach the accuracy gate at the lowest total cost, warmup included, without hurting funnels or isotropic targets.",
      start: 'uv run apin autoresearch eval --lane A1 --run-id <handle>_a1 \\\n  --dims 6,20 --repeats 2 \\\n  --hypothesis "baseline: unmodified local incumbent"',
    },
    {
      id: "A2",
      name: "Funnels and multiscale dynamics",
      kind: "algorithm",
      edits: "adapt() and transition(), focusing on local step size, multiscale moves and delayed rejection",
      fixed: "The judge and its target shapes",
      goal: "Handle funnels, bananas and Rosenbrock ridges without slowing down on rotated Gaussians.",
      start: 'uv run apin autoresearch eval --lane A2 --run-id <handle>_a2 \\\n  --dims 6,20 --repeats 2 \\\n  --hypothesis "baseline: unmodified local incumbent"',
    },
    {
      id: "A3",
      name: "Integrators, momentum and trajectories",
      kind: "algorithm",
      edits: "adapt() and transition(), focusing on the integrator, momentum refreshment and when a trajectory ends",
      fixed: "The judge and its target shapes",
      goal: "Get more accuracy per gradient from the dynamics, including non-reversible kernels and PDMPs.",
      start: 'uv run apin autoresearch eval --lane A3 --run-id <handle>_a3 \\\n  --dims 6,20 --repeats 2 \\\n  --hypothesis "baseline: unmodified local incumbent"',
    },
    {
      id: "A4",
      name: "Warm start and nonlinear transport",
      kind: "algorithm",
      edits: "Four transport functions: adapt_transport(), forward(), pullback_grad(), inverse()",
      fixed: "The kernel (unit-mass HMC in the transported space) and the Jacobian checks",
      goal: "Learn a change of variables that straightens the posterior before sampling starts.",
      start: 'uv run apin autoresearch eval --lane A4 --run-id <handle>_a4 \\\n  --dims 6,20 --repeats 2 \\\n  --hypothesis "baseline: unmodified local incumbent"',
    },
    {
      id: "S1",
      name: "Upstream sampler patches",
      kind: "systems",
      edits: "Source files in walnutpie, stan or nuts-rs, inside the paths you declare",
      fixed: "Upstream unit tests and apin's statistical gate must both pass",
      goal: "Make an existing engine sample better or faster without changing what it computes wrongly.",
      start: 'uv run apin autoresearch eval-s1 --upstream walnutpie \\\n  --allowed-path src/ --dry-run \\\n  --run-id <handle>_s1 --hypothesis "..."',
    },
    {
      id: "S2",
      name: "Gradient and compiler speed",
      kind: "systems",
      edits: "Autodiff and compiler code in stan-math, stan, stanc3 or bridgestan",
      fixed: "Draws at a fixed seed must stay byte-identical, or the full statistical gate applies",
      goal: "Cut the wall-clock cost of one gradient.",
      start: 'uv run apin autoresearch eval-s2 --upstream math \\\n  --allowed-path stan/math/rev/ --dry-run \\\n  --run-id <handle>_s2 --hypothesis "..."',
    },
    {
      id: "S3",
      name: "Rewind calibration",
      kind: "calibration",
      edits: "A buggy sampler taken from a real apin campaign",
      fixed: "A hidden check that the broken version fails and the reference fix passes",
      goal: "Grade agents on diagnosing real sampler bugs. These tasks feed the agent benchmark.",
      start: "uv run apin autoresearch eval-s3 --mode reference",
    },
  ];

  /* Tracks: where a finished approximator is ranked. Never pooled. */
  const TRACKS = [
    { id: "L1", name: "Adjusted kernels by cost", ranks: "Accuracy-gated ESS per cost unit, cost = density evaluations + r × gradient evaluations", cmd: "uv run apin score --models public --reps 3 --baseline cmdstan --track L1 --submission mine=path/to/mine.py" },
    { id: "L2", name: "Wall clock", ranks: "ESS per second, published with the machine fingerprint", cmd: "uv run apin score --models public --reps 3 --baseline cmdstan --track L2" },
    { id: "L3", name: "Rotation and scale stress", ranks: "The same kernel under random rotations and scales from 0.01 to 100", cmd: "uv run apin track run --track L3 --sampler path/to/mine.py --dims 6,12" },
    { id: "L4", name: "Unadjusted and approximate methods", ranks: "Measured bias and the cost to bring squared bias under 0.01 (ULA, VI, Pathfinder, Laplace)", cmd: "uv run apin track run --track L4 --sampler path/to/mine.py --dims 6" },
    { id: "L5", name: "Density only", ranks: "The L1 score for samplers that never call the gradient", cmd: "uv run apin track run --track L5 --sampler path/to/mine.py --dims 6,12" },
    { id: "S2", name: "Upstream patches", ranks: "Wall-clock gain of an autodiff or compiler patch with byte-identical draws", cmd: 'uv run apin autoresearch eval-s2 --upstream math --allowed-path stan/math/rev/ --run-id <handle>_s2 --hypothesis "..."' },
  ];

  const KEPT = new Set(["keep", "keep_baseline", "keep_simpler", "keep_incumbent_failed"]);

  /* glyph: "keep" (on the kept line), "open" (scored, not kept), "tick" (never scored). */
  const DECISIONS = {
    keep: { label: "kept", glyph: "keep", note: "Passed every check and beat the incumbent" },
    keep_baseline: { label: "baseline", glyph: "keep", note: "First passing trial of a run; later trials are compared to it" },
    keep_simpler: { label: "kept, simpler", glyph: "keep", note: "Tied the incumbent with less code" },
    keep_incumbent_failed: { label: "kept, unpaired", glyph: "keep", note: "Passed while the paired incumbent failed; gain not yet confirmed" },
    archive_only: { label: "archived", glyph: "open", note: "Passed, did not beat the incumbent, best in its niche" },
    discard_no_gain: { label: "no gain", glyph: "open", note: "Passed but did not beat the incumbent" },
    discard_tier0: { label: "failed checks", glyph: "tick", note: "Failed lint, replay or the invariance smoke test" },
    discard_tier1: { label: "failed accuracy", glyph: "tick", note: "Failed an accuracy or replay gate on the target suite" },
    discard_duplicate: { label: "duplicate", glyph: "tick", note: "Same code as an earlier trial" },
  };

  /* ---------- tiny DOM helpers ---------- */

  function esc(v) {
    return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  const SVG_NS = "http://www.w3.org/2000/svg";
  function svg(tag, attrs, text) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const k in attrs || {}) {
      if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, String(attrs[k]));
    }
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function $(sel, root) {
    return (root || document).querySelector(sel);
  }

  /* ---------- formatters ---------- */

  const isNum = (v) => v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v));

  const fmt = {
    eff(v) {
      if (!isNum(v)) return "—";
      const n = Number(v);
      if (n === 0) return "0";
      if (Math.abs(n) < 1e-4) return n.toExponential(2);
      return n.toPrecision(3);
    },
    num(v, digits) {
      return isNum(v) ? Number(v).toFixed(digits ?? 2) : "—";
    },
    times(v, digits) {
      return isNum(v) ? Number(v).toFixed(digits ?? 2) + "×" : "—";
    },
    int(v) {
      return isNum(v) ? Math.round(Number(v)).toLocaleString("en-US") : "—";
    },
    pct(v) {
      return isNum(v) ? Math.round(Number(v) * 100) + "%" : "—";
    },
    usd(v) {
      if (!isNum(v)) return "—";
      const n = Number(v);
      if (n === 0) return "$0";
      return "$" + n.toFixed(n < 0.1 ? 3 : 2);
    },
    tokens(v) {
      if (!isNum(v)) return "—";
      const n = Number(v);
      if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
      if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
      if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
      return String(n);
    },
    dur(v) {
      if (!isNum(v)) return "—";
      const s = Number(v);
      if (s >= 3600) return (s / 3600).toFixed(1) + " h";
      if (s >= 60) return Math.round(s / 60) + " min";
      return s.toFixed(0) + " s";
    },
    date(iso) {
      if (!iso) return "—";
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return String(iso);
      return d.toISOString().slice(0, 16).replace("T", " ") + " UTC";
    },
    day(iso) {
      const d = new Date(iso);
      return Number.isNaN(d.getTime()) ? "—" : d.toISOString().slice(0, 10);
    },
  };

  /* ---------- marks ---------- */

  const ICONS = {
    check: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3.2 3.2L13 4.8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    cross: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    dash: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8h9" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    alert: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2.2l6.2 11.1H1.8z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M8 6.6v3.2M8 11.6v.2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
    sun: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="3.1" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M8 1.3v1.8M8 12.9v1.8M1.3 8h1.8M12.9 8h1.8M3.3 3.3l1.3 1.3M11.4 11.4l1.3 1.3M3.3 12.7l1.3-1.3M11.4 4.6l1.3-1.3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
    close: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 3.5l9 9M12.5 3.5l-9 9" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
    mark: '<svg viewBox="0 0 24 24" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M3 18c4-9 14-9 18 0"/><path d="M6.5 18c2.6-5.4 8.4-5.4 11 0"/><path d="M10.2 18c1-1.6 2.6-1.6 3.6 0"/></g></svg>',
  };

  function status(kind, label) {
    const icon = kind === "good" ? ICONS.check : kind === "bad" ? ICONS.cross : kind === "warn" ? ICONS.alert : ICONS.dash;
    return `<span class="status ${kind}">${icon}${esc(label)}</span>`;
  }

  function laneColor(lane) {
    return LANES.some((l) => l.id === lane) ? `var(--lane-${lane})` : "var(--lane-none)";
  }

  function laneTag(lane) {
    return `<span class="lane-tag"><i class="swatch" style="background:${laneColor(lane)}"></i>${esc(lane)}</span>`;
  }

  function laneName(lane) {
    const hit = LANES.find((l) => l.id === lane);
    return hit ? hit.name : lane;
  }

  /* The three marks of the trial charts, reused by their legends and the ledger. */
  function glyph(kind, color, size) {
    const s = size || 14;
    const c = color || "currentColor";
    const open = `<svg viewBox="-8 -8 16 16" width="${s}" height="${s}" aria-hidden="true">`;
    if (kind === "open") return `${open}<circle r="3.4" fill="none" stroke="${c}" stroke-width="1.4"/></svg>`;
    if (kind === "tick") return `${open}<path d="M0-4.5V4.5" stroke="${c}" stroke-width="1.6" stroke-linecap="round"/></svg>`;
    if (kind === "stan") return `${open}<circle r="6.2" fill="none" stroke="${c}" stroke-width="1.1"/><circle r="3" fill="${c}"/></svg>`;
    if (kind === "line") return `${open}<path d="M-7 3H-1V-3H7" fill="none" stroke="${c}" stroke-width="1.5"/></svg>`;
    return `${open}<circle r="3.6" fill="${c}"/></svg>`;
  }
  function decisionMark(decision, lane) {
    const d = DECISIONS[decision] || { label: decision, glyph: "tick" };
    const quiet = d.glyph !== "keep";
    return `<span class="status${quiet ? " mute" : ""}" title="${esc(d.note || "")}">${glyph(d.glyph, quiet ? "var(--ink-3)" : laneColor(lane), 13)}${esc(d.label)}</span>`;
  }

  /* ---------- tooltip ---------- */

  let tipEl = null;
  const tip = {
    show(evt, html) {
      if (!tipEl) {
        tipEl = document.createElement("div");
        tipEl.className = "tip";
        tipEl.setAttribute("role", "tooltip");
        document.body.appendChild(tipEl);
      }
      tipEl.innerHTML = html;
      tipEl.hidden = false;
      const pad = 14;
      const x = Math.min(window.innerWidth - tipEl.offsetWidth - 8, evt.clientX + pad);
      const y = Math.min(window.innerHeight - tipEl.offsetHeight - 8, evt.clientY + pad);
      tipEl.style.left = Math.max(8, x) + "px";
      tipEl.style.top = Math.max(8, y) + "px";
    },
    hide() {
      if (tipEl) tipEl.hidden = true;
    },
    bind(node, html) {
      const get = typeof html === "function" ? html : () => html;
      node.addEventListener("mouseenter", (e) => tip.show(e, get()));
      node.addEventListener("mousemove", (e) => tip.show(e, get()));
      node.addEventListener("mouseleave", tip.hide);
    },
  };

  /* ---------- drawer ---------- */

  let drawerEl = null;
  let shadeEl = null;
  let lastFocus = null;
  const drawer = {
    open({ title, sub, body, onReady }) {
      if (!drawerEl) {
        shadeEl = document.createElement("div");
        shadeEl.className = "drawer-shade";
        shadeEl.hidden = true;
        shadeEl.addEventListener("click", drawer.close);
        drawerEl = document.createElement("aside");
        drawerEl.className = "drawer";
        drawerEl.setAttribute("role", "dialog");
        drawerEl.setAttribute("aria-modal", "true");
        drawerEl.setAttribute("aria-labelledby", "drawer-title");
        drawerEl.innerHTML =
          '<header><div><h3 id="drawer-title"></h3><div class="sub"></div></div>' +
          `<button class="icon-btn" type="button" aria-label="Close details">${ICONS.close}</button></header><div class="body"></div>`;
        drawerEl.querySelector("button").addEventListener("click", drawer.close);
        document.body.append(shadeEl, drawerEl);
        document.addEventListener("keydown", (e) => {
          if (e.key === "Escape") drawer.close();
        });
      }
      if (!drawerEl.classList.contains("open")) lastFocus = document.activeElement;
      $("#drawer-title", drawerEl).textContent = title || "";
      $(".sub", drawerEl).textContent = sub || "";
      const bodyEl = $(".body", drawerEl);
      bodyEl.innerHTML = body || "";
      bodyEl.scrollTop = 0;
      shadeEl.hidden = false;
      drawerEl.classList.add("open");
      drawerEl.querySelector("button").focus({ preventScroll: true });
      if (onReady) onReady(bodyEl);
    },
    close() {
      if (!drawerEl) return;
      drawerEl.classList.remove("open");
      shadeEl.hidden = true;
      if (lastFocus && lastFocus.focus) lastFocus.focus({ preventScroll: true });
    },
  };

  /* ---------- copyable commands ---------- */

  function cmd(text, opts) {
    const cls = opts && opts.prose ? "cmd wrap-text" : text.includes("\n") ? "cmd multi" : "cmd";
    return `<div class="${cls}"><pre>${esc(text)}</pre><button class="copy" type="button" data-copy="${esc(text.replace(/\\\n\s*/g, ""))}">Copy</button></div>`;
  }

  document.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-copy]");
    if (!btn) return;
    const text = btn.getAttribute("data-copy");
    try {
      await navigator.clipboard.writeText(text);
      const old = btn.textContent;
      btn.textContent = "Copied";
      setTimeout(() => (btn.textContent = old), 1400);
    } catch {
      btn.textContent = "Press Ctrl+C";
      const range = document.createRange();
      const pre = btn.parentElement.querySelector("pre");
      if (pre) {
        range.selectNodeContents(pre);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      }
    }
  });

  /* ---------- data ---------- */

  async function getJSON(url) {
    const resp = await fetch(url, { headers: { Accept: "application/json" } });
    if (!resp.ok) throw new Error(`${url}: HTTP ${resp.status}`);
    const type = resp.headers.get("content-type") || "";
    if (!type.includes("json")) throw new Error(`${url}: not JSON`);
    return resp.json();
  }

  /* The research graph: live from the Flywheel API, or the snapshot shipped with the site. */
  /* What has happened to a trial since it was posted. */
  const STATES = {
    unverified: { label: "not rerun yet", tone: "mute", note: "Only its author has run this trial." },
    "needs-replication": { label: "rerun requested", tone: "warn", note: "A contributor asked for an independent rerun." },
    replicated: { label: "rerun by someone else", tone: "good", note: "An independent rerun on fresh seeds reached the claimed efficiency." },
    verified: { label: "verified by a maintainer", tone: "good", note: "A maintainer reproduced it." },
    disputed: { label: "disputed", tone: "bad", note: "A rerun did not reach the claimed efficiency, or a contributor flagged it." },
    retracted: { label: "retracted", tone: "mute", note: "Its author withdrew it." },
  };
  const RELATIONS = { incumbent: "scored against", combines: "combines", fork: "forked from" };

  /* ---------- where the pages get trials from ----------
     The record can hold far more trials than a page should load, so no page asks for all of
     them. A store answers four bounded questions, always in the same shapes:
       overview()            totals, one summary per lane, claims, alignment points, model names
       runs({lane, offset})  a page of runs, newest first, each with slim trials for its chart
       trials(filters)       a page of full trials, newest first
       node(id)              one trial with its parents, children and the notes that cite it
     The live store asks the Flywheel API. The snapshot store answers the same questions from
     the file shipped with the site, and is used when the API does not answer. */

  const byTime = (a, b) => String(a.created_at).localeCompare(String(b.created_at)) || String(a.id).localeCompare(String(b.id));
  const keptScored = (n) => KEPT.has(n.decision) && isNum(n.soft_min_eff) && n.soft_min_eff > 0;
  const outcomeOf = (n) => (KEPT.has(n.decision) ? "kept" : n.decision === "archive_only" ? "archived" : "discarded");
  /* Rows written before trials had a state or stored source lack those fields. */
  const fullNode = (n) => ({ ...n, state: n.state || (n.verified === 1 ? "verified" : "unverified"), has_source: Boolean(n.has_source), author_tier: n.author_tier || null });
  const query = (params) => {
    const q = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "" && v !== false);
    return q.length ? "?" + q.map(([k, v]) => `${k}=${encodeURIComponent(v === true ? 1 : v)}`).join("&") : "";
  };

  function liveStore(overview) {
    return {
      source: "live",
      overview: async () => overview,
      runs: ({ lane, offset = 0, limit = 24 } = {}) => getJSON("/api/runs" + query({ lane, limit, offset })),
      trials: async ({ lane, model, outcome, state, stan, q, offset = 0, limit = 50 } = {}) => {
        const res = await getJSON("/api/nodes" + query({ lane, model, outcome, state, stan, q, limit, offset }));
        return { ...res, nodes: (res.nodes || []).map(fullNode) };
      },
      node: async (id) => {
        const res = await getJSON("/api/nodes/" + encodeURIComponent(id));
        return { node: fullNode(res.node), parents: res.parents || [], children: res.children || [], notes: res.notes || [] };
      },
    };
  }

  function snapshotStore(data) {
    const nodes = (Array.isArray(data.nodes) ? data.nodes : []).map(fullNode).sort(byTime);
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const edges = (Array.isArray(data.edges) ? data.edges : []).map((e) => ({ ...e, relation: e.relation || "incumbent" }));
    for (const n of nodes) {
      if (n.parent_id && !edges.some((e) => e.source === n.parent_id && e.target === n.id)) edges.push({ source: n.parent_id, target: n.id, relation: "incumbent" });
    }
    const brief = (id, relation) => {
      const n = byId.get(id) || {};
      return { id, relation, lane: n.lane || null, trial_id: n.trial_id || null, decision: n.decision || null };
    };
    const page = (list, offset, limit) => ({ count: list.length, limit, offset, next_offset: offset + limit < list.length ? offset + limit : null });

    function overview() {
      const lanes = [...new Set(nodes.map((n) => n.lane))].sort().map((lane) => {
        const mine = nodes.filter((n) => n.lane === lane);
        const frontier = (data.frontier || {})[lane] ? fullNode(data.frontier[lane]) : mine.filter(keptScored).sort((x, y) => y.soft_min_eff - x.soft_min_eff)[0] || null;
        const run = frontier ? mine.filter((n) => n.run_id === frontier.run_id) : [];
        const first = run.find(keptScored) || null;
        const steps = [];
        let best = null;
        for (const n of run) {
          if (keptScored(n) && (best === null || n.soft_min_eff > best)) best = n.soft_min_eff;
          if (best !== null) steps.push(best);
        }
        const stan = mine.filter((n) => n.real_stan_total > 0).sort((x, y) => y.real_stan_passed / y.real_stan_total - x.real_stan_passed / x.real_stan_total || byTime(y, x))[0] || null;
        return {
          lane,
          trials: mine.length,
          kept: mine.filter((n) => KEPT.has(n.decision)).length,
          runs: new Set(mine.map((n) => n.run_id)).size,
          latest_at: mine[mine.length - 1].created_at,
          frontier,
          run: frontier ? { run_id: frontier.run_id, first_eff: first ? first.soft_min_eff : null, gain: first && first.soft_min_eff > 0 ? frontier.soft_min_eff / first.soft_min_eff : null, steps } : null,
          stan,
        };
      });
      return {
        totals: {
          trials: nodes.length,
          kept: nodes.filter((n) => KEPT.has(n.decision)).length,
          runs: new Set(nodes.map((n) => n.lane + "/" + n.run_id)).size,
          contributors: new Set(nodes.map((n) => n.author || n.agent_model).filter(Boolean)).size,
          claims: (data.claims || []).length,
          latest_at: nodes.length ? nodes[nodes.length - 1].created_at : null,
        },
        lanes,
        claims: Array.isArray(data.claims) ? data.claims : [],
        alignment: Array.isArray(data.alignment) ? data.alignment : [],
        models: [...new Set(nodes.map((n) => n.agent_model).filter(Boolean))].sort(),
      };
    }

    function runs({ lane, offset = 0, limit = 24 } = {}) {
      const map = new Map();
      for (const n of nodes) {
        if (lane && n.lane !== lane) continue;
        const key = n.lane + "/" + n.run_id;
        if (map.has(key)) map.get(key).push(n);
        else map.set(key, [n]);
      }
      const all = [...map.values()]
        .map((list) => {
          const base = list.find(keptScored);
          return {
            lane: list[0].lane,
            run_id: list[0].run_id,
            author: list[0].author || null,
            author_tier: list[0].author_tier,
            agent_model: list[0].agent_model || null,
            trials: list.length,
            kept: list.filter((n) => KEPT.has(n.decision)).length,
            first_at: list[0].created_at,
            last_at: list[list.length - 1].created_at,
            base_eff: base ? base.soft_min_eff : null,
            truncated: list.length > 150,
            nodes: list.slice(-150).map((n) => ({ id: n.id, trial_id: n.trial_id, decision: n.decision, soft_min_eff: n.soft_min_eff, state: n.state, stan: n.real_stan_total > 0, created_at: n.created_at })),
          };
        })
        .sort((x, y) => String(y.last_at).localeCompare(String(x.last_at)));
      return { runs: all.slice(offset, offset + limit), ...page(all, offset, limit) };
    }

    function trials({ lane, model, outcome, state, stan, q, offset = 0, limit = 50 } = {}) {
      const terms = String(q || "").toLowerCase().split(/\s+/).filter(Boolean);
      const list = nodes
        .filter((n) => {
          if (lane && n.lane !== lane) return false;
          if (model && n.agent_model !== model) return false;
          if (outcome && outcomeOf(n) !== outcome) return false;
          if (state && n.state !== state) return false;
          if (stan && !(n.real_stan_total > 0)) return false;
          if (!terms.length) return true;
          const hay = `${n.id} ${n.title || ""} ${n.hypothesis || ""} ${n.code_sha || ""} ${(n.codex_refs || []).join(" ")} ${n.author || ""} ${n.agent_model || ""}`.toLowerCase();
          return terms.every((t) => hay.includes(t));
        })
        .reverse();
      return { nodes: list.slice(offset, offset + limit), ...page(list, offset, limit) };
    }

    function node(id) {
      if (!byId.has(id)) throw new Error("unknown trial " + id);
      return {
        node: byId.get(id),
        parents: edges.filter((e) => e.target === id).map((e) => brief(e.source, e.relation)),
        children: edges.filter((e) => e.source === id).map((e) => brief(e.target, e.relation)),
        notes: [],
      };
    }

    return { source: "snapshot", overview: async () => overview(), runs: async (a) => runs(a), trials: async (a) => trials(a), node: async (id) => node(id) };
  }

  async function openStore() {
    try {
      return liveStore(await getJSON("/api/overview"));
    } catch {
      return snapshotStore(await getJSON("/snapshot.json"));
    }
  }

  /* One row per lane of the guide, joined with what the overview knows about it. */
  function laneRows(overview) {
    const known = new Map((overview.lanes || []).map((l) => [l.lane, l]));
    return LANES.map((lane) => {
      const o = known.get(lane.id);
      const run = (o && o.run) || {};
      return {
        lane,
        trials: o ? o.trials : 0,
        kept: o ? o.kept : 0,
        runs: o ? o.runs : 0,
        top: o && o.frontier ? fullNode(o.frontier) : null,
        gain: isNum(run.gain) ? run.gain : null,
        first: isNum(run.first_eff) ? run.first_eff : null,
        steps: Array.isArray(run.steps) ? run.steps : [],
        stan: (o && o.stan) || null,
      };
    });
  }

  /* Hypotheses, insights and the rerun queue exist only on the live Flywheel. */
  async function loadNotes() {
    try {
      return (await getJSON("/api/notes?limit=100")).notes || [];
    } catch {
      return null;
    }
  }
  async function loadQueue() {
    try {
      return (await getJSON("/api/verify/queue?limit=50")).items || [];
    } catch {
      return null;
    }
  }

  /* Per-lane summary used by the home, frontier and lanes pages. */

  function sparkline(values, color) {
    if (!values || values.length < 2) return "";
    const w = 96;
    const h = 22;
    const logs = values.map((v) => Math.log10(Math.max(v, 1e-12)));
    const lo = Math.min(...logs);
    const hi = Math.max(...logs);
    const span = hi - lo || 1;
    const pts = logs.map((v, i) => {
      const x = (i / (logs.length - 1)) * (w - 4) + 2;
      const y = h - 3 - ((v - lo) / span) * (h - 6);
      return (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
    });
    return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="Best efficiency over trials, rising from ${fmt.eff(values[0])} to ${fmt.eff(values[values.length - 1])}"><path d="${pts.join(" ")}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/></svg>`;
  }

  /* ---------- agent benchmark helpers (shared by the overview and the leaderboard) ---------- */

  const benchIndex = new WeakMap();
  /* Benchmark data helpers. `rel` is one release block of /data/benchmark.json: releases are
     scored under different rules and are never mixed. A "cell" is one setting on one task,
     summed over all its attempts. */
  const bench = {
    release(data, tag) {
      const list = (data && data.releases) || [];
      return list.find((r) => r.release === tag) || list[0] || null;
    },
    /* Fix tasks first, then speed-up tasks; alphabetical inside each group. */
    tasks(rel) {
      const kind = (id) => bench.meta(rel, id).category;
      return [...(rel.tasks || [])].sort((a, b) => (kind(a) === kind(b) ? a.localeCompare(b) : kind(a) === "rewind" ? -1 : 1));
    },
    meta(rel, taskId) {
      const m = (rel.task_meta || {})[taskId] || {};
      return {
        id: taskId,
        category: m.category || (taskId.startsWith("s3-") ? "rewind" : "optimize"),
        title: m.title || taskId,
        lane: m.lane || null,
        difficulty: m.difficulty || null,
        denominators: m.denominators || null,
      };
    },
    trials(rel, configKey, taskId) {
      if (!benchIndex.has(rel)) {
        const map = new Map();
        for (const t of rel.trials || []) {
          const key = t.config_key + "||" + t.task_id;
          if (map.has(key)) map.get(key).push(t);
          else map.set(key, [t]);
        }
        benchIndex.set(rel, map);
      }
      return benchIndex.get(rel).get(configKey + "||" + taskId) || [];
    },
    cell(row, taskId) {
      return (row.tasks || {})[taskId] || null;
    },
    /* none: never attempted. unfinished: every attempt was cut off. Then, over the scored
       attempts: solved (all), partly (some), unsolved (valid but none solved), invalid (all failed). */
    state(cell) {
      if (!cell) return "none";
      if (!cell.attempts) return "unfinished";
      if (cell.solve_rate >= 1) return "solved";
      if (cell.solve_rate > 0) return "partly";
      const o = cell.outcomes || {};
      return (o.invalid || 0) + (o.unscorable || 0) >= cell.attempts ? "invalid" : "unsolved";
    },
    against(rel) {
      return rel.scoring === "reference" ? "the Stan-style reference" : "the starting code";
    },
    solvedCount(cell) {
      return Math.round((cell.solve_rate || 0) * cell.attempts);
    },
    label(rel, taskId, cell) {
      const st = bench.state(cell);
      if (st === "none") return "not run";
      if (st === "unfinished") return "did not finish";
      const many = cell.attempts > 1 ? `, solved in ${bench.solvedCount(cell)} of ${cell.attempts} attempts` : "";
      if (bench.meta(rel, taskId).category === "rewind") return (st === "solved" ? "bug fixed" : st === "partly" ? "bug fixed in some attempts" : "bug not fixed") + many;
      if (st === "invalid") return "failed the checks" + many;
      return `${fmt.times(cell.ratio_mean)} ${bench.against(rel)}${st === "unsolved" ? ", not solved" : ""}${many}`;
    },
    /* One attempt, in words. */
    outcome(rel, t) {
      const ends = { provider_limit: "the model provider cut it off", infrastructure: "the infrastructure failed", agent_crash: "the agent never started" };
      if (t.outcome === "interrupted") return `did not finish: ${ends[t.termination] || "interrupted"}`;
      if (bench.meta(rel, t.task_id).category === "rewind") return t.outcome === "solved" ? "bug fixed" : "bug not fixed";
      if (t.outcome === "invalid") return "failed the checks";
      if (t.outcome === "unscorable") return "could not be scored";
      const start = isNum(t.start_speedup) ? `${fmt.times(t.start_speedup)} the starting code` : "no ratio recorded";
      if (rel.scoring !== "reference") return t.outcome === "no_gain" ? `no gain: ${start}` : `improved: ${start}`;
      const ref = `${fmt.times(t.ref_speedup)} the Stan-style reference`;
      if (t.outcome === "solved") return `solved: ${ref}, ${start}`;
      if (t.outcome === "improved") return `improved, but below the reference: ${ref}, ${start}`;
      return `no gain: ${start}, ${ref}`;
    },
    pips(rel, row) {
      const tasks = bench.tasks(rel);
      const cls = { solved: "", partly: "part", unsolved: "open", invalid: "fail", unfinished: "infra", none: "none" };
      let html = '<span class="pips">';
      tasks.forEach((id, i) => {
        if (i > 0 && bench.meta(rel, id).category !== bench.meta(rel, tasks[i - 1]).category) html += '<span class="gap"></span>';
        const cell = bench.cell(row, id);
        const label = `${id}: ${bench.label(rel, id, cell)}`;
        html += `<button type="button" class="${cls[bench.state(cell)]}" ${cell ? `data-run="${esc(row.config_key + "||" + id)}"` : "disabled"} title="${esc(label)}" aria-label="${esc(label)}"></button>`;
      });
      return html + "</span>";
    },
  };

  /* Redraw a chart whenever its container changes width, so SVG text keeps its size. */
  function watchWidth(node, render) {
    let last = -1;
    const draw = () => {
      const w = Math.round(node.clientWidth);
      if (w > 0 && w !== last) {
        last = w;
        render(w);
      }
    };
    if ("ResizeObserver" in window) new ResizeObserver(draw).observe(node);
    else window.addEventListener("resize", draw);
    draw();
    return () => {
      last = -1;
      draw();
    };
  }

  /* Greedy label placement for a handful of scatter points: right, left, above, below. */
  function placeLabels(points, bounds) {
    const taken = points.map((p) => ({ x0: p.x - 8, x1: p.x + 8, y0: p.y - 8, y1: p.y + 8 }));
    const hits = (b) => taken.some((t) => b.x0 < t.x1 && b.x1 > t.x0 && b.y0 < t.y1 && b.y1 > t.y0);
    return points.map((p) => {
      const w = p.label.length * 6.7;
      const options = [
        { anchor: "start", x: p.x + 11, y: p.y + 4, x0: p.x + 11, x1: p.x + 11 + w },
        { anchor: "end", x: p.x - 11, y: p.y + 4, x0: p.x - 11 - w, x1: p.x - 11 },
        { anchor: "middle", x: p.x, y: p.y - 13, x0: p.x - w / 2, x1: p.x + w / 2 },
        { anchor: "middle", x: p.x, y: p.y + 21, x0: p.x - w / 2, x1: p.x + w / 2 },
      ];
      for (const o of options) {
        const box = { x0: o.x0, x1: o.x1, y0: o.y - 11, y1: o.y + 4 };
        if (box.x0 < bounds.x0 || box.x1 > bounds.x1 || box.y0 < bounds.y0 || box.y1 > bounds.y1 || hits(box)) continue;
        taken.push(box);
        return o;
      }
      return null;
    });
  }

  /* ---------- theme + chrome ---------- */

  function currentTheme() {
    const set = document.documentElement.getAttribute("data-theme");
    if (set) return set;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  try {
    const saved = localStorage.getItem("apin-theme");
    if (saved === "light" || saved === "dark") document.documentElement.setAttribute("data-theme", saved);
  } catch {
    /* storage unavailable: follow the system theme */
  }

  const themeListeners = [];

  function mount(pageId) {
    const head = document.createElement("header");
    head.className = "masthead";
    head.innerHTML =
      `<div class="wrap"><a class="wordmark" href="/" aria-label="apin, overview">${ICONS.mark}<span>apin</span></a>` +
      '<nav class="site-nav" aria-label="Site">' +
      NAV.filter((n) => n.id !== "home")
        .map((n) => `<a href="${n.href}"${n.id === pageId ? ' aria-current="page"' : ""}>${esc(n.label)}</a>`)
        .join("") +
      `<a class="push" href="${REPO}">GitHub</a></nav>` +
      `<button class="icon-btn" type="button" id="theme-toggle" aria-label="Switch between light and dark">${ICONS.sun}</button></div>`;
    document.body.prepend(head);

    const foot = document.createElement("footer");
    foot.className = "site-foot";
    foot.innerHTML =
      '<div class="wrap"><span>apin is BSD-3-Clause. Results show only that a sampler passed the stated checks at the stated budget.</span>' +
      `<nav aria-label="Project links"><a href="${REPO}">Repository</a><a href="${REPO}/blob/main/CONTRIBUTING.md">Contribute</a>` +
      `<a href="${REPO}/blob/main/BENCH.md">Scoring rules</a><a href="/api/frontier">Frontier API</a></nav></div>`;
    document.body.append(foot);

    $("#theme-toggle").addEventListener("click", () => {
      const next = currentTheme() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try {
        localStorage.setItem("apin-theme", next);
      } catch {
        /* not persisted */
      }
      themeListeners.forEach((fn) => fn(next));
    });
  }

  window.apin = {
    REPO,
    NAV,
    JOIN_PROMPT,
    LANES,
    TRACKS,
    KEPT,
    DECISIONS,
    ICONS,
    $,
    esc,
    svg,
    fmt,
    isNum,
    status,
    laneColor,
    laneTag,
    laneName,
    glyph,
    decisionMark,
    tip,
    drawer,
    cmd,
    getJSON,
    openStore,
    laneRows,
    outcomeOf,
    loadNotes,
    loadQueue,
    STATES,
    RELATIONS,
    sparkline,
    watchWidth,
    placeLabels,
    bench,
    mount,
    currentTheme,
    onTheme: (fn) => themeListeners.push(fn),
  };
})();
