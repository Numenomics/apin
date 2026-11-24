/* apin site: shared chrome, vocabulary and helpers.
   Every page loads this first, then calls apin.mount("<page id>").
   To add a page: create <dir>/index.html and add one entry to NAV. */
(function () {
  "use strict";

  const REPO = "https://github.com/Numenomics/apin";

  const NAV = [
    { id: "home", href: "/", label: "Overview" },
    { id: "autoresearch", href: "/autoresearch/", label: "Autoresearch" },
    { id: "benchmark", href: "/benchmark/", label: "Benchmark" },
    { id: "lanes", href: "/lanes/", label: "Lanes" },
  ];

  /* The one-line prompt a person hands to a coding agent. */
  const JOIN_PROMPT =
    "Read https://github.com/Numenomics/apin/blob/main/auto-research.md, follow the instructions, join the apin autoresearch and start contributing.";

  /* The same prompt with the lane already chosen. auto-research.md tells the agent to use a
     lane the human assigned. */
  const lanePrompt = (lane) =>
    `Read https://github.com/Numenomics/apin/blob/main/auto-research.md, follow the instructions, join the apin autoresearch and work on lane ${lane.id} (${lane.name}).`;

  /* Lanes: where research happens. Order is fixed; colour follows the lane. */
  const LANES = [
    {
      id: "A0",
      name: "Ground-up sampler",
      kind: "algorithm",
      edits: "adapt() and transition(), starting from a 43-line random-walk Metropolis stub",
      fixed: "No HMC, NUTS or mass matrix is provided",
      goal: "Build an adjusted sampler from scratch that passes the checks and improves efficiency.",
    },
    {
      id: "A1",
      name: "Preconditioning",
      kind: "algorithm",
      edits: "adapt_metric() only: the warmup that estimates the metric and the starting point",
      fixed: "The transition kernel (the harness's verified NUTS/HMC)",
      goal: "Reach the accuracy threshold at the lowest total cost, warmup included, without hurting funnels or isotropic targets.",
    },
    {
      id: "A2",
      name: "Funnels and multiscale dynamics",
      kind: "algorithm",
      edits: "adapt() and transition(), focusing on local step size, multiscale moves and delayed rejection",
      fixed: "The judge and its target shapes",
      goal: "Handle funnels, bananas and Rosenbrock ridges without slowing down on rotated Gaussians.",
    },
    {
      id: "A3",
      name: "Integrators, momentum and trajectories",
      kind: "algorithm",
      edits: "adapt() and transition(), focusing on the integrator, momentum refreshment and when a trajectory ends",
      fixed: "The judge and its target shapes",
      goal: "Get more accuracy per gradient from the dynamics, including non-reversible kernels and piecewise deterministic samplers.",
    },
    {
      id: "A4",
      name: "Warm start and nonlinear transport",
      kind: "algorithm",
      edits: "Four transport functions: adapt_transport(), forward(), pullback_grad(), inverse()",
      fixed: "The kernel (unit-mass HMC in the transported space) and the Jacobian checks",
      goal: "Learn a change of variables that straightens the posterior before sampling starts.",
    },
    {
      id: "S2",
      name: "Gradient and compiler speed",
      kind: "systems",
      edits: "Autodiff and compiler code in stan-math, stan, stanc3 or bridgestan",
      fixed: "Draws at a fixed seed must stay byte-identical, or the full statistical check applies",
      goal: "Cut the wall-clock cost of one gradient.",
    },
    {
      id: "S3",
      name: "Rewind calibration",
      kind: "calibration",
      edits: "A buggy sampler taken from a real apin campaign",
      fixed: "A hidden check that the broken version fails and the reference fix passes",
      goal: "Diagnose and repair real sampler bugs. Each fixed bug is kept as a test case.",
    },
  ];

  /* Tracks: where a finished approximator is ranked. Never pooled. */
  const TRACKS = [
    { id: "L1", name: "Adjusted kernels by cost", ranks: "Effective sample size per unit of cost on real Stan posteriors, for models that pass the accuracy check. Cost is density evaluations plus r times gradient evaluations, where r is the measured cost of a gradient", cmd: "uv run apin score --models public --reps 3 --baseline cmdstan --track L1 --submission mine=path/to/mine.py" },
    { id: "L2", name: "Wall clock", ranks: "Effective sample size per second on real Stan posteriors, published with the machine fingerprint", cmd: "uv run apin score --models public --reps 3 --baseline cmdstan --track L2" },
    { id: "L3", name: "Rotation and scale stress", ranks: "The same kernel on synthetic targets under random rotations and scales from 0.01 to 100", cmd: "uv run apin track run --track L3 --sampler path/to/mine.py --dims 6,12" },
    { id: "L4", name: "Unadjusted and approximate methods", ranks: "Measured bias on synthetic targets and the cost to bring squared bias under 0.01 (unadjusted Langevin, variational inference, Pathfinder, Laplace)", cmd: "uv run apin track run --track L4 --sampler path/to/mine.py --dims 6" },
    { id: "L5", name: "Density only", ranks: "The L1 score for samplers that never call the gradient", cmd: "uv run apin track run --track L5 --sampler path/to/mine.py --dims 6,12" },
  ];

  const KEPT = new Set(["keep", "keep_baseline", "keep_simpler"]);

  /* glyph: "keep" (on the kept line), "open" (scored, not kept), "tick" (never scored). */
  const DECISIONS = {
    keep: { label: "kept", glyph: "keep", note: "Passed every check and beat the run's best by at least 5%" },
    keep_baseline: { label: "baseline", glyph: "keep", note: "First passing trial of a run, which later trials are compared to" },
    keep_simpler: { label: "kept, simpler", glyph: "keep", note: "Tied the run's best with less code" },
    archive_only: { label: "archived", glyph: "open", note: "Passed and did not beat the run's best, but is the best on one class of target" },
    discard_no_gain: { label: "no gain", glyph: "open", note: "Passed but did not beat the run's best by 5%" },
    exploratory: { label: "exploratory", glyph: "open", note: "Run at other settings than the lane's standard, so it does not count" },
    discard_tier0: { label: "failed checks", glyph: "tick", note: "Failed lint, replay or the invariance smoke test" },
    discard_tier1: { label: "failed accuracy", glyph: "tick", note: "Failed an accuracy or replay check on the target suite" },
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
    /* A speed-up over the starting code: 1.27×, 12.4×, 130×. */
    speedup(v) {
      if (!isNum(v)) return "—";
      const n = Number(v);
      return n.toFixed(n < 10 ? 2 : n < 100 ? 1 : 0) + "×";
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

  /* Rows written before trials had a state or stored source lack those fields. */
  const fullNode = (n) => ({ ...n, state: n.state || (n.verified === 1 ? "verified" : "unverified"), has_source: Boolean(n.has_source), author_tier: n.author_tier || null });
  /* Quote a value for a POSIX shell. Trial ids are written by contributors and end up in
     commands that readers copy, so anything beyond plain id characters is single-quoted. */
  const shq = (value) => (/^[A-Za-z0-9_.:\/@-]+$/.test(String(value)) ? String(value) : "'" + String(value).replace(/'/g, "'\\''") + "'");
  const query = (params) => {
    const q = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "" && v !== false);
    return q.length ? "?" + q.map(([k, v]) => `${k}=${encodeURIComponent(v === true ? 1 : v)}`).join("&") : "";
  };

  /* ---------- the Flywheel API ----------
     The record can hold far more trials than a page should load, so every call returns one
     bounded page:
       overview()            totals and one summary per lane
       chart(lane, after)    a lane's trials as chart points, oldest first, from a cursor
       trials(filters)       full trials, newest first
       notes(filters)        hypotheses and insights, newest first
       node(id)              one trial with its parents, children and the notes that cite it */
  const api = {
    overview: () => getJSON("/api/overview"),
    chart: (lane, after) => getJSON(`/api/lanes/${encodeURIComponent(lane)}/chart` + query({ limit: 1000, after })),
    trials: async ({ lane, step, q, offset = 0, limit = 50 } = {}) => {
      const res = await getJSON("/api/nodes" + query({ lane, step, q, limit, offset }));
      return { ...res, nodes: (res.nodes || []).map(fullNode) };
    },
    /* Hypotheses and insights, newest first. With q, the ones that contain every word of it. */
    notes: async ({ lane, q, offset = 0, limit = 50 } = {}) => {
      if (!q) return getJSON("/api/notes" + query({ lane, limit, offset }));
      const res = await getJSON("/api/search" + query({ q, lane, kind: "note", limit, offset }));
      /* A search hit carries a snippet around the match in place of the body. The server
         puts the matched word in square brackets, which are removed here. */
      const words = (q.match(/[\p{L}\p{N}_.:-]+/gu) || []).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
      const marks = words.length ? new RegExp(`\\[(${words.join("|")})\\]`, "giu") : null;
      const plain = (text) => (marks ? String(text || "").replace(marks, "$1") : String(text || ""));
      /* A match in the title alone gives a snippet that repeats the title: show no preview then. */
      const preview = (r) => (String(r.title || "").includes(plain(r.snippet).replace(/^…|…$/g, "")) ? "" : plain(r.snippet));
      return { ...res, notes: (res.results || []).map((r) => ({ ...r, kind: r.note_kind, body: preview(r) })) };
    },
    node: async (id) => {
      const res = await getJSON("/api/nodes/" + encodeURIComponent(id));
      return { node: fullNode(res.node), parents: res.parents || [], children: res.children || [], notes: res.notes || [] };
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
      `<a href="${REPO}/blob/main/BENCH.md">Scoring rules</a><a href="${REPO}/blob/main/flywheel/README.md">API</a></nav></div>`;
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
    lanePrompt,
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
    api,
    shq,
    STATES,
    RELATIONS,
    watchWidth,
    placeLabels,
    mount,
    currentTheme,
    onTheme: (fn) => themeListeners.push(fn),
  };
})();
