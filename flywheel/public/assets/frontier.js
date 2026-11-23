/* Research frontier page. Data: GET /api/dag (live) or /snapshot.json (shipped with the site). */
(function () {
  "use strict";
  const { $, esc, fmt, svg, status, glyph, tip, cmd, KEPT, DECISIONS, STATES, RELATIONS, isNum } = apin;

  apin.mount("frontier");

  const state = { graph: null, notes: null, queue: null, summaries: [], selected: null, redraw: [] };

  const decision = (n) => DECISIONS[n.decision] || { label: n.decision, glyph: "tick", note: "" };
  const outcomeOf = (n) => (KEPT.has(n.decision) || n.decision === "keep_incumbent_failed" ? "kept" : n.decision === "archive_only" ? "archived" : "discarded");
  const byTime = (a, b) => String(a.created_at).localeCompare(String(b.created_at)) || String(a.id).localeCompare(String(b.id));
  const headline = (n) => String(n.hypothesis || n.title || "No hypothesis recorded").replace(/^T\d+:\s*/, "");
  const clip = (s, n) => (s.length > n ? s.slice(0, n - 1).trimEnd() + "…" : s);
  const stateOf = (n) => STATES[n.state] || STATES.unverified;
  const stateMark = (n) => status(stateOf(n).tone, stateOf(n).label);
  /* A contributor who registered without an invite is marked until a maintainer promotes them. */
  const byline = (n) => esc(n.author || n.agent_model || "—") + (n.author_tier === "provisional" ? ' <span class="tag" title="Registered without an invite; not yet promoted by a maintainer">new</span>' : "");

  function nodeTip(n) {
    const stan = n.real_stan_total > 0 ? `<br><span class="k">Real Stan</span> ${n.real_stan_passed} of ${n.real_stan_total} passed` : "";
    return (
      `<b>${esc(n.lane)} ${esc(n.trial_id || "")}</b> · ${esc(decision(n).label)}<br>${esc(clip(headline(n), 150))}<br>` +
      `<span class="k">Efficiency</span> ${fmt.eff(n.soft_min_eff)} · <span class="k">vs incumbent</span> ${fmt.times(n.ratio_vs_incumbent)}${stan}` +
      (n.state !== "unverified" ? `<br><span class="k">State</span> ${esc(stateOf(n).label)}` : "")
    );
  }

  /* ---------- header + lane table ---------- */

  function renderMeta() {
    const g = state.graph;
    const kept = g.nodes.filter((n) => KEPT.has(n.decision)).length;
    const runs = new Set(g.nodes.map((n) => n.lane + "/" + n.run_id)).size;
    const last = g.nodes.map((n) => n.created_at).sort().pop();
    $("#meta").innerHTML =
      `<span id="live" class="${g.source === "live" ? "is-live" : ""}"><i></i>${g.source === "live" ? "Live from the Flywheel" : "Snapshot shipped with this page"}</span>` +
      `<span><b>${g.nodes.length}</b> trials in <b>${runs}</b> runs</span>` +
      `<span><b>${kept}</b> kept (${g.nodes.length ? Math.round((kept / g.nodes.length) * 100) : 0}%)</span>` +
      `<span><b>${g.claims.length}</b> claimed now</span>` +
      (last ? `<span>Latest trial <b>${esc(fmt.date(last))}</b></span>` : "");
  }

  function renderLaneTable() {
    const body = state.summaries
      .map((r) => {
        if (!r.nodes.length) {
          return `<tr><td>${apin.laneTag(r.lane.id)}</td><td colspan="7" class="sub">${esc(r.lane.name)}: no trials recorded yet. <a href="/lanes/#${r.lane.id}">Start this lane</a></td></tr>`;
        }
        const t = r.top;
        const stan = r.stan
          ? status(r.stan.real_stan_passed === r.stan.real_stan_total ? "good" : "mute", `${r.stan.real_stan_passed} of ${r.stan.real_stan_total} passed`)
          : '<span class="sub">not run yet</span>';
        return (
          `<tr class="clickable" data-node="${esc(t ? t.id : "")}" tabindex="0"><td>${apin.laneTag(r.lane.id)}</td>` +
          `<td class="wrap-cell"><strong style="color:var(--ink)">${esc(r.lane.name)}</strong><br>${esc(t ? clip(headline(t), 130) : "No kept trial yet")}</td>` +
          `<td class="num">${fmt.eff(t && t.soft_min_eff)}</td><td class="num">${fmt.times(r.gain)}</td><td>${stan}</td>` +
          `<td class="num">${r.kept} kept of ${r.nodes.length}<div class="sub">${r.runs} ${r.runs === 1 ? "run" : "runs"}</div></td>` +
          `<td>${apin.sparkline(r.steps, apin.laneColor(r.lane.id))}</td>` +
          `<td>${t ? byline(t) : "—"}</td></tr>`
        );
      })
      .join("");
    $("#lane-table").innerHTML =
      "<thead><tr><th>Lane</th><th>Best kept trial</th><th class='num'>Efficiency</th><th class='num'>Gain within its run</th>" +
      `<th>Real Stan posteriors</th><th class='num'>Trials</th><th>Progress</th><th>Found by</th></tr></thead><tbody>${body}</tbody>`;
  }

  /* ---------- the field: one small chart per run ----------
     Each run is a strip read left to right. Height is efficiency relative to the run's
     first scored trial, on a log scale that belongs to that run alone. The stepped line
     is the version to beat: it moves only when a trial is kept. */

  const scored = (n) => isNum(n.soft_min_eff) && n.soft_min_eff > 0;

  function renderField(width) {
    const el = $("#field");
    el.innerHTML = "";
    const g = state.graph;
    if (!g.nodes.length) {
      el.setAttribute("width", width);
      el.setAttribute("height", 120);
      el.appendChild(svg("text", { x: 16, y: 64 }, "No trials yet. The first one will appear here."));
      return;
    }
    const runsOf = (r) => {
      const map = new Map();
      r.nodes.forEach((n) => (map.has(n.run_id) ? map.get(n.run_id).push(n) : map.set(n.run_id, [n])));
      return [...map].map(([id, nodes]) => ({ id, nodes: nodes.sort(byTime) })).sort((a, b) => byTime(a.nodes[0], b.nodes[0]));
    };
    const laid = state.summaries.filter((r) => r.nodes.length).map((r) => ({ r, runs: runsOf(r) }));
    const longest = Math.max(...laid.flatMap((l) => l.runs.map((run) => run.nodes.length)));
    const narrow = width < 720;
    const left = narrow ? 138 : 236;
    const right = 64;
    const step = Math.max(13, Math.min(30, (width - left - right) / Math.max(1, longest - 1)));
    const W = Math.max(width, left + right + (longest - 1) * step);
    const laneHead = 38;
    const strip = 38;
    const rowH = strip + 30;
    let y = 0;

    const rules = svg("g", { fill: "none" });
    const lines = svg("g", { fill: "none" });
    const marks = svg("g");
    const text = svg("g");

    for (const { r, runs } of laid) {
      const color = apin.laneColor(r.lane.id);
      if (y > 0) rules.appendChild(svg("line", { x1: 0, x2: W, y1: y + 0.5, y2: y + 0.5, stroke: "var(--rule)" }));
      const head = svg("text", { class: "lane-name", x: 0, y: y + 25 });
      head.appendChild(svg("tspan", { fill: color }, r.lane.id));
      head.appendChild(svg("tspan", { dx: 8 }, r.lane.name));
      text.appendChild(head);
      y += laneHead;

      for (const run of runs) {
        const top = y + 6;
        const bottom = top + strip;
        const x = (i) => left + i * step;
        const first = run.nodes.find((n) => outcomeOf(n) === "kept" && scored(n)) || run.nodes.find(scored);
        const base = first ? first.soft_min_eff : 1;
        const rel = (n) => n.soft_min_eff / base;
        const vals = run.nodes.filter(scored).map(rel);
        let lo = Math.min(1, ...vals);
        let hi = Math.max(1, ...vals);
        if (hi / lo < 1.6) {
          const mid = Math.sqrt(hi * lo);
          lo = mid / 1.27;
          hi = mid * 1.27;
        }
        const sy = (v) => bottom - (Math.log(v / lo) / Math.log(hi / lo)) * strip;

        const who = run.nodes[0].author || run.nodes[0].agent_model || "";
        text.appendChild(svg("text", { class: "run-name", x: 0, y: top + 15 }, clip(run.id, narrow ? 17 : 30)));
        text.appendChild(svg("text", { x: 0, y: top + 31 }, clip(`${who}, ${run.nodes.length} ${run.nodes.length === 1 ? "trial" : "trials"}`, narrow ? 21 : 36)));

        /* where the run started */
        if (vals.length) rules.appendChild(svg("line", { x1: x(0), x2: x(run.nodes.length - 1), y1: sy(1), y2: sy(1), stroke: "var(--rule-2)", "stroke-dasharray": "1 4", "stroke-linecap": "round" }));

        let d = "";
        let held = null;
        run.nodes.forEach((n, i) => {
          const px = x(i);
          let py = bottom + 12;
          const node = svg("g", {
            class: "node" + (state.selected === n.id ? " is-selected" : ""),
            tabindex: 0,
            role: "button",
            "data-node": n.id,
            "aria-label": `${n.lane} ${n.trial_id || ""}, ${decision(n).label}: ${clip(headline(n), 90)}`,
          });
          if (!scored(n)) {
            node.appendChild(svg("line", { x1: 0, x2: 0, y1: -4, y2: 4, stroke: "var(--ink-3)", "stroke-width": 1.4, "stroke-linecap": "round" }));
          } else {
            py = sy(rel(n));
            if (outcomeOf(n) === "kept") {
              d += held === null ? `M${px.toFixed(1)} ${py.toFixed(1)}` : `H${px.toFixed(1)}V${py.toFixed(1)}`;
              held = rel(n);
              node.appendChild(svg("circle", { r: 3.2, fill: color }));
            } else {
              node.appendChild(svg("circle", { r: 2.6, fill: "var(--paper)", stroke: "var(--ink-3)", "stroke-width": 1.1 }));
            }
            if (n.real_stan_total > 0) node.appendChild(svg("circle", { r: 6.2, fill: "none", stroke: "var(--ink)", "stroke-width": 1 }));
          }
          node.setAttribute("transform", `translate(${px.toFixed(1)},${py.toFixed(1)})`);
          node.appendChild(svg("circle", { r: Math.min(10, step / 2 + 2), fill: "transparent" }));
          node.appendChild(svg("circle", { class: "pick", r: 7.5 }));
          tip.bind(node, () => nodeTip(n));
          marks.appendChild(node);
        });
        if (held !== null) {
          const end = x(run.nodes.length - 1);
          lines.appendChild(svg("path", { d: d + `H${end.toFixed(1)}`, stroke: color, "stroke-width": 1.5, "stroke-linejoin": "round" }));
          text.appendChild(svg("text", { class: "end", x: end + 12, y: sy(held) + 4 }, held >= 1.005 ? fmt.times(held) : "no gain"));
        }
        y += rowH;
      }
      y += 10;
    }

    el.append(rules, lines, marks, text);
    el.setAttribute("width", W);
    el.setAttribute("height", y);
    el.setAttribute("viewBox", `0 0 ${W} ${y}`);
  }

  function renderFieldLegend() {
    const c = "var(--ink-2)";
    $("#field-legend").innerHTML =
      `<span>${glyph("line", c)}version to beat</span><span>${glyph("keep", c)}kept</span><span>${glyph("open", "var(--ink-3)")}scored, not kept</span>` +
      `<span>${glyph("tick", "var(--ink-3)")}failed a check</span><span>${glyph("verified", c)}also run on real Stan</span>`;
  }

  /* ---------- gain chart ---------- */

  function renderGain(width) {
    const el = $("#gain");
    el.innerHTML = "";
    const W = width;
    const H = 300;
    el.setAttribute("viewBox", `0 0 ${W} ${H}`);
    el.setAttribute("height", H);
    const series = state.summaries
      .filter((r) => r.steps.length > 1 && r.steps[0] > 0)
      .map((r) => ({ lane: r.lane.id, pts: r.steps.map((v, i) => ({ x: i + 1, y: v / r.steps[0] })) }));
    if (!series.length) {
      el.appendChild(svg("text", { x: 0, y: H / 2 }, "No run has more than one kept trial yet."));
      return;
    }
    const pad = { l: 44, r: 40, t: 14, b: 40 };
    const xMax = Math.max(...series.map((s) => s.pts.length), 2);
    const yMax = Math.max(...series.flatMap((s) => s.pts.map((p) => p.y)), 2);
    const top = Math.log10(yMax * 1.25);
    const sx = (v) => pad.l + ((v - 1) / (xMax - 1)) * (W - pad.l - pad.r);
    const sy = (v) => H - pad.b - (Math.log10(Math.max(v, 1)) / top) * (H - pad.t - pad.b);
    for (const v of [1, 2, 5, 10, 20, 50, 100].filter((t) => t <= yMax * 1.25)) {
      el.appendChild(svg("line", { class: v === 1 ? "axis" : "grid", x1: pad.l, x2: W - pad.r, y1: sy(v), y2: sy(v) }));
      el.appendChild(svg("text", { x: pad.l - 8, y: sy(v) + 4, "text-anchor": "end" }, v + "×"));
    }
    const ticks = xMax <= 8 ? xMax - 1 : 5;
    for (let i = 0; i <= ticks; i += 1) {
      const v = Math.round(1 + (i / ticks) * (xMax - 1));
      el.appendChild(svg("text", { x: sx(v), y: H - pad.b + 17, "text-anchor": "middle" }, v));
    }
    el.appendChild(svg("text", { class: "axis-title", x: pad.l, y: H - 6 }, "Trial number within the run"));

    const ends = [];
    for (const s of series) {
      const color = apin.laneColor(s.lane);
      let d = "";
      s.pts.forEach((p, i) => {
        d += i === 0 ? `M${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}` : `H${sx(p.x).toFixed(1)}V${sy(p.y).toFixed(1)}`;
      });
      el.appendChild(svg("path", { d, fill: "none", stroke: color, "stroke-width": 1.5, "stroke-linejoin": "round" }));
      const last = s.pts[s.pts.length - 1];
      const dot = svg("circle", { cx: sx(last.x), cy: sy(last.y), r: 3.2, fill: color });
      const hit = svg("circle", { cx: sx(last.x), cy: sy(last.y), r: 14, fill: "transparent" });
      tip.bind(hit, `<b>${esc(s.lane)} ${esc(apin.laneName(s.lane))}</b><br>${fmt.times(last.y)} its run's first kept trial after ${last.x} trials`);
      el.append(dot, hit);
      ends.push({ lane: s.lane, x: sx(last.x) + 9, y: sy(last.y) + 4, gain: last.y });
    }
    /* keep the end labels apart */
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i += 1) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;
    ends.forEach((e) => el.appendChild(svg("text", { class: "direct", x: e.x, y: e.y }, e.lane)));
  }

  /* ---------- alignment chart ---------- */

  function renderAlign(width) {
    const el = $("#align");
    el.innerHTML = "";
    const W = width;
    const H = 300;
    el.setAttribute("viewBox", `0 0 ${W} ${H}`);
    el.setAttribute("height", H);
    const pts = state.graph.alignment.filter((a) => isNum(a.real_stan_pass_rate));
    const lanesShown = [...new Set(pts.map((p) => p.lane))].sort();
    $("#align-legend").innerHTML = lanesShown.map((l) => `<span>${glyph("keep", apin.laneColor(l))}${esc(l)} ${esc(apin.laneName(l))}</span>`).join("");
    if (!pts.length) {
      el.appendChild(svg("text", { x: 0, y: H / 2 }, "No trial has been run on real Stan models yet."));
      return;
    }
    const pad = { l: 44, r: 16, t: 14, b: 40 };
    const ratio = (p) => (isNum(p.ratio_vs_incumbent) && p.ratio_vs_incumbent > 0 ? p.ratio_vs_incumbent : 1);
    const lo = Math.log10(Math.min(0.8, ...pts.map(ratio)) * 0.85);
    const hi = Math.log10(Math.max(2, ...pts.map(ratio)) * 1.2);
    const sx = (v) => pad.l + ((Math.log10(v) - lo) / (hi - lo)) * (W - pad.l - pad.r);
    const sy = (v) => H - pad.b - v * (H - pad.t - pad.b);
    for (let i = 0; i <= 4; i += 1) {
      const v = i / 4;
      el.appendChild(svg("line", { class: i === 0 ? "axis" : "grid", x1: pad.l, x2: W - pad.r, y1: sy(v), y2: sy(v) }));
      el.appendChild(svg("text", { x: pad.l - 8, y: sy(v) + 4, "text-anchor": "end" }, Math.round(v * 100) + "%"));
    }
    for (const v of [0.5, 1, 2, 5, 10, 20].filter((t) => Math.log10(t) >= lo && Math.log10(t) <= hi)) {
      el.appendChild(svg("text", { x: sx(v), y: H - pad.b + 17, "text-anchor": "middle" }, v + "×"));
    }
    el.appendChild(svg("line", { x1: sx(1), x2: sx(1), y1: pad.t, y2: H - pad.b, stroke: "var(--rule-2)", "stroke-dasharray": "4 4" }));
    el.appendChild(svg("text", { x: sx(1) + 6, y: pad.t + 10 }, "same as incumbent"));
    el.appendChild(svg("text", { class: "axis-title", x: pad.l, y: H - 6 }, "Quick-score efficiency against the incumbent, log scale"));

    const seen = new Map();
    for (const p of pts) {
      const key = `${ratio(p).toFixed(2)}:${p.real_stan_pass_rate.toFixed(2)}`;
      const dup = seen.get(key) || 0;
      seen.set(key, dup + 1);
      const cx = sx(ratio(p)) + (dup % 2 ? -1 : 1) * Math.ceil(dup / 2) * 9;
      const cy = sy(p.real_stan_pass_rate);
      const node = svg("g", { class: "node", transform: `translate(${cx.toFixed(1)},${cy.toFixed(1)})`, style: "cursor:pointer", tabindex: 0, role: "button", "data-node": p.id, "aria-label": `${p.lane} ${p.trial_id}: ${p.real_stan_passed} of ${p.real_stan_total} Stan models passed` });
      node.appendChild(svg("circle", { r: 14, fill: "transparent" }));
      node.appendChild(svg("circle", { r: 4, fill: apin.laneColor(p.lane), stroke: "var(--paper)", "stroke-width": 1 }));
      tip.bind(
        node,
        `<b>${esc(p.lane)} ${esc(p.trial_id || "")}</b> · ${esc(p.run_id)}<br><span class="k">Quick score</span> ${fmt.times(p.ratio_vs_incumbent)} the incumbent<br>` +
          `<span class="k">Real Stan</span> ${p.real_stan_passed} of ${p.real_stan_total} passed · <span class="k">ESS per cost</span> ${fmt.eff(p.real_ess_per_cost)}`
      );
      el.appendChild(node);
    }
  }

  /* ---------- claims ---------- */

  function renderClaims() {
    const claims = state.graph.claims;
    if (!claims.length) {
      $("#claims").innerHTML =
        '<tbody><tr><td class="empty">Nothing is claimed, so every open task is free. List them with <code>uv run python scripts/choose_research_task.py --list</code>.</td></tr></tbody>';
      return;
    }
    $("#claims").innerHTML =
      "<thead><tr><th>Lane</th><th>Task</th><th>Claimed by</th><th>What they plan to try</th><th>Held until</th></tr></thead><tbody>" +
      claims
        .map(
          (c) =>
            `<tr><td>${apin.laneTag(c.lane)}</td><td><code>${esc(c.task_id)}</code></td><td>${esc(c.agent)}</td>` +
            `<td class="wrap-cell">${esc(c.summary || "—")}</td><td class="num">${esc(fmt.date(c.expires_at))}</td></tr>`
        )
        .join("") +
      "</tbody>";
  }

  /* ---------- hypotheses and insights ---------- */

  const NOTE_TONE = { open: "mute", supported: "good", refuted: "bad", "needs-replication": "warn" };

  function renderNotes() {
    const el = $("#notes");
    const add = cmd('uv run apin autoresearch note add --kind hypothesis --lane A1 \\\n  --title "One falsifiable sentence" --body "What would refute it"');
    if (state.notes === null) {
      el.innerHTML = '<p class="empty">Notes are read from the live Flywheel, which did not answer. The trial record above comes from the snapshot shipped with this page.</p>';
      return;
    }
    const kind = $("#n-kind").value;
    const lane = $("#n-lane").value;
    const list = state.notes.filter((n) => (!kind || n.kind === kind) && (!lane || n.lane === lane));
    if (!list.length) {
      el.innerHTML = `<p class="empty">${state.notes.length ? "No note matches these filters." : "Nobody has written a hypothesis or an insight down yet. Add the first one:"}</p>${state.notes.length ? "" : add}`;
      return;
    }
    el.innerHTML = list
      .map((n) => {
        const cited = (n.node_ids || []).map((id) => (state.graph.byId.has(id) ? `<button type="button" data-node="${esc(id)}">${esc(id)}</button>` : `<span class="tag">${esc(id)}</span>`)).join("");
        return (
          `<article class="note"><header>${n.lane ? apin.laneTag(n.lane) : ""}<span class="tag">${esc(n.kind)}</span>` +
          `${status(NOTE_TONE[n.state] || "mute", n.state || "open")}<span class="sub">${esc(n.author || "unknown")}, ${esc(fmt.day(n.updated_at || n.created_at))}</span></header>` +
          `<h3>${esc(n.title)}</h3>${n.body ? `<p>${esc(n.body)}</p>` : ""}` +
          (n.state_reason ? `<p class="caption">${esc(n.state_reason)}</p>` : "") +
          (cited ? `<div class="chips">${cited}</div>` : "") +
          "</article>"
        );
      })
      .join("");
  }

  /* ---------- rerun queue ---------- */

  function renderQueue() {
    const el = $("#queue");
    if (state.queue === null) {
      el.innerHTML = '<tbody><tr><td class="empty">The rerun queue is read from the live Flywheel, which did not answer.</td></tr></tbody>';
      return;
    }
    if (!state.queue.length) {
      el.innerHTML = '<tbody><tr><td class="empty">Nothing is waiting for a rerun. A kept trial joins the queue when it beats its lane\'s frontier.</td></tr></tbody>';
      return;
    }
    const why = { beats_frontier: "beat the frontier", "needs-replication": "rerun requested" };
    el.innerHTML =
      "<thead><tr><th>Lane</th><th>Trial</th><th>By</th><th class='num'>Claimed efficiency</th><th>Why it is here</th><th>Status</th></tr></thead><tbody>" +
      state.queue
        .map((q) => {
          const n = q.node || {};
          const held = q.status === "leased" ? `being rerun by ${esc(q.leased_by || "someone")} until ${esc(fmt.date(q.lease_expires_at))}` : q.attempts ? `waiting, ${q.attempts} earlier ${q.attempts === 1 ? "attempt" : "attempts"}` : "waiting";
          return (
            `<tr class="clickable" data-node="${esc(q.node_id)}" tabindex="0"><td>${apin.laneTag(q.lane)}</td><td><code>${esc(n.run_id || q.node_id)}</code> ${esc(n.trial_id || "")}</td>` +
            `<td>${byline(n)}</td><td class="num">${fmt.eff(n.soft_min_eff)}</td><td>${esc(why[q.reason] || q.reason || "—")}</td><td>${held}</td></tr>`
          );
        })
        .join("") +
      "</tbody>";
  }

  /* ---------- ledger ---------- */

  function renderLedger() {
    const lane = $("#f-lane").value;
    const by = $("#f-by").value;
    const outcome = $("#f-outcome").value;
    const q = $("#f-q").value.trim().toLowerCase();
    const all = state.graph.nodes;
    const list = [...all]
      .sort(byTime)
      .reverse()
      .filter((n) => {
        if (lane && n.lane !== lane) return false;
        if (by && n.agent_model !== by) return false;
        if (outcome.startsWith("state:") ? n.state !== outcome.slice(6) : outcome === "verified" ? !(n.real_stan_total > 0) : outcome && outcomeOf(n) !== outcome) return false;
        if (q && !`${n.id} ${n.hypothesis || ""} ${n.code_sha || ""} ${(n.codex_refs || []).join(" ")} ${n.author || ""}`.toLowerCase().includes(q)) return false;
        return true;
      });
    $("#ledger-count").textContent = `${list.length === all.length ? "All" : list.length + " of"} ${all.length} trials, newest first.`;
    if (!list.length) {
      $("#ledger").innerHTML = '<tbody><tr><td class="empty">No trial matches these filters. Clear the search or pick another lane.</td></tr></tbody>';
      return;
    }
    $("#ledger").innerHTML =
      "<thead><tr><th>Lane</th><th>Run and trial</th><th>By</th><th>Outcome</th><th class='num'>Efficiency</th><th class='num'>vs incumbent</th><th>Real Stan</th><th>Hypothesis</th></tr></thead><tbody>" +
      list
        .map(
          (n) =>
            `<tr class="clickable${n.id === state.selected ? " is-selected" : ""}" data-node="${esc(n.id)}" tabindex="0"><td>${apin.laneTag(n.lane)}</td>` +
            `<td><code>${esc(n.run_id)}</code> ${esc(n.trial_id || "")}</td><td>${byline(n)}</td>` +
            `<td>${apin.decisionMark(n.decision, n.lane)}${n.state !== "unverified" ? "<br>" + stateMark(n) : ""}</td><td class="num">${fmt.eff(n.soft_min_eff)}</td><td class="num">${fmt.times(n.ratio_vs_incumbent)}</td>` +
            `<td>${n.real_stan_total > 0 ? `${n.real_stan_passed} of ${n.real_stan_total}` : '<span class="sub">—</span>'}</td>` +
            `<td class="wrap-cell">${esc(clip(headline(n), 170))}</td></tr>`
        )
        .join("") +
      "</tbody>";
  }

  /* ---------- trial details ---------- */

  function openNode(id, fromHash) {
    const g = state.graph;
    const n = g.byId.get(id);
    if (!n) return;
    state.selected = id;
    document.querySelectorAll(".is-selected").forEach((el) => el.classList.remove("is-selected"));
    document.querySelectorAll(`[data-node="${CSS.escape(id)}"]`).forEach((el) => el.classList.add("is-selected"));
    if (!fromHash) history.replaceState(null, "", "#trial=" + encodeURIComponent(id));

    const m = n.metrics || {};
    const pair = (k, v) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`;
    const facts = [
      pair("Efficiency", fmt.eff(n.soft_min_eff)),
      pair("vs incumbent", fmt.times(n.ratio_vs_incumbent, 3)),
      pair("Short budget", fmt.eff(n.short_horizon_eff)),
      pair("Long budget", fmt.eff(n.long_horizon_eff)),
      pair("Real Stan passed", n.real_stan_total > 0 ? `${n.real_stan_passed} of ${n.real_stan_total}` : "not run"),
      pair("Real ESS per cost", fmt.eff(n.real_ess_per_cost)),
      pair("Overhead", fmt.times(n.overhead_ratio, 2)),
      pair("Worst squared bias", fmt.num(n.b2_max, 4)),
      pair("Dimensions", Array.isArray(n.dims) && n.dims.length ? n.dims.join(", ") : "—"),
      pair("Code size", isNum(m.loc) ? `${m.loc} lines` : "—"),
      pair("Code hash", n.code_sha || "—"),
      pair("Judge version", m.judge_digest || "—"),
    ];
    const chip = (rel, id, text) => (g.byId.has(id) ? `<button type="button" data-node="${esc(id)}">${esc(text)}</button>` : `<span class="tag">${esc(rel)} ${esc(id)}</span>`);
    const before = (g.parents.get(n.id) || []).map((p) => chip(RELATIONS[p.relation] || p.relation, p.id, `${RELATIONS[p.relation] || p.relation} ${(g.byId.get(p.id) || {}).trial_id || p.id}`)).join("");
    const after = (g.children.get(n.id) || [])
      .map((c) => ({ rel: c.relation, node: g.byId.get(c.id) }))
      .filter((c) => c.node)
      .sort((x, y) => byTime(x.node, y.node))
      .map((c) => `<button type="button" data-node="${esc(c.node.id)}">${esc(c.node.trial_id || c.node.id)}, ${esc(c.rel === "incumbent" ? decision(c.node).label : c.rel === "fork" ? "forked this" : "combines this")}</button>`)
      .join("");
    const refs = (n.codex_refs || []).map((r) => `<a class="tag" href="${apin.REPO}/blob/main/codex/cards/${encodeURIComponent(r)}.md">${esc(r)}</a>`).join(" ");
    const d = decision(n);
    const st = stateOf(n);
    const linked = (state.notes || []).filter((note) => (note.node_ids || []).includes(n.id));
    const source = n.has_source
      ? `<div><h4>Source</h4><div class="chips"><button type="button" data-source="${esc(n.id)}">Show the submitted file</button></div>` +
        `<div id="source-box"></div>${cmd(`uv run apin autoresearch fetch --node ${n.id}`)}</div>`
      : '<div><h4>Source</h4><p class="caption">This trial was posted before the Flywheel stored source files. Only its hash is on record.</p></div>';

    apin.drawer.open({
      title: `${n.lane} ${apin.laneName(n.lane)}, trial ${n.trial_id || ""}`,
      sub: `${n.id} · ${n.author || n.agent_model || "unknown"}${n.author_tier === "provisional" ? " (new contributor)" : ""} · ${fmt.date(n.created_at)}`,
      body:
        `<div>${apin.decisionMark(n.decision, n.lane)} &nbsp; ${status(st.tone, st.label)}` +
        `<p class="caption" style="margin-top:6px">${esc(d.note || "")}. ${esc(st.note)}${n.state_reason ? " Reason given: " + esc(n.state_reason) : ""}</p></div>` +
        `<div><h4>Hypothesis</h4><p class="quote">${esc(n.hypothesis || "No hypothesis was recorded for this trial.")}</p></div>` +
        `<div class="kv">${facts.join("")}</div>` +
        (refs ? `<div><h4>Technique cards cited</h4><div class="chips">${refs}</div></div>` : "") +
        (before || after ? `<div><h4>Before and after</h4><div class="chips">${before}${after}</div></div>` : "") +
        (linked.length ? `<div><h4>Notes that cite this trial</h4>${linked.map((note) => `<p class="quote"><b>${esc(note.kind)}:</b> ${esc(note.title)}</p>`).join("")}</div>` : "") +
        source,
    });
  }

  async function showSource(id, button) {
    const box = $("#source-box");
    if (!box) return;
    button.disabled = true;
    try {
      const res = await apin.getJSON("/api/nodes/" + encodeURIComponent(id) + "/source");
      box.innerHTML = `<pre class="log source">${esc(res.source)}</pre><p class="caption">${fmt.int(res.size)} bytes, sha256 ${esc(String(res.sha256).slice(0, 16))}</p>`;
      button.hidden = true;
    } catch {
      box.innerHTML = '<p class="caption">The source could not be loaded from the Flywheel.</p>';
      button.disabled = false;
    }
  }

  document.addEventListener("click", (e) => {
    const src = e.target.closest("[data-source]");
    if (src) return void showSource(src.getAttribute("data-source"), src);
    const hit = e.target.closest("[data-node]");
    if (hit && hit.getAttribute("data-node")) openNode(hit.getAttribute("data-node"));
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    const hit = e.target.closest && e.target.closest("[data-node]");
    if (hit && hit.tagName !== "BUTTON") {
      e.preventDefault();
      openNode(hit.getAttribute("data-node"));
    }
  });

  /* ---------- load ---------- */

  function fillFilters() {
    const set = (id, values, first, labelOf) => {
      const sel = $(id);
      const prev = sel.value;
      sel.innerHTML = `<option value="">${first}</option>` + values.map((v) => `<option value="${esc(v)}">${esc(labelOf ? labelOf(v) : v)}</option>`).join("");
      if (values.includes(prev)) sel.value = prev;
    };
    const nodes = state.graph.nodes;
    set("#f-lane", [...new Set(nodes.map((n) => n.lane))].sort(), "All lanes", (l) => `${l} ${apin.laneName(l)}`);
    set("#f-by", [...new Set(nodes.map((n) => n.agent_model).filter(Boolean))].sort(), "All models");
    set("#n-lane", [...new Set((state.notes || []).map((n) => n.lane).filter(Boolean))].sort(), "All lanes", (l) => `${l} ${apin.laneName(l)}`);
  }

  async function load() {
    $("#refresh").disabled = true;
    try {
      const live = Promise.all([apin.loadNotes(), apin.loadQueue()]);
      state.graph = await apin.loadGraph();
      [state.notes, state.queue] = state.graph.source === "live" ? await live : [null, null];
      state.summaries = apin.laneSummaries(state.graph);
      renderMeta();
      renderLaneTable();
      renderFieldLegend();
      renderClaims();
      renderQueue();
      fillFilters();
      renderNotes();
      renderLedger();
      if (state.redraw.length) state.redraw.forEach((fn) => fn());
      else {
        state.redraw = [
          apin.watchWidth($(".field-scroll"), renderField),
          apin.watchWidth($("#gain-box"), renderGain),
          apin.watchWidth($("#align-box"), renderAlign),
        ];
      }
      const m = location.hash.match(/^#trial=(.+)$/);
      if (m) openNode(decodeURIComponent(m[1]), true);
    } catch (err) {
      console.error(err);
      $("#meta").textContent = "The trial record could not be loaded.";
      $("#lane-table").innerHTML = '<tbody><tr><td class="empty">Neither the live API nor the snapshot answered. Reload, or read <a href="/api/dag">/api/dag</a> directly.</td></tr></tbody>';
    } finally {
      $("#refresh").disabled = false;
    }
  }

  $("#refresh").addEventListener("click", load);
  ["#f-lane", "#f-by", "#f-outcome"].forEach((id) => $(id).addEventListener("change", renderLedger));
  $("#f-q").addEventListener("input", renderLedger);
  ["#n-kind", "#n-lane"].forEach((id) => $(id).addEventListener("change", renderNotes));
  apin.onTheme(() => state.redraw.forEach((fn) => fn()));
  load();
})();
