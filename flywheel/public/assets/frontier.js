/* Research frontier page. Data: GET /api/dag (live) or /snapshot.json (shipped with the site). */
(function () {
  "use strict";
  const { $, esc, fmt, svg, status, glyph, tip, KEPT, DECISIONS, isNum } = apin;

  apin.mount("frontier");

  const state = { graph: null, summaries: [], selected: null, redraw: [] };

  const decision = (n) => DECISIONS[n.decision] || { label: n.decision, glyph: "discard", note: "" };
  const outcomeOf = (n) => (KEPT.has(n.decision) || n.decision === "keep_incumbent_failed" ? "kept" : n.decision === "archive_only" ? "archived" : "discarded");
  const byTime = (a, b) => String(a.created_at).localeCompare(String(b.created_at)) || String(a.id).localeCompare(String(b.id));
  const headline = (n) => String(n.hypothesis || n.title || "No hypothesis recorded").replace(/^T\d+:\s*/, "");
  const clip = (s, n) => (s.length > n ? s.slice(0, n - 1).trimEnd() + "…" : s);

  function nodeTip(n) {
    const stan = n.real_stan_total > 0 ? `<br><span class="k">Real Stan</span> ${n.real_stan_passed} of ${n.real_stan_total} passed` : "";
    return (
      `<b>${esc(n.lane)} ${esc(n.trial_id || "")}</b> · ${esc(decision(n).label)}<br>${esc(clip(headline(n), 150))}<br>` +
      `<span class="k">Efficiency</span> ${fmt.eff(n.soft_min_eff)} · <span class="k">vs incumbent</span> ${fmt.times(n.ratio_vs_incumbent)}${stan}`
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
          `<td>${esc((t && (t.author || t.agent_model)) || "—")}</td></tr>`
        );
      })
      .join("");
    $("#lane-table").innerHTML =
      "<thead><tr><th>Lane</th><th>Best kept trial</th><th class='num'>Efficiency</th><th class='num'>Gain within its run</th>" +
      `<th>Real Stan posteriors</th><th class='num'>Trials</th><th>Progress</th><th>Found by</th></tr></thead><tbody>${body}</tbody>`;
  }

  /* ---------- the field: one row per run ---------- */

  function drawGlyph(g, n) {
    const d = decision(n).glyph;
    const color = apin.laneColor(n.lane);
    const ring = { stroke: "var(--paper)", "stroke-width": 2 };
    if (d === "base") g.appendChild(svg("rect", { x: -5.5, y: -5.5, width: 11, height: 11, rx: 1.5, fill: color, ...ring }));
    else if (d === "simpler") g.appendChild(svg("path", { d: "M0-7.5L7.5 0 0 7.5-7.5 0z", fill: color, ...ring }));
    else if (d === "archive") g.appendChild(svg("circle", { r: 4.6, fill: "var(--paper)", stroke: color, "stroke-width": 2 }));
    else if (d === "discard") g.appendChild(svg("path", { d: "M-3.4-3.4l6.8 6.8M3.4-3.4l-6.8 6.8", stroke: "var(--ink-3)", "stroke-width": 1.8, "stroke-linecap": "round", fill: "none" }));
    else g.appendChild(svg("circle", { r: 6, fill: color, ...ring }));
    if (n.real_stan_total > 0) g.appendChild(svg("circle", { r: 9.5, fill: "none", stroke: "var(--ink)", "stroke-width": 1.4 }));
  }

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
    const lanes = state.summaries.filter((r) => r.nodes.length);
    const runsOf = (r) => {
      const map = new Map();
      r.nodes.forEach((n) => (map.has(n.run_id) ? map.get(n.run_id).push(n) : map.set(n.run_id, [n])));
      return [...map].map(([id, nodes]) => ({ id, nodes: nodes.sort(byTime) })).sort((a, b) => byTime(a.nodes[0], b.nodes[0]));
    };
    const laid = lanes.map((r) => ({ r, runs: runsOf(r) }));
    const longest = Math.max(...laid.flatMap((l) => l.runs.map((run) => run.nodes.length)));
    const left = width < 720 ? 150 : 250;
    const step = Math.max(20, Math.min(34, (width - left - 30) / Math.max(1, longest)));
    const W = Math.max(width, left + 30 + longest * step);
    const laneHead = 34;
    const rowH = 48;
    const drop = 17;
    let y = 6;
    const pos = new Map();

    const lines = svg("g", { fill: "none" });
    const marks = svg("g");
    const text = svg("g");

    for (const { r, runs } of laid) {
      const color = apin.laneColor(r.lane.id);
      if (y > 6) el.appendChild(svg("line", { x1: 0, x2: W, y1: y, y2: y, stroke: "var(--rule)" }));
      text.appendChild(svg("rect", { x: 0, y: y + 13, width: 10, height: 10, rx: 2, fill: color }));
      text.appendChild(svg("text", { class: "lane-name", x: 18, y: y + 23 }, `${r.lane.id}  ${r.lane.name}`));
      y += laneHead;
      for (const run of runs) {
        const spine = y + 14;
        const who = run.nodes[0].author || run.nodes[0].agent_model || "";
        text.appendChild(svg("text", { class: "run-name", x: 18, y: spine + 1 }, clip(run.id, width < 720 ? 18 : 30)));
        text.appendChild(svg("text", { x: 18, y: spine + 15 }, clip(`${who}, ${run.nodes.length} ${run.nodes.length === 1 ? "trial" : "trials"}`, width < 720 ? 20 : 34)));
        const x0 = left + step / 2;
        lines.appendChild(svg("line", { x1: x0, x2: left + step * (run.nodes.length - 0.5), y1: spine, y2: spine, stroke: "var(--rule-2)", "stroke-width": 1 }));
        run.nodes.forEach((n, i) => {
          const onLine = outcomeOf(n) === "kept";
          pos.set(n.id, { x: left + step * (i + 0.5), y: onLine ? spine : spine + drop, onLine });
        });
        /* the kept line: each kept trial joined to the kept trial it replaced */
        let prev = null;
        for (const n of run.nodes) {
          const p = pos.get(n.id);
          if (!p.onLine) {
            const from = pos.get(n.parent_id);
            const fx = from ? from.x : p.x;
            lines.appendChild(svg("path", { d: `M${fx} ${spine}C${fx} ${spine + drop} ${p.x - 4} ${spine + drop} ${p.x} ${spine + drop}`, stroke: "var(--rule-2)", "stroke-width": 1, "stroke-dasharray": "2 3" }));
            continue;
          }
          if (prev) lines.appendChild(svg("line", { x1: prev.x, x2: p.x, y1: spine, y2: spine, stroke: color, "stroke-width": 2 }));
          prev = p;
        }
        y += rowH;
      }
      y += 8;
    }

    for (const n of g.nodes) {
      const p = pos.get(n.id);
      if (!p) continue;
      const node = svg("g", {
        class: "node" + (state.selected === n.id ? " is-selected" : ""),
        transform: `translate(${p.x.toFixed(1)},${p.y})`,
        tabindex: 0,
        role: "button",
        "data-node": n.id,
        "aria-label": `${n.lane} ${n.trial_id || ""}, ${decision(n).label}: ${clip(headline(n), 90)}`,
      });
      node.appendChild(svg("circle", { r: 11, fill: "transparent" }));
      drawGlyph(node, n);
      node.appendChild(svg("circle", { class: "pick", r: 12 }));
      tip.bind(node, () => nodeTip(n));
      marks.appendChild(node);
    }
    el.append(lines, marks, text);
    el.setAttribute("width", W);
    el.setAttribute("height", y);
    el.setAttribute("viewBox", `0 0 ${W} ${y}`);
  }

  function renderFieldLegend() {
    const c = "var(--ink-2)";
    $("#field-legend").innerHTML =
      `<span>${glyph("base", c)}baseline</span><span>${glyph("keep", c)}kept</span><span>${glyph("simpler", c)}kept, simpler</span>` +
      `<span>${glyph("archive", c)}archived</span><span>${glyph("discard", c)}not kept</span><span>${glyph("verified", c)}also run on real Stan</span>`;
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
      el.appendChild(svg("path", { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round" }));
      const last = s.pts[s.pts.length - 1];
      const dot = svg("circle", { cx: sx(last.x), cy: sy(last.y), r: 4.5, fill: color, stroke: "var(--paper)", "stroke-width": 2 });
      const hit = svg("circle", { cx: sx(last.x), cy: sy(last.y), r: 14, fill: "transparent" });
      tip.bind(hit, `<b>${esc(s.lane)} ${esc(apin.laneName(s.lane))}</b><br>${fmt.times(last.y)} its run's first kept trial after ${last.x} trials`);
      el.append(dot, hit);
      ends.push({ lane: s.lane, x: sx(last.x) + 9, y: sy(last.y) + 4 });
    }
    /* keep the end labels apart */
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i += 1) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
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
      const cx = sx(ratio(p)) + (dup % 2 ? -1 : 1) * Math.ceil(dup / 2) * 11;
      const cy = sy(p.real_stan_pass_rate);
      const node = svg("g", { class: "node", transform: `translate(${cx.toFixed(1)},${cy.toFixed(1)})`, style: "cursor:pointer", tabindex: 0, role: "button", "data-node": p.id, "aria-label": `${p.lane} ${p.trial_id}: ${p.real_stan_passed} of ${p.real_stan_total} Stan models passed` });
      node.appendChild(svg("circle", { r: 14, fill: "transparent" }));
      node.appendChild(svg("circle", { r: 6, fill: apin.laneColor(p.lane), stroke: "var(--paper)", "stroke-width": 2 }));
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
        if (outcome === "verified" ? !(n.real_stan_total > 0) : outcome && outcomeOf(n) !== outcome) return false;
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
            `<td><code>${esc(n.run_id)}</code> ${esc(n.trial_id || "")}</td><td>${esc(n.author || n.agent_model || "—")}</td>` +
            `<td>${apin.decisionMark(n.decision, n.lane)}</td><td class="num">${fmt.eff(n.soft_min_eff)}</td><td class="num">${fmt.times(n.ratio_vs_incumbent)}</td>` +
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
    const parent = n.parent_id && g.byId.has(n.parent_id) ? `<button type="button" data-node="${esc(n.parent_id)}">compared against ${esc(g.byId.get(n.parent_id).trial_id || n.parent_id)}</button>` : "";
    const kids = (g.children.get(n.id) || [])
      .sort(byTime)
      .map((c) => `<button type="button" data-node="${esc(c.id)}">${esc(c.trial_id || c.id)}, ${esc(decision(c).label)}</button>`)
      .join("");
    const refs = (n.codex_refs || []).map((r) => `<a class="tag" href="${apin.REPO}/blob/main/codex/cards/${encodeURIComponent(r)}.md">${esc(r)}</a>`).join(" ");
    const d = decision(n);

    apin.drawer.open({
      title: `${n.lane} ${apin.laneName(n.lane)}, trial ${n.trial_id || ""}`,
      sub: `${n.id} · ${n.author || n.agent_model || "unknown"} · ${fmt.date(n.created_at)}`,
      body:
        `<div>${apin.decisionMark(n.decision, n.lane)}${n.verified === 1 ? " &nbsp; " + status("good", "reproduced by a maintainer") : ""}` +
        `<p class="caption" style="margin-top:6px">${esc(d.note || "")}</p></div>` +
        `<div><h4>Hypothesis</h4><p class="quote">${esc(n.hypothesis || "No hypothesis was recorded for this trial.")}</p></div>` +
        `<div class="kv">${facts.join("")}</div>` +
        (refs ? `<div><h4>Technique cards cited</h4><div class="chips">${refs}</div></div>` : "") +
        (parent || kids ? `<div><h4>Before and after</h4><div class="chips">${parent}${kids}</div></div>` : ""),
    });
  }

  document.addEventListener("click", (e) => {
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
  }

  async function load() {
    $("#refresh").disabled = true;
    try {
      state.graph = await apin.loadGraph();
      state.summaries = apin.laneSummaries(state.graph);
      renderMeta();
      renderLaneTable();
      renderFieldLegend();
      renderClaims();
      fillFilters();
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
  apin.onTheme(() => state.redraw.forEach((fn) => fn()));
  load();
})();
