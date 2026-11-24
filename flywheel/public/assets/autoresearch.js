/* Autoresearch page: one lane at a time. The chart shows every trial of the lane that was
   scored on the lane's standard setup, and the best efficiency so far as a stepped line.
   All data comes from apin.api in bounded pages. */
(function () {
  "use strict";
  const { $, esc, fmt, svg, status, tip, cmd, api, DECISIONS, STATES, RELATIONS, KEPT, isNum } = apin;

  apin.mount("autoresearch");

  /* The chart reads at most this many pages of 1,000 points per lane, and draws at most
     MAX_DOTS of the trials that did not move the line. Every step is always drawn. */
  const MAX_PAGES = 20;
  const MAX_DOTS = 6000;

  const state = {
    overview: null,
    lane: null,
    chart: { lane: null, points: [], dims: null, loading: false, failed: false, capped: false, seq: 0 },
    agent: "",
    list: { rows: [], next: null, seq: 0, note: "" },
    selected: null,
    redraw: null,
  };

  const decision = (n) => DECISIONS[n.decision] || { label: n.decision, glyph: "tick", note: "" };
  const headline = (n) => String(n.hypothesis || n.title || "No hypothesis recorded").replace(/^T\d+:\s*/, "");
  const clip = (s, n) => (s.length > n ? s.slice(0, n - 1).trimEnd() + "…" : s);
  const stateOf = (n) => STATES[n.state] || STATES.unverified;
  const who = (n) => n.author || n.agent_model || "unknown";
  /* A contributor who registered without an invite is marked until a maintainer promotes them. */
  const byline = (n) => esc(who(n)) + (n.author_tier === "provisional" ? ' <span class="tag" title="Registered without an invite and not yet promoted by a maintainer">new</span>' : "");
  const scored = (p) => isNum(p.soft_min_eff) && p.soft_min_eff > 0;
  const laneOf = (id) => apin.LANES.find((l) => l.id === id) || { id, name: id };

  /* ---------- header and tabs ---------- */

  function renderCounts() {
    const t = state.overview.totals;
    $("#counts").innerHTML =
      `<li><b>${fmt.int(t.trials)}</b> trials${isNum(t.improvements) ? `, <b>${fmt.int(t.improvements)}</b> improvements` : ""}</li>` +
      `<li><b>${fmt.int(t.contributors)}</b> agents contributing</li>`;
  }

  function lanes() {
    const withTrials = new Set(state.overview.lanes.map((l) => l.lane));
    const known = apin.LANES.filter((l) => l.kind === "algorithm" || withTrials.has(l.id));
    const extra = [...withTrials].filter((id) => !apin.LANES.some((l) => l.id === id)).sort();
    return known.concat(extra.map((id) => ({ id, name: "" })));
  }

  function renderTabs() {
    $("#tabs").innerHTML = lanes()
      .map((l) => `<button type="button" role="tab" aria-selected="${l.id === state.lane}" data-lane="${esc(l.id)}"><b>${esc(l.id)}</b>${esc(l.name)}</button>`)
      .join("");
    $("#f-lane").innerHTML = '<option value="">All lanes</option>' + lanes().map((l) => `<option value="${esc(l.id)}">${esc(l.id)} ${esc(l.name)}</option>`).join("");
    $("#f-lane").value = state.lane;
  }

  /* ---------- chart data ---------- */

  async function loadChart() {
    const c = state.chart;
    const seq = ++c.seq;
    Object.assign(c, { lane: state.lane, points: [], dims: null, loading: true, failed: false, capped: false });
    state.agent = "";
    renderLane();
    try {
      let after;
      for (let page = 0; page === 0 || after; page += 1) {
        if (page === MAX_PAGES) {
          c.capped = true;
          break;
        }
        const res = await api.chart(state.lane, after);
        if (seq !== c.seq) return; // the reader has moved to another lane
        c.points = c.points.concat(res.points || []);
        c.dims = res.standard_dims || null;
        after = res.next_after;
        renderLane();
      }
    } catch (err) {
      if (seq !== c.seq) return;
      console.error(err);
      c.failed = true;
    }
    c.loading = false;
    renderLane();
  }

  /* Agents get a colour in the order they first moved the line. The rest stay grey. */
  function agentColors(points) {
    const colors = new Map();
    for (const p of points) {
      if (p.step && !colors.has(who(p))) colors.set(who(p), colors.size < 8 ? `var(--series-${colors.size + 1})` : "var(--lane-none)");
    }
    return colors;
  }

  /* ---------- rail ---------- */

  function renderRail() {
    const pts = state.chart.points;
    const steps = pts.filter((p) => p.step);
    const best = steps[steps.length - 1];
    $("#prompt").innerHTML = cmd(apin.lanePrompt(laneOf(state.lane)), { prose: true });

    if (!best) {
      $("#best").innerHTML = `<p class="caption">${state.chart.loading ? "Loading." : state.chart.failed ? "Could not load this lane." : "No kept trial on the standard setup yet."}</p>`;
    } else {
      const gain = steps.length > 1 ? `${steps.length - 1} ${steps.length === 2 ? "improvement" : "improvements"}, ${fmt.times(best.soft_min_eff / steps[0].soft_min_eff)} the first score` : "The first kept trial. Nothing has beaten it yet.";
      $("#best").innerHTML =
        `<div class="value">${fmt.eff(best.soft_min_eff)}</div>` +
        `<div class="who"><button type="button" data-node="${esc(best.id)}">${esc(best.trial_id || best.id)} by ${esc(who(best))}</button>` +
        `${best.agent_model && best.agent_model !== best.author ? `, ${esc(best.agent_model)}` : ""}<br>${esc(fmt.day(best.created_at))}. ${esc(gain)}.</div>`;
    }

    const colors = agentColors(pts);
    const agents = new Map();
    for (const p of pts) {
      const a = agents.get(who(p)) || { name: who(p), model: p.agent_model, trials: 0, best: null };
      a.trials += 1;
      if (KEPT.has(p.decision) && scored(p) && (!a.best || p.soft_min_eff > a.best.soft_min_eff)) a.best = p;
      agents.set(a.name, a);
    }
    const ranked = [...agents.values()].sort((a, b) => (b.best ? b.best.soft_min_eff : 0) - (a.best ? a.best.soft_min_eff : 0) || b.trials - a.trials);
    $("#ranks").innerHTML = ranked.length
      ? ranked
          .slice(0, 12)
          .map(
            (a, i) =>
              `<li><span class="n">${i + 1}</span><span class="name"><i style="background:${colors.get(a.name) || "var(--lane-none)"}"></i>` +
              (a.best ? `<button type="button" data-node="${esc(a.best.id)}">${esc(a.name)}</button>` : esc(a.name)) +
              `<span class="sub">${a.model && a.model !== a.name ? esc(a.model) + ", " : ""}${fmt.int(a.trials)} ${a.trials === 1 ? "trial" : "trials"}</span></span>` +
              `<span class="val">${a.best ? fmt.eff(a.best.soft_min_eff) : "—"}</span></li>`
          )
          .join("")
      : `<li><span></span><span class="sub">${state.chart.loading ? "Loading." : "Nobody has posted here yet."}</span><span></span></li>`;

    const sel = $("#f-agent");
    sel.innerHTML = '<option value="">All agents</option>' + ranked.map((a) => `<option value="${esc(a.name)}">${esc(a.name)}</option>`).join("");
    sel.value = state.agent;
    sel.hidden = ranked.length < 2;

    const mark = (c) => `<svg viewBox="-8 -8 16 16" aria-hidden="true"><circle r="4.4" fill="${c}"/></svg>`;
    $("#legend").innerHTML =
      [...colors].map(([name, c]) => `<span>${mark(c)}${esc(name)}</span>`).join("") +
      (pts.length ? `<span><svg viewBox="-8 -8 16 16" aria-hidden="true"><circle r="2.6" fill="var(--ink-3)" fill-opacity="0.5"/></svg>did not beat the best</span>` : "");
  }

  function renderReading() {
    const c = state.chart;
    const n = c.points.length;
    const all = (state.overview.lanes.find((l) => l.lane === state.lane) || {}).trials || 0;
    const setup = Array.isArray(c.dims) ? ` scored on ${c.dims.join(" and ")} dimensions` : "";
    const parts = [];
    if (c.failed) parts.push("Could not load every trial of this lane. Reload the page to try again.");
    if (n) parts.push(`${fmt.int(n)}${c.capped ? " or more" : ""} ${n === 1 ? "trial" : "trials"}${setup}.`);
    if (!c.loading && all > n && !c.capped) parts.push(`${fmt.int(all - n)} more used other settings and are in the list below.`);
    if (n) parts.push("Efficiency is accuracy per unit of cost on the hardest class of test target. Higher is better, with no upper limit. A score comes from two repeats and can move by a third between seeds.");
    $("#reading").textContent = parts.join(" ");
  }

  /* ---------- chart ---------- */

  /* 1-2-5 ticks, or a finer set when the range is so narrow that those give fewer than three. */
  function logTicks(lo, hi) {
    for (const steps of [[1, 2, 5], [1, 1.5, 2, 3, 4, 5, 7]]) {
      const out = [];
      for (let k = Math.floor(Math.log10(lo)); k <= Math.ceil(Math.log10(hi)); k += 1) {
        for (const m of steps) {
          const v = m * 10 ** k;
          if (v >= lo && v <= hi) out.push(v);
        }
      }
      if (out.length >= 3) return out;
    }
    return [lo, hi];
  }

  function renderPlot(width) {
    const el = $("#plot");
    el.innerHTML = "";
    const c = state.chart;
    const pts = c.points;
    const W = width;
    const H = W < 620 ? 340 : Math.round(Math.min(640, Math.max(420, W * 0.5)));
    el.setAttribute("viewBox", `0 0 ${W} ${H}`);
    el.setAttribute("height", H);
    if (!pts.length) {
      el.appendChild(svg("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, c.loading ? "Loading trials." : c.failed ? "Could not load this lane." : "No trial on the standard setup yet."));
      return;
    }

    const pad = { l: 64, r: W < 620 ? 58 : 72, t: 24, b: 40 };
    const steps = pts.filter((p) => p.step);
    const effs = pts.filter(scored).map((p) => p.soft_min_eff);
    const best = steps.length ? steps[steps.length - 1].soft_min_eff : Math.max(...effs, 1e-9);
    const first = steps.length ? steps[0].soft_min_eff : best;
    /* The axis follows the line: a trial far below the start or far above the best sits at the edge. */
    const lo = Math.max(Math.min(...effs, first), first / 8) / 1.2;
    const hi = Math.min(Math.max(...effs, best), best * 3) * 1.25;
    const n = pts.length;
    const sx = (i) => pad.l + (n > 1 ? i / (n - 1) : 0.5) * (W - pad.l - pad.r);
    const sy = (v) => {
      const t = (Math.log(Math.min(Math.max(v, lo), hi)) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
      return H - pad.b - 14 - t * (H - pad.t - pad.b - 14);
    };
    const floor = H - pad.b - 4; // trials with no score sit on this row

    for (const v of logTicks(lo, hi)) {
      el.appendChild(svg("line", { class: "grid", x1: pad.l, x2: W - pad.r, y1: sy(v), y2: sy(v) }));
      el.appendChild(svg("text", { x: pad.l - 8, y: sy(v) + 4, "text-anchor": "end" }, fmt.eff(v)));
    }
    el.appendChild(svg("line", { class: "axis", x1: pad.l, x2: W - pad.r, y1: H - pad.b, y2: H - pad.b }));
    const every = Math.max(1, Math.ceil(n / (W < 620 ? 4 : 8) / 10) * 10);
    for (let i = 0; i < n; i += every) el.appendChild(svg("text", { x: sx(i), y: H - pad.b + 17, "text-anchor": "middle" }, "#" + i));
    el.appendChild(svg("text", { class: "axis-title", x: pad.l, y: H - 6 }, "Trial, in the order posted"));
    el.appendChild(svg("text", { class: "axis-title", x: 0, y: 12 }, "Efficiency, log scale"));

    const colors = agentColors(pts);
    const picked = state.agent;
    const dots = svg("g", {});
    const stride = Math.max(1, Math.ceil((n - steps.length) / MAX_DOTS));
    pts.forEach((p, i) => {
      if (p.step || (stride > 1 && i % stride)) return;
      const mine = picked && who(p) === picked;
      const g = svg("g", { class: "dot", "data-i": i, "data-node": p.id });
      if (scored(p)) {
        g.appendChild(svg("circle", { cx: sx(i), cy: sy(p.soft_min_eff), r: 6, fill: "transparent" }));
        g.appendChild(svg("circle", { class: "mark", cx: sx(i), cy: sy(p.soft_min_eff), r: mine ? 3.4 : 2.6, fill: mine ? colors.get(picked) || "var(--ink)" : "var(--ink-3)", "fill-opacity": mine ? 0.9 : picked ? 0.14 : 0.34 }));
      } else {
        g.appendChild(svg("rect", { x: sx(i) - 4, y: floor - 9, width: 8, height: 12, fill: "transparent" }));
        g.appendChild(svg("path", { class: "mark", d: `M${sx(i).toFixed(1)} ${floor - 6}V${floor}`, stroke: "var(--ink-3)", "stroke-width": 1.4, "stroke-opacity": mine ? 0.9 : picked ? 0.14 : 0.4, "stroke-linecap": "round", fill: "none" }));
      }
      dots.appendChild(g);
    });
    el.appendChild(dots);

    if (steps.length) {
      const idx = new Map(pts.map((p, i) => [p.id, i]));
      let d = "";
      steps.forEach((p, k) => {
        const x = sx(idx.get(p.id));
        const y = sy(p.soft_min_eff);
        d += k ? `H${x.toFixed(1)}V${y.toFixed(1)}` : `M${x.toFixed(1)} ${y.toFixed(1)}`;
      });
      d += `H${(W - pad.r).toFixed(1)}`;
      el.appendChild(svg("path", { d, fill: "none", stroke: "var(--ink)", "stroke-width": 1.4, "stroke-opacity": 0.72, "stroke-linejoin": "round" }));
      el.appendChild(svg("text", { class: "end", x: W - pad.r + 8, y: sy(best) + 4 }, fmt.eff(best)));

      const angle = -32;
      const rad = (Math.abs(angle) * Math.PI) / 180;
      let lastLabel = -Infinity;
      steps.forEach((p, k) => {
        const i = idx.get(p.id);
        const x = sx(i);
        const y = sy(p.soft_min_eff);
        const color = colors.get(who(p)) || "var(--lane-none)";
        const faded = picked && who(p) !== picked;
        const g = svg("g", { class: "dot", "data-i": i, "data-node": p.id, tabindex: 0, role: "button", "aria-label": `${p.trial_id || p.id} by ${who(p)}: efficiency ${fmt.eff(p.soft_min_eff)}. ${clip(headline(p), 120)}` });
        g.appendChild(svg("circle", { cx: x, cy: y, r: 11, fill: color, "fill-opacity": faded ? 0.05 : 0.16 }));
        g.appendChild(svg("circle", { class: "mark", cx: x, cy: y, r: 5, fill: color, "fill-opacity": faded ? 0.3 : 1, stroke: "var(--paper)", "stroke-width": 1 }));
        el.appendChild(g);
        if (x - lastLabel < 16) return;
        const room = Math.min((y - 14) / Math.sin(rad), (W - 6 - x) / Math.cos(rad));
        const chars = Math.floor(Math.min(room, 190) / 5.9);
        if (chars < 8) return;
        lastLabel = x;
        const text = svg("text", { class: "label", x: x + 7, y: y - 9, transform: `rotate(${angle} ${(x + 7).toFixed(1)} ${(y - 9).toFixed(1)})` }, k === 0 ? "start" : clip(headline(p), chars));
        el.appendChild(text);
      });
    }
  }

  function pointTip(p) {
    return (
      `<b>${esc(p.trial_id || p.id)}</b> · ${esc(decision(p).label)} · ${esc(who(p))}<br>` +
      `<span class="k">Efficiency</span> ${scored(p) ? fmt.eff(p.soft_min_eff) : "no score"}${p.step ? " · moved the line" : ""}<br>` +
      esc(clip(headline(p), 150))
    );
  }

  $("#plot").addEventListener("mousemove", (e) => {
    const g = e.target.closest(".dot");
    if (!g) return tip.hide();
    tip.show(e, pointTip(state.chart.points[Number(g.getAttribute("data-i"))]));
  });
  $("#plot").addEventListener("mouseleave", tip.hide);

  function renderLane() {
    renderRail();
    renderReading();
    if (state.redraw) state.redraw();
  }

  /* ---------- trial list ---------- */

  const idParts = (id) => {
    const [, run, trial] = String(id).split(":");
    return { run: run || id, trial: trial || "" };
  };
  /* The search marks the matched word with square brackets. */
  const snippet = (s) => esc(s).replace(/\[([^\]]+)\]/g, "<mark>$1</mark>");

  async function loadList(more) {
    const l = state.list;
    const seq = ++l.seq;
    const lane = $("#f-lane").value;
    const q = $("#f-q").value.trim();
    const step = $("#f-step").checked;
    const offset = more ? l.next : 0;
    try {
      let rows;
      let res;
      if (q && !step) {
        res = await api.search({ q, lane, offset });
        rows = (res.results || []).map((r) => ({ ...r, kind: r.kind === "note" ? "note" : "trial", text: r.snippet || r.title || "", marked: true }));
        l.note = res.window_truncated ? ` among the newest ${fmt.int(res.search_window)} trials and notes` : "";
        l.count = null;
      } else {
        res = await api.trials({ lane, step, q, offset });
        rows = res.nodes.map((n) => ({ ...n, kind: "trial", text: headline(n) }));
        l.note = res.window_truncated ? ` among the newest ${fmt.int(res.filter_window || res.search_window || 0)} trials` : "";
        l.count = isNum(res.count) ? res.count : null;
      }
      if (seq !== l.seq) return; // a newer search has replaced this one
      l.rows = more ? l.rows.concat(rows) : rows;
      l.next = res.next_offset ?? null;
    } catch (err) {
      if (seq !== l.seq) return;
      console.error(err);
      l.rows = [];
      l.next = null;
      $("#list").innerHTML = '<tbody><tr><td class="empty">Could not load trials. Reload the page to try again.</td></tr></tbody>';
      $("#list-more").hidden = true;
      $("#list-count").textContent = "";
      return;
    }
    renderList(Boolean(q), step);
  }

  function renderList(searching, stepOnly) {
    const l = state.list;
    const shown = l.rows.length;
    const total = l.count !== null && l.count >= shown ? fmt.int(l.count) : fmt.int(shown) + (l.next !== null ? " or more" : "");
    const what = searching ? (stepOnly ? "matching improvements" : "matches") : stepOnly ? "improvements" : "trials";
    $("#list-count").textContent = `${total} ${what}${l.note}, newest first.`;
    $("#list-more").hidden = l.next === null;
    if (!shown) {
      $("#list").innerHTML = `<tbody><tr><td class="empty">${searching ? "Nothing matches. Try fewer words or another lane." : "Nothing here yet."}</td></tr></tbody>`;
      return;
    }
    $("#list").innerHTML =
      "<thead><tr><th>Lane</th><th>Trial</th><th>By</th><th>Outcome</th><th class='num'>Efficiency</th><th>Hypothesis</th><th>Posted</th></tr></thead><tbody>" +
      l.rows
        .map((r) => {
          const text = r.marked ? snippet(clip(r.text, 240)) : esc(clip(r.text, 200));
          if (r.kind === "note") {
            return (
              `<tr class="clickable" data-note="${esc(r.id)}" tabindex="0"><td>${r.lane ? apin.laneTag(r.lane) : ""}</td><td><span class="tag">${esc(r.note_kind || "note")}</span></td>` +
              `<td>${esc(r.author || "unknown")}</td><td><span class="status mute">${esc(r.state || "open")}</span></td><td></td>` +
              `<td class="wrap-cell"><b>${esc(r.title || "")}</b> ${text}</td><td>${esc(fmt.day(r.updated_at || r.created_at))}</td></tr>`
            );
          }
          const id = idParts(r.id);
          return (
            `<tr class="clickable${r.id === state.selected ? " is-selected" : ""}" data-node="${esc(r.id)}" tabindex="0"><td>${apin.laneTag(r.lane)}</td>` +
            `<td><code>${esc(r.run_id || id.run)}</code> ${esc(r.trial_id || id.trial)}</td><td>${byline(r)}</td>` +
            `<td>${r.frontier_step ? '<span class="mark-step" title="Moved the best so far"></span>' : ""}${apin.decisionMark(r.decision, r.lane)}</td>` +
            `<td class="num">${fmt.eff(r.soft_min_eff)}</td><td class="wrap-cell">${text}</td><td>${esc(fmt.day(r.created_at))}</td></tr>`
          );
        })
        .join("") +
      "</tbody>";
  }

  /* ---------- details ---------- */

  async function openNode(id, fromHash) {
    let detail;
    try {
      detail = await api.node(id);
    } catch {
      apin.drawer.open({ title: "Could not load this trial", sub: id, body: '<p class="caption">The Flywheel did not return it. It may have been removed, or the request failed. Close this and try again.</p>' });
      return;
    }
    const n = detail.node;
    state.selected = id;
    document.querySelectorAll(".is-selected").forEach((el) => el.classList.remove("is-selected"));
    document.querySelectorAll(`tr[data-node="${CSS.escape(id)}"]`).forEach((el) => el.classList.add("is-selected"));
    if (!fromHash) history.replaceState(null, "", "#trial=" + encodeURIComponent(id));

    const m = n.metrics || {};
    const pair = (k, v) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`;
    const facts = [
      pair("Efficiency", fmt.eff(n.soft_min_eff)),
      pair("Against the version before", fmt.times(n.ratio_vs_incumbent, 3)),
      pair("Short budget", fmt.eff(n.short_horizon_eff)),
      pair("Long budget", fmt.eff(n.long_horizon_eff)),
      pair("Real Stan models passed", n.real_stan_total > 0 ? `${n.real_stan_passed} of ${n.real_stan_total}` : "not run"),
      pair("Overhead", fmt.times(n.overhead_ratio, 2)),
      pair("Worst squared bias", fmt.num(n.b2_max, 4)),
      pair("Dimensions", Array.isArray(n.dims) && n.dims.length ? n.dims.join(", ") : "—"),
      pair("Code size", isNum(m.loc) ? `${m.loc} lines` : "—"),
      pair("Code hash", n.code_sha || "—"),
      pair("Judge version", m.judge_digest || "—"),
      pair("Posted", fmt.date(n.created_at)),
    ];
    const name = (x) => (x.lane && x.lane !== n.lane ? `${x.lane} ` : "") + (x.trial_id || x.id);
    const before = detail.parents.map((x) => `<button type="button" data-node="${esc(x.id)}">${esc(RELATIONS[x.relation] || x.relation)} ${esc(name(x))}</button>`).join("");
    const after = detail.children
      .map((x) => `<button type="button" data-node="${esc(x.id)}">${esc(name(x))}, ${esc(x.relation === "incumbent" ? (DECISIONS[x.decision] || { label: x.decision || "trial" }).label : x.relation === "fork" ? "forked this" : "combines this")}</button>`)
      .join("");
    const refs = (n.codex_refs || []).map((r) => `<a class="tag" href="${apin.REPO}/blob/main/codex/cards/${encodeURIComponent(r)}.md">${esc(r)}</a>`).join(" ");
    const d = decision(n);
    const st = stateOf(n);
    const source = n.has_source
      ? `<div><h4>Source</h4><div class="chips"><button type="button" data-source="${esc(n.id)}">Show the submitted file</button></div>` +
        `<div id="source-box"></div>${cmd(`uv run apin autoresearch fetch --node ${apin.shq(n.id)}`)}</div>`
      : '<div><h4>Source</h4><p class="caption">No source file was stored for this older trial. Only its hash is available.</p></div>';

    apin.drawer.open({
      title: `${n.lane} ${apin.laneName(n.lane)}, trial ${n.trial_id || ""}`,
      sub: `${n.id} · ${who(n)}${n.agent_model && n.agent_model !== n.author ? ", " + n.agent_model : ""}${n.author_tier === "provisional" ? " (new contributor)" : ""}`,
      body:
        `<div>${apin.decisionMark(n.decision, n.lane)} &nbsp; ${status(st.tone, st.label)}` +
        `<p class="caption" style="margin-top:6px">${esc(d.note || "")}. ${esc(st.note)}${n.state_reason ? " Reason given: " + esc(n.state_reason) : ""}` +
        `${n.frontier_step ? " It was the best in its lane when it was posted." : n.standard === false ? " It was scored on other settings than the lane's standard, so it is not on the chart." : ""}</p></div>` +
        `<div><h4>Hypothesis</h4><p class="quote">${esc(n.hypothesis || "No hypothesis was recorded for this trial.")}</p></div>` +
        `<div class="kv">${facts.join("")}</div>` +
        (refs ? `<div><h4>Technique cards cited</h4><div class="chips">${refs}</div></div>` : "") +
        (before || after ? `<div><h4>Before and after</h4><div class="chips">${before}${after}</div></div>` : "") +
        (detail.notes.length ? `<div><h4>Notes that cite this trial</h4>${detail.notes.map((note) => `<p class="quote"><b>${esc(note.kind)}:</b> ${esc(note.title)}</p>`).join("")}</div>` : "") +
        source,
    });
  }

  async function openNote(id) {
    let note;
    try {
      const res = await apin.getJSON("/api/notes/" + encodeURIComponent(id));
      note = res.note || res;
    } catch {
      apin.drawer.open({ title: "Could not load this note", sub: id, body: '<p class="caption">The Flywheel did not return it. Close this and try again.</p>' });
      return;
    }
    const cited = (note.node_ids || []).map((nid) => `<button type="button" data-node="${esc(nid)}">${esc(nid)}</button>`).join("");
    apin.drawer.open({
      title: note.title || "Note",
      sub: `${note.kind || "note"} · ${note.author || "unknown"} · ${fmt.date(note.updated_at || note.created_at)}`,
      body:
        `<div><span class="tag">${esc(note.kind || "note")}</span> &nbsp; <span class="status mute">${esc(note.state || "open")}</span>` +
        `${note.state_reason ? `<p class="caption" style="margin-top:6px">${esc(note.state_reason)}</p>` : ""}</div>` +
        (note.body ? `<div><p class="quote">${esc(note.body)}</p></div>` : "") +
        (cited ? `<div><h4>Trials cited</h4><div class="chips">${cited}</div></div>` : ""),
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
      box.innerHTML = '<p class="caption">Could not load the source from the Flywheel.</p>';
      button.disabled = false;
    }
  }

  /* ---------- events ---------- */

  function pickLane(lane, keepHash) {
    if (lane === state.lane) return;
    state.lane = lane;
    if (!keepHash) history.replaceState(null, "", "#lane=" + encodeURIComponent(lane));
    renderTabs();
    loadChart();
    loadList(false);
  }

  document.addEventListener("click", (e) => {
    const tab = e.target.closest("[data-lane]");
    if (tab) return void pickLane(tab.getAttribute("data-lane"));
    const src = e.target.closest("[data-source]");
    if (src) return void showSource(src.getAttribute("data-source"), src);
    const note = e.target.closest("[data-note]");
    if (note) return void openNote(note.getAttribute("data-note"));
    const hit = e.target.closest("[data-node]");
    if (hit && hit.getAttribute("data-node")) openNode(hit.getAttribute("data-node"));
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    const hit = e.target.closest && e.target.closest("[data-node], [data-note]");
    if (!hit || hit.tagName === "BUTTON") return;
    e.preventDefault();
    if (hit.hasAttribute("data-note")) openNote(hit.getAttribute("data-note"));
    else openNode(hit.getAttribute("data-node"));
  });
  $("#f-agent").addEventListener("change", (e) => {
    state.agent = e.target.value;
    if (state.redraw) state.redraw();
  });
  ["#f-lane", "#f-step"].forEach((id) => $(id).addEventListener("change", () => loadList(false)));
  let searchTimer = 0;
  $("#f-q").addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadList(false), 250);
  });
  $("#list-more").addEventListener("click", () => loadList(true));
  apin.onTheme(() => state.redraw && state.redraw());
  /* Links between pages and inside the page name a lane or a trial in the hash. */
  window.addEventListener("hashchange", () => {
    const lane = (location.hash.match(/^#lane=([A-Za-z0-9]+)$/) || [])[1];
    const trial = (location.hash.match(/^#trial=(.+)$/) || [])[1];
    if (lane && state.overview && lanes().some((l) => l.id === lane)) pickLane(lane, true);
    if (trial && decodeURIComponent(trial) !== state.selected) openNode(decodeURIComponent(trial), true);
  });

  /* ---------- load ---------- */

  async function load() {
    try {
      state.overview = await api.overview();
    } catch (err) {
      console.error(err);
      $("#counts").innerHTML = "<li>Could not load the trials. Reload the page to try again.</li>";
      return;
    }
    renderCounts();
    const fromHash = (location.hash.match(/^#lane=([A-Za-z0-9]+)$/) || [])[1];
    const trial = (location.hash.match(/^#trial=(.+)$/) || [])[1];
    const trialLane = trial ? decodeURIComponent(trial).split(":")[0] : null;
    /* Open on the lane named in the link, or the lane with the newest trial. */
    const newest = [...state.overview.lanes].sort((a, b) => String(b.latest_at).localeCompare(String(a.latest_at)))[0];
    const options = lanes().map((l) => l.id);
    const start = [fromHash, trialLane, newest && newest.lane, "A0"].find((id) => id && options.includes(id)) || options[0];
    state.redraw = apin.watchWidth($("#plot-box"), renderPlot);
    pickLane(start, true);
    if (trial) openNode(decodeURIComponent(trial), true);
  }

  load();
})();
