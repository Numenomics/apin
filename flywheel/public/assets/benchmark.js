/* Agent benchmark page. Data: /data/benchmark.json, written by scripts/collect_harbor_results.py. */
(function () {
  "use strict";
  const { $, esc, fmt, svg, status, cmd, bench, tip, ICONS } = apin;

  apin.mount("benchmark");

  $("#cmd-run").innerHTML = cmd(
    "uv tool install harbor\ndocker build -f harbor/base.Dockerfile -t apin-harbor-base:latest .\nharbor run -p harbor/tasks -a claude-code -m anthropic/claude-sonnet-4-5"
  );
  $("#cmd-verify").innerHTML = cmd(
    "uv run python harbor/verify.py optimize --lane A1 \\\n  --submission path/to/submission.py \\\n  --baseline harbor/tasks/a1-preconditioning/environment/baseline.py \\\n  --dims 6,20 --repeats 2"
  );
  $("#cmd-collect").innerHTML = cmd("uv run python scripts/collect_harbor_results.py --release bench-v0.1.1 --jobs jobs/");

  const SHAPES = { low: "circle", high: "square", max: "diamond" };
  const X_AXES = {
    cost: { get: (r) => r.total_cost_usd, tick: (v) => fmt.usd(v), title: "Total cost of the eight runs, US dollars" },
    tokens: { get: (r) => r.total_tokens, tick: (v) => fmt.tokens(v), title: "Total tokens, cached input included" },
    time: { get: (r) => r.total_wall_s, tick: (v) => fmt.dur(v), title: "Total agent time" },
  };

  const state = { data: null, xAxis: "cost", sort: { key: "rank", dir: 1 }, models: [], redraw: null };

  function modelColor(model) {
    const i = state.models.indexOf(model);
    return i >= 0 && i < 8 ? `var(--series-${i + 1})` : "var(--lane-none)";
  }

  function marker(shape, color, size) {
    const s = size || 14;
    const open = `<svg viewBox="-8 -8 16 16" width="${s}" height="${s}" aria-hidden="true">`;
    if (shape === "square") return `${open}<rect x="-5" y="-5" width="10" height="10" fill="${color}"/></svg>`;
    if (shape === "diamond") return `${open}<path d="M0-7L7 0 0 7-7 0z" fill="${color}"/></svg>`;
    return `${open}<circle r="5.5" fill="${color}"/></svg>`;
  }

  function rows() {
    const model = $("#f-model").value;
    const effort = $("#f-effort").value;
    return state.data.leaderboard.filter((r) => (!model || r.model === model) && (!effort || r.effort === effort));
  }

  /* ---------- leaderboard ---------- */

  const COLUMNS = [
    { key: "rank", label: "", sortable: false },
    { key: "label", label: "Model", sortable: false },
    { key: "effort", label: "Effort", sortable: false },
    { key: "tasks", label: "Tasks", sortable: false },
    { key: "passed", label: "Passed", num: true, value: (r) => r.passed },
    { key: "speedup", label: "Typical speed-up", num: true, value: (r) => r.opt_geomean_speedup },
    { key: "reward", label: "Mean reward", num: true, value: (r) => r.mean_reward },
    { key: "cost", label: "Cost", num: true, value: (r) => r.total_cost_usd },
    { key: "tokens", label: "Tokens", num: true, value: (r) => r.total_tokens },
    { key: "time", label: "Agent time", num: true, value: (r) => r.total_wall_s },
  ];

  function renderBoard() {
    const list = rows();
    const table = $("#board");
    if (!list.length) {
      table.innerHTML = '<tbody><tr><td class="empty">No setting matches these filters. Clear one to see results.</td></tr></tbody>';
      return;
    }
    const col = COLUMNS.find((c) => c.key === state.sort.key);
    if (col && col.value) list.sort((a, b) => (col.value(a) - col.value(b)) * state.sort.dir);
    const topReward = Math.max(...state.data.leaderboard.map((r) => r.mean_reward), 1e-9);

    const head = COLUMNS.map((c) => {
      const sorted = state.sort.key === c.key ? ` aria-sort="${state.sort.dir < 0 ? "descending" : "ascending"}"` : "";
      const inner = c.value ? `<button type="button" data-sort="${c.key}">${esc(c.label)}</button>` : esc(c.label);
      return `<th${c.num ? ' class="num"' : ""}${c.value ? sorted : ""}>${inner}</th>`;
    }).join("");

    const body = list
      .map((r) => {
        const harness = (r.agents || []).join(", ");
        const infra = r.infra_errors ? `<div class="sub">${r.infra_errors} did not run</div>` : "";
        return (
          `<tr><td class="rank">${r.rank}</td>` +
          `<td><span class="lane-tag" style="font-family:var(--sans);font-size:0.92rem"><i class="swatch round" style="background:${modelColor(r.model)}"></i>${esc(r.label)}</span>` +
          (harness ? `<div class="sub">via ${esc(harness)}</div>` : "") +
          `</td><td><span class="tag">${esc(r.effort)}</span></td>` +
          `<td>${bench.pips(state.data, r)}</td>` +
          `<td class="num">${r.passed} of ${r.n_trials}${infra}</td>` +
          `<td class="num">${r.opt_geomean_speedup ? fmt.times(r.opt_geomean_speedup) : "—"}<div class="sub">${r.opt_passed} of ${r.opt_total} passed</div></td>` +
          `<td class="num"><span class="bar"><i style="width:${Math.max(2, (r.mean_reward / topReward) * 100).toFixed(1)}%"></i></span>${fmt.num(r.mean_reward, 2)}</td>` +
          `<td class="num">${fmt.usd(r.total_cost_usd)}</td>` +
          `<td class="num">${fmt.tokens(r.total_tokens)}<div class="sub">${fmt.pct(r.cache_hit_rate)} cached</div></td>` +
          `<td class="num">${fmt.dur(r.total_wall_s)}</td></tr>`
        );
      })
      .join("");
    table.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
  }

  /* ---------- cost scatter ---------- */

  function niceStep(span, count) {
    const raw = span / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const norm = raw / mag;
    return (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  }

  function renderScatter(width) {
    const list = rows();
    const el = $("#scatter");
    el.innerHTML = "";
    const W = width;
    const H = W < 560 ? 300 : 380;
    el.setAttribute("viewBox", `0 0 ${W} ${H}`);
    el.setAttribute("height", H);
    if (!list.length) {
      el.appendChild(svg("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, "No setting matches these filters."));
      return;
    }
    const axis = X_AXES[state.xAxis];
    const pad = { l: 46, r: 18, t: 16, b: 46 };
    const xs = list.map(axis.get);
    const xStep = niceStep(Math.max(...xs, 1e-9) * 1.08, W < 560 ? 3 : 5);
    const xMax = Math.ceil((Math.max(...xs, 1e-9) * 1.04) / xStep) * xStep;
    const yStep = niceStep(Math.max(...list.map((r) => r.mean_reward), 1) * 1.1, 4);
    const yMax = Math.ceil((Math.max(...list.map((r) => r.mean_reward), 1) * 1.08) / yStep) * yStep;
    const sx = (v) => pad.l + (v / xMax) * (W - pad.l - pad.r);
    const sy = (v) => H - pad.b - (v / yMax) * (H - pad.t - pad.b);

    for (let v = 0; v <= yMax + 1e-9; v += yStep) {
      el.appendChild(svg("line", { class: v === 0 ? "axis" : "grid", x1: pad.l, x2: W - pad.r, y1: sy(v), y2: sy(v) }));
      el.appendChild(svg("text", { x: pad.l - 8, y: sy(v) + 4, "text-anchor": "end" }, v.toFixed(yStep < 1 ? 1 : 0)));
    }
    for (let v = 0; v <= xMax + 1e-9; v += xStep) {
      el.appendChild(svg("text", { x: sx(v), y: H - pad.b + 17, "text-anchor": v === 0 ? "start" : "middle" }, axis.tick(v)));
    }
    el.appendChild(svg("text", { class: "axis-title", x: pad.l, y: H - 8 }, axis.title));
    el.appendChild(svg("text", { class: "axis-title", x: pad.l, y: 10, "text-anchor": "start" }, "Mean reward"));

    /* settings nothing else beats on both axes: cheaper and at least as good */
    const sorted = [...list].sort((a, b) => axis.get(a) - axis.get(b) || b.mean_reward - a.mean_reward);
    const front = [];
    let best = -Infinity;
    for (const r of sorted) {
      if (r.mean_reward > best) {
        front.push(r);
        best = r.mean_reward;
      }
    }
    if (front.length > 1) {
      el.appendChild(
        svg("polyline", {
          points: front.map((r) => `${sx(axis.get(r)).toFixed(1)},${sy(r.mean_reward).toFixed(1)}`).join(" "),
          fill: "none",
          stroke: "var(--ink-3)",
          "stroke-width": 1.5,
          "stroke-dasharray": "5 4",
        })
      );
    }

    const pts = list.map((r) => ({ r, x: sx(axis.get(r)), y: sy(r.mean_reward), label: `${r.label} ${r.effort}` }));
    const spots = apin.placeLabels(pts, { x0: pad.l, x1: W - 4, y0: 14, y1: H - pad.b - 2 });
    pts.forEach((p, i) => {
      const g = svg("g", { transform: `translate(${p.x.toFixed(1)},${p.y.toFixed(1)})`, style: "cursor:pointer", tabindex: 0, role: "button", "aria-label": `${p.label}: mean reward ${fmt.num(p.r.mean_reward)}, ${axis.tick(axis.get(p.r))}` });
      g.appendChild(svg("circle", { r: 16, fill: "transparent" }));
      const color = modelColor(p.r.model);
      const shape = SHAPES[p.r.effort] || "circle";
      const ring = { stroke: "var(--paper)", "stroke-width": 2, fill: color };
      if (shape === "square") g.appendChild(svg("rect", { x: -6, y: -6, width: 12, height: 12, ...ring }));
      else if (shape === "diamond") g.appendChild(svg("path", { d: "M0-8L8 0 0 8-8 0z", ...ring }));
      else g.appendChild(svg("circle", { r: 6.5, ...ring }));
      tip.bind(
        g,
        `<b>${esc(p.r.label)}, ${esc(p.r.effort)} effort</b><br>` +
          `<span class="k">Mean reward</span> ${fmt.num(p.r.mean_reward)} · <span class="k">passed</span> ${p.r.passed} of ${p.r.n_trials}<br>` +
          `<span class="k">Typical speed-up</span> ${fmt.times(p.r.opt_geomean_speedup)}<br>` +
          `<span class="k">Cost</span> ${fmt.usd(p.r.total_cost_usd)} · <span class="k">tokens</span> ${fmt.tokens(p.r.total_tokens)} · <span class="k">time</span> ${fmt.dur(p.r.total_wall_s)}`
      );
      el.appendChild(g);
      if (spots[i]) el.appendChild(svg("text", { class: "direct", x: spots[i].x, y: spots[i].y, "text-anchor": spots[i].anchor }, p.label));
    });
  }

  function renderLegend() {
    const list = state.data.leaderboard;
    const order = ["low", "medium", "high", "max"];
    const efforts = [...new Set(list.map((r) => r.effort))].sort((a, b) => order.indexOf(a) - order.indexOf(b));
    const seen = new Map();
    list.forEach((r) => seen.set(r.model, r.label));
    $("#scatter-legend").innerHTML =
      [...seen].map(([model, label]) => `<span>${marker("circle", modelColor(model))}${esc(label)}</span>`).join("") +
      efforts.map((e) => `<span>${marker(SHAPES[e] || "circle", "var(--ink-3)")}${esc(e)} effort</span>`).join("");
  }

  /* ---------- matrix ---------- */

  function heatClass(reward) {
    if (reward <= 1.02) return "";
    const t = Math.min(1, Math.log(reward) / Math.log(8));
    return "h" + Math.max(1, Math.min(6, Math.ceil(t * 6)));
  }

  function renderMatrix() {
    const data = state.data;
    const list = rows();
    const table = $("#matrix");
    if (!list.length) {
      table.innerHTML = '<tbody><tr><td class="empty">No setting matches these filters.</td></tr></tbody>';
      return;
    }
    let html =
      "<thead><tr><th>Task</th>" +
      list.map((r) => `<th class="config">${esc(r.label)}<br><span class="tag">${esc(r.effort)}</span></th>`).join("") +
      "</tr></thead><tbody>";
    let group = null;
    for (const id of bench.tasks(data)) {
      const meta = bench.meta(data, id);
      if (meta.category !== group) {
        group = meta.category;
        html += `<tr class="group"><td colspan="${list.length + 1}">${group === "rewind" ? "Fix a bug: pass or fail" : "Make it faster: efficiency relative to the frozen baseline"}</td></tr>`;
      }
      const limit = meta.agent_timeout_s ? `${Math.round(meta.agent_timeout_s / 60)} minutes` : "";
      const facts = [limit, meta.lane ? `lane ${meta.lane}` : "", meta.difficulty ? `rated ${meta.difficulty}` : ""].filter(Boolean).join(", ");
      html +=
        `<tr><td class="task"><code>${esc(id)}</code><span>${esc(meta.title)}</span>` +
        `<a href="${apin.REPO}/tree/main/harbor/tasks/${esc(id)}">${esc(facts || "task files")}</a></td>`;
      for (const r of list) {
        const t = bench.trial(data, r.config_key, id);
        const st = bench.state(t);
        if (st === "none") {
          html += '<td class="cell"><span class="heat none">not run</span></td>';
          continue;
        }
        const key = esc(r.config_key + "||" + id);
        if (st === "infra") html += `<td class="cell"><button type="button" class="heat infra" data-run="${key}">did not run</button></td>`;
        else if (st === "fail") html += `<td class="cell"><button type="button" class="heat fail" data-run="${key}">${ICONS.cross}failed</button></td>`;
        else if (meta.category === "rewind") html += `<td class="cell"><button type="button" class="heat" data-run="${key}">${ICONS.check}fixed</button></td>`;
        else html += `<td class="cell"><button type="button" class="heat ${heatClass(t.reward)}" data-run="${key}">${fmt.times(t.reward)}</button></td>`;
      }
      html += "</tr>";
    }
    table.innerHTML = html + "</tbody>";
  }

  /* ---------- run details ---------- */

  function openRun(key) {
    const [configKey, taskId] = key.split("||");
    const data = state.data;
    const t = bench.trial(data, configKey, taskId);
    if (!t) return;
    const row = data.leaderboard.find((r) => r.config_key === configKey) || {};
    const st = bench.state(t);
    const sol = t.solution || {};
    const kind = bench.meta(data, taskId).category;
    const verdict = status(st === "pass" ? "good" : st === "infra" ? "warn" : "bad", bench.outcome(data, t));
    const pair = (k, v) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`;
    const facts = [
      pair("Reward", fmt.num(t.reward, 3)),
      pair("Cost", `${fmt.usd(t.cost_usd)}${t.cost_source ? " (" + String(t.cost_source).replace(/_/g, " ") + ")" : ""}`),
      pair("Agent time", fmt.dur(t.agent_wall_s)),
      pair("Verifier time", fmt.dur(t.verifier_wall_s)),
      pair("Input tokens", fmt.tokens(t.input_tokens)),
      pair("Cached tokens", `${fmt.tokens(t.cached_tokens)} (${fmt.pct(t.cache_hit_rate)})`),
      pair("Output tokens", fmt.tokens(t.output_tokens)),
      pair("Reasoning tokens", fmt.tokens(t.reasoning_tokens)),
      pair("Turns", fmt.int(t.turns)),
      pair("Tool calls", fmt.int(t.tool_calls)),
      pair("Test runs", fmt.int(t.test_runs)),
      pair("Solution size", sol.loc ? `${sol.loc} lines` : "—"),
    ];
    if (kind === "optimize" && st === "pass") {
      facts.push(
        pair("Efficiency", fmt.eff(t.soft_min_eff)),
        pair("Baseline efficiency", fmt.eff(t.baseline_soft_min)),
        pair("Short budget", fmt.eff(t.short_horizon_soft_min)),
        pair("Long budget", fmt.eff(t.long_horizon_soft_min)),
        pair("Overhead", t.overhead_ratio != null ? fmt.times(t.overhead_ratio, 1) : "—"),
        pair("Cyclomatic complexity", fmt.int(sol.cyclomatic_complexity))
      );
    }
    const classes = Object.entries(t.by_class || {}).sort((a, b) => a[1] - b[1]);
    const top = Math.max(...classes.map((c) => c[1]), 1e-12);
    const classHtml = classes.length
      ? "<div><h4>Efficiency by target shape, weakest first</h4><div class=\"class-bars\">" +
        classes.map(([k, v]) => `<div><span>${esc(k)}</span><i style="width:${Math.max(1, (v / top) * 100).toFixed(1)}%"></i><b>${Number(v).toExponential(2)}</b></div>`).join("") +
        '</div><p class="caption">The weakest shape sets the score.</p></div>'
      : "";
    const failed = (t.failed_targets || []).length ? `<div><h4>Targets that failed</h4><p class="quote">${esc(t.failed_targets.join("; "))}</p></div>` : "";
    const note = t.approach_summary ? `<div><h4>The agent's closing note</h4><pre class="log">${esc(t.approach_summary)}</pre></div>` : "";
    const error = t.error ? `<div><h4>How the run ended</h4><pre class="log">${esc(t.error)}</pre></div>` : "";

    document.querySelectorAll(".heat.is-open").forEach((n) => n.classList.remove("is-open"));
    document.querySelectorAll(`.heat[data-run="${CSS.escape(key)}"]`).forEach((n) => n.classList.add("is-open"));
    apin.drawer.open({
      title: bench.meta(data, taskId).title,
      sub: `${taskId} · ${row.label || t.model} at ${t.effort} effort · ${t.agent} · ${t.trial_id}`,
      body: `<div>${verdict}</div><div class="kv">${facts.join("")}</div>${classHtml}${failed}${note}${error}`,
    });
  }

  document.addEventListener("click", (e) => {
    const run = e.target.closest("[data-run]");
    if (run) return openRun(run.getAttribute("data-run"));
    const sort = e.target.closest("[data-sort]");
    if (sort) {
      const key = sort.getAttribute("data-sort");
      state.sort = state.sort.key === key ? { key, dir: -state.sort.dir } : { key, dir: key === "cost" || key === "tokens" || key === "time" ? 1 : -1 };
      renderBoard();
      return;
    }
    const seg = e.target.closest("[data-x]");
    if (seg) {
      state.xAxis = seg.getAttribute("data-x");
      document.querySelectorAll("[data-x]").forEach((b) => b.setAttribute("aria-pressed", String(b === seg)));
      state.redraw();
    }
  });

  function renderAll() {
    renderBoard();
    renderMatrix();
    if (state.redraw) state.redraw();
  }

  apin
    .getJSON("/data/benchmark.json")
    .then((data) => {
      state.data = data;
      state.models = [...new Set(data.leaderboard.map((r) => r.model))];
      const harness = [...new Set(data.trials.map((t) => t.agent))].join(", ");
      $("#meta").innerHTML =
        `<span>Release <b>${esc(data.release)}</b></span><span><b>${data.leaderboard.length}</b> settings</span>` +
        `<span><b>${data.trials.length}</b> runs on <b>${data.tasks.length}</b> tasks</span><span>Harness <b>${esc(harness)}</b></span>` +
        `<span>Updated <b>${esc(fmt.date(data.generated_at))}</b></span>`;
      const fill = (id, values, labelOf) => {
        $(id).insertAdjacentHTML("beforeend", values.map((v) => `<option value="${esc(v)}">${esc(labelOf(v))}</option>`).join(""));
        $(id).addEventListener("change", renderAll);
      };
      fill("#f-model", state.models, (m) => (data.leaderboard.find((r) => r.model === m) || {}).label || m);
      fill("#f-effort", [...new Set(data.leaderboard.map((r) => r.effort))], (e) => e + " effort");
      renderLegend();
      state.redraw = apin.watchWidth($("#scatter-box"), renderScatter);
      renderBoard();
      renderMatrix();
    })
    .catch((err) => {
      console.error(err);
      $("#meta").textContent = "No benchmark results are published with this build.";
      $("#board").innerHTML =
        '<tbody><tr><td class="empty">Results could not be loaded. Run <code>scripts/collect_harbor_results.py</code> to write <code>data/benchmark.json</code>, then reload.</td></tr></tbody>';
    });
})();
