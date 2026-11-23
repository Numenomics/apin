/* Agent benchmark page. Data: /data/benchmark.json, written by scripts/collect_harbor_results.py.
   The file holds one block per benchmark release; the page shows one release at a time. */
(function () {
  "use strict";
  const { $, esc, fmt, svg, status, cmd, bench, tip, ICONS, isNum } = apin;

  apin.mount("benchmark");

  $("#cmd-run").innerHTML = cmd(
    "uv tool install harbor==0.27.0\nuv run python scripts/build_harbor_base.py --tasks\nharbor run -p harbor/tasks -a claude-code -m anthropic/claude-sonnet-4-5 \\\n  --job-name bench-v0.2.0-claude-sonnet-4-5"
  );
  $("#cmd-verify").innerHTML = cmd(
    "uv run python harbor/verify.py optimize --lane A1 \\\n  --submission path/to/submission.py \\\n  --baseline harbor/tasks/a1-preconditioning/tests/baseline.py \\\n  --dims 6,20 --repeats 2"
  );
  $("#cmd-collect").innerHTML = cmd(
    "uv run python scripts/collect_harbor_results.py --jobs jobs/"
  );

  const SHAPES = { low: "circle", high: "square", max: "diamond" };
  const X_AXES = {
    cost: { get: (r) => r.total_cost_usd, tick: (v) => fmt.usd(v), title: "Total cost of the sweep, US dollars" },
    tokens: { get: (r) => r.total_tokens, tick: (v) => fmt.tokens(v), title: "Total tokens, cached input included" },
    time: { get: (r) => r.total_wall_s, tick: (v) => fmt.dur(v), title: "Total agent time" },
  };

  const state = { data: null, rel: null, xAxis: "cost", sort: { key: "rank", dir: 1 }, models: [], redraw: null };

  function modelColor(model) {
    const i = state.models.indexOf(model);
    return i >= 0 && i < 8 ? `var(--series-${i + 1})` : "var(--lane-none)";
  }

  function marker(shape, color, size) {
    const s = size || 14;
    const open = `<svg viewBox="-8 -8 16 16" width="${s}" height="${s}" aria-hidden="true">`;
    if (shape === "square") return `${open}<rect x="-4" y="-4" width="8" height="8" fill="${color}"/></svg>`;
    if (shape === "diamond") return `${open}<path d="M0-5.6L5.6 0 0 5.6-5.6 0z" fill="${color}"/></svg>`;
    return `${open}<circle r="4.4" fill="${color}"/></svg>`;
  }

  function rows() {
    const model = $("#f-model").value;
    const effort = $("#f-effort").value;
    return state.rel.leaderboard.filter((r) => (!model || r.model === model) && (!effort || r.effort === effort));
  }

  /* A count of tasks solved is whole when each task was attempted once, fractional otherwise. */
  const solvedText = (r, v) => fmt.num(v, r.attempts_per_task > 1 ? 1 : 0);

  /* ---------- leaderboard ---------- */

  const COLUMNS = [
    { key: "rank", label: "" },
    { key: "label", label: "Model" },
    { key: "effort", label: "Effort" },
    { key: "tasks", label: "Tasks" },
    { key: "solved", label: "Solved", num: true, value: (r) => r.solved },
    { key: "speedup", label: "Typical speed-up", num: true, value: (r) => r.opt_geomean || 0 },
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

    const head = COLUMNS.map((c) => {
      const sorted = state.sort.key === c.key ? ` aria-sort="${state.sort.dir < 0 ? "descending" : "ascending"}"` : "";
      const inner = c.value ? `<button type="button" data-sort="${c.key}">${esc(c.label)}</button>` : esc(c.label);
      return `<th${c.num ? ' class="num"' : ""}${c.value ? sorted : ""}>${inner}</th>`;
    }).join("");

    const body = list
      .map((r) => {
        const harness = (r.agents || []).join(", ");
        const unfinished = r.interrupted ? `<div class="sub">${r.interrupted} ${r.interrupted === 1 ? "run" : "runs"} did not finish</div>` : "";
        const attempts = r.attempts_per_task > 1 ? `<div class="sub">${r.attempts_per_task} attempts per task</div>` : "";
        const longest = isNum(r.max_wall_s) ? `<div class="sub">longest run ${fmt.dur(r.max_wall_s)}</div>` : "";
        return (
          `<tr><td class="rank">${r.rank}</td>` +
          `<td><span class="lane-tag" style="font-family:var(--sans);font-size:0.98rem"><i class="swatch round" style="background:${modelColor(r.model)}"></i>${esc(r.label)}</span>` +
          (harness ? `<div class="sub">via ${esc(harness)}</div>` : "") +
          `</td><td><span class="tag">${esc(r.effort)}</span>${attempts}</td>` +
          `<td>${bench.pips(state.rel, r)}</td>` +
          `<td class="num"><strong>${solvedText(r, r.solved)}</strong> of ${r.n_tasks}` +
          `<div class="sub">${solvedText(r, r.fix_solved)} of ${r.fix_total} fixes, ${solvedText(r, r.opt_solved)} of ${r.opt_total} speed-ups</div>${unfinished}</td>` +
          `<td class="num">${isNum(r.opt_geomean) ? fmt.times(r.opt_geomean) : "—"}<div class="sub">${solvedText(r, r.opt_valid)} of ${r.opt_total} valid</div></td>` +
          `<td class="num">${fmt.usd(r.total_cost_usd)}</td>` +
          `<td class="num">${fmt.tokens(r.total_tokens)}<div class="sub">${fmt.pct(r.cache_hit_rate)} cached</div></td>` +
          `<td class="num">${fmt.dur(r.total_wall_s)}${longest}</td></tr>`
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
    const H = W < 560 ? 300 : 360;
    el.setAttribute("viewBox", `0 0 ${W} ${H}`);
    el.setAttribute("height", H);
    if (!list.length) {
      el.appendChild(svg("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, "No setting matches these filters."));
      return;
    }
    const axis = X_AXES[state.xAxis];
    const pad = { l: 34, r: 18, t: 26, b: 46 };
    const xs = list.map(axis.get);
    const xStep = niceStep(Math.max(...xs, 1e-9) * 1.08, W < 560 ? 3 : 5);
    const xMax = Math.ceil((Math.max(...xs, 1e-9) * 1.04) / xStep) * xStep;
    const yMax = Math.max(...list.map((r) => r.n_tasks), 1);
    const sx = (v) => pad.l + (v / xMax) * (W - pad.l - pad.r);
    const sy = (v) => H - pad.b - (v / yMax) * (H - pad.t - pad.b);

    for (let v = 0; v <= yMax; v += yMax > 10 ? 5 : 2) {
      el.appendChild(svg("line", { class: v === 0 ? "axis" : "grid", x1: pad.l, x2: W - pad.r, y1: sy(v), y2: sy(v) }));
      el.appendChild(svg("text", { x: pad.l - 8, y: sy(v) + 4, "text-anchor": "end" }, v));
    }
    for (let v = 0; v <= xMax + 1e-9; v += xStep) {
      el.appendChild(svg("text", { x: sx(v), y: H - pad.b + 17, "text-anchor": v === 0 ? "start" : "middle" }, axis.tick(v)));
    }
    el.appendChild(svg("text", { class: "axis-title", x: pad.l, y: H - 8 }, axis.title));
    el.appendChild(svg("text", { class: "axis-title", x: 0, y: 12 }, `Tasks solved, of ${yMax}`));

    /* settings nothing else beats on both axes: cheaper and at least as many tasks solved */
    const sorted = list.filter((r) => r.complete).sort((a, b) => axis.get(a) - axis.get(b) || b.solved - a.solved);
    const front = [];
    let best = -Infinity;
    for (const r of sorted) {
      if (r.solved > best) {
        front.push(r);
        best = r.solved;
      }
    }
    if (front.length > 1) {
      el.appendChild(
        svg("polyline", {
          points: front.map((r) => `${sx(axis.get(r)).toFixed(1)},${sy(r.solved).toFixed(1)}`).join(" "),
          fill: "none",
          stroke: "var(--ink-3)",
          "stroke-width": 1,
          "stroke-dasharray": "1 4",
          "stroke-linecap": "round",
        })
      );
    }

    const pts = list.map((r) => ({ r, x: sx(axis.get(r)), y: sy(r.solved), label: `${r.label} ${r.effort}` }));
    const spots = apin.placeLabels(pts, { x0: pad.l, x1: W - 4, y0: pad.t - 6, y1: H - pad.b - 2 });
    pts.forEach((p, i) => {
      const g = svg("g", {
        transform: `translate(${p.x.toFixed(1)},${p.y.toFixed(1)})`,
        tabindex: 0,
        role: "img",
        "aria-label": `${p.label}: ${solvedText(p.r, p.r.solved)} of ${p.r.n_tasks} tasks solved, ${axis.tick(axis.get(p.r))}`,
      });
      g.appendChild(svg("circle", { r: 14, fill: "transparent" }));
      const color = modelColor(p.r.model);
      const shape = SHAPES[p.r.effort] || "circle";
      /* a setting with unfinished tasks is drawn hollow */
      const look = p.r.complete ? { fill: color, stroke: "var(--paper)", "stroke-width": 1 } : { fill: "var(--paper)", stroke: color, "stroke-width": 1.5 };
      if (shape === "square") g.appendChild(svg("rect", { x: -4.5, y: -4.5, width: 9, height: 9, ...look }));
      else if (shape === "diamond") g.appendChild(svg("path", { d: "M0-6L6 0 0 6-6 0z", ...look }));
      else g.appendChild(svg("circle", { r: 4.8, ...look }));
      tip.bind(
        g,
        `<b>${esc(p.r.label)}, ${esc(p.r.effort)} effort</b><br>` +
          `<span class="k">Solved</span> ${solvedText(p.r, p.r.solved)} of ${p.r.n_tasks}${p.r.complete ? "" : ` (${p.r.interrupted} did not finish)`}<br>` +
          `<span class="k">Typical speed-up</span> ${fmt.times(p.r.opt_geomean)}<br>` +
          `<span class="k">Cost</span> ${fmt.usd(p.r.total_cost_usd)} · <span class="k">tokens</span> ${fmt.tokens(p.r.total_tokens)} · <span class="k">time</span> ${fmt.dur(p.r.total_wall_s)}`
      );
      el.appendChild(g);
      if (spots[i]) el.appendChild(svg("text", { class: "direct", x: spots[i].x, y: spots[i].y, "text-anchor": spots[i].anchor }, p.label));
    });
  }

  function renderLegend() {
    const list = state.rel.leaderboard;
    const order = ["low", "medium", "high", "max"];
    const efforts = [...new Set(list.map((r) => r.effort))].sort((a, b) => order.indexOf(a) - order.indexOf(b));
    const seen = new Map();
    list.forEach((r) => seen.set(r.model, r.label));
    $("#scatter-legend").innerHTML =
      [...seen].map(([model, label]) => `<span>${marker("circle", modelColor(model))}${esc(label)}</span>`).join("") +
      efforts.map((e) => `<span>${marker(SHAPES[e] || "circle", "var(--ink-3)")}${esc(e)} effort</span>`).join("") +
      (list.some((r) => !r.complete) ? '<span><svg viewBox="-8 -8 16 16" width="14" height="14" aria-hidden="true"><circle r="4" fill="none" stroke="var(--ink-3)" stroke-width="1.5"/></svg>some runs did not finish</span>' : "");
  }

  /* ---------- matrix ---------- */

  /* The colour ramp ends at 4x a Stan-style reference, or at 8x a starting file in the earlier release. */
  const rampTop = () => (state.rel.scoring === "reference" ? 4 : 8);

  function heatClass(ratio) {
    if (!isNum(ratio) || ratio <= 1.02) return "h1";
    const t = Math.min(1, Math.log(ratio) / Math.log(rampTop()));
    return "h" + Math.max(1, Math.min(6, Math.ceil(t * 6)));
  }

  function renderMatrix() {
    const rel = state.rel;
    const list = rows();
    const table = $("#matrix");
    $("#ramp-top").textContent = `${rampTop()}× ${bench.against(rel)}`;
    if (!list.length) {
      table.innerHTML = '<tbody><tr><td class="empty">No setting matches these filters.</td></tr></tbody>';
      return;
    }
    let html =
      "<thead><tr><th>Task</th>" +
      list.map((r) => `<th class="config">${esc(r.label)}<br><span class="tag">${esc(r.effort)}</span></th>`).join("") +
      "</tr></thead><tbody>";
    let group = null;
    for (const id of bench.tasks(rel)) {
      const meta = bench.meta(rel, id);
      if (meta.category !== group) {
        group = meta.category;
        html += `<tr class="group"><td colspan="${list.length + 1}">${group === "rewind" ? "Fix a bug: solved or not" : `Make it faster: efficiency relative to ${bench.against(rel)}`}</td></tr>`;
      }
      const facts = [meta.lane ? `lane ${meta.lane}` : "", meta.difficulty ? `rated ${meta.difficulty}` : ""].filter(Boolean).join(", ");
      html +=
        `<tr><td class="task"><code>${esc(id)}</code><span>${esc(meta.title)}</span>` +
        `<a href="${apin.REPO}/tree/main/harbor/tasks/${esc(id)}">${esc(facts || "task files")}</a></td>`;
      for (const r of list) {
        const cell = bench.cell(r, id);
        const st = bench.state(cell);
        if (st === "none") {
          html += '<td class="cell"><span class="heat none">not run</span></td>';
          continue;
        }
        const open = `data-run="${esc(r.config_key + "||" + id)}" title="${esc(bench.label(rel, id, cell))}"`;
        const share = cell.attempts > 1 ? `<small>${bench.solvedCount(cell)}/${cell.attempts}</small>` : "";
        if (st === "unfinished") html += `<td class="cell"><button type="button" class="heat infra" ${open}>did not finish</button></td>`;
        else if (meta.category === "rewind") {
          html +=
            st === "invalid"
              ? `<td class="cell"><button type="button" class="heat fail" ${open}>${ICONS.cross}not fixed</button></td>`
              : `<td class="cell"><button type="button" class="heat" ${open}>${ICONS.check}fixed${share}</button></td>`;
        } else if (st === "invalid") html += `<td class="cell"><button type="button" class="heat fail" ${open}>${ICONS.cross}failed</button></td>`;
        else if (st === "unsolved") html += `<td class="cell"><button type="button" class="heat flat" ${open}>${fmt.times(cell.ratio_mean)}${share}</button></td>`;
        else html += `<td class="cell"><button type="button" class="heat ${heatClass(cell.ratio_mean)}" ${open}>${fmt.times(cell.ratio_mean)}${share}</button></td>`;
      }
      html += "</tr>";
    }
    table.innerHTML = html + "</tbody>";
  }

  /* ---------- run details ---------- */

  function openRun(key, pick) {
    const [configKey, taskId] = key.split("||");
    const rel = state.rel;
    const attempts = bench.trials(rel, configKey, taskId);
    if (!attempts.length) return;
    const row = rel.leaderboard.find((r) => r.config_key === configKey) || {};
    const cell = bench.cell(row, taskId) || {};
    const t = attempts.find((a) => a.trial_id === (pick || cell.best_trial)) || attempts[0];
    const sol = t.solution || {};
    const kind = bench.meta(rel, taskId).category;
    const tone = t.outcome === "solved" || (rel.scoring !== "reference" && t.outcome === "improved") ? "good" : t.outcome === "interrupted" ? "warn" : t.outcome === "invalid" || t.outcome === "unscorable" ? "bad" : "mute";
    const pair = (k, v) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`;
    const ended = { finished: "the agent stopped by itself", time_limit: "reached the time limit", agent_error: "the agent exited with an error", provider_limit: "cut off by the model provider", infrastructure: "infrastructure failure", agent_crash: "the agent never started" };
    const facts = [];
    if (kind === "optimize" && t.outcome !== "interrupted") {
      if (rel.scoring === "reference") facts.push(pair("Against the reference", fmt.times(t.ref_speedup)));
      facts.push(
        pair("Against the starting code", fmt.times(t.start_speedup)),
        pair("Efficiency", fmt.eff(t.soft_min_eff)),
        pair("Starting code", fmt.eff(t.start_soft_min))
      );
      if (rel.scoring === "reference") facts.push(pair("Reference", fmt.eff(t.reference_soft_min)));
      facts.push(pair("Short budget", fmt.eff(t.short_horizon_soft_min)), pair("Long budget", fmt.eff(t.long_horizon_soft_min)), pair("Overhead", isNum(t.overhead_ratio) ? fmt.times(t.overhead_ratio, 1) : "—"));
    }
    facts.push(
      pair("How it ended", ended[t.termination] || t.termination || "—"),
      pair("Agent time", fmt.dur(t.agent_wall_s)),
      pair("Cost", `${fmt.usd(t.cost_usd)}${t.cost_source ? " (" + String(t.cost_source).replace(/_/g, " ") + ")" : ""}`),
      pair("Input tokens", fmt.tokens(t.input_tokens)),
      pair("Cached tokens", `${fmt.tokens(t.cached_tokens)} (${fmt.pct(t.cache_hit_rate)})`),
      pair("Output tokens", fmt.tokens(t.output_tokens)),
      pair("Reasoning tokens", fmt.tokens(t.reasoning_tokens)),
      pair("Turns", fmt.int(t.turns)),
      pair("Tool calls", fmt.int(t.tool_calls)),
      pair("Local test runs", fmt.int(t.test_runs)),
      pair("Solution size", sol.loc ? `${sol.loc} lines` : "—"),
      pair("Verifier time", fmt.dur(t.verifier_wall_s))
    );
    const chips =
      attempts.length > 1
        ? '<div><h4>Attempts</h4><div class="chips">' +
          attempts.map((a, i) => `<button type="button" data-run="${esc(key)}" data-attempt="${esc(a.trial_id)}"${a === t ? ' aria-current="true"' : ""}>${i + 1}: ${esc(bench.outcome(rel, a).split(":")[0])}</button>`).join("") +
          "</div></div>"
        : "";
    const classes = Object.entries(t.by_class || {}).sort((a, b) => a[1] - b[1]);
    const top = Math.max(...classes.map((c) => c[1]), 1e-12);
    const classHtml = classes.length
      ? '<div><h4>Efficiency by target shape, weakest first</h4><div class="class-bars">' +
        classes.map(([k, v]) => `<div><span>${esc(k)}</span><i style="width:${Math.max(1, (v / top) * 100).toFixed(1)}%"></i><b>${Number(v).toExponential(2)}</b></div>`).join("") +
        '</div><p class="caption">The weakest shape sets the score.</p></div>'
      : "";
    const failed = (t.failed_targets || []).length ? `<div><h4>Targets that failed</h4><p class="quote">${esc(t.failed_targets.join("; "))}</p></div>` : "";
    const note = t.approach_summary ? `<div><h4>The agent's closing note</h4><pre class="log">${esc(t.approach_summary)}</pre></div>` : "";
    const error = t.error ? `<div><h4>Error reported by the harness</h4><pre class="log">${esc(t.error)}</pre></div>` : "";

    document.querySelectorAll(".heat.is-open").forEach((n) => n.classList.remove("is-open"));
    document.querySelectorAll(`.heat[data-run="${CSS.escape(key)}"]`).forEach((n) => n.classList.add("is-open"));
    apin.drawer.open({
      title: bench.meta(rel, taskId).title,
      sub: `${taskId} · ${row.label || t.model} at ${t.effort} effort · ${t.agent || "unknown harness"} · ${t.trial_id}`,
      body: `<div>${status(tone, bench.outcome(rel, t))}</div>${chips}<div class="kv">${facts.join("")}</div>${classHtml}${failed}${note}${error}`,
    });
  }

  document.addEventListener("click", (e) => {
    const run = e.target.closest("[data-run]");
    if (run) return openRun(run.getAttribute("data-run"), run.getAttribute("data-attempt"));
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

  function fill(id, first, values, labelOf) {
    $(id).innerHTML = `<option value="">${first}</option>` + values.map((v) => `<option value="${esc(v)}">${esc(labelOf(v))}</option>`).join("");
  }

  function showRelease(tag) {
    const data = state.data;
    const rel = bench.release(data, tag);
    state.rel = rel;
    state.sort = { key: "rank", dir: 1 };
    state.models = [...new Set(rel.leaderboard.map((r) => r.model))];
    const harness = [...new Set(rel.trials.map((t) => t.agent).filter(Boolean))].join(", ");
    const unfinished = rel.trials.filter((t) => t.outcome === "interrupted").length;
    $("#meta").innerHTML =
      `<span><b>${rel.leaderboard.length}</b> model and effort settings</span>` +
      `<span><b>${rel.trials.length}</b> runs on <b>${rel.tasks.length}</b> tasks${unfinished ? `, <b>${unfinished}</b> did not finish` : ""}</span>` +
      (harness ? `<span>Harness <b>${esc(harness)}</b></span>` : "") +
      `<span>Updated <b>${esc(fmt.date(data.generated_at))}</b></span>`;
    $("#release-note").textContent = rel.note || "";
    $("#release-note").classList.toggle("is-old", rel.release !== data.current_release);
    $("#speedup-rule").textContent =
      rel.scoring === "reference"
        ? "Typical speed-up is the geometric mean, over the five speed-up tasks, of efficiency relative to the Stan-style reference. A task with no valid result enters at 0.1×."
        : "Typical speed-up is the geometric mean, over the five speed-up tasks, of efficiency relative to the starting code. A task with no valid result enters at 0.1×.";
    fill("#f-model", "All models", state.models, (m) => (rel.leaderboard.find((r) => r.model === m) || {}).label || m);
    fill("#f-effort", "All efforts", [...new Set(rel.leaderboard.map((r) => r.effort))], (e) => e + " effort");
    renderLegend();
    renderAll();
  }

  apin
    .getJSON("/data/benchmark.json")
    .then((data) => {
      if (!data.releases || !data.releases.length) throw new Error("no releases in benchmark.json");
      state.data = data;
      const sel = $("#f-release");
      sel.innerHTML = data.releases.map((r) => `<option value="${esc(r.release)}">${esc(r.release)}${r.release === data.current_release ? " (current)" : ""}</option>`).join("");
      sel.hidden = data.releases.length < 2;
      sel.addEventListener("change", () => showRelease(sel.value));
      $("#f-model").addEventListener("change", renderAll);
      $("#f-effort").addEventListener("change", renderAll);
      state.rel = bench.release(data, data.current_release);
      state.redraw = apin.watchWidth($("#scatter-box"), renderScatter);
      showRelease(data.current_release);
    })
    .catch((err) => {
      console.error(err);
      $("#meta").textContent = "No benchmark results are published with this build.";
      $("#board").innerHTML =
        '<tbody><tr><td class="empty">Results could not be loaded. Run <code>scripts/collect_harbor_results.py</code> to write <code>data/benchmark.json</code>, then reload.</td></tr></tbody>';
    });
})();
