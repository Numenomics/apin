/* Benchmark page. Data: /data/benchmark.json, written by scripts/collect_harbor_results.py.
   The file holds one block per benchmark release, and the page shows one release at a time.
   A "row" is one model at one reasoning effort. A "cell" is that row on one task, over all
   its attempts. */
(function () {
  "use strict";
  const { $, esc, fmt, svg, status, cmd, tip, ICONS, isNum } = apin;

  apin.mount("benchmark");

  $("#cmd-run").innerHTML = cmd(
    "uv tool install harbor==0.27.0\nuv run python scripts/build_harbor_base.py --tasks\nharbor run -p harbor/tasks -a claude-code -m anthropic/claude-sonnet-4-5 \\\n  --job-name bench-v0.3.0-claude-sonnet-4-5"
  );
  $("#cmd-verify").innerHTML = cmd("uv run python harbor/verify.py optimize --lane A1 \\\n  --submission path/to/submission.py");
  $("#cmd-collect").innerHTML = cmd("uv run python scripts/collect_harbor_results.py --jobs jobs/");

  const GAIN = 1.05; /* below this a speed-up is within the noise of the measurement */
  const RAMP_TOP = 8;
  const SHAPES = { low: "circle", medium: "triangle", high: "square", max: "diamond" };
  const X_AXES = {
    cost: { get: (r) => r.total_cost_usd, tick: (v) => fmt.usd(v), title: "Total cost of all runs, US dollars" },
    tokens: { get: (r) => r.total_tokens, tick: (v) => fmt.tokens(v), title: "Total tokens, cached input included" },
    time: { get: (r) => r.total_wall_s, tick: (v) => fmt.dur(v), title: "Total agent time" },
  };

  const state = { data: null, rel: null, xAxis: "cost", sort: { key: "rank", dir: 1 }, models: [], redraw: null };

  /* ---------- reading a release ---------- */

  const index = new WeakMap();
  function attempts(rel, configKey, taskId) {
    if (!index.has(rel)) {
      const map = new Map();
      for (const t of rel.trials || []) {
        const key = t.config_key + "||" + t.task_id;
        if (map.has(key)) map.get(key).push(t);
        else map.set(key, [t]);
      }
      index.set(rel, map);
    }
    return index.get(rel).get(configKey + "||" + taskId) || [];
  }
  /* A task mirrors a lane, so it is shown under the lane's name. */
  const meta = (rel, taskId) => {
    const m = (rel.task_meta || {})[taskId] || {};
    return { id: taskId, ...m, title: m.lane ? apin.laneName(m.lane) : m.title || taskId };
  };
  const cellOf = (row, taskId) => (row.tasks || {})[taskId] || null;
  /* none: never attempted. unfinished: every attempt was cut off. invalid: no attempt passed
     the checks. Otherwise gain or flat, by the mean speed-up. */
  function cellState(cell) {
    if (!cell) return "none";
    if (!cell.attempts) return "unfinished";
    if (!cell.valid) return "invalid";
    return cell.speedup >= GAIN ? "gain" : "flat";
  }
  function cellLabel(cell) {
    const st = cellState(cell);
    if (st === "none") return "not run";
    if (st === "unfinished") return "did not finish";
    const many = cell.attempts > 1 ? `, mean of ${cell.attempts} attempts` : "";
    if (st === "invalid") return "failed the checks" + many;
    return `${fmt.times(cell.speedup)} the starting code${many}`;
  }
  /* One attempt, in words. */
  function attemptLabel(t) {
    const ends = { provider_limit: "the model provider cut it off", infrastructure: "the infrastructure failed", agent_crash: "the agent never started" };
    if (t.outcome === "interrupted") return `did not finish: ${ends[t.termination] || "interrupted"}`;
    if (t.outcome === "invalid") return t.rule_break ? `broke a rule (${t.rule_break.rules})` : "failed the checks";
    if (t.outcome === "unscorable") return "could not be scored";
    if (!isNum(t.start_speedup)) return "no ratio recorded";
    return `${fmt.times(t.start_speedup)} the starting code`;
  }

  function modelColor(model) {
    const i = state.models.indexOf(model);
    return i >= 0 && i < 8 ? `var(--series-${i + 1})` : "var(--lane-none)";
  }

  function marker(shape, color, size) {
    const s = size || 14;
    const open = `<svg viewBox="-8 -8 16 16" width="${s}" height="${s}" aria-hidden="true">`;
    if (shape === "square") return `${open}<rect x="-4" y="-4" width="8" height="8" fill="${color}"/></svg>`;
    if (shape === "diamond") return `${open}<path d="M0-5.6L5.6 0 0 5.6-5.6 0z" fill="${color}"/></svg>`;
    if (shape === "triangle") return `${open}<path d="M0-5.2L5.4 4.2H-5.4z" fill="${color}"/></svg>`;
    return `${open}<circle r="4.4" fill="${color}"/></svg>`;
  }

  function pips(rel, row) {
    const cls = { gain: "", flat: "open", invalid: "fail", unfinished: "infra", none: "none" };
    return (
      '<span class="pips">' +
      rel.tasks
        .map((id) => {
          const cell = cellOf(row, id);
          const label = `${id}: ${cellLabel(cell)}`;
          return `<button type="button" class="${cls[cellState(cell)]}" ${cell ? `data-run="${esc(row.config_key + "||" + id)}"` : "disabled"} title="${esc(label)}" aria-label="${esc(label)}"></button>`;
        })
        .join("") +
      "</span>"
    );
  }

  /* ---------- leaderboard ---------- */

  const COLUMNS = [
    { key: "rank", label: "#", value: (r) => r.rank },
    { key: "label", label: "Model" },
    { key: "effort", label: "Effort" },
    { key: "tasks", label: "Tasks" },
    { key: "speedup", label: "Speed-up", num: true, value: (r) => r.speedup || 0 },
    { key: "cost", label: "Cost", num: true, value: (r) => r.total_cost_usd },
    { key: "tokens", label: "Tokens", num: true, value: (r) => r.total_tokens },
    { key: "time", label: "Agent time", num: true, value: (r) => r.total_wall_s },
  ];

  function renderBoard() {
    const rel = state.rel;
    const list = [...rel.leaderboard];
    const table = $("#board");
    if (!list.length) {
      table.innerHTML = '<tbody><tr><td class="empty">This release has no results yet.</td></tr></tbody>';
      return;
    }
    const col = COLUMNS.find((c) => c.key === state.sort.key);
    /* Rows with an unfinished task stay last whichever way a column is sorted. */
    if (col && col.value) list.sort((a, b) => Number(b.complete) - Number(a.complete) || (col.value(a) - col.value(b)) * state.sort.dir);

    const head = COLUMNS.map((c) => {
      const sorted = state.sort.key === c.key ? ` aria-sort="${state.sort.dir < 0 ? "descending" : "ascending"}"` : "";
      const inner = c.value ? `<button type="button" data-sort="${c.key}">${esc(c.label)}</button>` : esc(c.label);
      return `<th${c.num ? ' class="num"' : ""}${c.value ? sorted : ""}>${inner}</th>`;
    }).join("");

    const body = list
      .map((r) => {
        const harness = (r.agents || []).join(", ");
        const open = r.n_tasks - r.tasks_run;
        const unfinished = open ? `<div class="sub">${open} of ${r.n_tasks} tasks did not finish</div>` : "";
        const tries = r.attempts_per_task > 1 ? `<div class="sub">${r.attempts_per_task} attempts per task</div>` : "";
        return (
          `<tr><td class="rank">${r.rank}</td>` +
          `<td><span class="lane-tag" style="font-family:var(--sans);font-size:0.98rem"><i class="swatch round" style="background:${modelColor(r.model)}"></i>${esc(r.label)}</span>` +
          (harness ? `<div class="sub">via ${esc(harness)}</div>` : "") +
          `</td><td><span class="tag">${esc(r.effort)}</span>${tries}</td>` +
          `<td>${pips(rel, r)}</td>` +
          `<td class="num"><strong>${isNum(r.speedup) ? fmt.times(r.speedup) : "—"}</strong>${unfinished}</td>` +
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
    const list = state.rel.leaderboard.filter((r) => isNum(r.speedup));
    const el = $("#scatter");
    el.innerHTML = "";
    const W = width;
    const H = W < 560 ? 300 : 360;
    el.setAttribute("viewBox", `0 0 ${W} ${H}`);
    el.setAttribute("height", H);
    if (!list.length) {
      el.appendChild(svg("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, "This release has no results yet."));
      return;
    }
    const axis = X_AXES[state.xAxis];
    const pad = { l: 40, r: 18, t: 26, b: 46 };
    const xs = list.map(axis.get);
    const xTop = Math.max(...xs) > 0 ? Math.max(...xs) : 1; // a release whose runs were all free
    const xStep = niceStep(xTop * 1.08, W < 560 ? 3 : 5);
    const xMax = Math.ceil((xTop * 1.04) / xStep) * xStep;
    const yStep = niceStep(Math.max(...list.map((r) => r.speedup), 1.2), 4);
    const yMax = Math.ceil((Math.max(...list.map((r) => r.speedup), 1.2) * 1.04) / yStep) * yStep;
    const sx = (v) => pad.l + (v / xMax) * (W - pad.l - pad.r);
    const sy = (v) => H - pad.b - (v / yMax) * (H - pad.t - pad.b);

    for (let v = 0; v <= yMax + 1e-9; v += yStep) {
      el.appendChild(svg("line", { class: v === 0 ? "axis" : "grid", x1: pad.l, x2: W - pad.r, y1: sy(v), y2: sy(v) }));
      el.appendChild(svg("text", { x: pad.l - 8, y: sy(v) + 4, "text-anchor": "end" }, fmt.times(v, yStep < 1 ? 1 : 0)));
    }
    /* 1x is the starting code */
    el.appendChild(svg("line", { x1: pad.l, x2: W - pad.r, y1: sy(1), y2: sy(1), stroke: "var(--ink-3)", "stroke-width": 1, "stroke-dasharray": "4 4" }));
    el.appendChild(svg("text", { x: W - pad.r, y: sy(1) - 6, "text-anchor": "end" }, "the starting code"));
    for (let v = 0; v <= xMax + 1e-9; v += xStep) {
      el.appendChild(svg("text", { x: sx(v), y: H - pad.b + 17, "text-anchor": v === 0 ? "start" : "middle" }, axis.tick(v)));
    }
    el.appendChild(svg("text", { class: "axis-title", x: pad.l, y: H - 8 }, axis.title));
    el.appendChild(svg("text", { class: "axis-title", x: 0, y: 12 }, "Speed-up over the starting code"));

    const pts = list.map((r) => ({ r, x: sx(axis.get(r)), y: sy(r.speedup), label: `${r.label} ${r.effort}` }));
    const spots = apin.placeLabels(pts, { x0: pad.l, x1: W - 4, y0: pad.t - 6, y1: H - pad.b - 2 });
    pts.forEach((p, i) => {
      const g = svg("g", {
        transform: `translate(${p.x.toFixed(1)},${p.y.toFixed(1)})`,
        tabindex: 0,
        role: "img",
        "aria-label": `${p.label}: ${fmt.times(p.r.speedup)}, ${axis.tick(axis.get(p.r))}`,
      });
      g.appendChild(svg("circle", { r: 14, fill: "transparent" }));
      const color = modelColor(p.r.model);
      const shape = SHAPES[p.r.effort] || "circle";
      /* a row with unfinished tasks is drawn hollow */
      const look = p.r.complete ? { fill: color, stroke: "var(--paper)", "stroke-width": 1 } : { fill: "var(--paper)", stroke: color, "stroke-width": 1.5 };
      if (shape === "square") g.appendChild(svg("rect", { x: -4.5, y: -4.5, width: 9, height: 9, ...look }));
      else if (shape === "diamond") g.appendChild(svg("path", { d: "M0-6L6 0 0 6-6 0z", ...look }));
      else if (shape === "triangle") g.appendChild(svg("path", { d: "M0-5.6L5.8 4.6H-5.8z", ...look }));
      else g.appendChild(svg("circle", { r: 4.8, ...look }));
      tip.bind(
        g,
        `<b>${esc(p.r.label)}, ${esc(p.r.effort)} effort</b><br>` +
          `<span class="k">Speed-up</span> ${fmt.times(p.r.speedup)}${p.r.complete ? "" : ` (${p.r.n_tasks - p.r.tasks_run} of ${p.r.n_tasks} tasks did not finish)`}<br>` +
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
      (list.some((r) => !r.complete) ? '<span><svg viewBox="-8 -8 16 16" width="14" height="14" aria-hidden="true"><circle r="4" fill="none" stroke="var(--ink-3)" stroke-width="1.5"/></svg>some tasks did not finish</span>' : "");
  }

  /* ---------- tasks ---------- */

  function heatClass(ratio) {
    const t = Math.min(1, Math.log(Math.max(ratio, 1)) / Math.log(RAMP_TOP));
    return "h" + Math.max(1, Math.min(6, Math.ceil(t * 6)));
  }

  function renderMatrix() {
    const rel = state.rel;
    const model = $("#f-model").value;
    const list = rel.leaderboard.filter((r) => r.model === model);
    const table = $("#matrix");
    if (!list.length) {
      table.innerHTML = '<tbody><tr><td class="empty">This release has no results yet.</td></tr></tbody>';
      return;
    }
    let html =
      "<thead><tr><th>Task</th>" +
      list.map((r) => `<th class="config"><span class="tag">${esc(r.effort)}</span></th>`).join("") +
      "</tr></thead><tbody>";
    for (const id of rel.tasks) {
      const m = meta(rel, id);
      /* How the Stan-style reference and the best autoresearch sampler compare on this task. */
      const den = m.denominators;
      const yard =
        den && isNum(den.start) && den.start > 0
          ? `<span class="yard">${[isNum(den.reference) ? `Stan-style reference ${fmt.times(den.reference / den.start)}` : "", isNum(den.frontier) ? `best autoresearch sampler ${fmt.times(den.frontier / den.start)}` : ""].filter(Boolean).join(", ")}</span>`
          : "";
      html +=
        `<tr><td class="task"><b>${esc(m.title)}</b><code>${esc(id)}</code>${yard}` +
        `<a href="${apin.REPO}/tree/main/harbor/tasks/${esc(id)}">${esc(m.lane ? `lane ${m.lane}, task files` : "task files")}</a></td>`;
      for (const r of list) {
        const cell = cellOf(r, id);
        const st = cellState(cell);
        if (st === "none") {
          html += '<td class="cell"><span class="heat none">not run</span></td>';
          continue;
        }
        const open = `data-run="${esc(r.config_key + "||" + id)}" title="${esc(cellLabel(cell))}"`;
        const tries = cell.attempts > 1 ? `<small>${cell.valid}/${cell.attempts} valid</small>` : "";
        if (st === "unfinished") html += `<td class="cell"><button type="button" class="heat infra" ${open}>did not finish</button></td>`;
        else if (st === "invalid") html += `<td class="cell"><button type="button" class="heat fail" ${open}>${ICONS.cross}failed</button></td>`;
        else if (st === "flat") html += `<td class="cell"><button type="button" class="heat flat" ${open}>${fmt.times(cell.speedup)}${tries}</button></td>`;
        else html += `<td class="cell"><button type="button" class="heat ${heatClass(cell.speedup)}" ${open}>${fmt.times(cell.speedup)}${tries}</button></td>`;
      }
      html += "</tr>";
    }
    table.innerHTML = html + "</tbody>";
  }

  /* ---------- run details ---------- */

  function openRun(key, pick) {
    const [configKey, taskId] = key.split("||");
    const rel = state.rel;
    const tries = attempts(rel, configKey, taskId);
    if (!tries.length) return;
    const row = rel.leaderboard.find((r) => r.config_key === configKey) || {};
    const cell = cellOf(row, taskId) || {};
    const t = tries.find((a) => a.trial_id === (pick || cell.best_trial)) || tries[0];
    const sol = t.solution || {};
    const tone = t.outcome === "interrupted" || t.outcome === "unscorable" ? "warn" : t.outcome === "invalid" ? "bad" : isNum(t.start_speedup) && t.start_speedup >= GAIN ? "good" : "mute";
    const pair = (k, v) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`;
    const ended = { finished: "the agent finished", time_limit: "reached the time limit", agent_error: "the agent exited with an error", provider_limit: "the model provider cut it off", infrastructure: "the infrastructure failed", agent_crash: "the agent never started" };
    const facts = [];
    if (t.outcome !== "interrupted") {
      facts.push(pair(t.rule_break ? "Measured against the starting code, not counted" : "Against the starting code", fmt.times(t.start_speedup)));
      if (isNum(t.ref_speedup)) facts.push(pair("Against the Stan-style reference", fmt.times(t.ref_speedup)));
      if (isNum(t.frontier_speedup)) facts.push(pair("Against the best autoresearch sampler", fmt.times(t.frontier_speedup)));
      facts.push(pair("Efficiency", fmt.eff(t.soft_min_eff)), pair("Starting code", fmt.eff(t.start_soft_min)));
      if (isNum(t.reference_soft_min)) facts.push(pair("Stan-style reference", fmt.eff(t.reference_soft_min)));
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
      tries.length > 1
        ? '<div><h4>Attempts</h4><div class="chips">' +
          tries.map((a, i) => `<button type="button" data-run="${esc(key)}" data-attempt="${esc(a.trial_id)}"${a === t ? ' aria-current="true"' : ""}>${i + 1}: ${esc(attemptLabel(a).split(":")[0])}</button>`).join("") +
          "</div></div>"
        : "";
    // An interrupted attempt has no result, so the numbers of the file it left behind are not shown.
    const classes = t.outcome === "interrupted" ? [] : Object.entries(t.by_class || {}).sort((a, b) => a[1] - b[1]);
    const top = Math.max(...classes.map((c) => c[1]), 1e-12);
    const classHtml = classes.length
      ? '<div><h4>Efficiency by target shape, weakest first</h4><div class="class-bars">' +
        classes.map(([k, v]) => `<div><span>${esc(k)}</span><i style="width:${Math.max(1, (v / top) * 100).toFixed(1)}%"></i><b>${Number(v).toExponential(2)}</b></div>`).join("") +
        "</div></div>"
      : "";
    const failed = (t.failed_targets || []).length ? `<div><h4>Failed targets</h4>${t.failed_targets.map((x) => `<p class="quote">${esc(x)}</p>`).join("")}</div>` : "";
    const note = t.approach_summary ? `<div><h4>The agent's closing note</h4><pre class="log">${esc(t.approach_summary)}</pre></div>` : "";
    const error = t.error ? `<div><h4>Run error</h4><pre class="log">${esc(t.error)}</pre></div>` : "";
    const ruling = t.rule_break ? `<div><h4>Why it does not count</h4><p class="quote">${esc(t.rule_break.reason)}</p></div>` : "";

    document.querySelectorAll(".heat.is-open").forEach((n) => n.classList.remove("is-open"));
    document.querySelectorAll(`.heat[data-run="${CSS.escape(key)}"]`).forEach((n) => n.classList.add("is-open"));
    apin.drawer.open({
      title: meta(rel, taskId).title,
      sub: `${taskId} · ${row.label || t.model} at ${t.effort} effort · ${t.agent || "unknown agent"} · ${t.trial_id}`,
      body: `<div>${status(tone, attemptLabel(t))}</div>${ruling}${chips}<div class="kv">${facts.join("")}</div>${classHtml}${failed}${note}${error}`,
    });
  }

  document.addEventListener("click", (e) => {
    const run = e.target.closest("[data-run]");
    if (run) return openRun(run.getAttribute("data-run"), run.getAttribute("data-attempt"));
    const sort = e.target.closest("[data-sort]");
    if (sort) {
      const key = sort.getAttribute("data-sort");
      state.sort = state.sort.key === key && key !== "rank" ? { key, dir: -state.sort.dir } : { key, dir: key === "speedup" ? -1 : 1 };
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

  function showRelease(tag) {
    const data = state.data;
    const rel = (data.releases || []).find((r) => r.release === tag) || data.releases[0];
    state.rel = rel;
    state.sort = { key: "rank", dir: 1 };
    state.models = [...new Set(rel.leaderboard.map((r) => r.model))];
    const runs = rel.trials.filter((t) => rel.tasks.includes(t.task_id));
    const harness = [...new Set(runs.map((t) => t.agent).filter(Boolean))].join(", ");
    const unfinished = runs.filter((t) => t.outcome === "interrupted" || t.outcome === "unscorable").length;
    $("#meta").innerHTML =
      `<span><b>${rel.leaderboard.length}</b> model and effort settings</span>` +
      `<span><b>${runs.length}</b> runs on <b>${rel.tasks.length}</b> tasks${unfinished ? `, <b>${unfinished}</b> did not finish` : ""}</span>` +
      (harness ? `<span>Agent <b>${esc(harness)}</b></span>` : "") +
      `<span>Updated <b>${esc(fmt.date(data.generated_at))}</b></span>`;
    $("#release-note").textContent = rel.release === data.current_release ? "" : rel.note || "";
    $("#release-note").classList.toggle("is-old", rel.release !== data.current_release);
    /* The task table shows one model at a time, so it stays readable as models are added. */
    const label = (m) => (rel.leaderboard.find((r) => r.model === m) || {}).label || m;
    $("#f-model").innerHTML = state.models.map((m) => `<option value="${esc(m)}">${esc(label(m))}</option>`).join("");
    $("#f-model").hidden = state.models.length < 2;
    renderLegend();
    renderBoard();
    renderMatrix();
    if (state.redraw) state.redraw();
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
      $("#f-model").addEventListener("change", renderMatrix);
      state.rel = data.releases[0];
      state.redraw = apin.watchWidth($("#scatter-box"), renderScatter);
      showRelease(data.current_release);
    })
    .catch((err) => {
      console.error(err);
      $("#meta").textContent = "This build has no published benchmark results.";
      $("#board").innerHTML =
        '<tbody><tr><td class="empty">Could not load the results. Run <code>scripts/collect_harbor_results.py</code> to write <code>data/benchmark.json</code>, then reload.</td></tr></tbody>';
    });
})();
