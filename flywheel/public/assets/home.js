/* Overview page: the live gate illustration and the two summary tables. */
(function () {
  "use strict";
  const { $, esc, fmt, status, cmd } = apin;

  apin.mount("home");

  $("#join-prompt").innerHTML = cmd(apin.JOIN_PROMPT, { prose: true });
  $("#way-research").innerHTML = cmd("uv run apin autoresearch brief --lane A1");
  $("#way-bench").innerHTML = cmd("harbor run -p harbor/tasks -a claude-code \\\n  -m anthropic/claude-sonnet-4-5");
  $("#way-score").innerHTML = cmd("uv run apin invariance --sampler mine.py");

  /* ---------- the gate, running ----------
     Target: x ~ N(0, s^2), y | x ~ N(b (x^2 - s^2), 1). Both means are 0 and both
     standard deviations are known, so the error of the chain's running mean is exact.
     The verdict follows the benchmark's first rule: "right" only when the upper
     99.5% bound on the error is inside the tolerance. */
  const S = 1.5;
  const B = 0.4;
  const SD = [S, Math.sqrt(1 + 2 * B * B * S ** 4)];
  const VIEW = { x0: -5.4, x1: 5.4, y0: -4.2, y1: 9.6 };
  const STEP = 1.1;
  const BATCH = 500;
  const Z = 2.576;
  const GATE = 0.1;
  const BUDGET = 600000;
  const PER_FRAME = 120; /* about six seconds to a verdict at 60 frames a second */

  const field = $("#plot-field");
  const draws = $("#plot-draws");
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let run = null;
  let frame = 0;

  const logp = (x, y) => -(x * x) / (2 * S * S) - (y - B * (x * x - S * S)) ** 2 / 2;

  function gauss() {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  }

  function sizeCanvas(canvas) {
    const box = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(box.width * dpr));
    canvas.height = Math.max(1, Math.round(box.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: box.width, h: box.height };
  }

  function drawField() {
    const { ctx, w, h } = sizeCanvas(field);
    const css = getComputedStyle(document.documentElement);
    const px = (x) => ((x - VIEW.x0) / (VIEW.x1 - VIEW.x0)) * w;
    const py = (y) => h - ((y - VIEW.y0) / (VIEW.y1 - VIEW.y0)) * h;
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 1;
    ctx.strokeStyle = css.getPropertyValue("--rule-2").trim();
    for (const r of [1, 2, 3]) {
      const c = (r * r) / 2;
      const xm = S * Math.sqrt(2 * c);
      const upper = [];
      const lower = [];
      for (let i = 0; i <= 120; i += 1) {
        const x = -xm + (2 * xm * i) / 120;
        const rest = Math.sqrt(Math.max(0, 2 * c - (x * x) / (S * S)));
        const mid = B * (x * x - S * S);
        upper.push([px(x), py(mid + rest)]);
        lower.push([px(x), py(mid - rest)]);
      }
      ctx.beginPath();
      upper.forEach(([a, b], i) => (i ? ctx.lineTo(a, b) : ctx.moveTo(a, b)));
      lower.reverse().forEach(([a, b]) => ctx.lineTo(a, b));
      ctx.closePath();
      ctx.stroke();
    }
    return { px, py };
  }

  function start() {
    cancelAnimationFrame(frame);
    const map = drawField();
    const { ctx } = sizeCanvas(draws);
    const css = getComputedStyle(document.documentElement);
    ctx.fillStyle = css.getPropertyValue("--ink").trim();
    ctx.globalAlpha = 0.32;
    run = {
      map,
      ctx,
      x: -3.4,
      y: 6.5,
      lp: logp(-3.4, 6.5),
      n: 0,
      acc: 0,
      sum: [0, 0],
      batchSum: [0, 0],
      batchMeans: [[], []],
      bound: null,
      verdict: "unproven",
    };
    report();
    if (still) {
      while (run.verdict === "unproven" && run.n < BUDGET) advance(PER_FRAME);
      report();
    } else {
      frame = requestAnimationFrame(tick);
    }
  }

  function advance(steps) {
    const r = run;
    for (let i = 0; i < steps; i += 1) {
      const nx = r.x + STEP * gauss();
      const ny = r.y + STEP * gauss();
      const nlp = logp(nx, ny);
      if (Math.log(Math.random()) < nlp - r.lp) {
        r.x = nx;
        r.y = ny;
        r.lp = nlp;
        r.acc += 1;
      }
      r.n += 1;
      r.sum[0] += r.x;
      r.sum[1] += r.y;
      r.batchSum[0] += r.x;
      r.batchSum[1] += r.y;
      if (r.n % BATCH === 0) {
        r.batchMeans[0].push(r.batchSum[0] / BATCH);
        r.batchMeans[1].push(r.batchSum[1] / BATCH);
        r.batchSum = [0, 0];
      }
      if (i % 8 === 0) r.ctx.fillRect(r.map.px(r.x) - 0.7, r.map.py(r.y) - 0.7, 1.4, 1.4);
    }
    const k = r.batchMeans[0].length;
    if (k >= 20) {
      let worst = 0;
      for (const d of [0, 1]) {
        const mean = r.sum[d] / r.n;
        const bm = r.batchMeans[d];
        const centre = bm.reduce((a, v) => a + v, 0) / k;
        const variance = bm.reduce((a, v) => a + (v - centre) ** 2, 0) / (k - 1);
        const mcse = Math.sqrt(variance / k);
        worst = Math.max(worst, (Math.abs(mean) + Z * mcse) / SD[d]);
      }
      r.bound = worst;
      if (worst < GATE) r.verdict = "right";
    }
    if (r.verdict === "unproven" && r.n >= BUDGET) r.verdict = "out";
  }

  function report() {
    const r = run;
    $("#r-draws").textContent = fmt.int(r.n);
    $("#r-acc").textContent = r.n ? Math.round((r.acc / r.n) * 100) + "%" : "—";
    $("#r-err").textContent = r.bound === null ? "—" : r.bound.toFixed(2) + " SD";
    const el = $("#r-verdict");
    const next =
      r.verdict === "right"
        ? status("good", `Right: within 0.10 SD after ${fmt.int(r.n)} draws`)
        : r.verdict === "out"
          ? status("warn", "Unproven at this budget")
          : status("mute", "Unproven so far");
    if (el.dataset.state !== r.verdict) {
      el.innerHTML = next;
      el.dataset.state = r.verdict;
    }
  }

  function tick() {
    advance(PER_FRAME);
    report();
    if (run.verdict === "unproven") frame = requestAnimationFrame(tick);
  }

  $("#r-again").addEventListener("click", start);
  apin.onTheme(start);
  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(start, 200);
  });
  start();

  /* ---------- frontier summary ---------- */

  apin
    .loadGraph()
    .then((graph) => {
      const rows = apin.laneSummaries(graph);
      const kept = graph.nodes.filter((n) => apin.KEPT.has(n.decision)).length;
      $("#front-note").textContent =
        `The best kept result in each lane so far: ${graph.nodes.length} trials recorded, ${kept} kept, ` +
        `${graph.claims.length} ${graph.claims.length === 1 ? "task" : "tasks"} claimed right now` +
        (graph.source === "snapshot" ? " (from the snapshot shipped with this page)." : ".");
      $("#front-table").innerHTML =
        "<thead><tr><th>Lane</th><th>What is being improved</th><th class='num'>Best efficiency</th><th class='num'>Gain within its run</th>" +
        "<th>Real Stan posteriors</th><th class='num'>Trials</th><th>Progress</th></tr></thead><tbody>" +
        rows
          .map((r) => {
            if (!r.nodes.length) {
              return `<tr><td>${apin.laneTag(r.lane.id)}</td><td>${esc(r.lane.name)}</td><td colspan="5" class="sub">No trials recorded yet. <a href="/lanes/#${r.lane.id}">Start this lane</a></td></tr>`;
            }
            const stan = r.stan
              ? status(r.stan.real_stan_passed === r.stan.real_stan_total ? "good" : "mute", `${r.stan.real_stan_passed} of ${r.stan.real_stan_total} passed`)
              : '<span class="sub">not run yet</span>';
            return (
              `<tr><td>${apin.laneTag(r.lane.id)}</td><td><a href="/lanes/#${r.lane.id}">${esc(r.lane.name)}</a></td>` +
              `<td class="num">${fmt.eff(r.top && r.top.soft_min_eff)}</td><td class="num">${fmt.times(r.gain)}</td>` +
              `<td>${stan}</td><td class="num">${r.kept} kept of ${r.nodes.length}</td>` +
              `<td>${apin.sparkline(r.steps, apin.laneColor(r.lane.id))}</td></tr>`
            );
          })
          .join("") +
        "</tbody>";
    })
    .catch(() => {
      $("#front-table").innerHTML = '<tbody><tr><td class="empty">The frontier could not be loaded. Reload the page, or read it from <a href="/api/frontier">the API</a>.</td></tr></tbody>';
    });

  /* ---------- benchmark summary ---------- */

  apin
    .getJSON("/data/benchmark.json")
    .then((data) => {
      const board = data.leaderboard || [];
      $("#board-note").textContent = `Release ${data.release}: ${board.length} model and effort settings, ${data.trials.length} runs across ${data.tasks.length} tasks.`;
      $("#board-table").innerHTML =
        "<thead><tr><th></th><th>Model</th><th>Effort</th><th>Tasks, fixes then speed-ups</th><th class='num'>Passed</th>" +
        "<th class='num'>Typical speed-up</th><th class='num'>Cost of the sweep</th></tr></thead><tbody>" +
        board
          .slice(0, 6)
          .map(
            (r) =>
              `<tr><td class="rank">${r.rank}</td><td>${esc(r.label)}</td><td><span class="tag">${esc(r.effort)}</span></td>` +
              `<td>${apin.bench.pips(data, r)}</td><td class="num">${r.passed} of ${r.n_trials}</td>` +
              `<td class="num">${r.opt_geomean_speedup ? fmt.times(r.opt_geomean_speedup) : "—"}</td><td class="num">${fmt.usd(r.total_cost_usd)}</td></tr>`
          )
          .join("") +
        "</tbody>";
    })
    .catch(() => {
      $("#board-table").innerHTML = '<tbody><tr><td class="empty">No benchmark results are published with this build yet.</td></tr></tbody>';
    });
})();
