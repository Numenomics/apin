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
  const PER_FRAME = 120; /* steps per 60 Hz frame: about six seconds to a typical verdict */
  const MAX_ANIMATED_MS = 12000; /* a slow chain is finished off-screen instead of animating on */

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

  /* The chain is cheap (two log-density calls per step), but it must never run where nobody
     can see it. It starts when the figure scrolls into view, pauses when it leaves or the tab
     is hidden, runs once, and then stops for good. Resizing or switching theme repaints the
     draws already made instead of sampling again. */
  let visible = false;
  let animatedMs = 0;
  let lastTime = 0;

  function paint() {
    run.map = drawField();
    const { ctx } = sizeCanvas(draws);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--ink").trim();
    ctx.globalAlpha = 0.32;
    run.ctx = ctx;
    for (let i = 0; i < run.pts.length; i += 2) ctx.fillRect(run.map.px(run.pts[i]) - 0.7, run.map.py(run.pts[i + 1]) - 0.7, 1.4, 1.4);
  }

  function start() {
    cancelAnimationFrame(frame);
    frame = 0;
    run = {
      map: null,
      ctx: null,
      pts: [],
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
    animatedMs = 0;
    paint();
    report();
    if (still) finish();
    else resume();
  }

  function finish() {
    while (run.verdict === "unproven") advance(PER_FRAME * 50);
    report();
  }

  function resume() {
    if (frame || !visible || document.hidden || !run || run.verdict !== "unproven") return;
    lastTime = 0;
    frame = requestAnimationFrame(tick);
  }

  function pause() {
    cancelAnimationFrame(frame);
    frame = 0;
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
      if (r.n % 8 === 0) {
        r.pts.push(r.x, r.y);
        r.ctx.fillRect(r.map.px(r.x) - 0.7, r.map.py(r.y) - 0.7, 1.4, 1.4);
      }
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

  function tick(now) {
    frame = 0;
    /* Step by elapsed time, so a 120 Hz screen does the same work per second as a 60 Hz one. */
    const dt = lastTime ? Math.min(now - lastTime, 50) : 16.7;
    lastTime = now;
    animatedMs += dt;
    if (animatedMs > MAX_ANIMATED_MS) finish();
    else {
      advance(Math.max(1, Math.round((PER_FRAME * dt) / 16.7)));
      report();
    }
    resume();
  }

  $("#r-again").addEventListener("click", () => {
    start();
  });
  apin.onTheme(() => run && paint());
  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => run && paint(), 200);
  });
  document.addEventListener("visibilitychange", () => (document.hidden ? pause() : resume()));
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(
      (entries) => {
        visible = entries[0].isIntersecting;
        if (!visible) pause();
        else if (!run) start();
        else resume();
      },
      { threshold: 0.2 }
    ).observe(draws);
  } else {
    visible = true;
    start();
  }

  /* ---------- frontier summary ---------- */

  apin
    .openStore()
    .then(async (store) => {
      const overview = await store.overview();
      const rows = apin.laneRows(overview);
      const t = overview.totals;
      $("#front-note").textContent =
        `The best kept result in each lane so far: ${fmt.int(t.trials)} trials recorded, ${fmt.int(t.kept)} kept, ` +
        `${t.claims} ${t.claims === 1 ? "task" : "tasks"} claimed now` +
        (store.source === "snapshot" ? " (from the snapshot shipped with this page)." : ".");
      $("#front-table").innerHTML =
        "<thead><tr><th>Lane</th><th>Focus</th><th class='num'>Best efficiency</th><th class='num'>Gain within its run</th>" +
        "<th>Real Stan posteriors</th><th class='num'>Trials</th><th>Progress</th></tr></thead><tbody>" +
        rows
          .map((r) => {
            if (!r.trials) {
              return `<tr><td>${apin.laneTag(r.lane.id)}</td><td>${esc(r.lane.name)}</td><td colspan="5" class="sub">No trials recorded yet. <a href="/lanes/#${r.lane.id}">Start this lane</a></td></tr>`;
            }
            const stan = r.stan
              ? status(r.stan.real_stan_passed === r.stan.real_stan_total ? "good" : "mute", `${r.stan.real_stan_passed} of ${r.stan.real_stan_total} passed`)
              : '<span class="sub">not run yet</span>';
            return (
              `<tr><td>${apin.laneTag(r.lane.id)}</td><td><a href="/lanes/#${r.lane.id}">${esc(r.lane.name)}</a></td>` +
              `<td class="num">${fmt.eff(r.top && r.top.soft_min_eff)}</td><td class="num">${fmt.times(r.gain)}</td>` +
              `<td>${stan}</td><td class="num">${fmt.int(r.kept)} kept of ${fmt.int(r.trials)}</td>` +
              `<td>${apin.sparkline(r.steps, apin.laneColor(r.lane.id))}</td></tr>`
            );
          })
          .join("") +
        "</tbody>";
    })
    .catch(() => {
      $("#front-table").innerHTML = '<tbody><tr><td class="empty">Could not load the frontier. Reload the page, or read <a href="/api/frontier">the API</a>.</td></tr></tbody>';
    });

  /* ---------- benchmark summary ---------- */

  /* The task blocks open their runs on the benchmark page. */
  $("#board-table").addEventListener("click", (e) => {
    if (e.target.closest("[data-run]")) location.href = "/benchmark/";
  });

  apin
    .getJSON("/data/benchmark.json")
    .then((data) => {
      const rel = apin.bench.release(data, data.current_release);
      const board = rel.leaderboard || [];
      const whole = (r, v) => fmt.num(v, r.attempts_per_task > 1 ? 1 : 0);
      $("#board-note").textContent = `Release ${rel.release}: ${board.length} model and effort settings, ${rel.trials.length} runs across ${rel.tasks.length} tasks.`;
      $("#board-table").innerHTML =
        "<thead><tr><th></th><th>Model</th><th>Effort</th><th>Tasks, fixes then speed-ups</th><th class='num'>Solved</th>" +
        "<th class='num'>Typical speed-up</th><th class='num'>Cost of the sweep</th></tr></thead><tbody>" +
        board
          .slice(0, 6)
          .map(
            (r) =>
              `<tr><td class="rank">${r.rank}</td><td>${esc(r.label)}</td><td><span class="tag">${esc(r.effort)}</span></td>` +
              `<td>${apin.bench.pips(rel, r)}</td><td class="num">${whole(r, r.solved)} of ${r.n_tasks}</td>` +
              `<td class="num">${r.opt_geomean ? fmt.times(r.opt_geomean) : "—"}</td><td class="num">${fmt.usd(r.total_cost_usd)}</td></tr>`
          )
          .join("") +
        "</tbody>";
    })
    .catch(() => {
      $("#board-table").innerHTML = '<tbody><tr><td class="empty">No benchmark results are published with this build yet.</td></tr></tbody>';
    });
})();
