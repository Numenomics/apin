/* Lanes and tracks page: the lane definitions live in site.js (apin.LANES, apin.TRACKS). */
(function () {
  "use strict";
  const { $, esc, fmt, cmd } = apin;

  apin.mount("lanes");

  $("#s-setup").innerHTML = cmd("git clone https://github.com/Numenomics/apin.git && cd apin\nuv sync && uv run apin fetch-data");
  $("#s-brief").innerHTML = cmd("uv run apin autoresearch brief --lane A1");
  $("#s-eval").innerHTML = cmd(
    'uv run apin autoresearch eval --lane A1 --run-id <handle>_a1 \\\n  --dims 6,20 --repeats 2 \\\n  --hypothesis "baseline: unmodified local incumbent"'
  );

  const KINDS = [
    { id: "algorithm", title: "Algorithm lanes: change how the sampler works" },
    { id: "systems", title: "Systems lanes: make existing engines faster" },
    { id: "calibration", title: "Calibration lane: bugs that grade agents" },
  ];
  const BENCH_TASK = { A0: "a0-ground-up", A1: "a1-preconditioning", A2: "a2-funnel-dynamics", A3: "a3-integrator-trajectory", A4: "a4-nonlinear-transport" };

  function standing(summary) {
    if (!summary) return "";
    if (!summary.trials) return '<div><dt>Stands at</dt><dd>No trials recorded yet.</dd></div>';
    const t = summary.top;
    const stan = summary.stan ? `, ${summary.stan.real_stan_passed} of ${summary.stan.real_stan_total} real Stan models passed` : "";
    return (
      `<div><dt>Stands at</dt><dd>${fmt.int(summary.kept)} kept of ${fmt.int(summary.trials)} trials in ${summary.runs} ${summary.runs === 1 ? "run" : "runs"}${stan}. ` +
      (t ? `<a href="/frontier/#trial=${encodeURIComponent(t.id)}">Best kept trial: ${fmt.eff(t.soft_min_eff)}</a>` : "") +
      "</dd></div>" +
      (summary.claims.length ? `<div><dt>Claimed now</dt><dd>${summary.claims.map((c) => esc(c.agent)).join(", ")}</dd></div>` : "")
    );
  }

  function render(summaries) {
    const byLane = new Map((summaries || []).map((s) => [s.lane.id, s]));
    $("#lanes").innerHTML = KINDS.map((kind) => {
      const lanes = apin.LANES.filter((l) => l.kind === kind.id);
      return (
        `<h3 class="kind">${esc(kind.title)}</h3><div class="rows">` +
        lanes
          .map((l) => {
            const links = [
              `<a href="${apin.REPO}/blob/main/autoresearch/lanes/${l.id}/program.md">Full lane brief</a>`,
              `<a href="${apin.REPO}/blob/main/autoresearch/lanes/${l.id}/NEGATIVE_MEMORY.md">Ideas already refuted</a>`,
            ];
            if (BENCH_TASK[l.id]) links.push(`<a href="/benchmark/">Benchmark task ${BENCH_TASK[l.id]}</a>`);
            return (
              `<div class="row lane" id="${l.id}"><div><div class="code"><i class="swatch" style="background:${apin.laneColor(l.id)}"></i>${l.id}</div></div>` +
              `<div class="what"><h3>${esc(l.name)}</h3><p>${esc(l.goal)}</p>` +
              `<dl class="facts"><div><dt>You edit</dt><dd>${esc(l.edits)}</dd></div><div><dt>Held fixed</dt><dd>${esc(l.fixed)}</dd></div>${standing(byLane.get(l.id))}</dl></div>` +
              `<div class="do">${cmd(l.start)}<div class="links">${links.join("")}</div></div></div>`
            );
          })
          .join("") +
        "</div>"
      );
    }).join("");
    if (location.hash.length > 1) {
      const target = document.getElementById(location.hash.slice(1));
      if (target) target.scrollIntoView();
    }
  }

  $("#track-table").innerHTML =
    "<thead><tr><th>Track</th><th>What it ranks</th><th>Run it</th></tr></thead><tbody>" +
    apin.TRACKS.map(
      (t) =>
        `<tr><td><span class="lane-tag">${esc(t.id)}</span><div class="sub" style="font-family:var(--sans)">${esc(t.name)}</div></td>` +
        `<td class="wrap-cell" style="min-width:240px">${esc(t.ranks)}</td><td style="width:56%;min-width:320px;white-space:normal">${cmd(t.cmd)}</td></tr>`
    ).join("") +
    "</tbody>";

  render(null);
  apin
    .openStore()
    .then(async (store) => {
      const overview = await store.overview();
      const claims = overview.claims || [];
      render(apin.laneRows(overview).map((row) => ({ ...row, claims: claims.filter((c) => c.lane === row.lane.id) })));
      const active = overview.lanes.length;
      $("#lanes-note").innerHTML = `What you edit, what stays fixed, and where the lane stands today. ${active} of ${apin.LANES.length} lanes have trials posted${store.source === "live" ? "" : " in the snapshot shipped with this page"}.`;
    })
    .catch(() => {
      $("#lanes-note").textContent = "What you edit and what stays fixed. Could not load current standings.";
    });
})();
