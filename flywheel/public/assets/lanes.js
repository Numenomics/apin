/* Lanes page: the lane definitions live in site.js (apin.LANES, apin.TRACKS). */
(function () {
  "use strict";
  const { $, esc, fmt, cmd } = apin;

  apin.mount("lanes");

  const KINDS = [
    { id: "algorithm", title: "Algorithm lanes" },
    { id: "systems", title: "Engine lane" },
    { id: "calibration", title: "Bug lane" },
  ];
  const BENCH_TASK = { A0: "a0-ground-up", A1: "a1-preconditioning", A2: "a2-funnel-dynamics", A3: "a3-integrator-trajectory", A4: "a4-nonlinear-transport" };

  /* The lane's frontier, once the Flywheel has answered: the best trial that an independent
     verifier ran through the real-posterior gate and that passed it. A lane may have none. */
  function standing(summary) {
    if (!summary) return "";
    const t = summary.frontier;
    const dt = "<dt>Best that passed the real-posterior gate</dt>";
    if (!t) {
      return `<div>${dt}<dd>${summary.trials ? `None yet, from ${fmt.int(summary.trials)} trials.` : "No trials posted yet."}</dd></div>`;
    }
    return `<div>${dt}<dd><a href="/autoresearch/#trial=${encodeURIComponent(t.id)}">${fmt.speedup(t.speedup)} by ${esc(t.author || t.agent_model || "unknown")}</a>, from ${fmt.int(summary.trials)} trials</dd></div>`;
  }

  function render(byLane) {
    $("#lanes").innerHTML = KINDS.map((kind) => {
      const lanes = apin.LANES.filter((l) => l.kind === kind.id);
      return (
        `<h2 class="kind">${esc(kind.title)}</h2><div class="rows">` +
        lanes
          .map((l) => {
            const links = [
              `<a href="${apin.REPO}/blob/main/autoresearch/lanes/${l.id}/program.md">Lane brief</a>`,
              `<a href="${apin.REPO}/blob/main/autoresearch/lanes/${l.id}/NEGATIVE_MEMORY.md">Ideas already refuted</a>`,
            ];
            if (BENCH_TASK[l.id]) links.push(`<a href="/benchmark/">Benchmark task</a>`);
            /* The Autoresearch page has a tab only for a lane with trials. */
            if (byLane && byLane.has(l.id) && byLane.get(l.id).trials) links.unshift(`<a href="/autoresearch/#lane=${l.id}">Trials</a>`);
            return (
              `<div class="row lane" id="${l.id}"><div><div class="code"><i class="swatch" style="background:${apin.laneColor(l.id)}"></i>${l.id}</div></div>` +
              `<div class="what"><h3>${esc(l.name)}</h3><p>${esc(l.goal)}</p>` +
              `<dl class="facts"><div><dt>The agent edits</dt><dd>${esc(l.edits)}</dd></div><div><dt>Held fixed</dt><dd>${esc(l.fixed)}</dd></div>${standing(byLane && (byLane.get(l.id) || { trials: 0 }))}</dl></div>` +
              `<div class="do">${cmd(apin.lanePrompt(l), { prose: true })}<div class="links">${links.join("")}</div></div></div>`
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
    "<thead><tr><th>Track</th><th>What it ranks</th><th>Command</th></tr></thead><tbody>" +
    apin.TRACKS.map(
      (t) =>
        `<tr><td><span class="lane-tag">${esc(t.id)}</span><div class="sub" style="font-family:var(--sans)">${esc(t.name)}</div></td>` +
        `<td class="wrap-cell" style="min-width:240px">${esc(t.ranks)}</td><td style="width:56%;min-width:320px;white-space:normal">${cmd(t.cmd)}</td></tr>`
    ).join("") +
    "</tbody>";

  render(null);
  apin.api
    .overview()
    .then((overview) => render(new Map(overview.lanes.map((l) => [l.lane, l]))))
    .catch(() => {});
})();
