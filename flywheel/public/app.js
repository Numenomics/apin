const SVG_NS = "http://www.w3.org/2000/svg";
const KEPT_DECISIONS = new Set(["keep", "keep_baseline", "keep_simpler"]);

const state = {
  dag: { nodes: [], edges: [], frontier: {}, claims: [], alignment: [] },
  nodeById: new Map(),
  childrenById: new Map(),
  selectedId: null,
  source: "loading",
  logScale: true,
};

function laneColor(lane) {
  const styles = getComputedStyle(document.documentElement);
  const map = {
    A0: styles.getPropertyValue("--lane-A0").trim() || "#f59e0b",
    A1: styles.getPropertyValue("--lane-A1").trim() || "#38bdf8",
    A2: styles.getPropertyValue("--lane-A2").trim() || "#a855f7",
    A3: styles.getPropertyValue("--lane-A3").trim() || "#10b981",
    A4: styles.getPropertyValue("--lane-A4").trim() || "#ec4899",
    S1: styles.getPropertyValue("--lane-S1").trim() || "#f97316",
    S2: styles.getPropertyValue("--lane-S2").trim() || "#06b6d4",
    S3: styles.getPropertyValue("--lane-S3").trim() || "#84cc16",
  };
  return map[lane] || styles.getPropertyValue("--lane-default").trim() || "#94a3b8";
}

function decisionClass(decision) {
  if (decision === "keep_baseline") return "badge-baseline";
  if (decision === "keep" || decision === "keep_simpler") return "badge-keep";
  if (decision === "archive_only") return "badge-archive";
  return "badge-discard";
}

function fmtEff(val) {
  if (val === null || val === undefined || !Number.isFinite(Number(val))) return "—";
  const n = Number(val);
  if (n === 0) return "0";
  if (Math.abs(n) < 0.001) return n.toExponential(2);
  return n.toFixed(5);
}

function fmtNum(val, digits = 2, suffix = "") {
  if (val === null || val === undefined || !Number.isFinite(Number(val))) return "—";
  return `${Number(val).toFixed(digits)}${suffix}`;
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function svgEl(tag, attrs = {}, text = null) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  }
  if (text !== null) el.textContent = text;
  return el;
}

function showTooltip(event, html) {
  const tip = document.getElementById("tooltip");
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14;
  const x = Math.min(window.innerWidth - tip.offsetWidth - pad, event.clientX + pad);
  const y = Math.min(window.innerHeight - tip.offsetHeight - pad, event.clientY + pad);
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}

function hideTooltip() {
  document.getElementById("tooltip").hidden = true;
}

async function loadData() {
  const badge = document.getElementById("data-source-badge");
  badge.textContent = "Loading...";
  badge.className = "badge badge-muted";

  let data = null;
  let source = "snapshot";

  try {
    const resp = await fetch("/api/dag", { headers: { Accept: "application/json" } });
    if (resp.ok) {
      const ct = resp.headers.get("content-type") || "";
      if (ct.includes("application/json")) {
        data = await resp.json();
        source = "live";
      }
    }
  } catch {
    // Fallback to static snapshot below
  }

  if (!data) {
    try {
      const resp = await fetch("./snapshot.json", { headers: { Accept: "application/json" } });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      data = await resp.json();
      source = "snapshot";
    } catch (err) {
      badge.textContent = "Load failed";
      badge.className = "badge badge-discard";
      console.error("Failed to load Flywheel data:", err);
      return;
    }
  }

  state.dag = {
    nodes: Array.isArray(data.nodes) ? data.nodes : [],
    edges: Array.isArray(data.edges) ? data.edges : [],
    frontier: data.frontier || {},
    claims: Array.isArray(data.claims) ? data.claims : [],
    alignment: Array.isArray(data.alignment) ? data.alignment : [],
  };
  state.source = source;
  state.nodeById = new Map(state.dag.nodes.map((n) => [n.id, n]));
  state.childrenById = new Map();
  for (const n of state.dag.nodes) {
    if (n.parent_id) {
      const list = state.childrenById.get(n.parent_id) || [];
      list.push(n);
      state.childrenById.set(n.parent_id, list);
    }
  }

  if (source === "live") {
    badge.textContent = "Live D1 API";
    badge.className = "badge badge-live";
  } else {
    badge.textContent = "Static Snapshot";
    badge.className = "badge badge-baseline";
  }

  populateFilters();
  renderAll();
}

function renderAll() {
  renderKpis();
  renderDag();
  renderFrontierChart();
  renderAlignmentChart();
  renderClaimsTable();
  renderLedgerTable();
  if (state.selectedId && state.nodeById.has(state.selectedId)) {
    openDrawer(state.selectedId, false);
  }
}

function renderKpis() {
  const { nodes, claims, alignment } = state.dag;
  const lanes = [...new Set(nodes.map((n) => n.lane))].sort();
  const runs = new Set(nodes.map((n) => `${n.lane}/${n.run_id}`));
  const kept = nodes.filter((n) => KEPT_DECISIONS.has(n.decision));
  const keepPct = nodes.length ? `${Math.round((kept.length / nodes.length) * 100)}% keep rate` : "—";

  document.getElementById("kpi-nodes").textContent = String(nodes.length);
  document.getElementById("kpi-runs").textContent = `${runs.size} runs`;
  document.getElementById("kpi-lanes").textContent = String(lanes.length);
  document.getElementById("kpi-lane-list").textContent = lanes.join(" · ") || "—";
  document.getElementById("kpi-kept").textContent = String(kept.length);
  document.getElementById("kpi-keep-rate").textContent = keepPct;
  document.getElementById("kpi-claims").textContent = String(claims.length);

  let bestStan = null;
  for (const a of alignment) {
    if (
      !bestStan ||
      a.real_stan_passed > bestStan.real_stan_passed ||
      (a.real_stan_passed === bestStan.real_stan_passed &&
        (a.real_ess_per_cost ?? 0) > (bestStan.real_ess_per_cost ?? 0))
    ) {
      bestStan = a;
    }
  }
  if (bestStan) {
    document.getElementById("kpi-real-stan").textContent = `${bestStan.real_stan_passed}/${bestStan.real_stan_total}`;
    document.getElementById("kpi-real-stan-sub").textContent = `${bestStan.lane} · ${bestStan.run_id}`;
  } else {
    document.getElementById("kpi-real-stan").textContent = "—";
    document.getElementById("kpi-real-stan-sub").textContent = "No verified runs yet";
  }
}

function computeDagLayout(nodes, edges) {
  const lanes = [...new Set(nodes.map((n) => n.lane))].sort();
  const byId = new Map(nodes.map((n) => [n.id, n]));

  // Topological depth per node within its lineage chain
  const depthMemo = new Map();
  function getDepth(id, visiting = new Set()) {
    if (depthMemo.has(id)) return depthMemo.get(id);
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const node = byId.get(id);
    if (!node || !node.parent_id || !byId.has(node.parent_id)) {
      depthMemo.set(id, 0);
      return 0;
    }
    const d = getDepth(node.parent_id, visiting) + 1;
    depthMemo.set(id, d);
    return d;
  }
  for (const n of nodes) getDepth(n.id);

  // Assign lane swimlanes & stack sibling trials cleanly at the same depth
  const colStep = 66;
  const leftPad = 92;
  const topPad = 24;
  const rowStep = 26;
  const laneGap = 30;

  const positions = new Map();
  const laneBands = [];
  let cursorY = topPad;
  let maxX = 600;

  for (const lane of lanes) {
    const laneNodes = nodes
      .filter((n) => n.lane === lane)
      .sort((a, b) => a.created_at.localeCompare(b.created_at) || a.id.localeCompare(b.id));

    // Group by depth; kept nodes sit on the primary spine (slot 0), branches stack below
    const byDepth = new Map();
    for (const n of laneNodes) {
      const d = depthMemo.get(n.id) || 0;
      const bucket = byDepth.get(d) || [];
      bucket.push(n);
      byDepth.set(d, bucket);
    }

    let maxSlot = 0;
    for (const [depth, bucket] of byDepth.entries()) {
      bucket.sort((a, b) => {
        const ak = KEPT_DECISIONS.has(a.decision) ? 0 : 1;
        const bk = KEPT_DECISIONS.has(b.decision) ? 0 : 1;
        if (ak !== bk) return ak - bk;
        return a.created_at.localeCompare(b.created_at);
      });
      bucket.forEach((n, slot) => {
        const x = leftPad + depth * colStep;
        const y = cursorY + 24 + slot * rowStep;
        positions.set(n.id, { x, y, depth, slot });
        if (slot > maxSlot) maxSlot = slot;
        if (x + 80 > maxX) maxX = x + 80;
      });
    }

    const bandHeight = Math.max(58, (maxSlot + 1) * rowStep + 28);
    laneBands.push({ lane, y: cursorY, height: bandHeight });
    cursorY += bandHeight + laneGap;
  }

  return {
    positions,
    laneBands,
    width: Math.max(820, maxX),
    height: Math.max(220, cursorY),
    edges: edges.filter((e) => positions.has(e.source) && positions.has(e.target)),
  };
}

function renderDag() {
  const svg = document.getElementById("dag-svg");
  svg.innerHTML = "";
  const { nodes, edges } = state.dag;
  if (!nodes.length) {
    svg.setAttribute("viewBox", "0 0 600 140");
    svg.setAttribute("height", "140");
    svg.appendChild(
      svgEl("text", { x: 300, y: 74, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 14 }, "No trial nodes yet.")
    );
    return;
  }

  const layout = computeDagLayout(nodes, edges);
  svg.setAttribute("width", String(layout.width));
  svg.setAttribute("height", String(layout.height));
  svg.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);

  // Swimlane backgrounds
  const bandGroup = svgEl("g");
  for (const band of layout.laneBands) {
    const color = laneColor(band.lane);
    bandGroup.appendChild(
      svgEl("rect", {
        x: 8,
        y: band.y,
        width: layout.width - 16,
        height: band.height,
        rx: 10,
        fill: "var(--bg-elevated)",
        stroke: "var(--border)",
        "stroke-width": 1,
      })
    );
    bandGroup.appendChild(
      svgEl("rect", {
        x: 8,
        y: band.y,
        width: 5,
        height: band.height,
        rx: 3,
        fill: color,
      })
    );
    bandGroup.appendChild(
      svgEl(
        "text",
        {
          x: 24,
          y: band.y + 28,
          fill: color,
          "font-size": 13,
          "font-weight": 700,
          "font-family": "var(--mono)",
        },
        band.lane
      )
    );
    const frontierNode = state.dag.frontier[band.lane];
    if (frontierNode) {
      bandGroup.appendChild(
        svgEl(
          "text",
          {
            x: 24,
            y: band.y + 44,
            fill: "var(--text-muted)",
            "font-size": 10,
            "font-family": "var(--mono)",
          },
          fmtEff(frontierNode.soft_min_eff)
        )
      );
    }
  }
  svg.appendChild(bandGroup);

  // Edges
  const edgeGroup = svgEl("g", { fill: "none" });
  for (const edge of layout.edges) {
    const s = layout.positions.get(edge.source);
    const t = layout.positions.get(edge.target);
    const dx = Math.max(24, (t.x - s.x) * 0.5);
    const d = `M ${s.x} ${s.y} C ${s.x + dx} ${s.y}, ${t.x - dx} ${t.y}, ${t.x} ${t.y}`;
    const isKept = KEPT_DECISIONS.has(edge.decision);
    edgeGroup.appendChild(
      svgEl("path", {
        d,
        stroke: isKept ? laneColor(edge.lane) : "var(--border)",
        "stroke-width": isKept ? 2 : 1.25,
        "stroke-opacity": isKept ? 0.85 : 0.7,
        "stroke-dasharray": isKept ? null : "3 3",
      })
    );
  }
  svg.appendChild(edgeGroup);

  // Nodes
  const nodeGroup = svgEl("g");
  for (const node of nodes) {
    const pos = layout.positions.get(node.id);
    if (!pos) continue;
    const color = laneColor(node.lane);
    const g = svgEl("g", {
      class: `dag-node${state.selectedId === node.id ? " selected" : ""}`,
      transform: `translate(${pos.x}, ${pos.y})`,
      "data-id": node.id,
    });

    g.appendChild(svgEl("circle", { r: 13, class: "node-select-ring" }));

    if (node.verified === 1) {
      g.appendChild(
        svgEl("circle", {
          r: 10.5,
          fill: "none",
          stroke: "var(--verified)",
          "stroke-width": 2,
        })
      );
    }

    if (node.decision === "keep_baseline") {
      g.appendChild(
        svgEl("rect", {
          x: -6.5,
          y: -6.5,
          width: 13,
          height: 13,
          rx: 2.5,
          fill: color,
          stroke: "var(--bg)",
          "stroke-width": 1.5,
          class: "node-glyph",
        })
      );
    } else if (node.decision === "keep_simpler") {
      g.appendChild(
        svgEl("polygon", {
          points: "0,-8 8,0 0,8 -8,0",
          fill: color,
          stroke: "var(--bg)",
          "stroke-width": 1.5,
          class: "node-glyph",
        })
      );
    } else if (node.decision === "keep") {
      g.appendChild(
        svgEl("circle", {
          r: 7,
          fill: color,
          stroke: "var(--bg)",
          "stroke-width": 1.5,
          class: "node-glyph",
        })
      );
    } else if (node.decision === "archive_only") {
      g.appendChild(
        svgEl("circle", {
          r: 5.8,
          fill: "var(--bg)",
          stroke: color,
          "stroke-width": 2,
          class: "node-glyph",
        })
      );
    } else {
      g.appendChild(
        svgEl("circle", {
          r: 4.3,
          fill: "var(--text-muted)",
          "fill-opacity": 0.45,
          stroke: "var(--bg)",
          "stroke-width": 1,
          class: "node-glyph",
        })
      );
    }

    g.addEventListener("mouseenter", (e) => {
      const stan =
        node.real_stan_total > 0
          ? `<br/>Real-Stan: <strong>${node.real_stan_passed}/${node.real_stan_total}</strong>`
          : "";
      showTooltip(
        e,
        `<strong>${escapeHtml(node.lane)} · ${escapeHtml(node.run_id)} · ${escapeHtml(node.trial_id || "")}</strong><br/>` +
          `Decision: <code>${escapeHtml(node.decision)}</code><br/>` +
          `soft_min_eff: <code>${fmtEff(node.soft_min_eff)}</code> (${fmtNum(node.ratio_vs_incumbent, 2, "×")})${stan}`
      );
    });
    g.addEventListener("mouseleave", hideTooltip);
    g.addEventListener("click", () => openDrawer(node.id, true));
    nodeGroup.appendChild(g);
  }
  svg.appendChild(nodeGroup);
}

function renderFrontierChart() {
  const svg = document.getElementById("frontier-svg");
  svg.innerHTML = "";
  const nodes = state.dag.nodes;
  const W = 620;
  const H = 290;
  const pad = { top: 20, right: 20, bottom: 36, left: 66 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  const lanes = [...new Set(nodes.map((n) => n.lane))].sort();
  const series = [];
  let maxTrial = 1;
  const allVals = [];

  for (const lane of lanes) {
    const laneNodes = nodes
      .filter((n) => n.lane === lane)
      .sort((a, b) => a.created_at.localeCompare(b.created_at) || a.id.localeCompare(b.id));
    let best = null;
    const pts = [];
    laneNodes.forEach((n, idx) => {
      if (KEPT_DECISIONS.has(n.decision) && n.soft_min_eff !== null && n.soft_min_eff > 0) {
        best = best === null ? n.soft_min_eff : Math.max(best, n.soft_min_eff);
      }
      if (best !== null) {
        pts.push({ x: idx + 1, y: best, node: n });
        allVals.push(best);
      }
    });
    if (pts.length) {
      maxTrial = Math.max(maxTrial, pts[pts.length - 1].x);
      series.push({ lane, pts });
    }
  }

  if (!allVals.length) {
    svg.appendChild(
      svgEl("text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 13 }, "No kept frontier points yet.")
    );
    return;
  }

  const useLog = state.logScale;
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const yMin = useLog ? Math.log10(minVal * 0.75) : 0;
  const yMax = useLog ? Math.log10(maxVal * 1.25) : maxVal * 1.12;
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const mapX = (x) => pad.left + ((x - 1) / Math.max(1, maxTrial - 1)) * plotW;
  const mapY = (v) => {
    const tv = useLog ? Math.log10(Math.max(v, 1e-12)) : v;
    const frac = (tv - yMin) / Math.max(1e-9, yMax - yMin);
    return pad.top + plotH - frac * plotH;
  };

  // Grid & axes
  const ticks = 4;
  for (let i = 0; i <= ticks; i += 1) {
    const frac = i / ticks;
    const tv = yMin + frac * (yMax - yMin);
    const val = useLog ? Math.pow(10, tv) : tv;
    const y = pad.top + plotH - frac * plotH;
    svg.appendChild(svgEl("line", { x1: pad.left, y1: y, x2: W - pad.right, y2: y, stroke: "var(--border)", "stroke-dasharray": "3 3" }));
    svg.appendChild(
      svgEl("text", { x: pad.left - 8, y: y + 4, "text-anchor": "end", fill: "var(--text-muted)", "font-size": 10, "font-family": "var(--mono)" }, fmtEff(val))
    );
  }
  svg.appendChild(svgEl("line", { x1: pad.left, y1: pad.top + plotH, x2: W - pad.right, y2: pad.top + plotH, stroke: "var(--border)" }));
  svg.appendChild(
    svgEl("text", { x: pad.left + plotW / 2, y: H - 8, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 11 }, "Trial sequence index within lane")
  );

  for (const s of series) {
    const color = laneColor(s.lane);
    const d = s.pts
      .map((p, idx) => `${idx === 0 ? "M" : "L"} ${mapX(p.x).toFixed(1)} ${mapY(p.y).toFixed(1)}`)
      .join(" ");
    svg.appendChild(svgEl("path", { d, fill: "none", stroke: color, "stroke-width": 2.4, "stroke-linejoin": "round" }));

    const last = s.pts[s.pts.length - 1];
    const dot = svgEl("circle", {
      cx: mapX(last.x),
      cy: mapY(last.y),
      r: 4.5,
      fill: color,
      stroke: "var(--bg-elevated)",
      "stroke-width": 1.5,
      style: "cursor:pointer",
    });
    dot.addEventListener("mouseenter", (e) => {
      showTooltip(e, `<strong>${escapeHtml(s.lane)} frontier</strong><br/>Best soft_min_eff: <code>${fmtEff(last.y)}</code>`);
    });
    dot.addEventListener("mouseleave", hideTooltip);
    dot.addEventListener("click", () => openDrawer(last.node.id, true));
    svg.appendChild(dot);

    svg.appendChild(
      svgEl(
        "text",
        {
          x: Math.min(W - pad.right - 22, mapX(last.x) + 7),
          y: mapY(last.y) + 4,
          fill: color,
          "font-size": 11,
          "font-weight": 700,
          "font-family": "var(--mono)",
        },
        s.lane
      )
    );
  }
}

function renderAlignmentChart() {
  const svg = document.getElementById("alignment-svg");
  svg.innerHTML = "";
  const W = 620;
  const H = 290;
  const pad = { top: 20, right: 24, bottom: 38, left: 58 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  const points = state.dag.alignment.map((a) => ({
    ...a,
    ratio: a.ratio_vs_incumbent !== null && a.ratio_vs_incumbent > 0 ? a.ratio_vs_incumbent : 1.0,
  }));

  if (!points.length) {
    svg.appendChild(
      svgEl("text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 13 }, "No nodes with both Tier-1 and Real-Stan scores yet.")
    );
    return;
  }

  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const minR = Math.min(0.8, ...points.map((p) => p.ratio));
  const maxR = Math.max(2.0, ...points.map((p) => p.ratio));
  const xMin = Math.log10(Math.max(0.25, minR * 0.8));
  const xMax = Math.log10(maxR * 1.25);

  const mapX = (r) => pad.left + ((Math.log10(Math.max(r, 0.1)) - xMin) / Math.max(1e-6, xMax - xMin)) * plotW;
  const mapY = (rate) => pad.top + plotH - rate * plotH;

  // Y grid (0%, 20%, 40%, 60%, 80%, 100%)
  for (let i = 0; i <= 5; i += 1) {
    const rate = i / 5;
    const y = mapY(rate);
    svg.appendChild(svgEl("line", { x1: pad.left, y1: y, x2: W - pad.right, y2: y, stroke: "var(--border)", "stroke-dasharray": "3 3" }));
    svg.appendChild(
      svgEl("text", { x: pad.left - 8, y: y + 4, "text-anchor": "end", fill: "var(--text-muted)", "font-size": 10, "font-family": "var(--mono)" }, `${Math.round(rate * 100)}%`)
    );
  }

  // Reference x=1.0x line
  const xOne = mapX(1.0);
  svg.appendChild(svgEl("line", { x1: xOne, y1: pad.top, x2: xOne, y2: pad.top + plotH, stroke: "var(--text-muted)", "stroke-dasharray": "4 4", "stroke-opacity": 0.55 }));

  const xTicks = [0.5, 1, 2, 5, 10, 15].filter((v) => Math.log10(v) >= xMin && Math.log10(v) <= xMax);
  for (const v of xTicks) {
    const x = mapX(v);
    svg.appendChild(svgEl("text", { x, y: pad.top + plotH + 15, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 10, "font-family": "var(--mono)" }, `${v}×`));
  }

  svg.appendChild(
    svgEl("text", { x: pad.left + plotW / 2, y: H - 6, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 11 }, "Tier-1 ratio_vs_incumbent (log scale)")
  );

  // Jitter overlapping points slightly for readability
  const seen = new Map();
  for (const p of points) {
    const key = `${p.ratio.toFixed(2)}:${p.real_stan_pass_rate.toFixed(2)}`;
    const dup = seen.get(key) || 0;
    seen.set(key, dup + 1);
    const cx = mapX(p.ratio) + (dup % 2 === 0 ? 1 : -1) * dup * 5;
    const cy = mapY(p.real_stan_pass_rate) + (dup > 0 ? (dup % 2 === 0 ? 7 : -7) : 0);
    const color = laneColor(p.lane);

    const g = svgEl("g", { style: "cursor:pointer" });
    g.appendChild(
      svgEl("circle", {
        cx,
        cy,
        r: 7.5,
        fill: color,
        "fill-opacity": 0.85,
        stroke: "var(--verified)",
        "stroke-width": 2,
      })
    );
    const anchor = cx > W - 150 ? "end" : "start";
    const dx = anchor === "end" ? -11 : 11;
    g.appendChild(
      svgEl(
        "text",
        {
          x: cx + dx,
          y: cy + 4,
          "text-anchor": anchor,
          fill: "var(--text)",
          "font-size": 10,
          "font-family": "var(--mono)",
        },
        `${p.lane}/${p.run_id}`
      )
    );
    g.addEventListener("mouseenter", (e) => {
      showTooltip(
        e,
        `<strong>${escapeHtml(p.lane)} · ${escapeHtml(p.run_id)} (${escapeHtml(p.trial_id || "")})</strong><br/>` +
          `Tier-1 ratio: <code>${fmtNum(p.ratio_vs_incumbent, 2, "×")}</code> · eff <code>${fmtEff(p.soft_min_eff)}</code><br/>` +
          `Real-Stan: <strong>${p.real_stan_passed}/${p.real_stan_total}</strong> (${Math.round(p.real_stan_pass_rate * 100)}%)<br/>` +
          `Real ESS/cost: <code>${fmtEff(p.real_ess_per_cost)}</code>`
      );
    });
    g.addEventListener("mouseleave", hideTooltip);
    g.addEventListener("click", () => openDrawer(p.id, true));
    svg.appendChild(g);
  }
}

function renderClaimsTable() {
  const tbody = document.getElementById("claims-tbody");
  const badge = document.getElementById("claims-count-badge");
  const claims = state.dag.claims;
  badge.textContent = `${claims.length} active`;

  if (!claims.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-cell">No active claims.</td></tr>';
    return;
  }

  tbody.innerHTML = claims
    .map((c) => {
      const expires = c.expires_at ? new Date(c.expires_at).toLocaleString() : "—";
      return `<tr>
        <td><code>${escapeHtml(c.task_id)}</code></td>
        <td><span class="lane-pill"><i class="dot lane-${escapeHtml(c.lane)}"></i>${escapeHtml(c.lane)}</span></td>
        <td>${escapeHtml(c.agent)}</td>
        <td>${escapeHtml(c.summary || "—")}</td>
        <td><code>${escapeHtml(expires)}</code></td>
      </tr>`;
    })
    .join("");
}

function populateFilters() {
  const nodes = state.dag.nodes;
  const setOptions = (id, values, label) => {
    const sel = document.getElementById(id);
    const prev = sel.value;
    sel.innerHTML = `<option value="">${label}</option>` +
      values.map((v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join("");
    if (values.includes(prev)) sel.value = prev;
  };

  setOptions("filter-lane", [...new Set(nodes.map((n) => n.lane))].sort(), "All lanes");
  setOptions("filter-model", [...new Set(nodes.map((n) => n.agent_model).filter(Boolean))].sort(), "All models");
  setOptions("filter-decision", [...new Set(nodes.map((n) => n.decision))].sort(), "All decisions");
}

function renderLedgerTable() {
  const tbody = document.getElementById("ledger-tbody");
  const lane = document.getElementById("filter-lane").value;
  const model = document.getElementById("filter-model").value;
  const decision = document.getElementById("filter-decision").value;
  const query = document.getElementById("filter-search").value.trim().toLowerCase();

  const filtered = [...state.dag.nodes]
    .reverse()
    .filter((n) => {
      if (lane && n.lane !== lane) return false;
      if (model && n.agent_model !== model) return false;
      if (decision && n.decision !== decision) return false;
      if (query) {
        const hay = `${n.id} ${n.run_id} ${n.trial_id || ""} ${n.hypothesis || ""} ${n.code_sha || ""} ${(n.codex_refs || []).join(" ")}`.toLowerCase();
        if (!hay.includes(query)) return false;
      }
      return true;
    });

  if (!filtered.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-cell">No trials match the current filters.</td></tr>';
    return;
  }

  tbody.innerHTML = filtered
    .map((n) => {
      const stan =
        n.real_stan_total > 0
          ? `<span class="badge badge-verified">${n.real_stan_passed}/${n.real_stan_total}</span>`
          : "—";
      const selected = n.id === state.selectedId ? ' class="selected-row"' : "";
      return `<tr data-id="${escapeHtml(n.id)}"${selected}>
        <td><span class="lane-pill"><i class="dot lane-${escapeHtml(n.lane)}"></i>${escapeHtml(n.lane)}</span></td>
        <td><code>${escapeHtml(n.run_id)}</code> / <strong>${escapeHtml(n.trial_id || "")}</strong></td>
        <td>${escapeHtml(n.agent_model || "—")}</td>
        <td><span class="badge ${decisionClass(n.decision)}">${escapeHtml(n.decision)}</span></td>
        <td class="num"><code>${fmtEff(n.soft_min_eff)}</code></td>
        <td class="num"><code>${fmtNum(n.ratio_vs_incumbent, 2, "×")}</code></td>
        <td class="num">${stan}</td>
        <td class="num"><code>${fmtNum(n.overhead_ratio, 2, "×")}</code></td>
        <td class="hyp-cell">${escapeHtml((n.hypothesis || "").slice(0, 140))}${(n.hypothesis || "").length > 140 ? "…" : ""}</td>
      </tr>`;
    })
    .join("");

  tbody.querySelectorAll("tr[data-id]").forEach((tr) => {
    tr.addEventListener("click", () => openDrawer(tr.getAttribute("data-id"), true));
  });
}

function openDrawer(nodeId, updateSelection) {
  const node = state.nodeById.get(nodeId);
  if (!node) return;
  state.selectedId = nodeId;

  if (updateSelection) {
    document.querySelectorAll(".dag-node.selected").forEach((el) => el.classList.remove("selected"));
    const svgNode = document.querySelector(`.dag-node[data-id="${CSS.escape(nodeId)}"]`);
    if (svgNode) svgNode.classList.add("selected");
    renderLedgerTable();
  }

  const badges = [
    `<span class="badge badge-muted">${escapeHtml(node.lane)}</span>`,
    `<span class="badge ${decisionClass(node.decision)}">${escapeHtml(node.decision)}</span>`,
  ];
  if (node.agent_model) badges.push(`<span class="badge badge-muted">${escapeHtml(node.agent_model)}</span>`);
  if (node.verified === 1) {
    const label = node.real_stan_total > 0 ? `Real-Stan ${node.real_stan_passed}/${node.real_stan_total}` : "Verified";
    badges.push(`<span class="badge badge-verified">★ ${escapeHtml(label)}</span>`);
  }

  document.getElementById("drawer-badges").innerHTML = badges.join("");
  document.getElementById("drawer-title").textContent = node.title || node.id;
  document.getElementById("drawer-id").textContent = node.id;

  const refs = (node.codex_refs || []).length
    ? node.codex_refs.map((r) => `<span class="badge badge-baseline">${escapeHtml(r)}</span>`).join("")
    : '<span class="badge badge-muted">none</span>';

  const parentHtml = node.parent_id && state.nodeById.has(node.parent_id)
    ? `<button class="link-btn" data-jump="${escapeHtml(node.parent_id)}">↑ ${escapeHtml(node.parent_id)}</button>`
    : `<span class="badge badge-muted">${escapeHtml(node.parent_id || "root")}</span>`;

  const children = state.childrenById.get(node.id) || [];
  const childrenHtml = children.length
    ? children
        .map((c) => `<button class="link-btn" data-jump="${escapeHtml(c.id)}">↓ ${escapeHtml(c.trial_id || c.id)} (${escapeHtml(c.decision)})</button>`)
        .join("")
    : '<span class="badge badge-muted">leaf</span>';

  const dimsText = Array.isArray(node.dims) && node.dims.length ? node.dims.join(", ") : "—";
  const stanText = node.real_stan_total > 0 ? `${node.real_stan_passed} / ${node.real_stan_total}` : "—";

  document.getElementById("drawer-body").innerHTML = `
    <div class="drawer-section">
      <h4>Hypothesis</h4>
      <div class="hypothesis-box">${escapeHtml(node.hypothesis || "No hypothesis recorded.")}</div>
    </div>
    <div class="drawer-section">
      <h4>Metrics</h4>
      <div class="metrics-grid">
        <div class="metric-tile"><span>soft_min_eff</span><strong>${fmtEff(node.soft_min_eff)}</strong></div>
        <div class="metric-tile"><span>ratio_vs_incumbent</span><strong>${fmtNum(node.ratio_vs_incumbent, 3, "×")}</strong></div>
        <div class="metric-tile"><span>short_horizon_eff</span><strong>${fmtEff(node.short_horizon_eff)}</strong></div>
        <div class="metric-tile"><span>long_horizon_eff</span><strong>${fmtEff(node.long_horizon_eff)}</strong></div>
        <div class="metric-tile"><span>real_stan_passed</span><strong>${stanText}</strong></div>
        <div class="metric-tile"><span>real_ess_per_cost</span><strong>${fmtEff(node.real_ess_per_cost)}</strong></div>
        <div class="metric-tile"><span>overhead_ratio</span><strong>${fmtNum(node.overhead_ratio, 2, "×")}</strong></div>
        <div class="metric-tile"><span>b2_max</span><strong>${fmtNum(node.b2_max, 4)}</strong></div>
        <div class="metric-tile"><span>dims</span><strong>${escapeHtml(dimsText)}</strong></div>
        <div class="metric-tile"><span>code_sha</span><strong>${escapeHtml(node.code_sha || "—")}</strong></div>
      </div>
    </div>
    <div class="drawer-section">
      <h4>Codex References</h4>
      <div class="pill-list">${refs}</div>
    </div>
    <div class="drawer-section">
      <h4>Lineage</h4>
      <div class="pill-list" style="margin-bottom:8px">${parentHtml}</div>
      <div class="pill-list">${childrenHtml}</div>
    </div>
    <div class="drawer-section">
      <h4>Timestamp</h4>
      <code>${escapeHtml(node.created_at || "—")}</code>
    </div>
  `;

  document.querySelectorAll("#drawer-body [data-jump]").forEach((btn) => {
    btn.addEventListener("click", () => openDrawer(btn.getAttribute("data-jump"), true));
  });

  const drawer = document.getElementById("node-drawer");
  const backdrop = document.getElementById("drawer-backdrop");
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  backdrop.hidden = false;
}

function closeDrawer() {
  const drawer = document.getElementById("node-drawer");
  const backdrop = document.getElementById("drawer-backdrop");
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  backdrop.hidden = true;
}

function initTheme() {
  const saved = localStorage.getItem("apin-flywheel-theme");
  if (saved === "light" || saved === "dark") {
    document.documentElement.setAttribute("data-theme", saved);
  }
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const next = current ? (current === "dark" ? "light" : "dark") : prefersDark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("apin-flywheel-theme", next);
    renderDag();
    renderFrontierChart();
    renderAlignmentChart();
  });
}

function initEvents() {
  document.getElementById("refresh-btn").addEventListener("click", loadData);
  document.getElementById("drawer-close").addEventListener("click", closeDrawer);
  document.getElementById("drawer-backdrop").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDrawer();
  });

  document.getElementById("frontier-log-toggle").addEventListener("change", (e) => {
    state.logScale = e.target.checked;
    renderFrontierChart();
  });

  ["filter-lane", "filter-model", "filter-decision"].forEach((id) => {
    document.getElementById(id).addEventListener("change", renderLedgerTable);
  });
  document.getElementById("filter-search").addEventListener("input", renderLedgerTable);
}

initTheme();
initEvents();
loadData();
