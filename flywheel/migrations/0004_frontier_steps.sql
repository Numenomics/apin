-- 0004: frontier steps. A lane's frontier is defined inside one standard setup, and the
-- trials that moved it are marked, so the site can draw one "best so far" line per lane.
--
--   * nodes.standard: 1 when the trial's dims are its lane's standard dims (STANDARD_DIMS in
--     src/index.ts: A0 to A4 use [6, 20]; a lane without an entry counts every trial). Scores
--     are only comparable on the same setup, so only standard trials can move a frontier.
--   * nodes.frontier_step: 1 when the trial moved its lane's frontier. Per lane, walk the
--     eligible trials in (created_at, id) order. Eligible: standard, soft_min_eff > 0, not
--     retracted or disputed, and verified or kept (keep, keep_baseline, keep_simpler). The
--     first one is a step. After that a trial is a step when its soft_min_eff is above the
--     latest step's (a tie is not a step). The lane frontier is the latest step.
--   * lane_stats.standard and lane_stats.steps: per-lane counts of both flags, kept current by
--     triggers on nodes, so /api/overview and the lane chart read one row per lane.
--   * Indexes for the lane chart (standard trials of a lane, oldest first), the latest step of
--     a lane, the step=1 filter of /api/nodes and the verification bar among standard trials.
--
-- Additive. Apply with: npx wrangler d1 migrations apply apin-flywheel --remote
-- The Worker sets both flags on every write. Trials loaded by other means (seed.sql, a
-- restore) need the statements between BACKFILL BEGIN and BACKFILL END below.
-- seed_from_apin.py copies that block to the end of seed.sql, so it is kept only here.
-- idx_nodes_frontier_pick, idx_nodes_frontier_any and idx_nodes_reviewed_eff (0003) are no
-- longer read by the Worker. They stay until a later migration, because the Worker that runs
-- between `migrations apply` and `deploy` still names them in INDEXED BY.
-- Like 0002 and 0003, the triggers avoid conflict clauses, and nodes must never be written
-- with INSERT OR REPLACE.
--
-- REBUILD (safe to re-run): run the BACKFILL block. It recomputes both flags from scratch and
-- then sets the lane_stats counters from them. After the REBUILD of 0003 (which recreates
-- lane_stats rows with zero counts), run this block again.

ALTER TABLE nodes ADD COLUMN standard INTEGER NOT NULL DEFAULT 0;
ALTER TABLE nodes ADD COLUMN frontier_step INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lane_stats ADD COLUMN standard INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lane_stats ADD COLUMN steps INTEGER NOT NULL DEFAULT 0;

-- The lane chart, the step walk and the "anything after this trial" checks (named in INDEXED BY).
CREATE INDEX IF NOT EXISTS idx_nodes_lane_standard ON nodes (lane, standard, created_at, id);
-- The latest step of a lane (the frontier) and /api/nodes?lane=…&step=1.
CREATE INDEX IF NOT EXISTS idx_nodes_steps ON nodes (lane, created_at, id) WHERE frontier_step = 1;
-- /api/nodes?step=1 across lanes.
CREATE INDEX IF NOT EXISTS idx_nodes_steps_created ON nodes (created_at, id) WHERE frontier_step = 1;
-- The verification bar: the best replicated or verified standard trial with source per lane.
CREATE INDEX IF NOT EXISTS idx_nodes_reviewed_std ON nodes (lane, soft_min_eff)
  WHERE state IN ('replicated', 'verified') AND source_sha IS NOT NULL AND standard = 1;

-- lane_stats.standard / steps ----------------------------------------------------
-- These may fire before or after the run_stats triggers of 0003, so each creates the
-- lane_stats row when it is missing. The run_stats trigger then finds it and adds its counts.

CREATE TRIGGER IF NOT EXISTS nodes_lane_flags_ai AFTER INSERT ON nodes
WHEN new.standard = 1 OR new.frontier_step = 1
BEGIN
  INSERT INTO lane_stats (lane) SELECT new.lane
    WHERE NOT EXISTS (SELECT 1 FROM lane_stats WHERE lane = new.lane);
  UPDATE lane_stats SET standard = standard + new.standard, steps = steps + new.frontier_step
  WHERE lane = new.lane;
END;

CREATE TRIGGER IF NOT EXISTS nodes_lane_flags_ad AFTER DELETE ON nodes
WHEN old.standard = 1 OR old.frontier_step = 1
BEGIN
  UPDATE lane_stats SET standard = standard - old.standard, steps = steps - old.frontier_step
  WHERE lane = old.lane;
END;

CREATE TRIGGER IF NOT EXISTS nodes_lane_flags_au AFTER UPDATE OF lane, standard, frontier_step ON nodes
WHEN old.lane IS NOT new.lane OR old.standard IS NOT new.standard OR old.frontier_step IS NOT new.frontier_step
BEGIN
  UPDATE lane_stats SET standard = standard - old.standard, steps = steps - old.frontier_step
  WHERE lane = old.lane;
  INSERT INTO lane_stats (lane) SELECT new.lane
    WHERE NOT EXISTS (SELECT 1 FROM lane_stats WHERE lane = new.lane);
  UPDATE lane_stats SET standard = standard + new.standard, steps = steps + new.frontier_step
  WHERE lane = new.lane;
END;

-- BACKFILL BEGIN
-- Standard setup per lane. Must match STANDARD_DIMS and isStandard() in src/index.ts: the
-- dims must be a JSON array of numbers equal to the lane's entry, element by element.
UPDATE nodes SET standard = CASE
  WHEN lane IN ('A0', 'A1', 'A2', 'A3', 'A4') THEN CASE WHEN json_valid(dims) THEN coalesce(
      json_type(dims) = 'array' AND json_array_length(dims) = 2
      AND json_type(dims, '$[0]') IN ('integer', 'real') AND json_extract(dims, '$[0]') = 6
      AND json_type(dims, '$[1]') IN ('integer', 'real') AND json_extract(dims, '$[1]') = 20, 0)
    ELSE 0 END
  ELSE 1 END;
-- Frontier steps. The walk takes each lane's next eligible trial with one index seek per step
-- (idx_nodes_lane_standard), so it is linear in the number of standard trials. The
-- eligibility test must match SQL_STEP_ELIGIBLE and the comparison must match the Worker's walk.
WITH RECURSIVE walk(lane, created_at, id, best, step) AS (
  SELECT n.lane, n.created_at, n.id, n.soft_min_eff, 1
  FROM (SELECT DISTINCT lane FROM nodes) AS l
  JOIN nodes AS n ON n.id = (
    SELECT id FROM nodes INDEXED BY idx_nodes_lane_standard
    WHERE lane = l.lane AND standard = 1 AND soft_min_eff > 0 AND state NOT IN ('retracted', 'disputed')
      AND (verified = 1 OR decision IN ('keep', 'keep_baseline', 'keep_simpler'))
    ORDER BY created_at, id LIMIT 1)
  UNION ALL
  SELECT n.lane, n.created_at, n.id,
    CASE WHEN n.soft_min_eff > w.best THEN n.soft_min_eff ELSE w.best END,
    n.soft_min_eff > w.best
  FROM walk AS w
  JOIN nodes AS n ON n.id = (
    SELECT id FROM nodes INDEXED BY idx_nodes_lane_standard
    WHERE lane = w.lane AND standard = 1 AND soft_min_eff > 0 AND state NOT IN ('retracted', 'disputed')
      AND (verified = 1 OR decision IN ('keep', 'keep_baseline', 'keep_simpler'))
      AND (created_at, id) > (w.created_at, w.id)
    ORDER BY created_at, id LIMIT 1)
)
UPDATE nodes SET frontier_step = (id IN (SELECT id FROM walk WHERE step = 1));
UPDATE lane_stats SET
  standard = (SELECT COUNT(*) FROM nodes n WHERE n.lane = lane_stats.lane AND n.standard = 1),
  steps = (SELECT COUNT(*) FROM nodes n INDEXED BY idx_nodes_steps WHERE n.lane = lane_stats.lane AND n.frontier_step = 1);
-- BACKFILL END
