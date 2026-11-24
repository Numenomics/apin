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
--   * lane_stats.steps_dirty: 1 when a change to a trial may have changed other flags of its
--     lane. Triggers on nodes set it in the writing transaction. Every Worker batch that writes
--     trials ends with the recompute statement of the BACKFILL block (SQL_SETTLE_STEPS in
--     src/index.ts), which recomputes the flags of every dirty lane and clears the mark. So a
--     write and the flags it changes commit together, and a lane left dirty by any other writer
--     is repaired by the next Worker write.
--   * Indexes for the lane chart (standard trials of a lane, oldest first), the latest step of
--     a lane, the step=1 filter of /api/nodes, the verification bar among standard trials and
--     the dirty lanes.
--
-- Additive. Rollout: apply this migration, deploy the Worker, then run the BACKFILL block once
-- (README.md, "Deploy"). The Worker that runs before the deploy sets neither flag on the trials
-- it writes, and the BACKFILL block repairs them. Trials loaded by other means (seed.sql, a
-- restore) need the same block. seed_from_apin.py copies it to the end of seed.sql and prints it
-- as a file with --backfill-sql, so it is kept only here.
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
ALTER TABLE lane_stats ADD COLUMN steps_dirty INTEGER NOT NULL DEFAULT 0;

-- The lane chart, the step walk and the "anything after this trial" checks (named in INDEXED BY).
CREATE INDEX IF NOT EXISTS idx_nodes_lane_standard ON nodes (lane, standard, created_at, id);
-- The latest step of a lane (the frontier) and /api/nodes?lane=…&step=1.
CREATE INDEX IF NOT EXISTS idx_nodes_steps ON nodes (lane, created_at, id) WHERE frontier_step = 1;
-- /api/nodes?step=1 across lanes.
CREATE INDEX IF NOT EXISTS idx_nodes_steps_created ON nodes (created_at, id) WHERE frontier_step = 1;
-- The verification bar: the best replicated or verified standard trial with source per lane.
CREATE INDEX IF NOT EXISTS idx_nodes_reviewed_std ON nodes (lane, soft_min_eff)
  WHERE state IN ('replicated', 'verified') AND source_sha IS NOT NULL AND standard = 1;
-- The dirty lanes: empty except inside a writing transaction, so the usual write reads no row.
CREATE INDEX IF NOT EXISTS idx_lane_stats_dirty ON lane_stats (lane) WHERE steps_dirty = 1;

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

-- lane_stats.steps_dirty ------------------------------------------------------
-- A trial that is not a step never changes the best score before a later trial, so a change
-- can move other flags only when the trial was a step or is eligible now. It cannot when the
-- trial stays eligible at the same position with the same score. An insert needs no mark: the
-- Worker sets the new trial's flag and clears the later steps it beats in its own batch.
-- The eligibility test must match sqlStepEligible() in src/index.ts.

CREATE TRIGGER IF NOT EXISTS nodes_steps_dirty_au
AFTER UPDATE OF lane, standard, soft_min_eff, decision, verified, state, created_at ON nodes
WHEN (old.frontier_step = 1
    OR (new.standard = 1 AND new.soft_min_eff > 0 AND new.state NOT IN ('retracted', 'disputed')
      AND (new.verified = 1 OR new.decision IN ('keep', 'keep_baseline', 'keep_simpler'))))
  AND NOT coalesce(
    old.standard = 1 AND old.soft_min_eff > 0 AND old.state NOT IN ('retracted', 'disputed')
      AND (old.verified = 1 OR old.decision IN ('keep', 'keep_baseline', 'keep_simpler'))
    AND new.standard = 1 AND new.soft_min_eff > 0 AND new.state NOT IN ('retracted', 'disputed')
      AND (new.verified = 1 OR new.decision IN ('keep', 'keep_baseline', 'keep_simpler'))
    AND old.lane IS new.lane AND old.created_at IS new.created_at AND old.soft_min_eff IS new.soft_min_eff, 0)
BEGIN
  UPDATE lane_stats SET steps_dirty = 1 WHERE lane IN (old.lane, new.lane) AND steps_dirty = 0;
END;

CREATE TRIGGER IF NOT EXISTS nodes_steps_dirty_ad AFTER DELETE ON nodes
WHEN old.frontier_step = 1
BEGIN
  UPDATE lane_stats SET steps_dirty = 1 WHERE lane = old.lane AND steps_dirty = 0;
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
-- Every lane is dirty, then the Worker's recompute runs (SQL_SETTLE_STEPS in src/index.ts, kept
-- equal to the statement below). Per dirty lane it reads the standard trials once through
-- idx_nodes_lane_standard and the steps through idx_nodes_steps, and toggles each flag that
-- differs from the walk: an eligible trial is a step when its score is above the best eligible
-- score before it (the first one has none), and an ineligible trial is never a step.
INSERT INTO lane_stats (lane) SELECT DISTINCT lane FROM nodes WHERE lane NOT IN (SELECT lane FROM lane_stats);
UPDATE lane_stats SET steps_dirty = 1;
UPDATE nodes SET frontier_step = 1 - frontier_step WHERE id IN (
  SELECT id FROM (
    SELECT n.id, n.frontier_step AS was, n.soft_min_eff > coalesce(max(n.soft_min_eff) OVER (
        PARTITION BY n.lane ORDER BY n.created_at, n.id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS step
    FROM lane_stats AS d INDEXED BY idx_lane_stats_dirty CROSS JOIN nodes AS n INDEXED BY idx_nodes_lane_standard
    WHERE d.steps_dirty = 1 AND n.lane = d.lane AND n.standard = 1 AND n.soft_min_eff > 0
      AND n.state NOT IN ('retracted', 'disputed') AND (n.verified = 1 OR n.decision IN ('keep', 'keep_baseline', 'keep_simpler'))
  ) WHERE step IS NOT was
  UNION ALL
  SELECT n.id FROM lane_stats AS d INDEXED BY idx_lane_stats_dirty CROSS JOIN nodes AS n INDEXED BY idx_nodes_steps
  WHERE d.steps_dirty = 1 AND n.lane = d.lane AND n.frontier_step = 1
    AND NOT coalesce(n.standard = 1 AND n.soft_min_eff > 0
      AND n.state NOT IN ('retracted', 'disputed') AND (n.verified = 1 OR n.decision IN ('keep', 'keep_baseline', 'keep_simpler')), 0));
UPDATE lane_stats SET steps_dirty = 0 WHERE steps_dirty = 1;
UPDATE lane_stats SET
  standard = (SELECT COUNT(*) FROM nodes n WHERE n.lane = lane_stats.lane AND n.standard = 1),
  steps = (SELECT COUNT(*) FROM nodes n INDEXED BY idx_nodes_steps WHERE n.lane = lane_stats.lane AND n.frontier_step = 1);
-- BACKFILL END
