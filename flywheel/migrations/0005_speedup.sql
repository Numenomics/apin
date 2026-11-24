-- 0005: speed-up. Trials are compared by their speed-up over the lane's starting code, only
-- within one yardstick, and a step must beat the latest step by a margin.
--
--   * nodes.speedup: the trial's efficiency divided by the frozen score of the lane's starting
--     code, from one grader at 8 repeats on fixed seeds. Null when the trial failed a check or
--     was exploratory.
--   * nodes.yardstick: the id of the grading setup, null for an exploratory trial. A trial is
--     standard (graded) when it has one. nodes.standard (0004) is dropped. Speed-ups on two
--     yardsticks are not comparable.
--   * Eligible: graded, speedup > 0, not retracted or disputed, and verified or kept (keep,
--     keep_baseline, keep_simpler).
--   * lane_stats.yardstick: the lane's current yardstick, the one whose first eligible trial
--     (in created_at, id order) is the newest. Null while the lane has no eligible trial. A
--     later trial on an older yardstick does not switch the lane back.
--   * nodes.frontier_step: walk the lane's eligible trials on its current yardstick in
--     (created_at, id) order. The first one is a step. A later one is a step when its speedup
--     is at least STEP_MARGIN (1.05) times the latest step's. A trial below that is not a step
--     and does not raise the bar. A trial on another yardstick is never a step. The lane
--     frontier is the latest step.
--   * lane_stats.standard and lane_stats.steps count the graded trials on the current yardstick
--     and the steps per lane, kept current by triggers and by the recompute below.
--   * lane_stats.steps_dirty: 1 when a change to a trial may have changed other flags of its
--     lane or its current yardstick. Triggers set it in the writing transaction. Every Worker
--     batch that writes trials ends with the recompute statements of the BACKFILL block
--     (SQL_SETTLE_YARDSTICK and SQL_SETTLE_STEPS in src/index.ts), which set each dirty lane's
--     yardstick, walk the lane again and clear the mark. So a write and the flags it changes
--     commit together.
--
-- On a database with trials from before this file, every trial is ungraded afterwards, so the
-- lanes have no steps until graded trials arrive. Like 0002 and 0003, the triggers avoid
-- conflict clauses, and nodes must never be written with INSERT OR REPLACE.
--
-- REBUILD (safe to re-run): run the BACKFILL block. It recomputes the yardsticks and the step
-- flags from scratch and then sets the lane_stats counters from them. seed_from_apin.py copies
-- it to the end of seed.sql and prints it as a file with --backfill-sql, so it is kept only here.

ALTER TABLE nodes ADD COLUMN speedup REAL;
ALTER TABLE nodes ADD COLUMN yardstick TEXT;
ALTER TABLE lane_stats ADD COLUMN yardstick TEXT;

-- What 0004 and 0003 built on the old rules.
DROP TRIGGER IF EXISTS nodes_lane_flags_ai;
DROP TRIGGER IF EXISTS nodes_lane_flags_ad;
DROP TRIGGER IF EXISTS nodes_lane_flags_au;
DROP TRIGGER IF EXISTS nodes_steps_dirty_au;
DROP TRIGGER IF EXISTS nodes_steps_dirty_ad;
DROP INDEX IF EXISTS idx_nodes_lane_standard;
DROP INDEX IF EXISTS idx_nodes_reviewed_std;
DROP INDEX IF EXISTS idx_nodes_reviewed_eff;
DROP INDEX IF EXISTS idx_nodes_frontier_pick;
DROP INDEX IF EXISTS idx_nodes_frontier_any;
ALTER TABLE nodes DROP COLUMN standard;

-- The lane chart and the graded count: a lane's graded trials per yardstick, oldest first.
CREATE INDEX IF NOT EXISTS idx_nodes_lane_graded ON nodes (lane, yardstick, created_at, id) WHERE yardstick IS NOT NULL;
-- The step walk, the yardsticks of a lane and the first eligible trial of each. The condition
-- must match sqlStepEligible() in src/index.ts, or the INDEXED BY clauses there fail.
CREATE INDEX IF NOT EXISTS idx_nodes_eligible ON nodes (lane, yardstick, created_at, id)
  WHERE yardstick IS NOT NULL AND speedup > 0 AND state NOT IN ('retracted', 'disputed')
      AND (verified = 1 OR decision IN ('keep', 'keep_baseline', 'keep_simpler'));
-- The verification bar: the best replicated or verified graded trial with source per lane and yardstick.
CREATE INDEX IF NOT EXISTS idx_nodes_reviewed_best ON nodes (lane, yardstick, speedup)
  WHERE state IN ('replicated', 'verified') AND source_sha IS NOT NULL AND yardstick IS NOT NULL;

-- lane_stats.standard / steps ----------------------------------------------------
-- These may fire before or after the run_stats triggers of 0003, so each creates the
-- lane_stats row when it is missing. The run_stats trigger then finds it and adds its counts.
-- standard counts only trials on the lane's current yardstick. When the recompute changes the
-- yardstick it counts them again.

CREATE TRIGGER IF NOT EXISTS nodes_lane_flags_ai AFTER INSERT ON nodes
WHEN new.yardstick IS NOT NULL OR new.frontier_step = 1
BEGIN
  INSERT INTO lane_stats (lane) SELECT new.lane
    WHERE NOT EXISTS (SELECT 1 FROM lane_stats WHERE lane = new.lane);
  UPDATE lane_stats SET standard = standard + coalesce(new.yardstick = lane_stats.yardstick, 0), steps = steps + new.frontier_step
  WHERE lane = new.lane;
END;

CREATE TRIGGER IF NOT EXISTS nodes_lane_flags_ad AFTER DELETE ON nodes
WHEN old.yardstick IS NOT NULL OR old.frontier_step = 1
BEGIN
  UPDATE lane_stats SET standard = standard - coalesce(old.yardstick = lane_stats.yardstick, 0), steps = steps - old.frontier_step
  WHERE lane = old.lane;
END;

CREATE TRIGGER IF NOT EXISTS nodes_lane_flags_au AFTER UPDATE OF lane, yardstick, frontier_step ON nodes
WHEN old.lane IS NOT new.lane OR old.yardstick IS NOT new.yardstick OR old.frontier_step IS NOT new.frontier_step
BEGIN
  UPDATE lane_stats SET standard = standard - coalesce(old.yardstick = lane_stats.yardstick, 0), steps = steps - old.frontier_step
  WHERE lane = old.lane;
  INSERT INTO lane_stats (lane) SELECT new.lane
    WHERE NOT EXISTS (SELECT 1 FROM lane_stats WHERE lane = new.lane);
  UPDATE lane_stats SET standard = standard + coalesce(new.yardstick = lane_stats.yardstick, 0), steps = steps + new.frontier_step
  WHERE lane = new.lane;
END;

-- lane_stats.steps_dirty ------------------------------------------------------
-- A trial that is not a step never changes the bar for a later trial, so a change can move
-- other flags only when the trial was a step or is eligible now. It cannot when the trial stays
-- eligible at the same position with the same speedup and yardstick. The current yardstick can
-- change only when the first eligible trial of some yardstick changes: a trial that is eligible
-- now, or one that was the first eligible trial of its yardstick. A new trial can move other
-- flags only when it is eligible and a step of its yardstick follows it, or when it is the first
-- eligible trial of its yardstick. For the newest trial on the current yardstick, the usual post,
-- neither holds, and the Worker sets its own flag in its batch (stepSelfStmt in src/index.ts).
-- The eligibility test must match sqlStepEligible() in src/index.ts.

CREATE TRIGGER IF NOT EXISTS nodes_steps_dirty_ai AFTER INSERT ON nodes
WHEN new.yardstick IS NOT NULL AND new.speedup > 0 AND new.state NOT IN ('retracted', 'disputed')
      AND (new.verified = 1 OR new.decision IN ('keep', 'keep_baseline', 'keep_simpler'))
  AND (EXISTS (SELECT 1 FROM nodes AS a WHERE a.lane = new.lane AND a.yardstick = new.yardstick AND a.frontier_step = 1
      AND (a.created_at, a.id) > (new.created_at, new.id))
    OR NOT EXISTS (SELECT 1 FROM nodes AS e WHERE e.lane = new.lane AND e.yardstick = new.yardstick
      AND e.yardstick IS NOT NULL AND e.speedup > 0 AND e.state NOT IN ('retracted', 'disputed')
        AND (e.verified = 1 OR e.decision IN ('keep', 'keep_baseline', 'keep_simpler'))
      AND (e.created_at, e.id) < (new.created_at, new.id)))
BEGIN
  INSERT INTO lane_stats (lane) SELECT new.lane
    WHERE NOT EXISTS (SELECT 1 FROM lane_stats WHERE lane = new.lane);
  UPDATE lane_stats SET steps_dirty = 1 WHERE lane = new.lane AND steps_dirty = 0;
END;

CREATE TRIGGER IF NOT EXISTS nodes_steps_dirty_au
AFTER UPDATE OF lane, yardstick, speedup, decision, verified, state, created_at ON nodes
WHEN (old.frontier_step = 1
    OR (new.yardstick IS NOT NULL AND new.speedup > 0 AND new.state NOT IN ('retracted', 'disputed')
      AND (new.verified = 1 OR new.decision IN ('keep', 'keep_baseline', 'keep_simpler')))
    OR (old.yardstick IS NOT NULL AND old.speedup > 0 AND old.state NOT IN ('retracted', 'disputed')
      AND (old.verified = 1 OR old.decision IN ('keep', 'keep_baseline', 'keep_simpler'))
      AND NOT EXISTS (SELECT 1 FROM nodes AS e WHERE e.lane = old.lane AND e.yardstick = old.yardstick
      AND e.yardstick IS NOT NULL AND e.speedup > 0 AND e.state NOT IN ('retracted', 'disputed')
        AND (e.verified = 1 OR e.decision IN ('keep', 'keep_baseline', 'keep_simpler'))
      AND (e.created_at, e.id) < (old.created_at, old.id))))
  AND NOT coalesce(
    old.yardstick IS NOT NULL AND old.speedup > 0 AND old.state NOT IN ('retracted', 'disputed')
      AND (old.verified = 1 OR old.decision IN ('keep', 'keep_baseline', 'keep_simpler'))
    AND new.yardstick IS NOT NULL AND new.speedup > 0 AND new.state NOT IN ('retracted', 'disputed')
      AND (new.verified = 1 OR new.decision IN ('keep', 'keep_baseline', 'keep_simpler'))
    AND old.lane IS new.lane AND old.created_at IS new.created_at AND old.speedup IS new.speedup
    AND old.yardstick IS new.yardstick, 0)
BEGIN
  INSERT INTO lane_stats (lane) SELECT new.lane
    WHERE NOT EXISTS (SELECT 1 FROM lane_stats WHERE lane = new.lane);
  UPDATE lane_stats SET steps_dirty = 1 WHERE lane IN (old.lane, new.lane) AND steps_dirty = 0;
END;

CREATE TRIGGER IF NOT EXISTS nodes_steps_dirty_ad AFTER DELETE ON nodes
WHEN old.frontier_step = 1
  OR (old.yardstick IS NOT NULL AND old.speedup > 0 AND old.state NOT IN ('retracted', 'disputed')
      AND (old.verified = 1 OR old.decision IN ('keep', 'keep_baseline', 'keep_simpler'))
    AND NOT EXISTS (SELECT 1 FROM nodes AS e WHERE e.lane = old.lane AND e.yardstick = old.yardstick
      AND e.yardstick IS NOT NULL AND e.speedup > 0 AND e.state NOT IN ('retracted', 'disputed')
        AND (e.verified = 1 OR e.decision IN ('keep', 'keep_baseline', 'keep_simpler'))
      AND (e.created_at, e.id) < (old.created_at, old.id)))
BEGIN
  UPDATE lane_stats SET steps_dirty = 1 WHERE lane = old.lane AND steps_dirty = 0;
END;

-- BACKFILL BEGIN
-- Every lane is dirty, then the Worker's recompute runs: the two statements after the dirty
-- mark equal SQL_SETTLE_YARDSTICK and SQL_SETTLE_STEPS in src/index.ts.
--
-- The first sets each dirty lane's current yardstick. It lists the lane's yardsticks that have
-- an eligible trial, one seek of idx_nodes_eligible each, takes the first eligible trial of
-- each, one more seek, and keeps the yardstick whose first trial is the newest. When that
-- changes the lane's yardstick, it counts the lane's graded trials on the new one again.
--
-- The second walks each dirty lane's eligible trials on its current yardstick from step to
-- step: from the latest step found (at first, nothing, with a bar of 0), the next step is the
-- first eligible trial after it whose speedup is at least 1.05 times that step's
-- (1.049999999999 absorbs rounding, so 1.26 over 1.2 counts). Each search seeks
-- idx_nodes_eligible once and reads on to the next step, so the walk reads each eligible trial
-- of the yardstick about once. Then it toggles each flag that differs: a walked trial that is
-- not marked, and a marked trial the walk skipped (on any yardstick).
INSERT INTO lane_stats (lane) SELECT DISTINCT lane FROM nodes WHERE lane NOT IN (SELECT lane FROM lane_stats);
UPDATE lane_stats SET steps_dirty = 1;
WITH RECURSIVE ys(lane, yardstick) AS (
    SELECT lane, '' FROM lane_stats INDEXED BY idx_lane_stats_dirty WHERE steps_dirty = 1
    UNION ALL
    SELECT ys.lane, (
      SELECT e.yardstick FROM nodes AS e INDEXED BY idx_nodes_eligible
      WHERE e.lane = ys.lane AND e.yardstick > ys.yardstick
        AND (e.yardstick IS NOT NULL AND e.speedup > 0 AND e.state NOT IN ('retracted', 'disputed')
          AND (e.verified = 1 OR e.decision IN ('keep', 'keep_baseline', 'keep_simpler')))
      ORDER BY e.yardstick LIMIT 1)
    FROM ys WHERE ys.yardstick IS NOT NULL
  ),
  cur(lane, yardstick) AS MATERIALIZED (
    SELECT d.lane, (
      SELECT ys.yardstick FROM ys JOIN nodes AS f ON f.id = (
        SELECT e.id FROM nodes AS e INDEXED BY idx_nodes_eligible
        WHERE e.lane = ys.lane AND e.yardstick = ys.yardstick
          AND (e.yardstick IS NOT NULL AND e.speedup > 0 AND e.state NOT IN ('retracted', 'disputed')
          AND (e.verified = 1 OR e.decision IN ('keep', 'keep_baseline', 'keep_simpler')))
        ORDER BY e.created_at, e.id LIMIT 1)
      WHERE ys.lane = d.lane ORDER BY f.created_at DESC, f.id DESC LIMIT 1)
    FROM lane_stats AS d INDEXED BY idx_lane_stats_dirty WHERE d.steps_dirty = 1
  )
UPDATE lane_stats SET yardstick = cur.yardstick,
  standard = (SELECT COUNT(*) FROM nodes AS g INDEXED BY idx_nodes_lane_graded WHERE g.lane = cur.lane AND g.yardstick = cur.yardstick)
FROM cur WHERE lane_stats.lane = cur.lane AND lane_stats.yardstick IS NOT cur.yardstick;
UPDATE nodes SET frontier_step = 1 - frontier_step WHERE id IN (
  WITH RECURSIVE walk(lane, yardstick, created_at, id, speedup) AS (
    SELECT lane, yardstick, '', '', 0 FROM lane_stats INDEXED BY idx_lane_stats_dirty WHERE steps_dirty = 1
    UNION ALL
    SELECT n.lane, w.yardstick, n.created_at, n.id, n.speedup FROM walk AS w JOIN nodes AS n ON n.id = (
      SELECT p.id FROM nodes AS p INDEXED BY idx_nodes_eligible
      WHERE p.lane = w.lane AND p.yardstick = w.yardstick AND (p.created_at, p.id) > (w.created_at, w.id)
        AND p.speedup >= w.speedup * 1.049999999999
        AND (p.yardstick IS NOT NULL AND p.speedup > 0 AND p.state NOT IN ('retracted', 'disputed')
          AND (p.verified = 1 OR p.decision IN ('keep', 'keep_baseline', 'keep_simpler')))
      ORDER BY p.created_at, p.id LIMIT 1)
  )
  SELECT n.id FROM walk AS w JOIN nodes AS n ON n.id = w.id WHERE n.frontier_step = 0
  UNION ALL
  SELECT n.id FROM lane_stats AS d INDEXED BY idx_lane_stats_dirty CROSS JOIN nodes AS n INDEXED BY idx_nodes_steps
  WHERE d.steps_dirty = 1 AND n.lane = d.lane AND n.frontier_step = 1 AND n.id NOT IN (SELECT id FROM walk));
UPDATE lane_stats SET steps_dirty = 0 WHERE steps_dirty = 1;
UPDATE lane_stats SET
  standard = (SELECT COUNT(*) FROM nodes n INDEXED BY idx_nodes_lane_graded WHERE n.lane = lane_stats.lane AND n.yardstick = lane_stats.yardstick),
  steps = (SELECT COUNT(*) FROM nodes n INDEXED BY idx_nodes_steps WHERE n.lane = lane_stats.lane AND n.frontier_step = 1);
-- BACKFILL END
