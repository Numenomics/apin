-- 0006: the real-posterior gate. A trial is a frontier step only when an independent verifier
-- (`apin autoresearch verify-next`) or a maintainer posted a real-posterior gate result for it
-- under the current gate id, and every required model passed. The gate is defined in
-- autoresearch/real_gate.json and its id is REAL_GATE in src/index.ts.
--
--   * nodes.real_gate: the gate id the trial's real_stan_* result was measured under. Null when
--     no gate result was posted. Rows with real_stan_* from earlier seeding (seed_from_apin.py)
--     or an earlier admin verify have no gate id, so they do not qualify.
--   * Yardstick-eligible, unchanged from 0005 (idx_nodes_eligible, sqlYardstickEligible in
--     src/index.ts): graded, speedup > 0, not retracted or disputed, and verified or kept. It
--     sets lane_stats.yardstick, and the verification queue works on that yardstick. It must not
--     need the gate: the verifier that runs the gate leases only trials on the lane's current
--     yardstick, so a yardstick that waited for the gate could never get a gated trial.
--   * Step-eligible (sqlStepEligible): yardstick-eligible, and real_gate = 'nuts-ref-v1' AND
--     real_stan_total > 0 AND real_stan_passed = real_stan_total. The step walk compares only
--     these. A lane with none has no frontier.
--   * nodes_steps_dirty_au is recreated: it also fires when real_stan_passed, real_stan_total or
--     real_gate change, and an eligible trial whose gate result changed counts as changed. So a
--     verifier's passing result marks the lane dirty and the Worker's settle statements in the
--     same batch move the frontier. Its tests use yardstick eligibility, which holds for every
--     step, so the trigger does not name the gate id. The insert and delete triggers of 0005 stay:
--     a new or deleted trial can move other flags only when it is (or was) a step, which needs
--     yardstick eligibility, or when it is the first yardstick-eligible trial of its yardstick.
--
-- On a database with trials from before this file no trial has a gate result, so after the
-- BACKFILL below no lane has a step or a frontier until a verifier posts a passing result.
--
-- REBUILD (safe to re-run): run the BACKFILL block. It recomputes the yardsticks and the step
-- flags from scratch and then the lane_stats counters. It supersedes the BACKFILL block of 0005.
-- seed_from_apin.py copies this one to the end of seed.sql and prints it with --backfill-sql.
--
-- To bump the gate id: change `id` in autoresearch/real_gate.json and REAL_GATE in src/index.ts,
-- and add a migration whose BACKFILL block is this one with the new id. Apply it and deploy the
-- Worker together. The triggers and indexes need no change.

ALTER TABLE nodes ADD COLUMN real_gate TEXT;

DROP TRIGGER IF EXISTS nodes_steps_dirty_au;

-- lane_stats.steps_dirty on an update ---------------------------------------------
-- As in 0005, with the gate result as part of what an eligible trial may change. A trial that is
-- not a step never changes the bar for a later trial, so a change can move other flags only when
-- the trial was a step or is yardstick-eligible now (every step is). It cannot when the trial
-- stays yardstick-eligible at the same position with the same speedup, yardstick and gate
-- result. The current yardstick can change only when the first yardstick-eligible trial of some
-- yardstick changes: a trial that is yardstick-eligible now, or one that was the first of its
-- yardstick. The eligibility test must match sqlYardstickEligible() in src/index.ts.

CREATE TRIGGER IF NOT EXISTS nodes_steps_dirty_au
AFTER UPDATE OF lane, yardstick, speedup, decision, verified, state, created_at,
  real_stan_passed, real_stan_total, real_gate ON nodes
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
    AND old.yardstick IS new.yardstick
    AND old.real_gate IS new.real_gate AND old.real_stan_passed IS new.real_stan_passed
    AND old.real_stan_total IS new.real_stan_total, 0)
BEGIN
  INSERT INTO lane_stats (lane) SELECT new.lane
    WHERE NOT EXISTS (SELECT 1 FROM lane_stats WHERE lane = new.lane);
  UPDATE lane_stats SET steps_dirty = 1 WHERE lane IN (old.lane, new.lane) AND steps_dirty = 0;
END;

-- BACKFILL BEGIN
-- Every lane is dirty, then the Worker's recompute runs: the two statements after the dirty
-- mark equal SQL_SETTLE_YARDSTICK and SQL_SETTLE_STEPS in src/index.ts.
--
-- The first sets each dirty lane's current yardstick, as in 0005: the gate plays no part. It
-- lists the lane's yardsticks that have a yardstick-eligible trial, one seek of
-- idx_nodes_eligible each, takes the first such trial of each, one more seek, and keeps the
-- yardstick whose first trial is the newest. When that changes the lane's yardstick, it counts
-- the lane's graded trials on the new one again.
--
-- The second walks each dirty lane's step-eligible trials on its current yardstick from step to
-- step: from the latest step found (at first, nothing, with a bar of 0), the next step is the
-- first trial after it that passed the current gate ('nuts-ref-v1', every required model) and
-- whose speedup is at least 1.05 times that step's (1.049999999999 absorbs rounding, so 1.26 over
-- 1.2 counts). Each search seeks idx_nodes_eligible once and reads on to the next step, so the
-- walk reads each yardstick-eligible trial of the yardstick about once. Then it toggles each flag
-- that differs: a walked trial that is not marked, and a marked trial the walk skipped (on any
-- yardstick).
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
        AND ((p.yardstick IS NOT NULL AND p.speedup > 0 AND p.state NOT IN ('retracted', 'disputed')
          AND (p.verified = 1 OR p.decision IN ('keep', 'keep_baseline', 'keep_simpler')))
          AND (p.real_gate = 'nuts-ref-v1' AND p.real_stan_total > 0 AND p.real_stan_passed = p.real_stan_total))
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
