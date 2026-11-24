-- 0004: frontier steps. The trials that moved a lane's frontier are marked, so the site can
-- draw one "best so far" line per lane.
--
--   * nodes.frontier_step: 1 when the trial moved its lane's frontier (the rule is in 0005).
--   * nodes.standard: this file's first rule for comparable trials (equal dims). Migration 0005
--     drops it: a trial is standard when it was graded (nodes.yardstick is not null).
--   * lane_stats.standard, lane_stats.steps: per-lane counts of standard trials and steps.
--   * lane_stats.steps_dirty: 1 while a lane's flags wait for the recompute (see 0005).
--   * Indexes for the latest step of a lane (the frontier), the step=1 filter of /api/nodes and
--     the dirty lanes.
--
-- This file first also held triggers, more indexes and a backfill. Migration 0005 drops those
-- on a database that has them and creates their replacements, so they are no longer here.

ALTER TABLE nodes ADD COLUMN standard INTEGER NOT NULL DEFAULT 0;
ALTER TABLE nodes ADD COLUMN frontier_step INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lane_stats ADD COLUMN standard INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lane_stats ADD COLUMN steps INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lane_stats ADD COLUMN steps_dirty INTEGER NOT NULL DEFAULT 0;

-- The latest step of a lane (the frontier) and /api/nodes?lane=…&step=1.
CREATE INDEX IF NOT EXISTS idx_nodes_steps ON nodes (lane, created_at, id) WHERE frontier_step = 1;
-- /api/nodes?step=1 across lanes.
CREATE INDEX IF NOT EXISTS idx_nodes_steps_created ON nodes (created_at, id) WHERE frontier_step = 1;
-- The dirty lanes: empty except inside a writing transaction, so the usual write reads no row.
CREATE INDEX IF NOT EXISTS idx_lane_stats_dirty ON lane_stats (lane) WHERE steps_dirty = 1;
