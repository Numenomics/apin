-- 0003: fixes from the Worker review.
--
--   * run_stats.author / author_tier are those of the run's earliest surviving trial (even when
--     that trial has no author), refreshed whenever
--     the run's membership changes (the three nodes triggers from 0002 are recreated).
--   * lane_stats: one row per lane, kept current by triggers on run_stats, so totals, lane
--     lists and counts read a row per lane instead of a row per run.
--   * author_stats + flywheel_counters: the number of distinct trial authors as one row.
--   * idx_nodes_reviewed_eff: the best replicated/verified trial with source per lane, the
--     bar a new trial must beat to enter the verification queue.
--
-- Additive for data. Apply with: npx wrangler d1 migrations apply apin-flywheel --remote
-- Like 0002, the triggers avoid conflict clauses (an outer statement's conflict mode
-- overrides them in SQLite), and nodes must never be written with INSERT OR REPLACE.
-- `+soft_min_eff` keeps the run-order lookups on idx_nodes_run_created: without the unary
-- plus SQLite may pick a (lane, soft_min_eff) index and read every scored trial of the lane per write.

CREATE INDEX IF NOT EXISTS idx_nodes_reviewed_eff ON nodes (lane, soft_min_eff)
  WHERE state IN ('replicated', 'verified') AND source_sha IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_run_stats_run ON run_stats (run_id);

-- lane_stats ----------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lane_stats (
  lane TEXT PRIMARY KEY,
  trials INTEGER NOT NULL DEFAULT 0,
  kept INTEGER NOT NULL DEFAULT 0,
  archived INTEGER NOT NULL DEFAULT 0,
  stan INTEGER NOT NULL DEFAULT 0,
  runs INTEGER NOT NULL DEFAULT 0,
  latest_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_lane_stats_trials ON lane_stats (trials);

CREATE TRIGGER IF NOT EXISTS run_stats_lane_ai AFTER INSERT ON run_stats BEGIN
  INSERT INTO lane_stats (lane) SELECT new.lane
    WHERE NOT EXISTS (SELECT 1 FROM lane_stats WHERE lane = new.lane);
  UPDATE lane_stats SET
    runs = runs + 1,
    trials = trials + new.trials,
    kept = kept + new.kept,
    archived = archived + new.archived,
    stan = stan + new.stan,
    latest_at = (SELECT MAX(last_at) FROM run_stats WHERE lane = new.lane)
  WHERE lane = new.lane;
END;

CREATE TRIGGER IF NOT EXISTS run_stats_lane_au AFTER UPDATE OF trials, kept, archived, stan, last_at ON run_stats BEGIN
  UPDATE lane_stats SET
    trials = trials + new.trials - old.trials,
    kept = kept + new.kept - old.kept,
    archived = archived + new.archived - old.archived,
    stan = stan + new.stan - old.stan,
    latest_at = (SELECT MAX(last_at) FROM run_stats WHERE lane = new.lane)
  WHERE lane = new.lane;
END;

CREATE TRIGGER IF NOT EXISTS run_stats_lane_ad AFTER DELETE ON run_stats BEGIN
  UPDATE lane_stats SET
    runs = runs - 1,
    trials = trials - old.trials,
    kept = kept - old.kept,
    archived = archived - old.archived,
    stan = stan - old.stan,
    latest_at = (SELECT MAX(last_at) FROM run_stats WHERE lane = old.lane)
  WHERE lane = old.lane;
  DELETE FROM lane_stats WHERE lane = old.lane AND runs <= 0;
END;

-- distinct authors -----------------------------------------------------------

CREATE TABLE IF NOT EXISTS author_stats (
  author_key TEXT PRIMARY KEY,  -- lower(author)
  trials INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS flywheel_counters (
  name TEXT PRIMARY KEY,
  value INTEGER NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO flywheel_counters (name, value) VALUES ('contributors', 0);

CREATE TRIGGER IF NOT EXISTS author_stats_ai AFTER INSERT ON author_stats BEGIN
  UPDATE flywheel_counters SET value = value + 1 WHERE name = 'contributors';
END;
CREATE TRIGGER IF NOT EXISTS author_stats_ad AFTER DELETE ON author_stats BEGIN
  UPDATE flywheel_counters SET value = value - 1 WHERE name = 'contributors';
END;

-- nodes triggers, recreated --------------------------------------------------

DROP TRIGGER IF EXISTS nodes_run_stats_ai;
DROP TRIGGER IF EXISTS nodes_run_stats_ad;
DROP TRIGGER IF EXISTS nodes_run_stats_au;

CREATE TRIGGER nodes_run_stats_ai AFTER INSERT ON nodes BEGIN
  INSERT INTO run_stats (lane, run_id)
    SELECT new.lane, new.run_id
    WHERE NOT EXISTS (SELECT 1 FROM run_stats WHERE lane = new.lane AND run_id = new.run_id);
  UPDATE run_stats SET
    trials = trials + 1,
    kept = kept + (new.decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')),
    archived = archived + (new.decision = 'archive_only'),
    stan = stan + (coalesce(new.real_stan_total, 0) > 0),
    agent_model = coalesce(new.agent_model, agent_model),
    first_at = (SELECT MIN(created_at) FROM nodes WHERE lane = new.lane AND run_id = new.run_id),
    last_at = (SELECT MAX(created_at) FROM nodes WHERE lane = new.lane AND run_id = new.run_id),
    author = (SELECT author FROM nodes WHERE lane = new.lane AND run_id = new.run_id
                ORDER BY created_at, id LIMIT 1),
    author_tier = (SELECT author_tier FROM nodes WHERE lane = new.lane AND run_id = new.run_id
                     ORDER BY created_at, id LIMIT 1),
    base_eff = (SELECT soft_min_eff FROM nodes WHERE lane = new.lane AND run_id = new.run_id
                  AND decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')
                  AND +soft_min_eff > 0 ORDER BY created_at, id LIMIT 1)
  WHERE lane = new.lane AND run_id = new.run_id;
  INSERT INTO author_stats (author_key) SELECT lower(new.author)
    WHERE new.author IS NOT NULL AND NOT EXISTS (SELECT 1 FROM author_stats WHERE author_key = lower(new.author));
  UPDATE author_stats SET trials = trials + 1 WHERE author_key = lower(new.author);
END;

CREATE TRIGGER nodes_run_stats_ad AFTER DELETE ON nodes BEGIN
  UPDATE run_stats SET
    trials = trials - 1,
    kept = kept - (old.decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')),
    archived = archived - (old.decision = 'archive_only'),
    stan = stan - (coalesce(old.real_stan_total, 0) > 0),
    first_at = (SELECT MIN(created_at) FROM nodes WHERE lane = old.lane AND run_id = old.run_id),
    last_at = (SELECT MAX(created_at) FROM nodes WHERE lane = old.lane AND run_id = old.run_id),
    author = (SELECT author FROM nodes WHERE lane = old.lane AND run_id = old.run_id
                ORDER BY created_at, id LIMIT 1),
    author_tier = (SELECT author_tier FROM nodes WHERE lane = old.lane AND run_id = old.run_id
                     ORDER BY created_at, id LIMIT 1),
    base_eff = (SELECT soft_min_eff FROM nodes WHERE lane = old.lane AND run_id = old.run_id
                  AND decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')
                  AND +soft_min_eff > 0 ORDER BY created_at, id LIMIT 1)
  WHERE lane = old.lane AND run_id = old.run_id;
  DELETE FROM run_stats WHERE lane = old.lane AND run_id = old.run_id AND trials <= 0;
  UPDATE author_stats SET trials = trials - 1 WHERE author_key = lower(old.author);
  DELETE FROM author_stats WHERE author_key = lower(old.author) AND trials <= 0;
END;

-- Re-posting an unchanged trial (the common idempotent case) does not fire this.
CREATE TRIGGER nodes_run_stats_au
AFTER UPDATE OF lane, run_id, decision, real_stan_total, created_at, soft_min_eff, author, author_tier, agent_model ON nodes
WHEN old.lane IS NOT new.lane OR old.run_id IS NOT new.run_id OR old.decision IS NOT new.decision
  OR coalesce(old.real_stan_total, 0) IS NOT coalesce(new.real_stan_total, 0) OR old.created_at IS NOT new.created_at
  OR old.soft_min_eff IS NOT new.soft_min_eff OR old.author IS NOT new.author
  OR old.author_tier IS NOT new.author_tier OR old.agent_model IS NOT new.agent_model
BEGIN
  UPDATE run_stats SET
    trials = trials - 1,
    kept = kept - (old.decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')),
    archived = archived - (old.decision = 'archive_only'),
    stan = stan - (coalesce(old.real_stan_total, 0) > 0)
  WHERE lane = old.lane AND run_id = old.run_id;
  INSERT INTO run_stats (lane, run_id)
    SELECT new.lane, new.run_id
    WHERE NOT EXISTS (SELECT 1 FROM run_stats WHERE lane = new.lane AND run_id = new.run_id);
  UPDATE run_stats SET
    trials = trials + 1,
    kept = kept + (new.decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')),
    archived = archived + (new.decision = 'archive_only'),
    stan = stan + (coalesce(new.real_stan_total, 0) > 0),
    agent_model = coalesce(new.agent_model, agent_model)
  WHERE lane = new.lane AND run_id = new.run_id;
  UPDATE run_stats SET
    first_at = (SELECT MIN(created_at) FROM nodes n WHERE n.lane = run_stats.lane AND n.run_id = run_stats.run_id),
    last_at = (SELECT MAX(created_at) FROM nodes n WHERE n.lane = run_stats.lane AND n.run_id = run_stats.run_id),
    author = (SELECT author FROM nodes n WHERE n.lane = run_stats.lane AND n.run_id = run_stats.run_id
                ORDER BY n.created_at, n.id LIMIT 1),
    author_tier = (SELECT author_tier FROM nodes n WHERE n.lane = run_stats.lane AND n.run_id = run_stats.run_id
                     ORDER BY n.created_at, n.id LIMIT 1),
    base_eff = (SELECT soft_min_eff FROM nodes n WHERE n.lane = run_stats.lane AND n.run_id = run_stats.run_id
                  AND n.decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')
                  AND +n.soft_min_eff > 0 ORDER BY n.created_at, n.id LIMIT 1)
  WHERE (lane = old.lane AND run_id = old.run_id) OR (lane = new.lane AND run_id = new.run_id);
  DELETE FROM run_stats WHERE lane = old.lane AND run_id = old.run_id AND trials <= 0;
  INSERT INTO author_stats (author_key) SELECT lower(new.author)
    WHERE new.author IS NOT NULL AND NOT EXISTS (SELECT 1 FROM author_stats WHERE author_key = lower(new.author));
  UPDATE author_stats SET trials = trials + 1 WHERE author_key = lower(new.author);
  UPDATE author_stats SET trials = trials - 1 WHERE author_key = lower(old.author);
  DELETE FROM author_stats WHERE author_key = lower(old.author) AND trials <= 0;
END;

-- Backfill ---------------------------------------------------------------------

UPDATE run_stats SET
  author = (SELECT author FROM nodes n WHERE n.lane = run_stats.lane AND n.run_id = run_stats.run_id
              ORDER BY n.created_at, n.id LIMIT 1),
  author_tier = (SELECT author_tier FROM nodes n WHERE n.lane = run_stats.lane AND n.run_id = run_stats.run_id
                   ORDER BY n.created_at, n.id LIMIT 1);

INSERT OR IGNORE INTO lane_stats (lane, trials, kept, archived, stan, runs, latest_at)
  SELECT lane, SUM(trials), SUM(kept), SUM(archived), SUM(stan), COUNT(*), MAX(last_at)
  FROM run_stats GROUP BY lane;

INSERT OR IGNORE INTO author_stats (author_key, trials)
  SELECT lower(author), COUNT(*) FROM nodes WHERE author IS NOT NULL GROUP BY lower(author);
UPDATE flywheel_counters SET value = (SELECT COUNT(*) FROM author_stats) WHERE name = 'contributors';

-- REBUILD (safe to re-run, after the REBUILD of 0002):
--   DELETE FROM lane_stats; then the lane_stats INSERT above;
--   DELETE FROM author_stats; then the author_stats INSERT and the flywheel_counters UPDATE above.

-- Newest notes of a lane, for the bounded search window.
CREATE INDEX IF NOT EXISTS idx_notes_lane_updated ON notes (lane, updated_at, id);

-- /api/nodes reads each filter through an index (see DRIVING_INDEXES in src/index.ts).
CREATE INDEX IF NOT EXISTS idx_nodes_runid_created ON nodes (run_id, created_at, id);
CREATE INDEX IF NOT EXISTS idx_nodes_kept_created ON nodes (created_at, id)
  WHERE decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed');
CREATE INDEX IF NOT EXISTS idx_nodes_archived_created ON nodes (created_at, id) WHERE decision = 'archive_only';
CREATE INDEX IF NOT EXISTS idx_nodes_discarded_created ON nodes (created_at, id)
  WHERE decision NOT IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed', 'archive_only');
CREATE INDEX IF NOT EXISTS idx_nodes_lane_kept ON nodes (lane, created_at, id)
  WHERE decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed');
CREATE INDEX IF NOT EXISTS idx_nodes_lane_archived ON nodes (lane, created_at, id) WHERE decision = 'archive_only';
CREATE INDEX IF NOT EXISTS idx_nodes_lane_discarded ON nodes (lane, created_at, id)
  WHERE decision NOT IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed', 'archive_only');

-- Overview `stan`: the lane's best real-Stan pass share (newest on ties), read as one index row.
-- The expression must match laneStanStmt() in src/index.ts exactly.
CREATE INDEX IF NOT EXISTS idx_nodes_stan_share ON nodes
  (lane, (CAST(coalesce(real_stan_passed, 0) AS REAL) / real_stan_total) DESC, created_at DESC, id DESC)
  WHERE real_stan_total > 0;

-- Newest-first reads by lane or model carry id, so ORDER BY created_at DESC, id DESC LIMIT n
-- never sorts a whole timestamp tie. They replace idx_nodes_lane_created (base schema, kept
-- because schema.sql recreates it) and idx_nodes_model_created (0002, dropped here).
CREATE INDEX IF NOT EXISTS idx_nodes_lane_newest ON nodes (lane, created_at, id);
CREATE INDEX IF NOT EXISTS idx_nodes_model_newest ON nodes (agent_model, created_at, id);
DROP INDEX IF EXISTS idx_nodes_model_created;

-- Frontier pick per lane, in the exact frontier order, so it reads one index row even when the
-- lane has no eligible trial or a long score tie. The expressions and WHERE clauses must match
-- laneFrontiers() in src/index.ts. They replace idx_nodes_lane_eff (0002), dropped here.
CREATE INDEX IF NOT EXISTS idx_nodes_frontier_pick ON nodes
  (lane, soft_min_eff DESC, verified DESC, (coalesce(real_stan_passed, -1)) DESC, created_at DESC, id DESC)
  WHERE state NOT IN ('retracted', 'disputed') AND (verified = 1 OR decision IN ('keep', 'keep_baseline', 'keep_simpler'));
CREATE INDEX IF NOT EXISTS idx_nodes_frontier_any ON nodes
  (lane, soft_min_eff DESC, verified DESC, (coalesce(real_stan_passed, -1)) DESC, created_at DESC, id DESC)
  WHERE state NOT IN ('retracted', 'disputed');
DROP INDEX IF EXISTS idx_nodes_lane_eff;
