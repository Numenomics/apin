-- 0002: bounded reads. Every page read touches a bounded number of rows in the Worker,
-- whatever the number of trials.
--
--   * Indexes for newest-first lists, per-run slices, per-lane frontier and Stan lookups.
--   * run_stats: one row per (lane, run_id), kept current by triggers on nodes, so
--     /api/overview, /api/runs, /api/health and list counts never aggregate over all trials.
--   * note_links: which notes cite which trial, kept current by triggers on notes.
--
-- Additive only. Apply with: npx wrangler d1 migrations apply apin-flywheel --remote
--
-- The triggers count a trial once per INSERT and adjust on UPDATE/DELETE. INSERT OR REPLACE
-- into nodes deletes rows without firing the delete trigger, so it would double-count:
-- write trials with plain INSERT or an upsert (ON CONFLICT DO UPDATE), as the Worker and
-- seed.sql do. If the counters ever drift, rebuild them with the two statements marked
-- "REBUILD" at the end of this file. Triggers avoid conflict clauses because an outer
-- statement's conflict mode overrides them in SQLite.

CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes (created_at, id);
CREATE INDEX IF NOT EXISTS idx_nodes_run_created ON nodes (lane, run_id, created_at, id);
CREATE INDEX IF NOT EXISTS idx_nodes_lane_eff ON nodes (lane, soft_min_eff);  -- frontier (named in INDEXED BY)
CREATE INDEX IF NOT EXISTS idx_nodes_model_created ON nodes (agent_model, created_at);
CREATE INDEX IF NOT EXISTS idx_nodes_stan_created ON nodes (created_at, id) WHERE real_stan_total > 0;
CREATE INDEX IF NOT EXISTS idx_nodes_stan_lane ON nodes (lane, created_at, id) WHERE real_stan_total > 0;

CREATE TABLE IF NOT EXISTS run_stats (
  lane TEXT NOT NULL,
  run_id TEXT NOT NULL,
  trials INTEGER NOT NULL DEFAULT 0,
  -- kept: keep, keep_baseline, keep_simpler, keep_incumbent_failed
  kept INTEGER NOT NULL DEFAULT 0,
  -- archived: archive_only; discarded = trials - kept - archived
  archived INTEGER NOT NULL DEFAULT 0,
  -- stan: trials with real_stan_total > 0
  stan INTEGER NOT NULL DEFAULT 0,
  first_at TEXT,
  last_at TEXT,
  -- author of the run's first trial; author_tier follows that author's promotion
  author TEXT,
  author_tier TEXT,
  -- most recently written non-null agent_model
  agent_model TEXT,
  -- soft_min_eff of the run's first kept trial with soft_min_eff > 0
  base_eff REAL,
  PRIMARY KEY (lane, run_id)
);
CREATE INDEX IF NOT EXISTS idx_run_stats_last ON run_stats (last_at, lane, run_id);
CREATE INDEX IF NOT EXISTS idx_run_stats_lane_last ON run_stats (lane, last_at, run_id);
CREATE INDEX IF NOT EXISTS idx_run_stats_model ON run_stats (agent_model);

CREATE TRIGGER IF NOT EXISTS nodes_run_stats_ai AFTER INSERT ON nodes BEGIN
  INSERT INTO run_stats (lane, run_id, author, author_tier)
    SELECT new.lane, new.run_id, new.author, new.author_tier
    WHERE NOT EXISTS (SELECT 1 FROM run_stats WHERE lane = new.lane AND run_id = new.run_id);
  UPDATE run_stats SET
    trials = trials + 1,
    kept = kept + (new.decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')),
    archived = archived + (new.decision = 'archive_only'),
    stan = stan + (coalesce(new.real_stan_total, 0) > 0),
    agent_model = coalesce(new.agent_model, agent_model),
    author = coalesce(author, new.author),
    author_tier = CASE WHEN author IS NULL OR author IS new.author THEN coalesce(new.author_tier, author_tier) ELSE author_tier END,
    first_at = (SELECT MIN(created_at) FROM nodes WHERE lane = new.lane AND run_id = new.run_id),
    last_at = (SELECT MAX(created_at) FROM nodes WHERE lane = new.lane AND run_id = new.run_id),
    base_eff = (SELECT soft_min_eff FROM nodes WHERE lane = new.lane AND run_id = new.run_id
                  AND decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')
                  AND soft_min_eff > 0 ORDER BY created_at, id LIMIT 1)
  WHERE lane = new.lane AND run_id = new.run_id;
END;

CREATE TRIGGER IF NOT EXISTS nodes_run_stats_ad AFTER DELETE ON nodes BEGIN
  UPDATE run_stats SET
    trials = trials - 1,
    kept = kept - (old.decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')),
    archived = archived - (old.decision = 'archive_only'),
    stan = stan - (coalesce(old.real_stan_total, 0) > 0),
    first_at = (SELECT MIN(created_at) FROM nodes WHERE lane = old.lane AND run_id = old.run_id),
    last_at = (SELECT MAX(created_at) FROM nodes WHERE lane = old.lane AND run_id = old.run_id),
    base_eff = (SELECT soft_min_eff FROM nodes WHERE lane = old.lane AND run_id = old.run_id
                  AND decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')
                  AND soft_min_eff > 0 ORDER BY created_at, id LIMIT 1)
  WHERE lane = old.lane AND run_id = old.run_id;
  DELETE FROM run_stats WHERE lane = old.lane AND run_id = old.run_id AND trials <= 0;
END;

-- Re-posting an unchanged trial (the common idempotent case) does not fire this.
CREATE TRIGGER IF NOT EXISTS nodes_run_stats_au
AFTER UPDATE OF lane, run_id, decision, real_stan_total, created_at, soft_min_eff, author, author_tier, agent_model ON nodes
WHEN old.lane IS NOT new.lane OR old.run_id IS NOT new.run_id OR old.decision IS NOT new.decision
  OR coalesce(old.real_stan_total, 0) IS NOT coalesce(new.real_stan_total, 0) OR old.created_at IS NOT new.created_at
  OR old.soft_min_eff IS NOT new.soft_min_eff OR old.author IS NOT new.author
  OR old.author_tier IS NOT new.author_tier OR old.agent_model IS NOT new.agent_model
BEGIN
  -- Take the old row out of its run...
  UPDATE run_stats SET
    trials = trials - 1,
    kept = kept - (old.decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')),
    archived = archived - (old.decision = 'archive_only'),
    stan = stan - (coalesce(old.real_stan_total, 0) > 0)
  WHERE lane = old.lane AND run_id = old.run_id;
  -- ...put the new row into its run (usually the same one)...
  INSERT INTO run_stats (lane, run_id, author, author_tier)
    SELECT new.lane, new.run_id, new.author, new.author_tier
    WHERE NOT EXISTS (SELECT 1 FROM run_stats WHERE lane = new.lane AND run_id = new.run_id);
  UPDATE run_stats SET
    trials = trials + 1,
    kept = kept + (new.decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')),
    archived = archived + (new.decision = 'archive_only'),
    stan = stan + (coalesce(new.real_stan_total, 0) > 0),
    agent_model = coalesce(new.agent_model, agent_model),
    author = coalesce(author, new.author),
    author_tier = CASE WHEN author IS NULL OR author IS new.author THEN coalesce(new.author_tier, author_tier) ELSE author_tier END
  WHERE lane = new.lane AND run_id = new.run_id;
  -- ...and refresh the derived columns of both runs.
  UPDATE run_stats SET
    first_at = (SELECT MIN(created_at) FROM nodes n WHERE n.lane = run_stats.lane AND n.run_id = run_stats.run_id),
    last_at = (SELECT MAX(created_at) FROM nodes n WHERE n.lane = run_stats.lane AND n.run_id = run_stats.run_id),
    base_eff = (SELECT soft_min_eff FROM nodes n WHERE n.lane = run_stats.lane AND n.run_id = run_stats.run_id
                  AND n.decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')
                  AND n.soft_min_eff > 0 ORDER BY n.created_at, n.id LIMIT 1)
  WHERE (lane = old.lane AND run_id = old.run_id) OR (lane = new.lane AND run_id = new.run_id);
  DELETE FROM run_stats WHERE lane = old.lane AND run_id = old.run_id AND trials <= 0;
END;

-- Which notes cite which trial (from notes.node_ids, a JSON array).
CREATE TABLE IF NOT EXISTS note_links (
  note_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  PRIMARY KEY (note_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_note_links_node ON note_links (node_id);

CREATE TRIGGER IF NOT EXISTS notes_links_ai AFTER INSERT ON notes BEGIN
  INSERT INTO note_links (note_id, node_id)
    SELECT DISTINCT new.id, value FROM json_each(new.node_ids) WHERE json_valid(new.node_ids);
END;

CREATE TRIGGER IF NOT EXISTS notes_links_au AFTER UPDATE OF node_ids ON notes BEGIN
  DELETE FROM note_links WHERE note_id = old.id;
  INSERT INTO note_links (note_id, node_id)
    SELECT DISTINCT new.id, value FROM json_each(new.node_ids) WHERE json_valid(new.node_ids);
END;

CREATE TRIGGER IF NOT EXISTS notes_links_ad AFTER DELETE ON notes BEGIN
  DELETE FROM note_links WHERE note_id = old.id;
END;

-- Backfill from existing rows.
INSERT OR IGNORE INTO note_links (note_id, node_id)
  SELECT DISTINCT n.id, j.value FROM notes n, json_each(n.node_ids) j WHERE json_valid(n.node_ids);

-- REBUILD (safe to re-run): DELETE FROM run_stats; then the INSERT below.
INSERT OR IGNORE INTO run_stats (lane, run_id, trials, kept, archived, stan, first_at, last_at, author, author_tier, agent_model, base_eff)
  SELECT g.lane, g.run_id, g.trials, g.kept, g.archived, g.stan, g.first_at, g.last_at,
    (SELECT author FROM nodes n WHERE n.lane = g.lane AND n.run_id = g.run_id AND n.author IS NOT NULL ORDER BY n.created_at, n.id LIMIT 1),
    (SELECT author_tier FROM nodes n WHERE n.lane = g.lane AND n.run_id = g.run_id AND n.author IS NOT NULL ORDER BY n.created_at, n.id LIMIT 1),
    (SELECT agent_model FROM nodes n WHERE n.lane = g.lane AND n.run_id = g.run_id AND n.agent_model IS NOT NULL ORDER BY n.created_at DESC, n.id DESC LIMIT 1),
    (SELECT soft_min_eff FROM nodes n WHERE n.lane = g.lane AND n.run_id = g.run_id
       AND n.decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')
       AND n.soft_min_eff > 0 ORDER BY n.created_at, n.id LIMIT 1)
  FROM (
    SELECT lane, run_id, COUNT(*) AS trials,
      SUM(decision IN ('keep', 'keep_baseline', 'keep_simpler', 'keep_incumbent_failed')) AS kept,
      SUM(decision = 'archive_only') AS archived,
      SUM(coalesce(real_stan_total, 0) > 0) AS stan,
      MIN(created_at) AS first_at, MAX(created_at) AS last_at
    FROM nodes GROUP BY lane, run_id
  ) g;
