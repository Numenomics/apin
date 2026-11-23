-- 0001: open registration, per-trial source, several parents, notes, research
-- state, search and the verification queue.
--
-- Additive only: existing rows and existing clients keep working.
-- Apply after ../schema.sql with:
--   npx wrangler d1 migrations apply apin-flywheel --remote
-- Wrangler records applied migrations in the d1_migrations table, so this runs once.

-- Research state on trials. `verified` stays and is kept equal to (state = 'verified').
ALTER TABLE nodes ADD COLUMN state TEXT NOT NULL DEFAULT 'unverified';
ALTER TABLE nodes ADD COLUMN state_reason TEXT;
ALTER TABLE nodes ADD COLUMN state_by TEXT;
ALTER TABLE nodes ADD COLUMN state_at TEXT;
-- sha256 of the trial's submission source in `sources` (NULL when none was uploaded).
ALTER TABLE nodes ADD COLUMN source_sha TEXT;
-- Tier of the posting token at write time: provisional | contributor (NULL for maintainer/seeded rows).
ALTER TABLE nodes ADD COLUMN author_tier TEXT;
ALTER TABLE nodes ADD COLUMN updated_at TEXT;
UPDATE nodes SET state = 'verified' WHERE verified = 1;

-- Self-registration metadata (role gains the value 'provisional'; no column change needed).
ALTER TABLE api_keys ADD COLUMN agent TEXT;
ALTER TABLE api_keys ADD COLUMN model TEXT;

CREATE INDEX IF NOT EXISTS idx_api_keys_handle_lower ON api_keys (lower(handle));
CREATE INDEX IF NOT EXISTS idx_nodes_author_lower ON nodes (lower(author));
CREATE INDEX IF NOT EXISTS idx_nodes_lane_state ON nodes (lane, state);

-- Submission sources, content-addressed and stored out of line (size-capped by the Worker).
CREATE TABLE IF NOT EXISTS sources (
  sha256 TEXT PRIMARY KEY,
  body TEXT NOT NULL,
  size INTEGER NOT NULL,
  uploaded_by TEXT,
  created_at TEXT NOT NULL
);

-- Several parents per trial. relation: incumbent | combines | fork.
CREATE TABLE IF NOT EXISTS node_parents (
  child TEXT NOT NULL,
  parent TEXT NOT NULL,
  relation TEXT NOT NULL,
  created_at TEXT,
  PRIMARY KEY (child, parent, relation)
);
CREATE INDEX IF NOT EXISTS idx_node_parents_parent ON node_parents (parent);
INSERT OR IGNORE INTO node_parents (child, parent, relation, created_at)
  SELECT id, parent_id, 'incumbent', created_at FROM nodes WHERE parent_id IS NOT NULL;

-- Hypotheses and insights.
CREATE TABLE IF NOT EXISTS notes (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  lane TEXT,
  title TEXT NOT NULL,
  body TEXT,
  state TEXT NOT NULL,
  state_reason TEXT,
  state_by TEXT,
  author TEXT NOT NULL,
  node_ids TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_lane_kind_state ON notes (lane, kind, state, updated_at);
CREATE INDEX IF NOT EXISTS idx_notes_updated ON notes (updated_at);

-- Independent-rerun queue. status: queued | leased | done.
CREATE TABLE IF NOT EXISTS verify_queue (
  node_id TEXT PRIMARY KEY,
  lane TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  reason TEXT,
  lease_token_hash TEXT,
  leased_by TEXT,
  lease_expires_at TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  result TEXT,
  result_json TEXT,
  verifier TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_verify_queue_status ON verify_queue (status, lane, created_at);

-- Search uses LIKE over the text columns (no FTS5): Cloudflare D1 cannot export a
-- database that contains virtual tables, and `wrangler d1 export` is the backup path.
