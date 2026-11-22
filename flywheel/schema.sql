-- apin Flywheel D1 SQLite schema

CREATE TABLE IF NOT EXISTS nodes (
  id TEXT PRIMARY KEY,
  parent_id TEXT,
  lane TEXT NOT NULL,
  run_id TEXT NOT NULL,
  trial_id TEXT,
  agent_model TEXT,
  author TEXT,
  title TEXT,
  hypothesis TEXT,
  decision TEXT NOT NULL,
  soft_min_eff REAL,
  ratio_vs_incumbent REAL,
  short_horizon_eff REAL,
  long_horizon_eff REAL,
  real_stan_passed INTEGER,
  real_stan_total INTEGER,
  real_ess_per_cost REAL,
  overhead_ratio REAL,
  b2_max REAL,
  dims TEXT,
  codex_refs TEXT,
  code_sha TEXT,
  metrics_json TEXT,
  verified INTEGER DEFAULT 0,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  lane TEXT NOT NULL,
  agent TEXT NOT NULL,
  summary TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  expires_at TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
  id TEXT PRIMARY KEY,
  key_hash TEXT NOT NULL UNIQUE,
  handle TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'contributor',
  hourly_limit INTEGER NOT NULL DEFAULT 120,
  revoked_at TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rate_limits (
  key_id TEXT NOT NULL,
  bucket TEXT NOT NULL,
  count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (key_id, bucket)
);

CREATE INDEX IF NOT EXISTS idx_nodes_lane_created ON nodes (lane, created_at);
CREATE INDEX IF NOT EXISTS idx_nodes_parent_id ON nodes (parent_id);
CREATE INDEX IF NOT EXISTS idx_claims_status_expires ON claims (status, expires_at);
CREATE INDEX IF NOT EXISTS idx_claims_task_status_expires ON claims (task_id, status, expires_at);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys (key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_handle ON api_keys (handle);
