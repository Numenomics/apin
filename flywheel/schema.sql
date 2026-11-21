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

CREATE INDEX IF NOT EXISTS idx_nodes_lane_created ON nodes (lane, created_at);
CREATE INDEX IF NOT EXISTS idx_nodes_parent_id ON nodes (parent_id);
CREATE INDEX IF NOT EXISTS idx_claims_status_expires ON claims (status, expires_at);
CREATE INDEX IF NOT EXISTS idx_claims_task_status_expires ON claims (task_id, status, expires_at);
