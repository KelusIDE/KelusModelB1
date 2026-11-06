

CREATE TABLE IF NOT EXISTS api_keys (
  id TEXT PRIMARY KEY,
  hash TEXT NOT NULL UNIQUE,
  hint TEXT NOT NULL,
  owner TEXT NOT NULL,
  name TEXT NOT NULL,
  created INTEGER NOT NULL,
  last_used INTEGER,
  requests INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS api_keys_owner ON api_keys (owner);


CREATE TABLE IF NOT EXISTS api_usage (
  owner TEXT NOT NULL,
  day TEXT NOT NULL,
  requests INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (owner, day)
);

CREATE TABLE IF NOT EXISTS auth_email_limits (
  key TEXT NOT NULL,
  day TEXT NOT NULL,
  requests INTEGER NOT NULL DEFAULT 0,
  last_sent INTEGER NOT NULL,
  PRIMARY KEY (key, day)
);
