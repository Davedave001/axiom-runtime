"""Database schema definitions (SQL DDL for PostgreSQL/SQLite persistence)."""

POSTGRES_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    idempotency_key VARCHAR(128) UNIQUE NOT NULL,
    agent_name VARCHAR(128) NOT NULL,
    goal TEXT NOT NULL,
    status VARCHAR(32) NOT NULL,
    current_step INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1,
    conversation_history JSONB NOT NULL DEFAULT '[]'::jsonb,
    pending_tool_call JSONB,
    pending_approval_id VARCHAR(64),
    final_output TEXT,
    failure_category VARCHAR(64) NOT NULL DEFAULT 'NONE',
    error_message TEXT,
    total_prompt_tokens INTEGER NOT NULL DEFAULT 0,
    total_completion_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    status_at_checkpoint VARCHAR(32) NOT NULL,
    conversation_history JSONB NOT NULL,
    pending_tool_call JSONB,
    pending_approval_id VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run_step ON checkpoints(run_id, step_index DESC);

CREATE TABLE IF NOT EXISTS tool_invocations (
    invocation_id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    idempotency_key VARCHAR(128) UNIQUE NOT NULL,
    tool_name VARCHAR(128) NOT NULL,
    arguments JSONB NOT NULL,
    status VARCHAR(32) NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 1,
    result JSONB,
    error_message TEXT,
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS approval_requests (
    approval_id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    tool_call JSONB NOT NULL,
    status VARCHAR(32) NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    decided_by VARCHAR(128),
    decided_at TIMESTAMPTZ,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS trace_events (
    event_id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    state_from VARCHAR(32),
    state_to VARCHAR(32),
    failure_category VARCHAR(64) NOT NULL DEFAULT 'NONE',
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""
