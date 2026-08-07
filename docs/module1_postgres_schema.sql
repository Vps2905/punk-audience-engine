-- Module 1: Ingestion & Privacy Layer
-- Production Postgres schema reference
--
-- Purpose:
--   Defines the DB tables used by Module 1 privacy ingestion, lineage,
--   job tracking, privacy budget enforcement, and approval audit support.
--
-- Important:
--   This repository currently does not include Alembic.
--   Treat this file as the production schema reference.
--   If Alembic is later introduced, convert these CREATE TABLE statements
--   into Alembic revision files.

BEGIN;

CREATE TABLE IF NOT EXISTS audience_ingestion_jobs (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT UNIQUE NOT NULL,
    run_id TEXT,
    source_type TEXT NOT NULL,
    source_ref TEXT,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'blocked')),
    input_rows INTEGER CHECK (input_rows IS NULL OR input_rows >= 0),
    output_rows INTEGER CHECK (output_rows IS NULL OR output_rows >= 0),
    dropped_rows INTEGER CHECK (dropped_rows IS NULL OR dropped_rows >= 0),
    error_message TEXT,
    actor TEXT NOT NULL DEFAULT 'system',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audience_ingestion_jobs_job
ON audience_ingestion_jobs(job_id);

CREATE INDEX IF NOT EXISTS idx_audience_ingestion_jobs_run
ON audience_ingestion_jobs(run_id);

CREATE INDEX IF NOT EXISTS idx_audience_ingestion_jobs_status
ON audience_ingestion_jobs(status);


CREATE TABLE IF NOT EXISTS audience_lineage_events (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT NOT NULL,
    run_id TEXT,
    stage TEXT NOT NULL,
    transformation TEXT NOT NULL,
    input_ref TEXT,
    output_ref TEXT,
    input_rows INTEGER CHECK (input_rows IS NULL OR input_rows >= 0),
    output_rows INTEGER CHECK (output_rows IS NULL OR output_rows >= 0),
    dropped_rows INTEGER CHECK (dropped_rows IS NULL OR dropped_rows >= 0),
    actor TEXT NOT NULL DEFAULT 'system',
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audience_lineage_job
ON audience_lineage_events(job_id);

CREATE INDEX IF NOT EXISTS idx_audience_lineage_run
ON audience_lineage_events(run_id);

CREATE INDEX IF NOT EXISTS idx_audience_lineage_stage
ON audience_lineage_events(stage);


CREATE TABLE IF NOT EXISTS audience_privacy_budget_ledger (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL CHECK (
        tenant_id ~ '^[a-z0-9][a-z0-9_.-]{0,127}$'
    ),
    run_id TEXT NOT NULL,
    cohort_id TEXT,
    budget_scope TEXT NOT NULL,
    query_type TEXT NOT NULL,
    mechanism TEXT NOT NULL,
    epsilon DOUBLE PRECISION NOT NULL CHECK (epsilon > 0),
    delta DOUBLE PRECISION CHECK (delta IS NULL OR (delta > 0 AND delta < 1)),
    sensitivity DOUBLE PRECISION NOT NULL CHECK (sensitivity > 0),
    budget_before DOUBLE PRECISION NOT NULL CHECK (budget_before >= 0),
    budget_after DOUBLE PRECISION NOT NULL CHECK (budget_after >= 0),
    max_budget DOUBLE PRECISION NOT NULL CHECK (max_budget > 0),
    decision TEXT NOT NULL CHECK (decision IN ('allowed', 'blocked')),
    reason TEXT,
    actor TEXT NOT NULL DEFAULT 'system',
    note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audience_privacy_budget_scope
ON audience_privacy_budget_ledger(tenant_id, budget_scope);

CREATE INDEX IF NOT EXISTS idx_audience_privacy_budget_run
ON audience_privacy_budget_ledger(tenant_id, run_id);

CREATE INDEX IF NOT EXISTS idx_audience_privacy_budget_decision
ON audience_privacy_budget_ledger(decision);


-- Approval/run-history support tables.
-- These are used by the approval/audit workflow connected to Module 1 export safety.

CREATE TABLE IF NOT EXISTS audience_run_history (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL CHECK (
        tenant_id ~ '^[a-z0-9][a-z0-9_.-]{0,127}$'
    ),
    run_id TEXT NOT NULL,
    prompt TEXT,
    status TEXT,
    source_mode TEXT,
    source_rows INTEGER,
    privacy_cohorts INTEGER,
    prompt_selected_cohorts INTEGER,
    exported_cohorts INTEGER,
    approval_status TEXT,
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT false,
    filter_mode TEXT,
    locations_detected JSONB NOT NULL DEFAULT '[]'::jsonb,
    poi_terms_detected JSONB NOT NULL DEFAULT '[]'::jsonb,
    dayparts_detected JSONB NOT NULL DEFAULT '[]'::jsonb,
    coverage_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    prompt_filter_report JSONB NOT NULL DEFAULT '{}'::jsonb,
    sensitive_poi_privacy_risk JSONB NOT NULL DEFAULT '{}'::jsonb,
    hybrid_retrieval_intelligence JSONB NOT NULL DEFAULT '{}'::jsonb,
    privacy_guarantees JSONB NOT NULL DEFAULT '{}'::jsonb,
    v2_autonomous JSONB NOT NULL DEFAULT '{}'::jsonb,
    v2_swarm_review JSONB NOT NULL DEFAULT '{}'::jsonb,
    safe_export JSONB NOT NULL DEFAULT '{}'::jsonb,
    final_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    run_dir TEXT,
    final_summary_path TEXT,
    business_summary_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_audience_run_history_run
ON audience_run_history(tenant_id, run_id);

CREATE INDEX IF NOT EXISTS idx_audience_run_history_approval
ON audience_run_history(tenant_id, approval_status);

CREATE INDEX IF NOT EXISTS idx_audience_run_history_created
ON audience_run_history(tenant_id, created_at DESC);


CREATE TABLE IF NOT EXISTS audience_run_cohorts (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL CHECK (
        tenant_id ~ '^[a-z0-9][a-z0-9_.-]{0,127}$'
    ),
    run_id TEXT NOT NULL,
    export_cohort_id TEXT NOT NULL,
    audience_name TEXT,
    location_name TEXT,
    primary_poi_type TEXT,
    created_day_part TEXT,
    lookback_bucket TEXT,
    quality_score DOUBLE PRECISION,
    management_quality_score DOUBLE PRECISION,
    approval_status TEXT,
    privacy_mode TEXT,
    data_safety_status TEXT,
    risk_decision TEXT,
    risk_level TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, run_id, export_cohort_id),
    FOREIGN KEY (tenant_id, run_id)
    REFERENCES audience_run_history(tenant_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_audience_run_cohorts_run
ON audience_run_cohorts(tenant_id, run_id);

CREATE INDEX IF NOT EXISTS idx_audience_run_cohorts_cohort
ON audience_run_cohorts(tenant_id, export_cohort_id);


CREATE TABLE IF NOT EXISTS audience_run_artifacts (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL CHECK (
        tenant_id ~ '^[a-z0-9][a-z0-9_.-]{0,127}$'
    ),
    run_id TEXT NOT NULL,
    artifact_type TEXT,
    artifact_path TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (tenant_id, run_id)
    REFERENCES audience_run_history(tenant_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_audience_run_artifacts_run
ON audience_run_artifacts(tenant_id, run_id);


CREATE TABLE IF NOT EXISTS audience_run_warnings (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL CHECK (
        tenant_id ~ '^[a-z0-9][a-z0-9_.-]{0,127}$'
    ),
    run_id TEXT NOT NULL,
    warning_type TEXT,
    warning_message TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (tenant_id, run_id)
    REFERENCES audience_run_history(tenant_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_audience_run_warnings_run
ON audience_run_warnings(tenant_id, run_id);


CREATE TABLE IF NOT EXISTS audience_run_events (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL CHECK (
        tenant_id ~ '^[a-z0-9][a-z0-9_.-]{0,127}$'
    ),
    run_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL DEFAULT 'system',
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (tenant_id, run_id)
    REFERENCES audience_run_history(tenant_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_audience_run_events_run
ON audience_run_events(tenant_id, run_id);

CREATE INDEX IF NOT EXISTS idx_audience_run_events_type
ON audience_run_events(event_type);


CREATE TABLE IF NOT EXISTS audience_run_approvals (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL CHECK (
        tenant_id ~ '^[a-z0-9][a-z0-9_.-]{0,127}$'
    ),
    run_id TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    note TEXT,
    previous_status TEXT,
    new_status TEXT,
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT false,
    privacy_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifacts_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (tenant_id, run_id)
    REFERENCES audience_run_history(tenant_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_audience_run_approvals_run
ON audience_run_approvals(tenant_id, run_id);

-- Every connection must set the transaction-local tenant before access:
-- SELECT set_config('app.tenant_id', '<canonical-tenant-id>', true);
ALTER TABLE audience_run_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_run_history FORCE ROW LEVEL SECURITY;
ALTER TABLE audience_run_cohorts ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_run_cohorts FORCE ROW LEVEL SECURITY;
ALTER TABLE audience_run_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_run_artifacts FORCE ROW LEVEL SECURITY;
ALTER TABLE audience_run_warnings ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_run_warnings FORCE ROW LEVEL SECURITY;
ALTER TABLE audience_run_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_run_events FORCE ROW LEVEL SECURITY;
ALTER TABLE audience_run_approvals ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_run_approvals FORCE ROW LEVEL SECURITY;
ALTER TABLE audience_privacy_budget_ledger ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_privacy_budget_ledger FORCE ROW LEVEL SECURITY;

CREATE POLICY audience_run_history_tenant_policy
ON audience_run_history
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

CREATE POLICY audience_run_cohorts_tenant_policy
ON audience_run_cohorts
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

CREATE POLICY audience_run_artifacts_tenant_policy
ON audience_run_artifacts
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

CREATE POLICY audience_run_warnings_tenant_policy
ON audience_run_warnings
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

CREATE POLICY audience_run_events_tenant_policy
ON audience_run_events
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

CREATE POLICY audience_run_approvals_tenant_policy
ON audience_run_approvals
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

CREATE POLICY audience_privacy_budget_ledger_tenant_policy
ON audience_privacy_budget_ledger
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

REVOKE ALL ON audience_run_history FROM PUBLIC;
REVOKE ALL ON audience_run_cohorts FROM PUBLIC;
REVOKE ALL ON audience_run_artifacts FROM PUBLIC;
REVOKE ALL ON audience_run_warnings FROM PUBLIC;
REVOKE ALL ON audience_run_events FROM PUBLIC;
REVOKE ALL ON audience_run_approvals FROM PUBLIC;
REVOKE ALL ON audience_privacy_budget_ledger FROM PUBLIC;

COMMIT;
