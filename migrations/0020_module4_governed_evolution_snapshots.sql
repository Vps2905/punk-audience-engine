-- Module 4.1 immutable aggregate cohort evolution snapshots.
-- No membership, automatic evolution, routing, activation, or export.

CREATE TABLE IF NOT EXISTS audience_evolution_snapshot_runs (
    tenant_id TEXT NOT NULL,
    snapshot_fingerprint CHAR(64) NOT NULL,
    source_run_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    execution_mode TEXT NOT NULL CHECK (execution_mode IN (
        'historical_preview', 'offline_evaluation', 'production'
    )),
    source_cohort_count INTEGER NOT NULL CHECK (source_cohort_count > 0),
    monitoring_count INTEGER NOT NULL CHECK (monitoring_count >= 0),
    review_required_count INTEGER NOT NULL CHECK (review_required_count >= 0),
    paused_count INTEGER NOT NULL CHECK (paused_count >= 0),
    blocked_count INTEGER NOT NULL CHECK (blocked_count >= 0),
    historical_count INTEGER NOT NULL CHECK (historical_count >= 0),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        raw_identifiers_stored = FALSE
    ),
    audience_membership_read BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        audience_membership_read = FALSE
    ),
    individual_behavior_inferred BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        individual_behavior_inferred = FALSE
    ),
    cohort_lifecycle_mutated BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        cohort_lifecycle_mutated = FALSE
    ),
    automatic_evolution_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        automatic_evolution_performed = FALSE
    ),
    automatic_approval_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        automatic_approval_performed = FALSE
    ),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (
        manual_approval_required = TRUE
    ),
    routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (routing_enabled = FALSE),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        activation_or_export_performed = FALSE
    ),
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        downstream_export_enabled = FALSE
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, snapshot_fingerprint),
    FOREIGN KEY (tenant_id, source_run_id)
    REFERENCES public.audience_run_history (tenant_id, run_id)
);

CREATE TABLE IF NOT EXISTS audience_evolution_cohort_snapshots (
    tenant_id TEXT NOT NULL,
    snapshot_fingerprint CHAR(64) NOT NULL,
    cohort_snapshot_fingerprint CHAR(64) NOT NULL,
    source_run_id TEXT NOT NULL,
    export_cohort_id TEXT NOT NULL,
    quality_score DOUBLE PRECISION NOT NULL CHECK (
        quality_score >= 0.0 AND quality_score <= 1.0
    ),
    freshness_status TEXT NOT NULL,
    approval_status TEXT NOT NULL,
    data_safety_status TEXT NOT NULL,
    risk_decision TEXT NOT NULL,
    monitoring_status TEXT NOT NULL CHECK (monitoring_status IN (
        'monitoring_baseline', 'historical_baseline',
        'review_required_quality', 'paused_stale_source',
        'blocked_policy', 'blocked_approval'
    )),
    reason_codes JSONB NOT NULL CHECK (jsonb_typeof(reason_codes) = 'array'),
    lifecycle_mutated BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        lifecycle_mutated = FALSE
    ),
    automatic_approval_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        automatic_approval_performed = FALSE
    ),
    routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (routing_enabled = FALSE),
    eligible_for_activation BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        eligible_for_activation = FALSE
    ),
    eligible_for_export BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        eligible_for_export = FALSE
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, cohort_snapshot_fingerprint),
    UNIQUE (tenant_id, snapshot_fingerprint, export_cohort_id),
    FOREIGN KEY (tenant_id, snapshot_fingerprint)
    REFERENCES audience_evolution_snapshot_runs (
        tenant_id, snapshot_fingerprint
    )
);

CREATE OR REPLACE FUNCTION prevent_module4_evolution_snapshot_mutation()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Module 4.1 evolution snapshot evidence is immutable';
END;
$$;

DROP TRIGGER IF EXISTS trg_module4_evolution_runs_immutable
ON audience_evolution_snapshot_runs;
CREATE TRIGGER trg_module4_evolution_runs_immutable
BEFORE UPDATE OR DELETE ON audience_evolution_snapshot_runs
FOR EACH ROW EXECUTE FUNCTION prevent_module4_evolution_snapshot_mutation();

DROP TRIGGER IF EXISTS trg_module4_evolution_cohorts_immutable
ON audience_evolution_cohort_snapshots;
CREATE TRIGGER trg_module4_evolution_cohorts_immutable
BEFORE UPDATE OR DELETE ON audience_evolution_cohort_snapshots
FOR EACH ROW EXECUTE FUNCTION prevent_module4_evolution_snapshot_mutation();

CREATE INDEX IF NOT EXISTS idx_evolution_runs_source
ON audience_evolution_snapshot_runs (tenant_id, source_run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evolution_cohorts_monitoring
ON audience_evolution_cohort_snapshots (
    tenant_id, monitoring_status, source_run_id
);

ALTER TABLE audience_evolution_snapshot_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_evolution_snapshot_runs FORCE ROW LEVEL SECURITY;
ALTER TABLE audience_evolution_cohort_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_evolution_cohort_snapshots FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS audience_evolution_runs_tenant_policy
ON audience_evolution_snapshot_runs;
CREATE POLICY audience_evolution_runs_tenant_policy
ON audience_evolution_snapshot_runs
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
DROP POLICY IF EXISTS audience_evolution_cohorts_tenant_policy
ON audience_evolution_cohort_snapshots;
CREATE POLICY audience_evolution_cohorts_tenant_policy
ON audience_evolution_cohort_snapshots
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

REVOKE ALL ON audience_evolution_snapshot_runs FROM PUBLIC;
REVOKE ALL ON audience_evolution_cohort_snapshots FROM PUBLIC;
