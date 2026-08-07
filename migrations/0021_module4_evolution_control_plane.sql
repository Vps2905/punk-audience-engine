-- Module 4.2-4.5 immutable audience evolution control-plane evidence.
-- All records are review evidence. No table enables mutation or routing.

CREATE TABLE IF NOT EXISTS audience_evolution_drift_reports (
    tenant_id TEXT NOT NULL,
    drift_report_fingerprint CHAR(64) NOT NULL,
    baseline_snapshot_fingerprint CHAR(64) NOT NULL,
    current_snapshot_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    cohort_count INTEGER NOT NULL CHECK (cohort_count > 0),
    changed_count INTEGER NOT NULL CHECK (changed_count >= 0),
    critical_count INTEGER NOT NULL CHECK (critical_count >= 0),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (raw_identifiers_stored = FALSE),
    audience_membership_read BOOLEAN NOT NULL DEFAULT FALSE CHECK (audience_membership_read = FALSE),
    individual_behavior_inferred BOOLEAN NOT NULL DEFAULT FALSE CHECK (individual_behavior_inferred = FALSE),
    cohort_lifecycle_mutated BOOLEAN NOT NULL DEFAULT FALSE CHECK (cohort_lifecycle_mutated = FALSE),
    automatic_evolution_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_evolution_performed = FALSE),
    automatic_approval_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_approval_performed = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (routing_enabled = FALSE),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (activation_or_export_performed = FALSE),
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (downstream_export_enabled = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, drift_report_fingerprint),
    FOREIGN KEY (tenant_id, baseline_snapshot_fingerprint)
    REFERENCES audience_evolution_snapshot_runs (tenant_id, snapshot_fingerprint),
    FOREIGN KEY (tenant_id, current_snapshot_fingerprint)
    REFERENCES audience_evolution_snapshot_runs (tenant_id, snapshot_fingerprint),
    CHECK (baseline_snapshot_fingerprint <> current_snapshot_fingerprint)
);

CREATE TABLE IF NOT EXISTS audience_evolution_recommendation_reports (
    tenant_id TEXT NOT NULL,
    recommendation_report_fingerprint CHAR(64) NOT NULL,
    source_drift_report_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    recommendation_count INTEGER NOT NULL CHECK (recommendation_count > 0),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    automatic_evolution_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_evolution_performed = FALSE),
    automatic_approval_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_approval_performed = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (routing_enabled = FALSE),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (activation_or_export_performed = FALSE),
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (downstream_export_enabled = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, recommendation_report_fingerprint),
    FOREIGN KEY (tenant_id, source_drift_report_fingerprint)
    REFERENCES audience_evolution_drift_reports (tenant_id, drift_report_fingerprint)
);

CREATE TABLE IF NOT EXISTS audience_evolution_recommendations (
    tenant_id TEXT NOT NULL,
    recommendation_report_fingerprint CHAR(64) NOT NULL,
    recommendation_fingerprint CHAR(64) NOT NULL,
    export_cohort_id TEXT NOT NULL,
    recommended_action TEXT NOT NULL CHECK (recommended_action IN (
        'onboarding_review', 'rollback_review', 'pause_and_review',
        'investigate', 'maintain_monitoring'
    )),
    reason_codes JSONB NOT NULL CHECK (jsonb_typeof(reason_codes) = 'array'),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    lifecycle_mutated BOOLEAN NOT NULL DEFAULT FALSE CHECK (lifecycle_mutated = FALSE),
    routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (routing_enabled = FALSE),
    eligible_for_activation BOOLEAN NOT NULL DEFAULT FALSE CHECK (eligible_for_activation = FALSE),
    eligible_for_export BOOLEAN NOT NULL DEFAULT FALSE CHECK (eligible_for_export = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, recommendation_fingerprint),
    FOREIGN KEY (tenant_id, recommendation_report_fingerprint)
    REFERENCES audience_evolution_recommendation_reports (
        tenant_id, recommendation_report_fingerprint
    )
);

CREATE TABLE IF NOT EXISTS audience_evolution_approval_shadow_reports (
    tenant_id TEXT NOT NULL,
    approval_shadow_report_fingerprint CHAR(64) NOT NULL,
    source_recommendation_report_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    manual_review_count INTEGER NOT NULL CHECK (manual_review_count > 0),
    shadow_observation_count INTEGER NOT NULL CHECK (shadow_observation_count >= 0),
    shadow_validation_passed BOOLEAN NOT NULL DEFAULT FALSE,
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    automatic_evolution_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_evolution_performed = FALSE),
    automatic_approval_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_approval_performed = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (routing_enabled = FALSE),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (activation_or_export_performed = FALSE),
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (downstream_export_enabled = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, approval_shadow_report_fingerprint),
    FOREIGN KEY (tenant_id, source_recommendation_report_fingerprint)
    REFERENCES audience_evolution_recommendation_reports (
        tenant_id, recommendation_report_fingerprint
    )
);

CREATE TABLE IF NOT EXISTS audience_evolution_recovery_reports (
    tenant_id TEXT NOT NULL,
    recovery_report_fingerprint CHAR(64) NOT NULL,
    source_approval_shadow_report_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    recovery_plan_count INTEGER NOT NULL CHECK (recovery_plan_count > 0),
    recovery_coverage_complete BOOLEAN NOT NULL CHECK (recovery_coverage_complete = TRUE),
    recovery_executed BOOLEAN NOT NULL DEFAULT FALSE CHECK (recovery_executed = FALSE),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    automatic_evolution_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_evolution_performed = FALSE),
    automatic_approval_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_approval_performed = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (routing_enabled = FALSE),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (activation_or_export_performed = FALSE),
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (downstream_export_enabled = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, recovery_report_fingerprint),
    FOREIGN KEY (tenant_id, source_approval_shadow_report_fingerprint)
    REFERENCES audience_evolution_approval_shadow_reports (
        tenant_id, approval_shadow_report_fingerprint
    )
);

CREATE OR REPLACE FUNCTION prevent_module4_evolution_control_evidence_mutation()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Module 4 evolution control evidence is immutable';
END;
$$;

DO $$
DECLARE table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'audience_evolution_drift_reports',
        'audience_evolution_recommendation_reports',
        'audience_evolution_recommendations',
        'audience_evolution_approval_shadow_reports',
        'audience_evolution_recovery_reports'
    ] LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS trg_module4_control_immutable ON %I', table_name);
        EXECUTE format(
            'CREATE TRIGGER trg_module4_control_immutable BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION prevent_module4_evolution_control_evidence_mutation()',
            table_name
        );
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
        EXECUTE format('DROP POLICY IF EXISTS module4_control_tenant_policy ON %I', table_name);
        EXECUTE format(
            'CREATE POLICY module4_control_tenant_policy ON %I USING (tenant_id = current_setting(''app.tenant_id'', true)) WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))',
            table_name
        );
        EXECUTE format('REVOKE ALL ON %I FROM PUBLIC', table_name);
    END LOOP;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_evolution_drift_current
ON audience_evolution_drift_reports (tenant_id, current_snapshot_fingerprint, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evolution_recommendations_action
ON audience_evolution_recommendations (tenant_id, recommended_action, export_cohort_id);
