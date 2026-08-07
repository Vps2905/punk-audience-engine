-- Cross-module immutable aggregate quality-monitoring evidence.
-- No table stores row-level data or authorizes remediation, routing or export.

CREATE TABLE IF NOT EXISTS production_quality_snapshots (
    tenant_id TEXT NOT NULL,
    quality_snapshot_fingerprint CHAR(64) NOT NULL,
    snapshot_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    metric_count INTEGER NOT NULL CHECK (metric_count > 0),
    healthy_count INTEGER NOT NULL CHECK (healthy_count >= 0),
    warning_count INTEGER NOT NULL CHECK (warning_count >= 0),
    critical_count INTEGER NOT NULL CHECK (critical_count >= 0),
    overall_health TEXT NOT NULL CHECK (overall_health IN (
        'healthy', 'warning', 'critical'
    )),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    aggregate_metrics_only BOOLEAN NOT NULL DEFAULT TRUE CHECK (aggregate_metrics_only = TRUE),
    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (raw_identifiers_stored = FALSE),
    individual_records_read BOOLEAN NOT NULL DEFAULT FALSE CHECK (individual_records_read = FALSE),
    individual_behavior_inferred BOOLEAN NOT NULL DEFAULT FALSE CHECK (individual_behavior_inferred = FALSE),
    automatic_remediation_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_remediation_performed = FALSE),
    external_alert_dispatched BOOLEAN NOT NULL DEFAULT FALSE CHECK (external_alert_dispatched = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    production_routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_routing_enabled = FALSE),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (activation_or_export_performed = FALSE),
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (downstream_export_enabled = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, quality_snapshot_fingerprint),
    UNIQUE (tenant_id, snapshot_id)
);

CREATE TABLE IF NOT EXISTS production_quality_drift_reports (
    tenant_id TEXT NOT NULL,
    quality_drift_report_fingerprint CHAR(64) NOT NULL,
    baseline_snapshot_fingerprint CHAR(64) NOT NULL,
    current_snapshot_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    metric_count INTEGER NOT NULL CHECK (metric_count > 0),
    warning_count INTEGER NOT NULL CHECK (warning_count >= 0),
    critical_count INTEGER NOT NULL CHECK (critical_count >= 0),
    drift_detected BOOLEAN NOT NULL,
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    aggregate_metrics_only BOOLEAN NOT NULL DEFAULT TRUE CHECK (aggregate_metrics_only = TRUE),
    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (raw_identifiers_stored = FALSE),
    automatic_remediation_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_remediation_performed = FALSE),
    external_alert_dispatched BOOLEAN NOT NULL DEFAULT FALSE CHECK (external_alert_dispatched = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    production_routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_routing_enabled = FALSE),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (activation_or_export_performed = FALSE),
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (downstream_export_enabled = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, quality_drift_report_fingerprint),
    FOREIGN KEY (tenant_id, baseline_snapshot_fingerprint)
    REFERENCES production_quality_snapshots (tenant_id, quality_snapshot_fingerprint),
    FOREIGN KEY (tenant_id, current_snapshot_fingerprint)
    REFERENCES production_quality_snapshots (tenant_id, quality_snapshot_fingerprint),
    CHECK (baseline_snapshot_fingerprint <> current_snapshot_fingerprint)
);

CREATE TABLE IF NOT EXISTS production_quality_alert_plans (
    tenant_id TEXT NOT NULL,
    quality_alert_plan_fingerprint CHAR(64) NOT NULL,
    source_quality_drift_report_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    alert_count INTEGER NOT NULL CHECK (alert_count > 0),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    aggregate_metrics_only BOOLEAN NOT NULL DEFAULT TRUE CHECK (aggregate_metrics_only = TRUE),
    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (raw_identifiers_stored = FALSE),
    automatic_remediation_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_remediation_performed = FALSE),
    external_alert_dispatched BOOLEAN NOT NULL DEFAULT FALSE CHECK (external_alert_dispatched = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    production_routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_routing_enabled = FALSE),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (activation_or_export_performed = FALSE),
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (downstream_export_enabled = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, quality_alert_plan_fingerprint),
    FOREIGN KEY (tenant_id, source_quality_drift_report_fingerprint)
    REFERENCES production_quality_drift_reports (
        tenant_id, quality_drift_report_fingerprint
    )
);

CREATE OR REPLACE FUNCTION prevent_production_quality_evidence_mutation()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Production quality-monitoring evidence is immutable';
END;
$$;

DO $$
DECLARE table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'production_quality_snapshots',
        'production_quality_drift_reports',
        'production_quality_alert_plans'
    ] LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS trg_quality_evidence_immutable ON %I', table_name);
        EXECUTE format(
            'CREATE TRIGGER trg_quality_evidence_immutable BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION prevent_production_quality_evidence_mutation()',
            table_name
        );
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
        EXECUTE format('DROP POLICY IF EXISTS quality_monitoring_tenant_policy ON %I', table_name);
        EXECUTE format(
            'CREATE POLICY quality_monitoring_tenant_policy ON %I USING (tenant_id = current_setting(''app.tenant_id'', true)) WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))',
            table_name
        );
        EXECUTE format('REVOKE ALL ON %I FROM PUBLIC', table_name);
    END LOOP;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_production_quality_snapshot_health
ON production_quality_snapshots (tenant_id, overall_health, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_production_quality_drift_current
ON production_quality_drift_reports (
    tenant_id, current_snapshot_fingerprint, created_at DESC
);
