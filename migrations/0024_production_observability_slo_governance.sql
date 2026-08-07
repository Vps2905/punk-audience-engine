-- Immutable aggregate telemetry, SLO and incident-review evidence.
-- Payloads, sensitive attributes, dispatch and remediation are prohibited.

CREATE TABLE IF NOT EXISTS production_observability_snapshots (
    tenant_id TEXT NOT NULL,
    observability_snapshot_fingerprint CHAR(64) NOT NULL,
    snapshot_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    telemetry_coverage_count INTEGER NOT NULL CHECK (telemetry_coverage_count > 0),
    service_level_count INTEGER NOT NULL CHECK (service_level_count > 0),
    warning_count INTEGER NOT NULL CHECK (warning_count >= 0),
    critical_count INTEGER NOT NULL CHECK (critical_count >= 0),
    overall_status TEXT NOT NULL CHECK (overall_status IN (
        'healthy', 'at_risk', 'warning', 'critical', 'insufficient_data'
    )),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    aggregate_telemetry_only BOOLEAN NOT NULL DEFAULT TRUE CHECK (aggregate_telemetry_only = TRUE),
    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (raw_identifiers_stored = FALSE),
    prompt_content_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (prompt_content_stored = FALSE),
    request_payload_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (request_payload_stored = FALSE),
    response_payload_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (response_payload_stored = FALSE),
    sensitive_attributes_exported BOOLEAN NOT NULL DEFAULT FALSE CHECK (sensitive_attributes_exported = FALSE),
    external_alert_dispatched BOOLEAN NOT NULL DEFAULT FALSE CHECK (external_alert_dispatched = FALSE),
    automatic_remediation_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_remediation_performed = FALSE),
    incident_declared_automatically BOOLEAN NOT NULL DEFAULT FALSE CHECK (incident_declared_automatically = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    production_routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_routing_enabled = FALSE),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (activation_or_export_performed = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, observability_snapshot_fingerprint),
    UNIQUE (tenant_id, snapshot_id)
);

CREATE TABLE IF NOT EXISTS production_incident_review_plans (
    tenant_id TEXT NOT NULL,
    incident_review_plan_fingerprint CHAR(64) NOT NULL,
    source_observability_snapshot_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    action_count INTEGER NOT NULL CHECK (action_count > 0),
    review_required_count INTEGER NOT NULL CHECK (review_required_count >= 0),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    aggregate_telemetry_only BOOLEAN NOT NULL DEFAULT TRUE CHECK (aggregate_telemetry_only = TRUE),
    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (raw_identifiers_stored = FALSE),
    prompt_content_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (prompt_content_stored = FALSE),
    request_payload_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (request_payload_stored = FALSE),
    response_payload_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (response_payload_stored = FALSE),
    sensitive_attributes_exported BOOLEAN NOT NULL DEFAULT FALSE CHECK (sensitive_attributes_exported = FALSE),
    external_alert_dispatched BOOLEAN NOT NULL DEFAULT FALSE CHECK (external_alert_dispatched = FALSE),
    automatic_remediation_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_remediation_performed = FALSE),
    incident_declared_automatically BOOLEAN NOT NULL DEFAULT FALSE CHECK (incident_declared_automatically = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    production_routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_routing_enabled = FALSE),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (activation_or_export_performed = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, incident_review_plan_fingerprint),
    FOREIGN KEY (tenant_id, source_observability_snapshot_fingerprint)
    REFERENCES production_observability_snapshots (
        tenant_id, observability_snapshot_fingerprint
    )
);

CREATE OR REPLACE FUNCTION prevent_production_observability_evidence_mutation()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Production observability evidence is immutable';
END;
$$;

DO $$
DECLARE table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'production_observability_snapshots',
        'production_incident_review_plans'
    ] LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS trg_observability_evidence_immutable ON %I', table_name);
        EXECUTE format(
            'CREATE TRIGGER trg_observability_evidence_immutable BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION prevent_production_observability_evidence_mutation()',
            table_name
        );
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
        EXECUTE format('DROP POLICY IF EXISTS observability_tenant_policy ON %I', table_name);
        EXECUTE format(
            'CREATE POLICY observability_tenant_policy ON %I USING (tenant_id = current_setting(''app.tenant_id'', true)) WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))',
            table_name
        );
        EXECUTE format('REVOKE ALL ON %I FROM PUBLIC', table_name);
    END LOOP;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_production_observability_status
ON production_observability_snapshots (tenant_id, overall_status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_production_incident_review_source
ON production_incident_review_plans (
    tenant_id, source_observability_snapshot_fingerprint, created_at DESC
);
