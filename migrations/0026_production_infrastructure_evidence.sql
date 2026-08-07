-- Immutable, tenant-scoped Infrastructure as Code assessment and review evidence.
-- Resource identifiers, credentials, deployment execution and traffic changes are prohibited.

CREATE TABLE IF NOT EXISTS production_infrastructure_assessments (
    tenant_id TEXT NOT NULL,
    infrastructure_assessment_fingerprint CHAR(64) NOT NULL,
    assessment_id TEXT NOT NULL,
    environment_name TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    assessment_status TEXT NOT NULL CHECK (assessment_status IN ('pass', 'fail_closed')),
    control_count INTEGER NOT NULL CHECK (control_count > 0),
    passed_control_count INTEGER NOT NULL CHECK (passed_control_count >= 0),
    failed_control_count INTEGER NOT NULL CHECK (failed_control_count >= 0),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    secret_values_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (secret_values_stored = FALSE),
    credentials_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (credentials_returned = FALSE),
    resource_identifiers_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (resource_identifiers_returned = FALSE),
    environment_values_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (environment_values_returned = FALSE),
    cloud_resources_mutated BOOLEAN NOT NULL DEFAULT FALSE CHECK (cloud_resources_mutated = FALSE),
    change_set_executed BOOLEAN NOT NULL DEFAULT FALSE CHECK (change_set_executed = FALSE),
    automatic_deployment_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_deployment_performed = FALSE),
    production_traffic_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_traffic_enabled = FALSE),
    production_release_authorized BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_release_authorized = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, infrastructure_assessment_fingerprint),
    UNIQUE (tenant_id, assessment_id)
);

CREATE TABLE IF NOT EXISTS production_infrastructure_change_set_reviews (
    tenant_id TEXT NOT NULL,
    infrastructure_review_fingerprint CHAR(64) NOT NULL,
    source_infrastructure_assessment_fingerprint CHAR(64) NOT NULL,
    review_reference TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN (
        'approved_for_preproduction_change_set', 'requires_changes', 'rejected'
    )),
    policy_version TEXT NOT NULL,
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    secret_values_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (secret_values_stored = FALSE),
    credentials_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (credentials_returned = FALSE),
    resource_identifiers_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (resource_identifiers_returned = FALSE),
    environment_values_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (environment_values_returned = FALSE),
    cloud_resources_mutated BOOLEAN NOT NULL DEFAULT FALSE CHECK (cloud_resources_mutated = FALSE),
    change_set_executed BOOLEAN NOT NULL DEFAULT FALSE CHECK (change_set_executed = FALSE),
    automatic_deployment_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_deployment_performed = FALSE),
    production_traffic_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_traffic_enabled = FALSE),
    production_release_authorized BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_release_authorized = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, infrastructure_review_fingerprint),
    FOREIGN KEY (tenant_id, source_infrastructure_assessment_fingerprint)
    REFERENCES production_infrastructure_assessments (
        tenant_id, infrastructure_assessment_fingerprint
    )
);

CREATE OR REPLACE FUNCTION prevent_production_infrastructure_evidence_mutation()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Production infrastructure evidence is immutable';
END;
$$;

DO $$
DECLARE table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'production_infrastructure_assessments',
        'production_infrastructure_change_set_reviews'
    ] LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS trg_infrastructure_evidence_immutable ON %I', table_name);
        EXECUTE format(
            'CREATE TRIGGER trg_infrastructure_evidence_immutable BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION prevent_production_infrastructure_evidence_mutation()',
            table_name
        );
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
        EXECUTE format('DROP POLICY IF EXISTS infrastructure_tenant_policy ON %I', table_name);
        EXECUTE format(
            'CREATE POLICY infrastructure_tenant_policy ON %I USING (tenant_id = current_setting(''app.tenant_id'', true)) WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))',
            table_name
        );
        EXECUTE format('REVOKE ALL ON %I FROM PUBLIC', table_name);
    END LOOP;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_production_infrastructure_assessment_status
ON production_infrastructure_assessments (
    tenant_id, assessment_status, created_at DESC
);
CREATE INDEX IF NOT EXISTS idx_production_infrastructure_review_source
ON production_infrastructure_change_set_reviews (
    tenant_id, source_infrastructure_assessment_fingerprint, created_at DESC
);
