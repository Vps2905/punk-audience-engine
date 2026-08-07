-- Immutable, tenant-scoped, non-secret security posture and review evidence.
-- Credentials, environment values, automatic remediation and release are prohibited.

CREATE TABLE IF NOT EXISTS production_security_posture_reports (
    tenant_id TEXT NOT NULL,
    security_posture_fingerprint CHAR(64) NOT NULL,
    assessment_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    posture_status TEXT NOT NULL CHECK (posture_status IN ('pass', 'fail_closed')),
    control_count INTEGER NOT NULL CHECK (control_count > 0),
    passed_control_count INTEGER NOT NULL CHECK (passed_control_count >= 0),
    failed_control_count INTEGER NOT NULL CHECK (failed_control_count >= 0),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    secret_values_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (secret_values_stored = FALSE),
    credentials_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (credentials_returned = FALSE),
    database_urls_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (database_urls_returned = FALSE),
    environment_values_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (environment_values_returned = FALSE),
    vulnerability_exploit_attempted BOOLEAN NOT NULL DEFAULT FALSE CHECK (vulnerability_exploit_attempted = FALSE),
    security_configuration_mutated BOOLEAN NOT NULL DEFAULT FALSE CHECK (security_configuration_mutated = FALSE),
    automatic_remediation_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_remediation_performed = FALSE),
    production_release_authorized BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_release_authorized = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, security_posture_fingerprint),
    UNIQUE (tenant_id, assessment_id)
);

CREATE TABLE IF NOT EXISTS production_security_manual_reviews (
    tenant_id TEXT NOT NULL,
    security_review_fingerprint CHAR(64) NOT NULL,
    source_security_posture_fingerprint CHAR(64) NOT NULL,
    review_reference TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN (
        'approved_for_preproduction_review', 'needs_remediation', 'rejected'
    )),
    policy_version TEXT NOT NULL,
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    secret_values_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (secret_values_stored = FALSE),
    credentials_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (credentials_returned = FALSE),
    database_urls_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (database_urls_returned = FALSE),
    environment_values_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (environment_values_returned = FALSE),
    vulnerability_exploit_attempted BOOLEAN NOT NULL DEFAULT FALSE CHECK (vulnerability_exploit_attempted = FALSE),
    security_configuration_mutated BOOLEAN NOT NULL DEFAULT FALSE CHECK (security_configuration_mutated = FALSE),
    automatic_remediation_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_remediation_performed = FALSE),
    production_release_authorized BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_release_authorized = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, security_review_fingerprint),
    FOREIGN KEY (tenant_id, source_security_posture_fingerprint)
    REFERENCES production_security_posture_reports (
        tenant_id, security_posture_fingerprint
    )
);

CREATE OR REPLACE FUNCTION prevent_production_security_evidence_mutation()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Production security evidence is immutable';
END;
$$;

DO $$
DECLARE table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'production_security_posture_reports',
        'production_security_manual_reviews'
    ] LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS trg_security_evidence_immutable ON %I', table_name);
        EXECUTE format(
            'CREATE TRIGGER trg_security_evidence_immutable BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION prevent_production_security_evidence_mutation()',
            table_name
        );
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
        EXECUTE format('DROP POLICY IF EXISTS security_tenant_policy ON %I', table_name);
        EXECUTE format(
            'CREATE POLICY security_tenant_policy ON %I USING (tenant_id = current_setting(''app.tenant_id'', true)) WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))',
            table_name
        );
        EXECUTE format('REVOKE ALL ON %I FROM PUBLIC', table_name);
    END LOOP;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_production_security_posture_status
ON production_security_posture_reports (tenant_id, posture_status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_production_security_review_source
ON production_security_manual_reviews (
    tenant_id, source_security_posture_fingerprint, created_at DESC
);
