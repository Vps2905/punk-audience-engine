-- Immutable, tenant-scoped evidence for measured preproduction deployments.
-- This ledger never stores AWS resource identifiers, secret values or data rows.

CREATE TABLE IF NOT EXISTS preproduction_deployment_certifications (
    tenant_id TEXT NOT NULL,
    deployment_certification_fingerprint CHAR(64) NOT NULL,
    infrastructure_review_fingerprint CHAR(64) NOT NULL,
    certification_id TEXT NOT NULL,
    environment_name TEXT NOT NULL,
    candidate_image_digest TEXT NOT NULL CHECK (
        candidate_image_digest ~ '^sha256:[0-9a-f]{64}$'
    ),
    expected_migration_head TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    certification_status TEXT NOT NULL CHECK (certification_status IN (
        'preproduction_deployment_certified',
        'preproduction_deployment_certification_failed_closed'
    )),
    control_count INTEGER NOT NULL CHECK (control_count > 0),
    passed_control_count INTEGER NOT NULL CHECK (passed_control_count >= 0),
    failed_control_count INTEGER NOT NULL CHECK (failed_control_count >= 0),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    inspection_read_only BOOLEAN NOT NULL DEFAULT TRUE CHECK (
        inspection_read_only = TRUE
    ),
    secret_values_read BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        secret_values_read = FALSE
    ),
    secret_values_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        secret_values_returned = FALSE
    ),
    resource_identifiers_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        resource_identifiers_returned = FALSE
    ),
    data_rows_read BOOLEAN NOT NULL DEFAULT FALSE CHECK (data_rows_read = FALSE),
    live_provider_data_required BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        live_provider_data_required = FALSE
    ),
    production_resources_mutated BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        production_resources_mutated = FALSE
    ),
    production_traffic_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        production_traffic_enabled = FALSE
    ),
    audience_activation_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        audience_activation_performed = FALSE
    ),
    downstream_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        downstream_export_performed = FALSE
    ),
    live_production_certified BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        live_production_certified = FALSE
    ),
    manual_production_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (
        manual_production_approval_required = TRUE
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, deployment_certification_fingerprint),
    UNIQUE (tenant_id, certification_id),
    FOREIGN KEY (tenant_id, infrastructure_review_fingerprint)
    REFERENCES production_infrastructure_change_set_reviews (
        tenant_id, infrastructure_review_fingerprint
    )
);

CREATE OR REPLACE FUNCTION prevent_preproduction_deployment_evidence_mutation()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Preproduction deployment certification evidence is immutable';
END;
$$;

DROP TRIGGER IF EXISTS trg_preproduction_deployment_evidence_immutable
ON preproduction_deployment_certifications;
CREATE TRIGGER trg_preproduction_deployment_evidence_immutable
BEFORE UPDATE OR DELETE ON preproduction_deployment_certifications
FOR EACH ROW EXECUTE FUNCTION prevent_preproduction_deployment_evidence_mutation();

ALTER TABLE preproduction_deployment_certifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE preproduction_deployment_certifications FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS preproduction_deployment_tenant_policy
ON preproduction_deployment_certifications;
CREATE POLICY preproduction_deployment_tenant_policy
ON preproduction_deployment_certifications
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

REVOKE ALL ON preproduction_deployment_certifications FROM PUBLIC;

CREATE INDEX IF NOT EXISTS idx_preproduction_deployment_review
ON preproduction_deployment_certifications (
    tenant_id, infrastructure_review_fingerprint, created_at DESC
);
