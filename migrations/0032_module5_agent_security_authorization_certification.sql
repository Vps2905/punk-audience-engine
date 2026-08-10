-- Immutable Module 5.10 agent authorization certification evidence.
-- This evidence never grants production identity, approval, delivery or cutover.

CREATE TABLE IF NOT EXISTS module5_agent_security_certifications (
    tenant_id TEXT NOT NULL,
    agent_security_certification_fingerprint CHAR(64) NOT NULL,
    source_functional_shadow_report_fingerprint CHAR(64) NOT NULL,
    source_security_posture_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    certification_status TEXT NOT NULL CHECK (certification_status IN (
        'engineering_preview_ready', 'engineering_preview_blocked'
    )),
    scenario_count INTEGER NOT NULL CHECK (
        scenario_count >= 0 AND scenario_count <= 100
    ),
    passed_scenario_count INTEGER NOT NULL CHECK (
        passed_scenario_count >= 0
        AND passed_scenario_count <= scenario_count
    ),
    failed_scenario_count INTEGER NOT NULL CHECK (
        failed_scenario_count >= 0
        AND failed_scenario_count <= scenario_count
        AND passed_scenario_count + failed_scenario_count = scenario_count
    ),
    eligible_for_staging_review BOOLEAN NOT NULL DEFAULT FALSE,
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    credentials_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        credentials_stored = FALSE
    ),
    authentication_tokens_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        authentication_tokens_stored = FALSE
    ),
    prompt_content_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        prompt_content_stored = FALSE
    ),
    tool_arguments_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        tool_arguments_stored = FALSE
    ),
    raw_identifiers_returned BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        raw_identifiers_returned = FALSE
    ),
    cross_tenant_access_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        cross_tenant_access_performed = FALSE
    ),
    automatic_approval_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        automatic_approval_performed = FALSE
    ),
    production_identity_provider_certified BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        production_identity_provider_certified = FALSE
    ),
    external_penetration_test_completed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        external_penetration_test_completed = FALSE
    ),
    live_cutover_authorized BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        live_cutover_authorized = FALSE
    ),
    production_effect_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        production_effect_performed = FALSE
    ),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        activation_or_export_performed = FALSE
    ),
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        downstream_export_enabled = FALSE
    ),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (
        manual_approval_required = TRUE
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, agent_security_certification_fingerprint)
);

CREATE OR REPLACE FUNCTION prevent_module5_agent_security_certification_mutation()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Module 5 agent security certification evidence is immutable';
END;
$$;

DROP TRIGGER IF EXISTS trg_module5_agent_security_certification_immutable
ON module5_agent_security_certifications;
CREATE TRIGGER trg_module5_agent_security_certification_immutable
BEFORE UPDATE OR DELETE ON module5_agent_security_certifications
FOR EACH ROW EXECUTE FUNCTION prevent_module5_agent_security_certification_mutation();

ALTER TABLE module5_agent_security_certifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE module5_agent_security_certifications FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS module5_agent_security_certification_tenant_policy
ON module5_agent_security_certifications;
CREATE POLICY module5_agent_security_certification_tenant_policy
ON module5_agent_security_certifications
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

REVOKE ALL ON module5_agent_security_certifications FROM PUBLIC;

CREATE INDEX IF NOT EXISTS idx_module5_agent_security_certification_status
ON module5_agent_security_certifications (
    tenant_id, certification_status, created_at DESC
);
