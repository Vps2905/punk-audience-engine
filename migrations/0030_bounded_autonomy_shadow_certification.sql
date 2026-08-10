-- Immutable Module 5.8 repeated shadow-certification evidence.
-- Staging-review eligibility never constitutes production authorization.

CREATE TABLE IF NOT EXISTS bounded_autonomy_shadow_certifications (
    tenant_id TEXT NOT NULL,
    bounded_autonomy_certification_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    certification_status TEXT NOT NULL CHECK (certification_status IN (
        'engineering_preview_ready', 'engineering_preview_blocked'
    )),
    submitted_case_count INTEGER NOT NULL CHECK (
        submitted_case_count >= 0 AND submitted_case_count <= 10000
    ),
    evaluated_run_count INTEGER NOT NULL CHECK (
        evaluated_run_count >= 0 AND evaluated_run_count <= 10000
    ),
    unique_goal_hash_count INTEGER NOT NULL CHECK (
        unique_goal_hash_count >= 0 AND unique_goal_hash_count <= 10000
    ),
    terminal_safety_run_count INTEGER NOT NULL CHECK (
        terminal_safety_run_count >= 0 AND terminal_safety_run_count <= 10000
    ),
    nonterminal_review_run_count INTEGER NOT NULL CHECK (
        nonterminal_review_run_count >= 0
        AND nonterminal_review_run_count <= 10000
    ),
    route_agreement_rate DOUBLE PRECISION NOT NULL CHECK (
        route_agreement_rate >= 0.0 AND route_agreement_rate <= 1.0
    ),
    overall_divergence_rate DOUBLE PRECISION NOT NULL CHECK (
        overall_divergence_rate >= 0.0 AND overall_divergence_rate <= 1.0
    ),
    critical_divergence_rate DOUBLE PRECISION NOT NULL CHECK (
        critical_divergence_rate >= 0.0
        AND critical_divergence_rate <= 1.0
    ),
    shadow_p95_latency_ms DOUBLE PRECISION NOT NULL CHECK (
        shadow_p95_latency_ms >= 0.0
        AND shadow_p95_latency_ms <= 3600000.0
    ),
    eligible_for_staging_review BOOLEAN NOT NULL DEFAULT FALSE,
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    shadow_only BOOLEAN NOT NULL DEFAULT TRUE CHECK (shadow_only = TRUE),
    prompt_content_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        prompt_content_stored = FALSE
    ),
    tool_arguments_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        tool_arguments_stored = FALSE
    ),
    tool_results_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        tool_results_stored = FALSE
    ),
    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        raw_identifiers_stored = FALSE
    ),
    live_cutover_authorized BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        live_cutover_authorized = FALSE
    ),
    automatic_cutover_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        automatic_cutover_performed = FALSE
    ),
    production_routing_changed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        production_routing_changed = FALSE
    ),
    fresh_data_certified BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        fresh_data_certified = FALSE
    ),
    automatic_approval_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        automatic_approval_performed = FALSE
    ),
    automatic_mutation_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        automatic_mutation_performed = FALSE
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
    PRIMARY KEY (tenant_id, bounded_autonomy_certification_fingerprint)
);

CREATE OR REPLACE FUNCTION prevent_bounded_autonomy_certification_mutation()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Bounded autonomy certification evidence is immutable';
END;
$$;

DROP TRIGGER IF EXISTS trg_bounded_autonomy_certification_immutable
ON bounded_autonomy_shadow_certifications;
CREATE TRIGGER trg_bounded_autonomy_certification_immutable
BEFORE UPDATE OR DELETE ON bounded_autonomy_shadow_certifications
FOR EACH ROW EXECUTE FUNCTION prevent_bounded_autonomy_certification_mutation();

ALTER TABLE bounded_autonomy_shadow_certifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE bounded_autonomy_shadow_certifications FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS bounded_autonomy_certification_tenant_policy
ON bounded_autonomy_shadow_certifications;
CREATE POLICY bounded_autonomy_certification_tenant_policy
ON bounded_autonomy_shadow_certifications
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

REVOKE ALL ON bounded_autonomy_shadow_certifications FROM PUBLIC;

CREATE INDEX IF NOT EXISTS idx_bounded_autonomy_certification_status
ON bounded_autonomy_shadow_certifications (
    tenant_id, certification_status, created_at DESC
);

CREATE INDEX IF NOT EXISTS idx_bounded_autonomy_certification_latency
ON bounded_autonomy_shadow_certifications (
    tenant_id, shadow_p95_latency_ms, created_at DESC
);
