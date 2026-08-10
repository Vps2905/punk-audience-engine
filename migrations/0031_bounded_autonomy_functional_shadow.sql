-- Immutable Module 5.9 functional shadow evidence.
-- FUNCTIONAL shadow evidence is immutable and never authorizes production.

CREATE TABLE IF NOT EXISTS bounded_autonomy_functional_shadows (
    tenant_id TEXT NOT NULL,
    functional_shadow_report_fingerprint CHAR(64) NOT NULL,
    source_certification_report_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    functional_status TEXT NOT NULL CHECK (functional_status IN (
        'engineering_preview_ready', 'engineering_preview_blocked'
    )),
    stage_count INTEGER NOT NULL CHECK (
        stage_count >= 0 AND stage_count <= 32
    ),
    failed_stage_count INTEGER NOT NULL CHECK (
        failed_stage_count >= 0 AND failed_stage_count <= stage_count
    ),
    overall_divergence_count INTEGER NOT NULL CHECK (
        overall_divergence_count >= 0 AND overall_divergence_count <= 32
    ),
    critical_divergence_count INTEGER NOT NULL CHECK (
        critical_divergence_count >= 0
        AND critical_divergence_count <= overall_divergence_count
    ),
    total_latency_ms DOUBLE PRECISION NOT NULL CHECK (
        total_latency_ms >= 0.0 AND total_latency_ms <= 3600000.0
    ),
    peak_python_bytes BIGINT NOT NULL CHECK (
        peak_python_bytes >= 0 AND peak_python_bytes <= 16000000000
    ),
    eligible_for_staging_review BOOLEAN NOT NULL DEFAULT FALSE,
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    shadow_only BOOLEAN NOT NULL DEFAULT TRUE CHECK (shadow_only = TRUE),
    isolated_historical_snapshot_only BOOLEAN NOT NULL DEFAULT TRUE CHECK (
        isolated_historical_snapshot_only = TRUE
    ),
    read_only_feature_access BOOLEAN NOT NULL DEFAULT TRUE CHECK (
        read_only_feature_access = TRUE
    ),
    prompt_content_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        prompt_content_stored = FALSE
    ),
    query_text_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        query_text_stored = FALSE
    ),
    tool_arguments_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        tool_arguments_stored = FALSE
    ),
    tool_results_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        tool_results_stored = FALSE
    ),
    full_service_outputs_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        full_service_outputs_stored = FALSE
    ),
    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        raw_identifiers_stored = FALSE
    ),
    database_write_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        database_write_performed = FALSE
    ),
    live_cutover_authorized BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        live_cutover_authorized = FALSE
    ),
    fresh_data_certified BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        fresh_data_certified = FALSE
    ),
    automatic_cutover_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        automatic_cutover_performed = FALSE
    ),
    production_routing_changed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        production_routing_changed = FALSE
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
    PRIMARY KEY (tenant_id, functional_shadow_report_fingerprint)
);

CREATE OR REPLACE FUNCTION prevent_bounded_autonomy_functional_shadow_mutation()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Bounded autonomy FUNCTIONAL shadow evidence is immutable';
END;
$$;

DROP TRIGGER IF EXISTS trg_bounded_autonomy_functional_shadow_immutable
ON bounded_autonomy_functional_shadows;
CREATE TRIGGER trg_bounded_autonomy_functional_shadow_immutable
BEFORE UPDATE OR DELETE ON bounded_autonomy_functional_shadows
FOR EACH ROW EXECUTE FUNCTION prevent_bounded_autonomy_functional_shadow_mutation();

ALTER TABLE bounded_autonomy_functional_shadows ENABLE ROW LEVEL SECURITY;
ALTER TABLE bounded_autonomy_functional_shadows FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS bounded_autonomy_functional_shadow_tenant_policy
ON bounded_autonomy_functional_shadows;
CREATE POLICY bounded_autonomy_functional_shadow_tenant_policy
ON bounded_autonomy_functional_shadows
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

REVOKE ALL ON bounded_autonomy_functional_shadows FROM PUBLIC;

CREATE INDEX IF NOT EXISTS idx_bounded_autonomy_functional_shadow_status
ON bounded_autonomy_functional_shadows (
    tenant_id, functional_status, created_at DESC
);

CREATE INDEX IF NOT EXISTS idx_bounded_autonomy_functional_shadow_latency
ON bounded_autonomy_functional_shadows (
    tenant_id, total_latency_ms, created_at DESC
);
