-- Immutable Module 5.7 legacy/autonomous dual-run comparison evidence.
-- This ledger records minimized comparisons and can never authorize cutover.

CREATE TABLE IF NOT EXISTS bounded_autonomy_dual_run_comparisons (
    tenant_id TEXT NOT NULL,
    shadow_comparison_report_fingerprint CHAR(64) NOT NULL,
    bounded_autonomy_report_fingerprint CHAR(64) NOT NULL,
    final_plan_fingerprint CHAR(64) NOT NULL,
    request_id TEXT NOT NULL,
    goal_id TEXT NOT NULL,
    objective_sha256 CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    comparison_status TEXT NOT NULL CHECK (comparison_status IN (
        'engineering_preview_ready', 'engineering_preview_blocked'
    )),
    legacy_route TEXT NOT NULL,
    autonomous_route TEXT NOT NULL,
    divergence_count INTEGER NOT NULL CHECK (
        divergence_count >= 0 AND divergence_count <= 64
    ),
    critical_divergence_count INTEGER NOT NULL CHECK (
        critical_divergence_count >= 0
        AND critical_divergence_count <= divergence_count
    ),
    eligible_for_human_review BOOLEAN NOT NULL DEFAULT FALSE,
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
    PRIMARY KEY (tenant_id, shadow_comparison_report_fingerprint),
    UNIQUE (tenant_id, request_id, goal_id, final_plan_fingerprint)
);

CREATE OR REPLACE FUNCTION prevent_bounded_autonomy_comparison_mutation()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Bounded autonomy dual-run evidence is immutable';
END;
$$;

DROP TRIGGER IF EXISTS trg_bounded_autonomy_comparison_immutable
ON bounded_autonomy_dual_run_comparisons;
CREATE TRIGGER trg_bounded_autonomy_comparison_immutable
BEFORE UPDATE OR DELETE ON bounded_autonomy_dual_run_comparisons
FOR EACH ROW EXECUTE FUNCTION prevent_bounded_autonomy_comparison_mutation();

ALTER TABLE bounded_autonomy_dual_run_comparisons ENABLE ROW LEVEL SECURITY;
ALTER TABLE bounded_autonomy_dual_run_comparisons FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS bounded_autonomy_comparison_tenant_policy
ON bounded_autonomy_dual_run_comparisons;
CREATE POLICY bounded_autonomy_comparison_tenant_policy
ON bounded_autonomy_dual_run_comparisons
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

REVOKE ALL ON bounded_autonomy_dual_run_comparisons FROM PUBLIC;

CREATE INDEX IF NOT EXISTS idx_bounded_autonomy_comparison_goal
ON bounded_autonomy_dual_run_comparisons (
    tenant_id, goal_id, created_at DESC
);

CREATE INDEX IF NOT EXISTS idx_bounded_autonomy_comparison_divergence
ON bounded_autonomy_dual_run_comparisons (
    tenant_id, critical_divergence_count, divergence_count, created_at DESC
);
