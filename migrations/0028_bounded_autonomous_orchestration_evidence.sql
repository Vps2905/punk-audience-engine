-- Immutable Module 5.6 bounded-autonomy shadow evidence.
-- Plans contain minimized metadata only and can never authorize production effects.

CREATE TABLE IF NOT EXISTS bounded_autonomy_shadow_reports (
    tenant_id TEXT NOT NULL,
    bounded_autonomy_report_fingerprint CHAR(64) NOT NULL,
    goal_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    objective_sha256 CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    execution_mode TEXT NOT NULL CHECK (execution_mode IN (
        'historical_preview', 'offline_evaluation', 'shadow', 'production'
    )),
    execution_status TEXT NOT NULL CHECK (execution_status IN (
        'completed', 'failed', 'blocked'
    )),
    plan_fingerprint CHAR(64) NOT NULL,
    selected_capability_count INTEGER NOT NULL CHECK (
        selected_capability_count > 0 AND selected_capability_count <= 64
    ),
    total_attempts INTEGER NOT NULL CHECK (
        total_attempts >= 0 AND total_attempts <= 256
    ),
    replan_count INTEGER NOT NULL CHECK (
        replan_count >= 0 AND replan_count <= 5
    ),
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
    prompt_specific_routing_used BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        prompt_specific_routing_used = FALSE
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
    PRIMARY KEY (tenant_id, bounded_autonomy_report_fingerprint),
    UNIQUE (tenant_id, request_id, goal_id, plan_fingerprint)
);

CREATE OR REPLACE FUNCTION prevent_bounded_autonomy_evidence_mutation()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Bounded autonomy shadow evidence is immutable';
END;
$$;

DROP TRIGGER IF EXISTS trg_bounded_autonomy_evidence_immutable
ON bounded_autonomy_shadow_reports;
CREATE TRIGGER trg_bounded_autonomy_evidence_immutable
BEFORE UPDATE OR DELETE ON bounded_autonomy_shadow_reports
FOR EACH ROW EXECUTE FUNCTION prevent_bounded_autonomy_evidence_mutation();

ALTER TABLE bounded_autonomy_shadow_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE bounded_autonomy_shadow_reports FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS bounded_autonomy_tenant_policy
ON bounded_autonomy_shadow_reports;
CREATE POLICY bounded_autonomy_tenant_policy
ON bounded_autonomy_shadow_reports
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

REVOKE ALL ON bounded_autonomy_shadow_reports FROM PUBLIC;

CREATE INDEX IF NOT EXISTS idx_bounded_autonomy_goal
ON bounded_autonomy_shadow_reports (
    tenant_id, goal_id, created_at DESC
);

CREATE INDEX IF NOT EXISTS idx_bounded_autonomy_status
ON bounded_autonomy_shadow_reports (
    tenant_id, execution_status, created_at DESC
);
