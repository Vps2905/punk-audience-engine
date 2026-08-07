-- Module 5.1-5.5 immutable governed-agent control-plane evidence.
-- These records never authorize execution, mutation, approval, routing or export.

CREATE TABLE IF NOT EXISTS governed_agent_execution_reports (
    tenant_id TEXT NOT NULL,
    execution_report_fingerprint CHAR(64) NOT NULL,
    request_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    supervisor_route TEXT NOT NULL CHECK (supervisor_route IN (
        'blocked', 'failed', 'needs_clarification',
        'needs_existing_approval', 'pending_approval', 'completed_safe'
    )),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (raw_identifiers_stored = FALSE),
    prompt_content_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (prompt_content_stored = FALSE),
    tool_arguments_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (tool_arguments_stored = FALSE),
    tool_results_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (tool_results_stored = FALSE),
    agent_execution_triggered BOOLEAN NOT NULL DEFAULT FALSE CHECK (agent_execution_triggered = FALSE),
    automatic_mutation_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_mutation_performed = FALSE),
    automatic_approval_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_approval_performed = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    production_routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_routing_enabled = FALSE),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (activation_or_export_performed = FALSE),
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (downstream_export_enabled = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, execution_report_fingerprint),
    UNIQUE (tenant_id, request_id, run_id)
);

CREATE TABLE IF NOT EXISTS governed_agent_policy_reports (
    tenant_id TEXT NOT NULL,
    policy_report_fingerprint CHAR(64) NOT NULL,
    source_execution_report_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    recommended_action TEXT NOT NULL CHECK (recommended_action IN (
        'quarantine_and_review', 'hold_and_review',
        'request_manual_clarification', 'await_existing_approval',
        'await_manual_approval', 'shadow_review_only'
    )),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    production_execution_authorized BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_execution_authorized = FALSE),
    agent_mutation_authorized BOOLEAN NOT NULL DEFAULT FALSE CHECK (agent_mutation_authorized = FALSE),
    automatic_approval_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_approval_performed = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    eligible_for_activation BOOLEAN NOT NULL DEFAULT FALSE CHECK (eligible_for_activation = FALSE),
    eligible_for_export BOOLEAN NOT NULL DEFAULT FALSE CHECK (eligible_for_export = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, policy_report_fingerprint),
    FOREIGN KEY (tenant_id, source_execution_report_fingerprint)
    REFERENCES governed_agent_execution_reports (
        tenant_id, execution_report_fingerprint
    )
);

CREATE TABLE IF NOT EXISTS governed_agent_circuit_breaker_reports (
    tenant_id TEXT NOT NULL,
    circuit_breaker_report_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    execution_report_fingerprints JSONB NOT NULL CHECK (
        jsonb_typeof(execution_report_fingerprints) = 'array'
    ),
    sample_size INTEGER NOT NULL CHECK (sample_size > 0),
    failure_count INTEGER NOT NULL CHECK (failure_count >= 0),
    circuit_state TEXT NOT NULL CHECK (circuit_state IN (
        'closed', 'review_required', 'open'
    )),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    automatic_retry_triggered BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_retry_triggered = FALSE),
    production_execution_authorized BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_execution_authorized = FALSE),
    automatic_mutation_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_mutation_performed = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    production_routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_routing_enabled = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, circuit_breaker_report_fingerprint)
);

CREATE TABLE IF NOT EXISTS governed_agent_human_shadow_review_reports (
    tenant_id TEXT NOT NULL,
    human_review_report_fingerprint CHAR(64) NOT NULL,
    source_policy_report_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    manual_decision TEXT NOT NULL CHECK (manual_decision IN (
        'approved_for_shadow_review', 'rejected', 'needs_review'
    )),
    shadow_validation_passed BOOLEAN NOT NULL DEFAULT FALSE,
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    automatic_approval_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_approval_performed = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    production_routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_routing_enabled = FALSE),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (activation_or_export_performed = FALSE),
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (downstream_export_enabled = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, human_review_report_fingerprint),
    FOREIGN KEY (tenant_id, source_policy_report_fingerprint)
    REFERENCES governed_agent_policy_reports (tenant_id, policy_report_fingerprint)
);

CREATE TABLE IF NOT EXISTS governed_agent_recovery_reports (
    tenant_id TEXT NOT NULL,
    recovery_report_fingerprint CHAR(64) NOT NULL,
    source_circuit_breaker_report_fingerprint CHAR(64) NOT NULL,
    source_human_review_report_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    engineering_evidence_complete BOOLEAN NOT NULL DEFAULT FALSE,
    production_certified BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_certified = FALSE),
    recovery_executed BOOLEAN NOT NULL DEFAULT FALSE CHECK (recovery_executed = FALSE),
    production_execution_authorized BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_execution_authorized = FALSE),
    manual_execution_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_execution_required = TRUE),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    automatic_mutation_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_mutation_performed = FALSE),
    automatic_approval_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (automatic_approval_performed = FALSE),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (manual_approval_required = TRUE),
    production_routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (production_routing_enabled = FALSE),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (activation_or_export_performed = FALSE),
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (downstream_export_enabled = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, recovery_report_fingerprint),
    FOREIGN KEY (tenant_id, source_circuit_breaker_report_fingerprint)
    REFERENCES governed_agent_circuit_breaker_reports (
        tenant_id, circuit_breaker_report_fingerprint
    ),
    FOREIGN KEY (tenant_id, source_human_review_report_fingerprint)
    REFERENCES governed_agent_human_shadow_review_reports (
        tenant_id, human_review_report_fingerprint
    )
);

CREATE OR REPLACE FUNCTION prevent_module5_agent_evidence_mutation()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Module 5 governed-agent evidence is immutable';
END;
$$;

DO $$
DECLARE table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'governed_agent_execution_reports',
        'governed_agent_policy_reports',
        'governed_agent_circuit_breaker_reports',
        'governed_agent_human_shadow_review_reports',
        'governed_agent_recovery_reports'
    ] LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS trg_module5_evidence_immutable ON %I', table_name);
        EXECUTE format(
            'CREATE TRIGGER trg_module5_evidence_immutable BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION prevent_module5_agent_evidence_mutation()',
            table_name
        );
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
        EXECUTE format('DROP POLICY IF EXISTS module5_agent_tenant_policy ON %I', table_name);
        EXECUTE format(
            'CREATE POLICY module5_agent_tenant_policy ON %I USING (tenant_id = current_setting(''app.tenant_id'', true)) WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))',
            table_name
        );
        EXECUTE format('REVOKE ALL ON %I FROM PUBLIC', table_name);
    END LOOP;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_governed_agent_execution_run
ON governed_agent_execution_reports (tenant_id, run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_governed_agent_circuit_state
ON governed_agent_circuit_breaker_reports (tenant_id, circuit_state, created_at DESC);
