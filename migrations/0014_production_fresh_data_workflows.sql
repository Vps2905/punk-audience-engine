-- Production Module 1 -> Module 2 -> Module 3 fresh-data orchestration.
-- Stores only privacy-safe manifests, aggregate feature/candidate evidence,
-- workflow leases, and immutable transition events. It does not enable
-- lookalikes, activation, routing, or export.

CREATE TABLE IF NOT EXISTS audience_fresh_data_workflows (
    tenant_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    request_fingerprint CHAR(64) NOT NULL,
    ingestion_id TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    canonical_source_fingerprint CHAR(64) NOT NULL,
    feature_build_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'claimed',
            'validating_ingestion',
            'reading_canonical',
            'building_features',
            'generating_candidates',
            'analyzing_overlap',
            'awaiting_review',
            'blocked',
            'quarantined',
            'failed'
        )
    ),
    reason_code TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 1 CHECK (attempt_count >= 1),
    lease_owner TEXT,
    lease_expires_at TIMESTAMPTZ,
    request_manifest JSONB NOT NULL CHECK (
        jsonb_typeof(request_manifest) = 'object'
    ),
    feature_receipt JSONB,
    candidate_report JSONB,
    overlap_report JSONB,
    result_receipt JSONB,
    approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (
        approval_required = TRUE
    ),
    activation_requested BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        activation_requested = FALSE
    ),
    export_requested BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        export_requested = FALSE
    ),
    lookalike_generation_requested BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        lookalike_generation_requested = FALSE
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    PRIMARY KEY (tenant_id, workflow_id),
    UNIQUE (tenant_id, request_fingerprint),
    UNIQUE (
        tenant_id,
        ingestion_id,
        canonical_source_fingerprint,
        feature_build_id
    ),
    CHECK (
        (lease_owner IS NULL) = (lease_expires_at IS NULL)
    ),
    CHECK (
        status <> 'awaiting_review'
        OR (
            feature_receipt IS NOT NULL
            AND candidate_report IS NOT NULL
            AND overlap_report IS NOT NULL
            AND result_receipt IS NOT NULL
            AND completed_at IS NOT NULL
            AND lease_owner IS NULL
            AND lease_expires_at IS NULL
        )
    ),
    CHECK (
        status NOT IN ('blocked', 'quarantined', 'failed')
        OR completed_at IS NOT NULL
    )
);

CREATE TABLE IF NOT EXISTS audience_fresh_data_workflow_events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY,
    tenant_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason_code TEXT,
    details JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (
        jsonb_typeof(details) = 'object'
    ),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id),
    FOREIGN KEY (tenant_id, workflow_id)
    REFERENCES audience_fresh_data_workflows (tenant_id, workflow_id)
);

CREATE INDEX IF NOT EXISTS idx_fresh_data_workflows_status
ON audience_fresh_data_workflows (
    tenant_id,
    status,
    updated_at
);

CREATE INDEX IF NOT EXISTS idx_fresh_data_workflows_ingestion
ON audience_fresh_data_workflows (
    tenant_id,
    ingestion_id,
    created_at
);

CREATE INDEX IF NOT EXISTS idx_fresh_data_workflows_lease
ON audience_fresh_data_workflows (
    status,
    lease_expires_at
)
WHERE lease_expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_fresh_data_workflow_events_lookup
ON audience_fresh_data_workflow_events (
    tenant_id,
    workflow_id,
    occurred_at,
    event_id
);

CREATE OR REPLACE FUNCTION prevent_fresh_data_workflow_identity_change()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF
        NEW.tenant_id IS DISTINCT FROM OLD.tenant_id
        OR NEW.workflow_id IS DISTINCT FROM OLD.workflow_id
        OR NEW.request_fingerprint IS DISTINCT FROM OLD.request_fingerprint
        OR NEW.ingestion_id IS DISTINCT FROM OLD.ingestion_id
        OR NEW.provider_id IS DISTINCT FROM OLD.provider_id
        OR NEW.dataset_id IS DISTINCT FROM OLD.dataset_id
        OR NEW.canonical_source_fingerprint
            IS DISTINCT FROM OLD.canonical_source_fingerprint
        OR NEW.feature_build_id IS DISTINCT FROM OLD.feature_build_id
        OR NEW.request_manifest IS DISTINCT FROM OLD.request_manifest
        OR NEW.approval_required IS DISTINCT FROM OLD.approval_required
        OR NEW.activation_requested IS DISTINCT FROM OLD.activation_requested
        OR NEW.export_requested IS DISTINCT FROM OLD.export_requested
        OR NEW.lookalike_generation_requested
            IS DISTINCT FROM OLD.lookalike_generation_requested
    THEN
        RAISE EXCEPTION
            'Fresh-data workflow identity and safety contract are immutable';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_fresh_data_workflow_identity_immutable
ON audience_fresh_data_workflows;

CREATE TRIGGER trg_fresh_data_workflow_identity_immutable
BEFORE UPDATE ON audience_fresh_data_workflows
FOR EACH ROW
EXECUTE FUNCTION prevent_fresh_data_workflow_identity_change();

CREATE OR REPLACE FUNCTION prevent_fresh_data_workflow_event_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION
        'Fresh-data workflow audit events are append-only';
END;
$$;

DROP TRIGGER IF EXISTS trg_fresh_data_workflow_events_immutable
ON audience_fresh_data_workflow_events;

CREATE TRIGGER trg_fresh_data_workflow_events_immutable
BEFORE UPDATE OR DELETE ON audience_fresh_data_workflow_events
FOR EACH ROW
EXECUTE FUNCTION prevent_fresh_data_workflow_event_mutation();

ALTER TABLE audience_fresh_data_workflows
ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_fresh_data_workflows
FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS audience_fresh_data_workflows_tenant_policy
ON audience_fresh_data_workflows;
CREATE POLICY audience_fresh_data_workflows_tenant_policy
ON audience_fresh_data_workflows
USING (
    tenant_id = current_setting('app.tenant_id', true)
)
WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
);

ALTER TABLE audience_fresh_data_workflow_events
ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_fresh_data_workflow_events
FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS audience_fresh_data_workflow_events_tenant_policy
ON audience_fresh_data_workflow_events;
CREATE POLICY audience_fresh_data_workflow_events_tenant_policy
ON audience_fresh_data_workflow_events
USING (
    tenant_id = current_setting('app.tenant_id', true)
)
WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
);

REVOKE ALL ON audience_fresh_data_workflows FROM PUBLIC;
REVOKE ALL ON audience_fresh_data_workflow_events FROM PUBLIC;
