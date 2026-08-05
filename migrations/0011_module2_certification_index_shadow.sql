-- Module 2 completion control plane:
-- certification evidence, governed index lifecycle, and privacy-safe shadow metrics.
-- This migration does not enable model registration, production routing, activation,
-- proposal creation, or export.

CREATE TABLE IF NOT EXISTS module2_certification_reports (
    tenant_id TEXT NOT NULL,
    report_fingerprint CHAR(64) NOT NULL,
    taxonomy_fingerprint CHAR(64) NOT NULL,
    benchmark_report_fingerprint CHAR(64) NOT NULL,
    module2_evidence_ready BOOLEAN NOT NULL DEFAULT FALSE,
    production_certification_ready BOOLEAN NOT NULL DEFAULT FALSE,
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, report_fingerprint),
    CHECK (
        NOT production_certification_ready
        OR (
            module2_evidence_ready
            AND approved_by IS NOT NULL
            AND approved_at IS NOT NULL
        )
    )
);

CREATE TABLE IF NOT EXISTS audience_retrieval_indexes (
    tenant_id TEXT NOT NULL,
    index_id TEXT NOT NULL,
    index_version INTEGER NOT NULL CHECK (index_version >= 1),
    manifest_fingerprint CHAR(64) NOT NULL,
    manifest JSONB NOT NULL CHECK (jsonb_typeof(manifest) = 'object'),
    status TEXT NOT NULL CHECK (
        status IN (
            'candidate', 'building', 'built', 'validated',
            'shadow', 'active', 'retired', 'failed'
        )
    ),
    event_count INTEGER NOT NULL DEFAULT 0 CHECK (event_count >= 0),
    latest_evidence JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(latest_evidence) = 'object'),
    activated_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, index_id, index_version),
    UNIQUE (tenant_id, manifest_fingerprint),
    CHECK (
        status <> 'active'
        OR (activated_at IS NOT NULL AND retired_at IS NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_audience_retrieval_active_index
ON audience_retrieval_indexes (tenant_id)
WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_audience_retrieval_indexes_status
ON audience_retrieval_indexes (tenant_id, status, updated_at);

CREATE TABLE IF NOT EXISTS audience_retrieval_index_events (
    tenant_id TEXT NOT NULL,
    index_id TEXT NOT NULL,
    index_version INTEGER NOT NULL,
    event_fingerprint CHAR(64) NOT NULL,
    previous_status TEXT NOT NULL,
    next_status TEXT NOT NULL,
    evidence JSONB NOT NULL CHECK (jsonb_typeof(evidence) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, event_fingerprint),
    FOREIGN KEY (tenant_id, index_id, index_version)
        REFERENCES audience_retrieval_indexes (
            tenant_id, index_id, index_version
        )
);

CREATE TABLE IF NOT EXISTS audience_retrieval_shadow_observations (
    tenant_id TEXT NOT NULL,
    request_fingerprint CHAR(64) NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    incumbent_signature CHAR(64) NOT NULL,
    candidate_signature CHAR(64) NOT NULL,
    incumbent_status TEXT NOT NULL,
    candidate_status TEXT NOT NULL,
    incumbent_latency_ms DOUBLE PRECISION NOT NULL
        CHECK (incumbent_latency_ms >= 0),
    candidate_latency_ms DOUBLE PRECISION NOT NULL
        CHECK (candidate_latency_ms >= 0),
    candidate_error BOOLEAN NOT NULL,
    safety_divergence BOOLEAN NOT NULL,
    agreement BOOLEAN NOT NULL,
    raw_query_stored BOOLEAN NOT NULL DEFAULT FALSE
        CHECK (raw_query_stored = FALSE),
    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE
        CHECK (raw_identifiers_stored = FALSE),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, request_fingerprint, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_shadow_observations_window
ON audience_retrieval_shadow_observations (tenant_id, observed_at DESC);

CREATE OR REPLACE FUNCTION prevent_module2_certification_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION
        'Module 2 certification reports are immutable; create a new report';
END;
$$;

DROP TRIGGER IF EXISTS trg_module2_certification_immutable
ON module2_certification_reports;

CREATE TRIGGER trg_module2_certification_immutable
BEFORE UPDATE OR DELETE ON module2_certification_reports
FOR EACH ROW EXECUTE FUNCTION prevent_module2_certification_mutation();

CREATE OR REPLACE FUNCTION prevent_module2_index_identity_change()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF
        NEW.tenant_id IS DISTINCT FROM OLD.tenant_id
        OR NEW.index_id IS DISTINCT FROM OLD.index_id
        OR NEW.index_version IS DISTINCT FROM OLD.index_version
        OR NEW.manifest_fingerprint IS DISTINCT FROM OLD.manifest_fingerprint
        OR NEW.manifest IS DISTINCT FROM OLD.manifest
    THEN
        RAISE EXCEPTION
            'Module 2 index identity and manifest are immutable';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_module2_index_identity_immutable
ON audience_retrieval_indexes;

CREATE TRIGGER trg_module2_index_identity_immutable
BEFORE UPDATE ON audience_retrieval_indexes
FOR EACH ROW EXECUTE FUNCTION prevent_module2_index_identity_change();

ALTER TABLE module2_certification_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE module2_certification_reports FORCE ROW LEVEL SECURITY;
ALTER TABLE audience_retrieval_indexes ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_retrieval_indexes FORCE ROW LEVEL SECURITY;
ALTER TABLE audience_retrieval_index_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_retrieval_index_events FORCE ROW LEVEL SECURITY;
ALTER TABLE audience_retrieval_shadow_observations ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_retrieval_shadow_observations FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS module2_certification_tenant_policy
ON module2_certification_reports;
CREATE POLICY module2_certification_tenant_policy
ON module2_certification_reports
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS audience_retrieval_indexes_tenant_policy
ON audience_retrieval_indexes;
CREATE POLICY audience_retrieval_indexes_tenant_policy
ON audience_retrieval_indexes
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS audience_retrieval_index_events_tenant_policy
ON audience_retrieval_index_events;
CREATE POLICY audience_retrieval_index_events_tenant_policy
ON audience_retrieval_index_events
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS audience_retrieval_shadow_tenant_policy
ON audience_retrieval_shadow_observations;
CREATE POLICY audience_retrieval_shadow_tenant_policy
ON audience_retrieval_shadow_observations
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

REVOKE ALL ON module2_certification_reports FROM PUBLIC;
REVOKE ALL ON audience_retrieval_indexes FROM PUBLIC;
REVOKE ALL ON audience_retrieval_index_events FROM PUBLIC;
REVOKE ALL ON audience_retrieval_shadow_observations FROM PUBLIC;
