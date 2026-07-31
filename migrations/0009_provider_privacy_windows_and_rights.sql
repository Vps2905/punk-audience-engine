-- Module 1 production closure: privacy-window composition, corrections,
-- rights propagation, canonical publication state, and scale evidence.

ALTER TABLE provider_privacy_releases
ADD COLUMN IF NOT EXISTS composition_group TEXT;

ALTER TABLE provider_privacy_releases
ADD COLUMN IF NOT EXISTS charged_epsilon DOUBLE PRECISION;

ALTER TABLE provider_privacy_releases
ADD COLUMN IF NOT EXISTS privacy_partition_index INTEGER;

UPDATE provider_privacy_releases
SET composition_group = 'legacy:' || release_id
WHERE composition_group IS NULL;

UPDATE provider_privacy_releases
SET charged_epsilon = epsilon
WHERE charged_epsilon IS NULL;

UPDATE provider_privacy_releases
SET privacy_partition_index = -1
WHERE privacy_partition_index IS NULL;

ALTER TABLE provider_privacy_releases
ALTER COLUMN composition_group SET NOT NULL;

ALTER TABLE provider_privacy_releases
ALTER COLUMN charged_epsilon SET NOT NULL;

ALTER TABLE provider_privacy_releases
ALTER COLUMN privacy_partition_index SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'provider_privacy_releases_charged_epsilon_check'
    ) THEN
        ALTER TABLE provider_privacy_releases
        ADD CONSTRAINT provider_privacy_releases_charged_epsilon_check
        CHECK (charged_epsilon >= 0 AND charged_epsilon <= epsilon)
        NOT VALID;
    END IF;
END
$$;

ALTER TABLE provider_privacy_releases
VALIDATE CONSTRAINT provider_privacy_releases_charged_epsilon_check;

CREATE INDEX IF NOT EXISTS idx_provider_privacy_release_composition
ON provider_privacy_releases (
    budget_scope,
    composition_group,
    privacy_partition_index,
    decision
);

CREATE TABLE IF NOT EXISTS provider_privacy_windows (
    window_key TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    delivery_window_id TEXT NOT NULL,
    event_time_start TIMESTAMPTZ NOT NULL,
    event_time_end TIMESTAMPTZ NOT NULL,
    partition_count INTEGER NOT NULL
        CHECK (partition_count BETWEEN 1 AND 100000),
    partition_algorithm TEXT NOT NULL
        CHECK (partition_algorithm = 'spark_xxhash64_v1'),
    status TEXT NOT NULL
        CHECK (status IN ('open', 'sealed', 'revoked')),
    revocation_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    sealed_at TIMESTAMPTZ,
    CHECK (event_time_start < event_time_end),
    UNIQUE (
        tenant_id,
        provider_id,
        dataset_id,
        schema_version,
        delivery_window_id
    )
);

CREATE TABLE IF NOT EXISTS provider_privacy_window_partitions (
    fingerprint CHAR(64) PRIMARY KEY,
    window_key TEXT NOT NULL
        REFERENCES provider_privacy_windows(window_key),
    partition_index INTEGER NOT NULL CHECK (partition_index >= 0),
    partition_count INTEGER NOT NULL CHECK (partition_count >= 1),
    ingestion_id TEXT NOT NULL
        REFERENCES provider_ingestion_objects(ingestion_id),
    row_count BIGINT CHECK (row_count >= 0),
    delivery_type TEXT NOT NULL
        CHECK (delivery_type IN ('snapshot', 'correction')),
    supersedes_fingerprint CHAR(64),
    status TEXT NOT NULL
        CHECK (status IN ('registered', 'ready', 'superseded')),
    is_current INTEGER NOT NULL CHECK (is_current IN (0, 1)),
    canonical_ref TEXT,
    privacy_release_id TEXT
        REFERENCES provider_privacy_releases(release_id),
    output_rows BIGINT CHECK (output_rows >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (
        (delivery_type = 'correction' AND supersedes_fingerprint IS NOT NULL)
        OR
        (delivery_type = 'snapshot' AND supersedes_fingerprint IS NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_provider_privacy_partition_current
ON provider_privacy_window_partitions (window_key, partition_index)
WHERE is_current = 1;

CREATE INDEX IF NOT EXISTS idx_provider_privacy_windows_status
ON provider_privacy_windows (
    tenant_id,
    provider_id,
    dataset_id,
    status,
    event_time_end DESC
);

CREATE TABLE IF NOT EXISTS provider_canonical_partitions (
    canonical_partition_id TEXT PRIMARY KEY,
    window_key TEXT NOT NULL
        REFERENCES provider_privacy_windows(window_key),
    partition_index INTEGER NOT NULL CHECK (partition_index >= 0),
    canonical_ref TEXT NOT NULL,
    checksum_sha256 CHAR(64) NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('candidate', 'active', 'revoked')),
    revocation_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (window_key, partition_index, canonical_ref)
);

CREATE TABLE IF NOT EXISTS provider_data_rights_requests (
    request_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    request_type TEXT NOT NULL
        CHECK (request_type IN ('delete', 'opt_out')),
    subject_token_sha256 CHAR(64) NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL,
    event_time_start TIMESTAMPTZ,
    event_time_end TIMESTAMPTZ,
    source_request_ref TEXT,
    status TEXT NOT NULL CHECK (status IN ('received', 'applied')),
    affected_window_count INTEGER NOT NULL DEFAULT 0,
    rebuild_required INTEGER NOT NULL DEFAULT 0
        CHECK (rebuild_required IN (0, 1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    applied_at TIMESTAMPTZ,
    CHECK (
        (event_time_start IS NULL AND event_time_end IS NULL)
        OR
        (event_time_start < event_time_end)
    )
);

CREATE INDEX IF NOT EXISTS idx_provider_data_rights_scope
ON provider_data_rights_requests (
    tenant_id,
    provider_id,
    dataset_id,
    status,
    requested_at
);

CREATE TABLE IF NOT EXISTS provider_scale_acceptance_runs (
    evidence_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    environment TEXT NOT NULL
        CHECK (environment IN ('staging', 'preproduction')),
    report_fingerprint CHAR(64) NOT NULL,
    status TEXT NOT NULL
        CHECK (
            status IN (
                'blocked_production_scale_gate',
                'production_scale_gate_passed'
            )
        ),
    measured_events BIGINT NOT NULL CHECK (measured_events >= 0),
    blockers JSONB NOT NULL DEFAULT '[]'::jsonb,
    report JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, evidence_id),
    UNIQUE (tenant_id, report_fingerprint)
);

ALTER TABLE provider_privacy_windows ENABLE ROW LEVEL SECURITY;
ALTER TABLE provider_privacy_windows FORCE ROW LEVEL SECURITY;
ALTER TABLE provider_privacy_window_partitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE provider_privacy_window_partitions FORCE ROW LEVEL SECURITY;
ALTER TABLE provider_canonical_partitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE provider_canonical_partitions FORCE ROW LEVEL SECURITY;
ALTER TABLE provider_data_rights_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE provider_data_rights_requests FORCE ROW LEVEL SECURITY;
ALTER TABLE provider_scale_acceptance_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE provider_scale_acceptance_runs FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS provider_privacy_windows_tenant
ON provider_privacy_windows;
CREATE POLICY provider_privacy_windows_tenant
ON provider_privacy_windows
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS provider_privacy_window_partitions_tenant
ON provider_privacy_window_partitions;
CREATE POLICY provider_privacy_window_partitions_tenant
ON provider_privacy_window_partitions
USING (
    EXISTS (
        SELECT 1 FROM provider_privacy_windows AS privacy_window
        WHERE privacy_window.window_key = provider_privacy_window_partitions.window_key
          AND privacy_window.tenant_id = current_setting('app.tenant_id', true)
    )
)
WITH CHECK (
    EXISTS (
        SELECT 1 FROM provider_privacy_windows AS privacy_window
        WHERE privacy_window.window_key = provider_privacy_window_partitions.window_key
          AND privacy_window.tenant_id = current_setting('app.tenant_id', true)
    )
);

DROP POLICY IF EXISTS provider_canonical_partitions_tenant
ON provider_canonical_partitions;
CREATE POLICY provider_canonical_partitions_tenant
ON provider_canonical_partitions
USING (
    EXISTS (
        SELECT 1 FROM provider_privacy_windows AS privacy_window
        WHERE privacy_window.window_key = provider_canonical_partitions.window_key
          AND privacy_window.tenant_id = current_setting('app.tenant_id', true)
    )
)
WITH CHECK (
    EXISTS (
        SELECT 1 FROM provider_privacy_windows AS privacy_window
        WHERE privacy_window.window_key = provider_canonical_partitions.window_key
          AND privacy_window.tenant_id = current_setting('app.tenant_id', true)
    )
);

DROP POLICY IF EXISTS provider_data_rights_requests_tenant
ON provider_data_rights_requests;
CREATE POLICY provider_data_rights_requests_tenant
ON provider_data_rights_requests
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS provider_scale_acceptance_runs_tenant
ON provider_scale_acceptance_runs;
CREATE POLICY provider_scale_acceptance_runs_tenant
ON provider_scale_acceptance_runs
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
