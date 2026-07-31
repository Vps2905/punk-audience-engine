CREATE TABLE IF NOT EXISTS provider_dataset_contracts (
    contract_key TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    contract_checksum TEXT NOT NULL,
    contract JSONB NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    actor TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, provider_id, dataset_id, schema_version)
);

CREATE INDEX IF NOT EXISTS idx_provider_contract_lookup
ON provider_dataset_contracts (
    tenant_id,
    provider_id,
    dataset_id,
    schema_version,
    active
);

CREATE TABLE IF NOT EXISTS provider_ingestion_objects (
    ingestion_id TEXT PRIMARY KEY,
    fingerprint TEXT UNIQUE NOT NULL,
    tenant_id TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    object_version TEXT,
    checksum_sha256 TEXT,
    status TEXT NOT NULL CHECK (
        status IN (
            'received',
            'validating',
            'processing',
            'completed',
            'blocked',
            'quarantined',
            'failed'
        )
    ),
    reason_code TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    privacy_job_id TEXT,
    canonical_ref TEXT,
    input_rows BIGINT CHECK (input_rows IS NULL OR input_rows >= 0),
    output_rows BIGINT CHECK (output_rows IS NULL OR output_rows >= 0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_provider_ingestion_tenant_dataset
ON provider_ingestion_objects (
    tenant_id,
    provider_id,
    dataset_id,
    updated_at
);

CREATE INDEX IF NOT EXISTS idx_provider_ingestion_status
ON provider_ingestion_objects (status, updated_at);

CREATE TABLE IF NOT EXISTS provider_ingestion_replays (
    replay_id TEXT PRIMARY KEY,
    ingestion_id TEXT NOT NULL REFERENCES
        provider_ingestion_objects(ingestion_id),
    requested_by TEXT NOT NULL,
    reason TEXT NOT NULL,
    approval_reference TEXT NOT NULL,
    replay_mode TEXT NOT NULL CHECK (
        replay_mode IN (
            'idempotency_verification',
            'terminal_failure_retry'
        )
    ),
    status TEXT NOT NULL CHECK (status IN ('pending', 'queued')),
    queue_message_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_provider_ingestion_replays_ingestion
ON provider_ingestion_replays (ingestion_id, created_at);
