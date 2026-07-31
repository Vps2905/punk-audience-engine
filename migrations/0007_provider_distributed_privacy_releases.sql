CREATE TABLE IF NOT EXISTS provider_privacy_releases (
    release_id TEXT PRIMARY KEY,
    ingestion_id TEXT NOT NULL
        REFERENCES provider_ingestion_objects(ingestion_id),
    fingerprint CHAR(64) NOT NULL,
    dispatch_attempt INTEGER NOT NULL CHECK (dispatch_attempt >= 1),
    tenant_id TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    budget_scope TEXT NOT NULL,
    epsilon DOUBLE PRECISION NOT NULL CHECK (epsilon > 0),
    delta DOUBLE PRECISION NOT NULL CHECK (delta > 0 AND delta < 1),
    sensitivity DOUBLE PRECISION NOT NULL CHECK (sensitivity > 0),
    mechanism TEXT NOT NULL CHECK (mechanism = 'gaussian'),
    max_budget DOUBLE PRECISION NOT NULL CHECK (max_budget > 0),
    budget_before DOUBLE PRECISION NOT NULL CHECK (budget_before >= 0),
    budget_after DOUBLE PRECISION NOT NULL CHECK (budget_after >= 0),
    decision TEXT NOT NULL CHECK (decision IN ('allowed', 'blocked')),
    status TEXT NOT NULL
        CHECK (status IN ('reserved', 'completed', 'blocked', 'failed')),
    canonical_ref TEXT,
    reason_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    CHECK (
        (status = 'completed' AND canonical_ref IS NOT NULL)
        OR
        (status <> 'completed' AND canonical_ref IS NULL)
    ),
    UNIQUE (ingestion_id, dispatch_attempt)
);

CREATE INDEX IF NOT EXISTS idx_provider_privacy_releases_scope
ON provider_privacy_releases (budget_scope, decision);

CREATE INDEX IF NOT EXISTS idx_provider_privacy_releases_tenant
ON provider_privacy_releases (
    tenant_id,
    provider_id,
    dataset_id,
    created_at DESC
);
