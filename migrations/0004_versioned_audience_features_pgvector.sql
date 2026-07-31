CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS audience_feature_sets (
    tenant_id TEXT NOT NULL,
    feature_set_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 1),
    status TEXT NOT NULL,
    source_mode TEXT NOT NULL,
    data_use_mode TEXT NOT NULL CHECK (
        data_use_mode IN (
            'historical_preview',
            'offline_evaluation',
            'production'
        )
    ),
    source_ref TEXT NOT NULL,
    source_version TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    source_latest_at TIMESTAMPTZ,
    freshness_status TEXT NOT NULL CHECK (
        freshness_status IN (
            'fresh',
            'stale',
            'unknown'
        )
    ),
    stale_after_hours INTEGER NOT NULL CHECK (stale_after_hours >= 1),
    model_backend TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (
        embedding_dimension = 384
    ),
    privacy_policy_version TEXT NOT NULL,
    rights_policy_id TEXT NOT NULL,
    purpose TEXT NOT NULL,
    eligible_for_retrieval BOOLEAN NOT NULL DEFAULT FALSE,
    eligible_for_activation BOOLEAN NOT NULL DEFAULT FALSE,
    feature_count INTEGER NOT NULL CHECK (feature_count >= 0),
    lineage JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (
        tenant_id,
        feature_set_id,
        version
    ),
    UNIQUE (
        tenant_id,
        source_fingerprint
    ),
    CHECK (
        data_use_mode = 'production'
        OR eligible_for_activation = FALSE
    ),
    CHECK (
        freshness_status = 'fresh'
        OR eligible_for_activation = FALSE
    )
);

CREATE INDEX IF NOT EXISTS idx_audience_feature_sets_lookup
ON audience_feature_sets (
    tenant_id,
    data_use_mode,
    eligible_for_retrieval,
    version DESC,
    updated_at DESC
);

CREATE TABLE IF NOT EXISTS audience_feature_vectors (
    tenant_id TEXT NOT NULL,
    feature_set_id TEXT NOT NULL,
    feature_set_version INTEGER NOT NULL CHECK (
        feature_set_version >= 1
    ),
    feature_id TEXT NOT NULL,
    location_name TEXT NOT NULL,
    primary_poi_type TEXT NOT NULL,
    created_day_part TEXT NOT NULL,
    lookback_bucket TEXT,
    cohort_size BIGINT NOT NULL CHECK (cohort_size >= 0),
    quality_score DOUBLE PRECISION NOT NULL CHECK (
        quality_score >= 0.0
        AND quality_score <= 1.0
    ),
    privacy_status TEXT NOT NULL,
    rights_status TEXT NOT NULL,
    purpose TEXT NOT NULL,
    source_latest_at TIMESTAMPTZ,
    freshness_status TEXT NOT NULL CHECK (
        freshness_status IN (
            'fresh',
            'stale',
            'unknown'
        )
    ),
    data_use_mode TEXT NOT NULL CHECK (
        data_use_mode IN (
            'historical_preview',
            'offline_evaluation',
            'production'
        )
    ),
    eligible_for_retrieval BOOLEAN NOT NULL DEFAULT FALSE,
    eligible_for_activation BOOLEAN NOT NULL DEFAULT FALSE,
    trait_text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_document TSVECTOR GENERATED ALWAYS AS (
        to_tsvector(
            'simple',
            coalesce(replace(location_name, '_', ' '), '')
            || ' '
            || coalesce(replace(primary_poi_type, '_', ' '), '')
            || ' '
            || coalesce(replace(created_day_part, '_', ' '), '')
            || ' '
            || coalesce(trait_text, '')
        )
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (
        tenant_id,
        feature_set_id,
        feature_set_version,
        feature_id
    ),
    FOREIGN KEY (
        tenant_id,
        feature_set_id,
        feature_set_version
    )
    REFERENCES audience_feature_sets (
        tenant_id,
        feature_set_id,
        version
    ),
    CHECK (
        data_use_mode = 'production'
        OR eligible_for_activation = FALSE
    ),
    CHECK (
        freshness_status = 'fresh'
        OR eligible_for_activation = FALSE
    )
);

CREATE INDEX IF NOT EXISTS idx_audience_feature_vectors_hard_filters
ON audience_feature_vectors (
    tenant_id,
    feature_set_id,
    feature_set_version,
    eligible_for_retrieval,
    freshness_status,
    location_name,
    primary_poi_type,
    created_day_part,
    quality_score DESC
);

CREATE INDEX IF NOT EXISTS idx_audience_feature_vectors_search
ON audience_feature_vectors
USING GIN (search_document);

CREATE INDEX IF NOT EXISTS idx_audience_feature_vectors_embedding_hnsw
ON audience_feature_vectors
USING hnsw (embedding vector_cosine_ops);

ALTER TABLE audience_feature_sets
ENABLE ROW LEVEL SECURITY;

ALTER TABLE audience_feature_sets
FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS audience_feature_sets_tenant_policy
ON audience_feature_sets;

CREATE POLICY audience_feature_sets_tenant_policy
ON audience_feature_sets
USING (
    tenant_id = current_setting(
        'app.tenant_id',
        true
    )
)
WITH CHECK (
    tenant_id = current_setting(
        'app.tenant_id',
        true
    )
);

ALTER TABLE audience_feature_vectors
ENABLE ROW LEVEL SECURITY;

ALTER TABLE audience_feature_vectors
FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS audience_feature_vectors_tenant_policy
ON audience_feature_vectors;

CREATE POLICY audience_feature_vectors_tenant_policy
ON audience_feature_vectors
USING (
    tenant_id = current_setting(
        'app.tenant_id',
        true
    )
)
WITH CHECK (
    tenant_id = current_setting(
        'app.tenant_id',
        true
    )
);
