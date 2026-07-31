CREATE TABLE IF NOT EXISTS audience_embedding_models (
    tenant_id TEXT NOT NULL,
    model_fingerprint CHAR(64) NOT NULL,
    backend TEXT NOT NULL CHECK (
        backend IN (
            'sentence_transformers',
            'external_embedding_service'
        )
    ),
    model_name TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (
        embedding_dimension = 384
    ),
    normalize_embeddings BOOLEAN NOT NULL,
    document_prefix TEXT NOT NULL DEFAULT '',
    query_prefix TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK (
        status IN (
            'evaluation',
            'approved',
            'retired',
            'blocked'
        )
    ),
    benchmark_status TEXT NOT NULL CHECK (
        benchmark_status IN (
            'pending',
            'passed',
            'failed'
        )
    ),
    benchmark_report JSONB NOT NULL DEFAULT '{}'::jsonb,
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (
        tenant_id,
        model_fingerprint
    ),
    UNIQUE (
        tenant_id,
        model_name,
        model_revision
    ),
    CHECK (
        status <> 'approved'
        OR (
            benchmark_status = 'passed'
            AND approved_by IS NOT NULL
            AND approved_at IS NOT NULL
        )
    )
);

CREATE TABLE IF NOT EXISTS audience_feature_build_jobs (
    tenant_id TEXT NOT NULL,
    feature_build_id TEXT NOT NULL,
    request_fingerprint CHAR(64) NOT NULL,
    canonical_source_fingerprint CHAR(64) NOT NULL,
    source_ref TEXT NOT NULL,
    source_version TEXT NOT NULL,
    model_fingerprint CHAR(64) NOT NULL,
    data_use_mode TEXT NOT NULL CHECK (
        data_use_mode IN (
            'historical_preview',
            'offline_evaluation',
            'production'
        )
    ),
    status TEXT NOT NULL CHECK (
        status IN (
            'claimed',
            'validating',
            'embedding',
            'publishing',
            'completed',
            'blocked',
            'quarantined',
            'failed'
        )
    ),
    reason_code TEXT,
    expected_feature_count BIGINT NOT NULL CHECK (
        expected_feature_count >= 1
    ),
    processed_feature_count BIGINT NOT NULL DEFAULT 0 CHECK (
        processed_feature_count >= 0
    ),
    feature_set_id TEXT,
    feature_set_version INTEGER CHECK (
        feature_set_version IS NULL
        OR feature_set_version >= 1
    ),
    request_manifest JSONB NOT NULL,
    result_receipt JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    PRIMARY KEY (
        tenant_id,
        feature_build_id
    ),
    UNIQUE (
        tenant_id,
        request_fingerprint
    ),
    FOREIGN KEY (
        tenant_id,
        model_fingerprint
    )
    REFERENCES audience_embedding_models (
        tenant_id,
        model_fingerprint
    ),
    CHECK (
        status <> 'completed'
        OR (
            feature_set_id IS NOT NULL
            AND feature_set_version IS NOT NULL
            AND processed_feature_count = expected_feature_count
            AND completed_at IS NOT NULL
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_audience_embedding_models_runtime
ON audience_embedding_models (
    tenant_id,
    status,
    benchmark_status,
    model_name,
    model_revision
);

CREATE INDEX IF NOT EXISTS idx_audience_feature_build_jobs_status
ON audience_feature_build_jobs (
    tenant_id,
    status,
    updated_at
);

CREATE OR REPLACE FUNCTION prevent_approved_embedding_model_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status = 'approved' THEN
        RAISE EXCEPTION
            'Approved embedding model records are immutable; register a new revision';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_approved_embedding_model_immutable
ON audience_embedding_models;

CREATE TRIGGER trg_approved_embedding_model_immutable
BEFORE UPDATE OR DELETE ON audience_embedding_models
FOR EACH ROW
EXECUTE FUNCTION prevent_approved_embedding_model_mutation();

CREATE OR REPLACE FUNCTION prevent_feature_build_identity_change()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF
        NEW.tenant_id IS DISTINCT FROM OLD.tenant_id
        OR NEW.feature_build_id IS DISTINCT FROM OLD.feature_build_id
        OR NEW.request_fingerprint IS DISTINCT FROM OLD.request_fingerprint
        OR NEW.canonical_source_fingerprint
            IS DISTINCT FROM OLD.canonical_source_fingerprint
        OR NEW.source_ref IS DISTINCT FROM OLD.source_ref
        OR NEW.source_version IS DISTINCT FROM OLD.source_version
        OR NEW.model_fingerprint IS DISTINCT FROM OLD.model_fingerprint
        OR NEW.data_use_mode IS DISTINCT FROM OLD.data_use_mode
        OR NEW.expected_feature_count
            IS DISTINCT FROM OLD.expected_feature_count
        OR NEW.request_manifest IS DISTINCT FROM OLD.request_manifest
    THEN
        RAISE EXCEPTION
            'Feature build identity and request manifest are immutable';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_feature_build_identity_immutable
ON audience_feature_build_jobs;

CREATE TRIGGER trg_feature_build_identity_immutable
BEFORE UPDATE ON audience_feature_build_jobs
FOR EACH ROW
EXECUTE FUNCTION prevent_feature_build_identity_change();

CREATE OR REPLACE FUNCTION prevent_feature_artifact_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION
        'Published audience feature artifacts are immutable; create a new version';
END;
$$;

DROP TRIGGER IF EXISTS trg_audience_feature_sets_immutable
ON audience_feature_sets;

CREATE TRIGGER trg_audience_feature_sets_immutable
BEFORE UPDATE OR DELETE ON audience_feature_sets
FOR EACH ROW
EXECUTE FUNCTION prevent_feature_artifact_mutation();

DROP TRIGGER IF EXISTS trg_audience_feature_vectors_immutable
ON audience_feature_vectors;

CREATE TRIGGER trg_audience_feature_vectors_immutable
BEFORE UPDATE OR DELETE ON audience_feature_vectors
FOR EACH ROW
EXECUTE FUNCTION prevent_feature_artifact_mutation();

ALTER TABLE audience_embedding_models
ENABLE ROW LEVEL SECURITY;

ALTER TABLE audience_embedding_models
FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS audience_embedding_models_tenant_policy
ON audience_embedding_models;

CREATE POLICY audience_embedding_models_tenant_policy
ON audience_embedding_models
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

ALTER TABLE audience_feature_build_jobs
ENABLE ROW LEVEL SECURITY;

ALTER TABLE audience_feature_build_jobs
FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS audience_feature_build_jobs_tenant_policy
ON audience_feature_build_jobs;

CREATE POLICY audience_feature_build_jobs_tenant_policy
ON audience_feature_build_jobs
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
