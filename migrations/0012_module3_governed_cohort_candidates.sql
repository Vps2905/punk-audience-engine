-- Module 3.1-3.2 governed cohort candidate control plane.
-- This migration stores only aggregate candidate metadata and evidence.
-- It does not enable lookalikes, production routing, activation, or export.

CREATE TABLE IF NOT EXISTS audience_cohort_candidate_batches (
    tenant_id TEXT NOT NULL,
    batch_fingerprint CHAR(64) NOT NULL,
    source_feature_set_id TEXT NOT NULL,
    source_feature_set_version INTEGER NOT NULL CHECK (
        source_feature_set_version >= 1
    ),
    policy_version TEXT NOT NULL,
    execution_mode TEXT NOT NULL CHECK (
        execution_mode IN (
            'historical_preview',
            'offline_evaluation',
            'production'
        )
    ),
    source_freshness_status TEXT NOT NULL CHECK (
        source_freshness_status IN ('fresh', 'stale', 'unknown')
    ),
    source_feature_count INTEGER NOT NULL CHECK (source_feature_count >= 0),
    generated_candidate_count INTEGER NOT NULL CHECK (
        generated_candidate_count >= 0
    ),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        raw_identifiers_stored = FALSE
    ),
    overlap_or_unique_reach_claimed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        overlap_or_unique_reach_claimed = FALSE
    ),
    lookalike_generation_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        lookalike_generation_performed = FALSE
    ),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        activation_or_export_performed = FALSE
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, batch_fingerprint),
    FOREIGN KEY (
        tenant_id,
        source_feature_set_id,
        source_feature_set_version
    ) REFERENCES audience_feature_sets (
        tenant_id,
        feature_set_id,
        version
    )
);

CREATE TABLE IF NOT EXISTS audience_cohort_candidates (
    tenant_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    candidate_version INTEGER NOT NULL CHECK (candidate_version >= 1),
    candidate_fingerprint CHAR(64) NOT NULL,
    batch_fingerprint CHAR(64) NOT NULL,
    source_feature_set_id TEXT NOT NULL,
    source_feature_set_version INTEGER NOT NULL CHECK (
        source_feature_set_version >= 1
    ),
    source_feature_id TEXT NOT NULL,
    location_name TEXT NOT NULL,
    primary_poi_type TEXT NOT NULL,
    created_day_part TEXT NOT NULL,
    lookback_bucket TEXT,
    cohort_size BIGINT NOT NULL CHECK (cohort_size >= 1000),
    source_quality_score DOUBLE PRECISION NOT NULL CHECK (
        source_quality_score >= 0.0 AND source_quality_score <= 1.0
    ),
    quality_score DOUBLE PRECISION NOT NULL CHECK (
        quality_score >= 0.0 AND quality_score <= 1.0
    ),
    quality_components JSONB NOT NULL CHECK (
        jsonb_typeof(quality_components) = 'object'
    ),
    metric_disclosure JSONB NOT NULL CHECK (
        jsonb_typeof(metric_disclosure) = 'object'
    ),
    privacy_status TEXT NOT NULL,
    privacy_decision TEXT NOT NULL,
    sensitive_poi_decision TEXT NOT NULL CHECK (
        sensitive_poi_decision IN (
            'block_export',
            'review_required',
            'allow_approval_gated_export'
        )
    ),
    rights_status TEXT NOT NULL,
    freshness_status TEXT NOT NULL CHECK (
        freshness_status IN ('fresh', 'stale', 'unknown')
    ),
    data_use_mode TEXT NOT NULL CHECK (
        data_use_mode IN (
            'historical_preview',
            'offline_evaluation',
            'production'
        )
    ),
    lifecycle_status TEXT NOT NULL CHECK (
        lifecycle_status IN (
            'historical_preview_only',
            'quality_review_pending',
            'review_required_sensitive_poi',
            'blocked_sensitive_poi'
        )
    ),
    approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (
        approval_required = TRUE
    ),
    eligible_for_activation BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        eligible_for_activation = FALSE
    ),
    eligible_for_export BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        eligible_for_export = FALSE
    ),
    lineage JSONB NOT NULL CHECK (jsonb_typeof(lineage) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, candidate_id, candidate_version),
    UNIQUE (tenant_id, candidate_fingerprint),
    FOREIGN KEY (tenant_id, batch_fingerprint)
        REFERENCES audience_cohort_candidate_batches (
            tenant_id,
            batch_fingerprint
        ),
    FOREIGN KEY (
        tenant_id,
        source_feature_set_id,
        source_feature_set_version,
        source_feature_id
    ) REFERENCES audience_feature_vectors (
        tenant_id,
        feature_set_id,
        feature_set_version,
        feature_id
    )
);

CREATE INDEX IF NOT EXISTS idx_cohort_candidates_review_queue
ON audience_cohort_candidates (
    tenant_id,
    lifecycle_status,
    quality_score DESC,
    candidate_id
);

CREATE INDEX IF NOT EXISTS idx_cohort_candidates_constraints
ON audience_cohort_candidates (
    tenant_id,
    location_name,
    primary_poi_type,
    created_day_part,
    lookback_bucket
);

CREATE OR REPLACE FUNCTION prevent_module3_candidate_batch_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION
        'Module 3 candidate batches are immutable; create a new fingerprinted batch';
END;
$$;

DROP TRIGGER IF EXISTS trg_module3_candidate_batch_immutable
ON audience_cohort_candidate_batches;

CREATE TRIGGER trg_module3_candidate_batch_immutable
BEFORE UPDATE OR DELETE ON audience_cohort_candidate_batches
FOR EACH ROW EXECUTE FUNCTION prevent_module3_candidate_batch_mutation();

CREATE OR REPLACE FUNCTION prevent_module3_candidate_identity_change()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION
            'Module 3 cohort candidates are immutable; create a new version';
    END IF;
    IF
        NEW.tenant_id IS DISTINCT FROM OLD.tenant_id
        OR NEW.candidate_id IS DISTINCT FROM OLD.candidate_id
        OR NEW.candidate_version IS DISTINCT FROM OLD.candidate_version
        OR NEW.candidate_fingerprint IS DISTINCT FROM OLD.candidate_fingerprint
        OR NEW.batch_fingerprint IS DISTINCT FROM OLD.batch_fingerprint
        OR NEW.source_feature_set_id IS DISTINCT FROM OLD.source_feature_set_id
        OR NEW.source_feature_set_version IS DISTINCT FROM OLD.source_feature_set_version
        OR NEW.source_feature_id IS DISTINCT FROM OLD.source_feature_id
        OR NEW.location_name IS DISTINCT FROM OLD.location_name
        OR NEW.primary_poi_type IS DISTINCT FROM OLD.primary_poi_type
        OR NEW.created_day_part IS DISTINCT FROM OLD.created_day_part
        OR NEW.lookback_bucket IS DISTINCT FROM OLD.lookback_bucket
        OR NEW.cohort_size IS DISTINCT FROM OLD.cohort_size
        OR NEW.lineage IS DISTINCT FROM OLD.lineage
    THEN
        RAISE EXCEPTION
            'Module 3 candidate identity, source lineage, and aggregate size are immutable';
    END IF;
    IF
        NEW.eligible_for_activation IS DISTINCT FROM FALSE
        OR NEW.eligible_for_export IS DISTINCT FROM FALSE
    THEN
        RAISE EXCEPTION
            'Module 3.1-3.2 candidates cannot be activated or exported';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_module3_candidate_identity_immutable
ON audience_cohort_candidates;

CREATE TRIGGER trg_module3_candidate_identity_immutable
BEFORE UPDATE OR DELETE ON audience_cohort_candidates
FOR EACH ROW EXECUTE FUNCTION prevent_module3_candidate_identity_change();

ALTER TABLE audience_cohort_candidate_batches ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_cohort_candidate_batches FORCE ROW LEVEL SECURITY;
ALTER TABLE audience_cohort_candidates ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_cohort_candidates FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS audience_cohort_candidate_batches_tenant_policy
ON audience_cohort_candidate_batches;
CREATE POLICY audience_cohort_candidate_batches_tenant_policy
ON audience_cohort_candidate_batches
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS audience_cohort_candidates_tenant_policy
ON audience_cohort_candidates;
CREATE POLICY audience_cohort_candidates_tenant_policy
ON audience_cohort_candidates
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

REVOKE ALL ON audience_cohort_candidate_batches FROM PUBLIC;
REVOKE ALL ON audience_cohort_candidates FROM PUBLIC;
