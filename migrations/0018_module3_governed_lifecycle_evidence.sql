-- Module 3.5 immutable governed lifecycle and monitoring review evidence.
--
-- These tables store recommendations, not lifecycle state. Inserts cannot
-- approve candidates, enable shadow routing, activate audiences, or export.

CREATE TABLE IF NOT EXISTS audience_cohort_lifecycle_evaluations (
    tenant_id TEXT NOT NULL,
    evaluation_fingerprint CHAR(64) NOT NULL,
    source_overlap_report_fingerprint CHAR(64) NOT NULL,
    source_lookalike_report_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    execution_mode TEXT NOT NULL CHECK (
        execution_mode IN (
            'historical_preview',
            'offline_evaluation',
            'production'
        )
    ),
    purpose TEXT NOT NULL,
    source_candidate_count INTEGER NOT NULL CHECK (
        source_candidate_count >= 0
    ),
    recommendation_count INTEGER NOT NULL CHECK (
        recommendation_count >= 0
    ),
    shadow_review_pending_count INTEGER NOT NULL CHECK (
        shadow_review_pending_count >= 0
    ),
    review_required_count INTEGER NOT NULL CHECK (
        review_required_count >= 0
    ),
    paused_count INTEGER NOT NULL CHECK (paused_count >= 0),
    blocked_count INTEGER NOT NULL CHECK (blocked_count >= 0),
    historical_preview_count INTEGER NOT NULL CHECK (
        historical_preview_count >= 0
    ),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),

    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        raw_identifiers_stored = FALSE
    ),
    audience_membership_read BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        audience_membership_read = FALSE
    ),
    membership_intersection_read BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        membership_intersection_read = FALSE
    ),
    overlap_rate_computed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        overlap_rate_computed = FALSE
    ),
    unique_reach_claimed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        unique_reach_claimed = FALSE
    ),
    cohort_sizes_summed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        cohort_sizes_summed = FALSE
    ),
    candidate_lifecycle_mutated BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        candidate_lifecycle_mutated = FALSE
    ),
    automatic_approval_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        automatic_approval_performed = FALSE
    ),
    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (
        manual_approval_required = TRUE
    ),
    monitoring_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (
        monitoring_required = TRUE
    ),
    shadow_routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        shadow_routing_enabled = FALSE
    ),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        activation_or_export_performed = FALSE
    ),
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        downstream_export_enabled = FALSE
    ),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (tenant_id, evaluation_fingerprint),

    FOREIGN KEY (
        tenant_id,
        source_overlap_report_fingerprint
    ) REFERENCES audience_cohort_overlap_analysis_runs (
        tenant_id,
        analysis_fingerprint
    ),

    FOREIGN KEY (
        tenant_id,
        source_lookalike_report_fingerprint
    ) REFERENCES audience_cohort_lookalike_analysis_runs (
        tenant_id,
        analysis_fingerprint
    )
);


CREATE TABLE IF NOT EXISTS audience_cohort_lifecycle_recommendations (
    tenant_id TEXT NOT NULL,
    evaluation_fingerprint CHAR(64) NOT NULL,
    recommendation_fingerprint CHAR(64) NOT NULL,
    candidate_id TEXT NOT NULL,
    candidate_version INTEGER NOT NULL CHECK (candidate_version >= 1),
    candidate_fingerprint CHAR(64) NOT NULL,
    current_lifecycle_status TEXT NOT NULL,
    recommended_lifecycle_status TEXT NOT NULL CHECK (
        recommended_lifecycle_status IN (
            'historical_preview_only',
            'shadow_review_pending',
            'review_required_overlap',
            'review_required_sensitive_poi',
            'paused_stale_source',
            'paused_quality_degraded',
            'blocked_policy',
            'blocked_rights'
        )
    ),
    quality_score DOUBLE PRECISION NOT NULL CHECK (
        quality_score >= 0.0 AND quality_score <= 1.0
    ),
    previous_quality_score DOUBLE PRECISION CHECK (
        previous_quality_score IS NULL
        OR (
            previous_quality_score >= 0.0
            AND previous_quality_score <= 1.0
        )
    ),
    quality_delta DOUBLE PRECISION CHECK (
        quality_delta IS NULL
        OR (quality_delta >= -1.0 AND quality_delta <= 1.0)
    ),
    reason_codes JSONB NOT NULL CHECK (
        jsonb_typeof(reason_codes) = 'array'
    ),

    manual_approval_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (
        manual_approval_required = TRUE
    ),
    monitoring_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (
        monitoring_required = TRUE
    ),
    lifecycle_mutated BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        lifecycle_mutated = FALSE
    ),
    shadow_routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        shadow_routing_enabled = FALSE
    ),
    eligible_for_activation BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        eligible_for_activation = FALSE
    ),
    eligible_for_export BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        eligible_for_export = FALSE
    ),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (tenant_id, recommendation_fingerprint),

    UNIQUE (
        tenant_id,
        evaluation_fingerprint,
        candidate_id,
        candidate_version
    ),

    FOREIGN KEY (
        tenant_id,
        evaluation_fingerprint
    ) REFERENCES audience_cohort_lifecycle_evaluations (
        tenant_id,
        evaluation_fingerprint
    ),

    FOREIGN KEY (
        tenant_id,
        candidate_id,
        candidate_version
    ) REFERENCES audience_cohort_candidates (
        tenant_id,
        candidate_id,
        candidate_version
    )
);


CREATE OR REPLACE FUNCTION validate_module3_lifecycle_candidate_fingerprint()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM audience_cohort_candidates candidate
        WHERE candidate.tenant_id = NEW.tenant_id
          AND candidate.candidate_id = NEW.candidate_id
          AND candidate.candidate_version = NEW.candidate_version
          AND candidate.candidate_fingerprint = NEW.candidate_fingerprint
    ) THEN
        RAISE EXCEPTION
            'Module 3.5 candidate fingerprint does not match immutable candidate';
    END IF;

    RETURN NEW;
END;
$$;


DROP TRIGGER IF EXISTS trg_module3_lifecycle_candidate_validate
ON audience_cohort_lifecycle_recommendations;

CREATE TRIGGER trg_module3_lifecycle_candidate_validate
BEFORE INSERT ON audience_cohort_lifecycle_recommendations
FOR EACH ROW
EXECUTE FUNCTION validate_module3_lifecycle_candidate_fingerprint();


CREATE OR REPLACE FUNCTION prevent_module3_lifecycle_evidence_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION
        'Module 3.5 lifecycle evidence is immutable; create a new fingerprinted evaluation';
END;
$$;


DROP TRIGGER IF EXISTS trg_module3_lifecycle_evaluations_immutable
ON audience_cohort_lifecycle_evaluations;

CREATE TRIGGER trg_module3_lifecycle_evaluations_immutable
BEFORE UPDATE OR DELETE ON audience_cohort_lifecycle_evaluations
FOR EACH ROW
EXECUTE FUNCTION prevent_module3_lifecycle_evidence_mutation();


DROP TRIGGER IF EXISTS trg_module3_lifecycle_recommendations_immutable
ON audience_cohort_lifecycle_recommendations;

CREATE TRIGGER trg_module3_lifecycle_recommendations_immutable
BEFORE UPDATE OR DELETE ON audience_cohort_lifecycle_recommendations
FOR EACH ROW
EXECUTE FUNCTION prevent_module3_lifecycle_evidence_mutation();


CREATE INDEX IF NOT EXISTS idx_cohort_lifecycle_evaluations_source
ON audience_cohort_lifecycle_evaluations (
    tenant_id,
    source_overlap_report_fingerprint,
    source_lookalike_report_fingerprint,
    created_at DESC
);


CREATE INDEX IF NOT EXISTS idx_cohort_lifecycle_recommendations_review
ON audience_cohort_lifecycle_recommendations (
    tenant_id,
    recommended_lifecycle_status,
    candidate_id,
    candidate_version
);


ALTER TABLE audience_cohort_lifecycle_evaluations
ENABLE ROW LEVEL SECURITY;

ALTER TABLE audience_cohort_lifecycle_evaluations
FORCE ROW LEVEL SECURITY;

ALTER TABLE audience_cohort_lifecycle_recommendations
ENABLE ROW LEVEL SECURITY;

ALTER TABLE audience_cohort_lifecycle_recommendations
FORCE ROW LEVEL SECURITY;


DROP POLICY IF EXISTS audience_cohort_lifecycle_evaluations_tenant_policy
ON audience_cohort_lifecycle_evaluations;

CREATE POLICY audience_cohort_lifecycle_evaluations_tenant_policy
ON audience_cohort_lifecycle_evaluations
USING (
    tenant_id = current_setting('app.tenant_id', true)
)
WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
);


DROP POLICY IF EXISTS audience_cohort_lifecycle_recommendations_tenant_policy
ON audience_cohort_lifecycle_recommendations;

CREATE POLICY audience_cohort_lifecycle_recommendations_tenant_policy
ON audience_cohort_lifecycle_recommendations
USING (
    tenant_id = current_setting('app.tenant_id', true)
)
WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
);


REVOKE ALL
ON audience_cohort_lifecycle_evaluations
FROM PUBLIC;

REVOKE ALL
ON audience_cohort_lifecycle_recommendations
FROM PUBLIC;
