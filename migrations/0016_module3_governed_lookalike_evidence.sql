-- Module 3.4 governed aggregate-only lookalike review evidence.
--
-- This schema stores candidate-to-candidate review suggestions only.
-- It cannot store audience membership, overlap percentages, unique reach,
-- activation state, export eligibility, or generated member identifiers.

CREATE TABLE IF NOT EXISTS audience_cohort_lookalike_analysis_runs (
    tenant_id TEXT NOT NULL,
    analysis_fingerprint CHAR(64) NOT NULL,
    source_overlap_analysis_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    execution_mode TEXT NOT NULL CHECK (
        execution_mode IN (
            'historical_preview',
            'offline_evaluation',
            'production'
        )
    ),
    source_retained_candidate_count INTEGER NOT NULL CHECK (
        source_retained_candidate_count >= 0
    ),
    eligible_seed_count INTEGER NOT NULL CHECK (
        eligible_seed_count >= 0
    ),
    generated_lookalike_candidate_count INTEGER NOT NULL CHECK (
        generated_lookalike_candidate_count >= 0
    ),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),

    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        raw_identifiers_stored = FALSE
    ),
    audience_membership_read BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        audience_membership_read = FALSE
    ),
    membership_similarity_computed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        membership_similarity_computed = FALSE
    ),
    audience_membership_generated BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        audience_membership_generated = FALSE
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
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        activation_or_export_performed = FALSE
    ),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (tenant_id, analysis_fingerprint),

    FOREIGN KEY (
        tenant_id,
        source_overlap_analysis_fingerprint
    ) REFERENCES audience_cohort_overlap_analysis_runs (
        tenant_id,
        analysis_fingerprint
    )
);


CREATE TABLE IF NOT EXISTS audience_cohort_lookalike_candidates (
    tenant_id TEXT NOT NULL,
    lookalike_id TEXT NOT NULL,
    lookalike_version INTEGER NOT NULL CHECK (
        lookalike_version >= 1
    ),
    lookalike_fingerprint CHAR(64) NOT NULL,
    analysis_fingerprint CHAR(64) NOT NULL,

    seed_candidate_id TEXT NOT NULL,
    seed_candidate_version INTEGER NOT NULL CHECK (
        seed_candidate_version >= 1
    ),
    seed_candidate_fingerprint CHAR(64) NOT NULL,

    target_candidate_id TEXT NOT NULL,
    target_candidate_version INTEGER NOT NULL CHECK (
        target_candidate_version >= 1
    ),
    target_candidate_fingerprint CHAR(64) NOT NULL,

    similarity_score DOUBLE PRECISION NOT NULL CHECK (
        similarity_score >= 0.0
        AND similarity_score <= 1.0
    ),
    similarity_components JSONB NOT NULL CHECK (
        jsonb_typeof(similarity_components) = 'object'
    ),
    reason_codes JSONB NOT NULL CHECK (
        jsonb_typeof(reason_codes) = 'array'
    ),

    review_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (
        review_required = TRUE
    ),
    membership_generated BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        membership_generated = FALSE
    ),
    eligible_for_activation BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        eligible_for_activation = FALSE
    ),
    eligible_for_export BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        eligible_for_export = FALSE
    ),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (
        tenant_id,
        lookalike_id,
        lookalike_version
    ),

    UNIQUE (
        tenant_id,
        lookalike_fingerprint
    ),

    FOREIGN KEY (
        tenant_id,
        analysis_fingerprint
    ) REFERENCES audience_cohort_lookalike_analysis_runs (
        tenant_id,
        analysis_fingerprint
    ),

    FOREIGN KEY (
        tenant_id,
        seed_candidate_id,
        seed_candidate_version
    ) REFERENCES audience_cohort_candidates (
        tenant_id,
        candidate_id,
        candidate_version
    ),

    FOREIGN KEY (
        tenant_id,
        target_candidate_id,
        target_candidate_version
    ) REFERENCES audience_cohort_candidates (
        tenant_id,
        candidate_id,
        candidate_version
    ),

    CHECK (
        seed_candidate_id <> target_candidate_id
        OR seed_candidate_version <> target_candidate_version
    )
);


CREATE OR REPLACE FUNCTION validate_module3_lookalike_candidate_fingerprints()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM audience_cohort_candidates candidate
        WHERE candidate.tenant_id = NEW.tenant_id
          AND candidate.candidate_id = NEW.seed_candidate_id
          AND candidate.candidate_version = NEW.seed_candidate_version
          AND candidate.candidate_fingerprint = NEW.seed_candidate_fingerprint
    ) THEN
        RAISE EXCEPTION
            'Module 3.4 seed candidate fingerprint does not match immutable candidate';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM audience_cohort_candidates candidate
        WHERE candidate.tenant_id = NEW.tenant_id
          AND candidate.candidate_id = NEW.target_candidate_id
          AND candidate.candidate_version = NEW.target_candidate_version
          AND candidate.candidate_fingerprint = NEW.target_candidate_fingerprint
    ) THEN
        RAISE EXCEPTION
            'Module 3.4 target candidate fingerprint does not match immutable candidate';
    END IF;

    RETURN NEW;
END;
$$;


DROP TRIGGER IF EXISTS trg_module3_lookalike_candidate_validate
ON audience_cohort_lookalike_candidates;

CREATE TRIGGER trg_module3_lookalike_candidate_validate
BEFORE INSERT ON audience_cohort_lookalike_candidates
FOR EACH ROW
EXECUTE FUNCTION validate_module3_lookalike_candidate_fingerprints();


CREATE OR REPLACE FUNCTION prevent_module3_lookalike_evidence_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION
        'Module 3.4 lookalike evidence is immutable; create a new fingerprinted analysis';
END;
$$;


DROP TRIGGER IF EXISTS trg_module3_lookalike_runs_immutable
ON audience_cohort_lookalike_analysis_runs;

CREATE TRIGGER trg_module3_lookalike_runs_immutable
BEFORE UPDATE OR DELETE ON audience_cohort_lookalike_analysis_runs
FOR EACH ROW
EXECUTE FUNCTION prevent_module3_lookalike_evidence_mutation();


DROP TRIGGER IF EXISTS trg_module3_lookalike_candidates_immutable
ON audience_cohort_lookalike_candidates;

CREATE TRIGGER trg_module3_lookalike_candidates_immutable
BEFORE UPDATE OR DELETE ON audience_cohort_lookalike_candidates
FOR EACH ROW
EXECUTE FUNCTION prevent_module3_lookalike_evidence_mutation();


CREATE INDEX IF NOT EXISTS idx_cohort_lookalike_seed
ON audience_cohort_lookalike_candidates (
    tenant_id,
    seed_candidate_id,
    seed_candidate_version,
    similarity_score DESC
);


CREATE INDEX IF NOT EXISTS idx_cohort_lookalike_target
ON audience_cohort_lookalike_candidates (
    tenant_id,
    target_candidate_id,
    target_candidate_version
);


ALTER TABLE audience_cohort_lookalike_analysis_runs
ENABLE ROW LEVEL SECURITY;

ALTER TABLE audience_cohort_lookalike_analysis_runs
FORCE ROW LEVEL SECURITY;

ALTER TABLE audience_cohort_lookalike_candidates
ENABLE ROW LEVEL SECURITY;

ALTER TABLE audience_cohort_lookalike_candidates
FORCE ROW LEVEL SECURITY;


DROP POLICY IF EXISTS audience_cohort_lookalike_runs_tenant_policy
ON audience_cohort_lookalike_analysis_runs;

CREATE POLICY audience_cohort_lookalike_runs_tenant_policy
ON audience_cohort_lookalike_analysis_runs
USING (
    tenant_id = current_setting('app.tenant_id', true)
)
WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
);


DROP POLICY IF EXISTS audience_cohort_lookalike_candidates_tenant_policy
ON audience_cohort_lookalike_candidates;

CREATE POLICY audience_cohort_lookalike_candidates_tenant_policy
ON audience_cohort_lookalike_candidates
USING (
    tenant_id = current_setting('app.tenant_id', true)
)
WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
);


REVOKE ALL
ON audience_cohort_lookalike_analysis_runs
FROM PUBLIC;

REVOKE ALL
ON audience_cohort_lookalike_candidates
FROM PUBLIC;
