-- Module 3.3 governed overlap-risk and exact-duplicate evidence control plane.
--
-- This schema never stores raw membership, overlap percentages, unique reach,
-- merged cohort sizes, lookalikes, production routing, activation, or export.

CREATE TABLE IF NOT EXISTS audience_cohort_overlap_analysis_runs (
    tenant_id TEXT NOT NULL,
    analysis_fingerprint CHAR(64) NOT NULL,
    source_batch_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    execution_mode TEXT NOT NULL CHECK (
        execution_mode IN (
            'historical_preview',
            'offline_evaluation',
            'production'
        )
    ),
    source_candidate_count INTEGER NOT NULL CHECK (
        source_candidate_count >= 0
    ),
    retained_candidate_count INTEGER NOT NULL CHECK (
        retained_candidate_count >= 0
    ),
    exact_duplicate_group_count INTEGER NOT NULL CHECK (
        exact_duplicate_group_count >= 0
    ),
    suppressed_exact_duplicate_occurrence_count INTEGER NOT NULL CHECK (
        suppressed_exact_duplicate_occurrence_count >= 0
    ),
    potential_overlap_group_count INTEGER NOT NULL CHECK (
        potential_overlap_group_count >= 0
    ),
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),
    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        raw_identifiers_stored = FALSE
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
    lookalike_generation_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        lookalike_generation_performed = FALSE
    ),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        activation_or_export_performed = FALSE
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, analysis_fingerprint),
    FOREIGN KEY (tenant_id, source_batch_fingerprint)
        REFERENCES audience_cohort_candidate_batches (
            tenant_id,
            batch_fingerprint
        )
);

CREATE TABLE IF NOT EXISTS audience_cohort_overlap_groups (
    tenant_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    group_version INTEGER NOT NULL CHECK (group_version >= 1),
    group_fingerprint CHAR(64) NOT NULL,
    analysis_fingerprint CHAR(64) NOT NULL,
    group_type TEXT NOT NULL CHECK (
        group_type IN (
            'potential_constraint_overlap',
            'potential_temporal_overlap'
        )
    ),
    normalized_signature JSONB NOT NULL CHECK (
        jsonb_typeof(normalized_signature) = 'object'
    ),
    candidate_count INTEGER NOT NULL CHECK (candidate_count >= 2),
    review_required BOOLEAN NOT NULL DEFAULT TRUE CHECK (
        review_required = TRUE
    ),
    overlap_estimate_available BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        overlap_estimate_available = FALSE
    ),
    unique_reach_claimed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        unique_reach_claimed = FALSE
    ),
    eligible_for_activation BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        eligible_for_activation = FALSE
    ),
    eligible_for_export BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        eligible_for_export = FALSE
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, group_id, group_version),
    UNIQUE (tenant_id, group_fingerprint),
    FOREIGN KEY (tenant_id, analysis_fingerprint)
        REFERENCES audience_cohort_overlap_analysis_runs (
            tenant_id,
            analysis_fingerprint
        )
);

CREATE TABLE IF NOT EXISTS audience_cohort_overlap_group_members (
    tenant_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    group_version INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    candidate_version INTEGER NOT NULL CHECK (candidate_version >= 1),
    candidate_fingerprint CHAR(64) NOT NULL,
    member_role TEXT NOT NULL DEFAULT 'potential_overlap_member' CHECK (
        member_role = 'potential_overlap_member'
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (
        tenant_id,
        group_id,
        group_version,
        candidate_id,
        candidate_version
    ),
    FOREIGN KEY (tenant_id, group_id, group_version)
        REFERENCES audience_cohort_overlap_groups (
            tenant_id,
            group_id,
            group_version
        ),
    FOREIGN KEY (tenant_id, candidate_id, candidate_version)
        REFERENCES audience_cohort_candidates (
            tenant_id,
            candidate_id,
            candidate_version
        )
);

CREATE TABLE IF NOT EXISTS audience_cohort_duplicate_suppressions (
    tenant_id TEXT NOT NULL,
    analysis_fingerprint CHAR(64) NOT NULL,
    retained_candidate_id TEXT NOT NULL,
    retained_candidate_version INTEGER NOT NULL CHECK (
        retained_candidate_version >= 1
    ),
    candidate_fingerprint CHAR(64) NOT NULL,
    suppressed_occurrence_count INTEGER NOT NULL CHECK (
        suppressed_occurrence_count >= 1
    ),
    suppression_reason TEXT NOT NULL CHECK (
        suppression_reason = 'identical_candidate_fingerprint'
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (
        tenant_id,
        analysis_fingerprint,
        retained_candidate_id,
        retained_candidate_version
    ),
    FOREIGN KEY (tenant_id, analysis_fingerprint)
        REFERENCES audience_cohort_overlap_analysis_runs (
            tenant_id,
            analysis_fingerprint
        ),
    FOREIGN KEY (
        tenant_id,
        retained_candidate_id,
        retained_candidate_version
    ) REFERENCES audience_cohort_candidates (
        tenant_id,
        candidate_id,
        candidate_version
    )
);

CREATE OR REPLACE FUNCTION validate_module3_overlap_member_candidate()
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
            'Module 3.3 overlap member candidate fingerprint does not match immutable candidate';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_module3_overlap_member_candidate_validate
ON audience_cohort_overlap_group_members;
CREATE TRIGGER trg_module3_overlap_member_candidate_validate
BEFORE INSERT ON audience_cohort_overlap_group_members
FOR EACH ROW EXECUTE FUNCTION validate_module3_overlap_member_candidate();

CREATE OR REPLACE FUNCTION validate_module3_duplicate_suppression_candidate()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM audience_cohort_candidates candidate
        WHERE candidate.tenant_id = NEW.tenant_id
          AND candidate.candidate_id = NEW.retained_candidate_id
          AND candidate.candidate_version = NEW.retained_candidate_version
          AND candidate.candidate_fingerprint = NEW.candidate_fingerprint
    ) THEN
        RAISE EXCEPTION
            'Module 3.3 suppression fingerprint does not match immutable candidate';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_module3_duplicate_suppression_candidate_validate
ON audience_cohort_duplicate_suppressions;
CREATE TRIGGER trg_module3_duplicate_suppression_candidate_validate
BEFORE INSERT ON audience_cohort_duplicate_suppressions
FOR EACH ROW EXECUTE FUNCTION validate_module3_duplicate_suppression_candidate();

CREATE INDEX IF NOT EXISTS idx_cohort_overlap_runs_source
ON audience_cohort_overlap_analysis_runs (
    tenant_id,
    source_batch_fingerprint,
    created_at DESC
);

CREATE INDEX IF NOT EXISTS idx_cohort_overlap_groups_review
ON audience_cohort_overlap_groups (
    tenant_id,
    review_required,
    group_type,
    group_id
);

CREATE INDEX IF NOT EXISTS idx_cohort_overlap_members_candidate
ON audience_cohort_overlap_group_members (
    tenant_id,
    candidate_id,
    candidate_version
);

CREATE OR REPLACE FUNCTION prevent_module3_overlap_evidence_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION
        'Module 3.3 overlap and deduplication evidence is immutable; create a new fingerprinted analysis';
END;
$$;

DROP TRIGGER IF EXISTS trg_module3_overlap_runs_immutable
ON audience_cohort_overlap_analysis_runs;
CREATE TRIGGER trg_module3_overlap_runs_immutable
BEFORE UPDATE OR DELETE ON audience_cohort_overlap_analysis_runs
FOR EACH ROW EXECUTE FUNCTION prevent_module3_overlap_evidence_mutation();

DROP TRIGGER IF EXISTS trg_module3_overlap_groups_immutable
ON audience_cohort_overlap_groups;
CREATE TRIGGER trg_module3_overlap_groups_immutable
BEFORE UPDATE OR DELETE ON audience_cohort_overlap_groups
FOR EACH ROW EXECUTE FUNCTION prevent_module3_overlap_evidence_mutation();

DROP TRIGGER IF EXISTS trg_module3_overlap_members_immutable
ON audience_cohort_overlap_group_members;
CREATE TRIGGER trg_module3_overlap_members_immutable
BEFORE UPDATE OR DELETE ON audience_cohort_overlap_group_members
FOR EACH ROW EXECUTE FUNCTION prevent_module3_overlap_evidence_mutation();

DROP TRIGGER IF EXISTS trg_module3_duplicate_suppressions_immutable
ON audience_cohort_duplicate_suppressions;
CREATE TRIGGER trg_module3_duplicate_suppressions_immutable
BEFORE UPDATE OR DELETE ON audience_cohort_duplicate_suppressions
FOR EACH ROW EXECUTE FUNCTION prevent_module3_overlap_evidence_mutation();

ALTER TABLE audience_cohort_overlap_analysis_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_cohort_overlap_analysis_runs FORCE ROW LEVEL SECURITY;
ALTER TABLE audience_cohort_overlap_groups ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_cohort_overlap_groups FORCE ROW LEVEL SECURITY;
ALTER TABLE audience_cohort_overlap_group_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_cohort_overlap_group_members FORCE ROW LEVEL SECURITY;
ALTER TABLE audience_cohort_duplicate_suppressions ENABLE ROW LEVEL SECURITY;
ALTER TABLE audience_cohort_duplicate_suppressions FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS audience_cohort_overlap_runs_tenant_policy
ON audience_cohort_overlap_analysis_runs;
CREATE POLICY audience_cohort_overlap_runs_tenant_policy
ON audience_cohort_overlap_analysis_runs
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS audience_cohort_overlap_groups_tenant_policy
ON audience_cohort_overlap_groups;
CREATE POLICY audience_cohort_overlap_groups_tenant_policy
ON audience_cohort_overlap_groups
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS audience_cohort_overlap_members_tenant_policy
ON audience_cohort_overlap_group_members;
CREATE POLICY audience_cohort_overlap_members_tenant_policy
ON audience_cohort_overlap_group_members
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS audience_cohort_duplicate_suppressions_tenant_policy
ON audience_cohort_duplicate_suppressions;
CREATE POLICY audience_cohort_duplicate_suppressions_tenant_policy
ON audience_cohort_duplicate_suppressions
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

REVOKE ALL ON audience_cohort_overlap_analysis_runs FROM PUBLIC;
REVOKE ALL ON audience_cohort_overlap_groups FROM PUBLIC;
REVOKE ALL ON audience_cohort_overlap_group_members FROM PUBLIC;
REVOKE ALL ON audience_cohort_duplicate_suppressions FROM PUBLIC;
