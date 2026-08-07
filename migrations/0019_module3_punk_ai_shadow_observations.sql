-- Module 3.6 immutable Punk AI proposal shadow observations.
--
-- This schema records alignment evidence for existing proposals. It cannot
-- create or modify proposals, mutate candidate lifecycle, route traffic,
-- activate audiences, or export data.

CREATE TABLE IF NOT EXISTS audience_module3_shadow_observation_runs (
    tenant_id TEXT NOT NULL,
    observation_fingerprint CHAR(64) NOT NULL,
    proposal_id TEXT NOT NULL,
    source_overlap_report_fingerprint CHAR(64) NOT NULL,
    lifecycle_evaluation_fingerprint CHAR(64) NOT NULL,
    policy_version TEXT NOT NULL,
    execution_mode TEXT NOT NULL CHECK (
        execution_mode IN (
            'historical_preview',
            'offline_evaluation',
            'production'
        )
    ),
    proposal_status TEXT NOT NULL,
    proposal_candidate_count INTEGER NOT NULL CHECK (
        proposal_candidate_count >= 0
    ),
    observation_count INTEGER NOT NULL CHECK (observation_count >= 0),
    aligned_candidate_count INTEGER NOT NULL CHECK (
        aligned_candidate_count >= 0
    ),
    blocked_candidate_count INTEGER NOT NULL CHECK (
        blocked_candidate_count >= 0
    ),
    unmatched_candidate_count INTEGER NOT NULL CHECK (
        unmatched_candidate_count >= 0
    ),
    ambiguous_candidate_count INTEGER NOT NULL CHECK (
        ambiguous_candidate_count >= 0
    ),
    shadow_alignment_passed BOOLEAN NOT NULL DEFAULT FALSE,
    report JSONB NOT NULL CHECK (jsonb_typeof(report) = 'object'),

    raw_identifiers_stored BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        raw_identifiers_stored = FALSE
    ),
    audience_membership_read BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        audience_membership_read = FALSE
    ),
    proposal_created BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        proposal_created = FALSE
    ),
    proposal_modified BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        proposal_modified = FALSE
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
    production_routing_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        production_routing_enabled = FALSE
    ),
    activation_or_export_performed BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        activation_or_export_performed = FALSE
    ),
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        downstream_export_enabled = FALSE
    ),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (tenant_id, observation_fingerprint),

    FOREIGN KEY (tenant_id, proposal_id)
    REFERENCES public.punk_ai_audience_proposals (
        tenant_id,
        proposal_id
    ),

    FOREIGN KEY (
        tenant_id,
        source_overlap_report_fingerprint
    ) REFERENCES audience_cohort_overlap_analysis_runs (
        tenant_id,
        analysis_fingerprint
    ),

    FOREIGN KEY (
        tenant_id,
        lifecycle_evaluation_fingerprint
    ) REFERENCES audience_cohort_lifecycle_evaluations (
        tenant_id,
        evaluation_fingerprint
    )
);


CREATE TABLE IF NOT EXISTS audience_module3_shadow_candidate_observations (
    tenant_id TEXT NOT NULL,
    observation_fingerprint CHAR(64) NOT NULL,
    candidate_observation_fingerprint CHAR(64) NOT NULL,
    proposal_id TEXT NOT NULL,
    proposal_rank INTEGER NOT NULL CHECK (proposal_rank >= 1),
    feature_id TEXT NOT NULL,
    candidate_id TEXT,
    candidate_version INTEGER CHECK (
        candidate_version IS NULL OR candidate_version >= 1
    ),
    candidate_fingerprint CHAR(64),
    recommended_lifecycle_status TEXT,
    alignment_status TEXT NOT NULL CHECK (
        alignment_status IN (
            'aligned_historical_preview',
            'aligned_manual_shadow_review',
            'blocked_lifecycle',
            'unmatched_candidate',
            'ambiguous_candidate'
        )
    ),
    reason_codes JSONB NOT NULL CHECK (jsonb_typeof(reason_codes) = 'array'),

    proposal_modified BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        proposal_modified = FALSE
    ),
    candidate_lifecycle_mutated BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        candidate_lifecycle_mutated = FALSE
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

    PRIMARY KEY (tenant_id, candidate_observation_fingerprint),

    UNIQUE (tenant_id, observation_fingerprint, proposal_rank),

    FOREIGN KEY (
        tenant_id,
        observation_fingerprint
    ) REFERENCES audience_module3_shadow_observation_runs (
        tenant_id,
        observation_fingerprint
    ),

    FOREIGN KEY (tenant_id, proposal_id)
    REFERENCES public.punk_ai_audience_proposals (
        tenant_id,
        proposal_id
    ),

    FOREIGN KEY (tenant_id, candidate_id, candidate_version)
    REFERENCES audience_cohort_candidates (
        tenant_id,
        candidate_id,
        candidate_version
    ),

    CHECK (
        (candidate_id IS NULL
            AND candidate_version IS NULL
            AND candidate_fingerprint IS NULL)
        OR
        (candidate_id IS NOT NULL
            AND candidate_version IS NOT NULL
            AND candidate_fingerprint IS NOT NULL)
    )
);


CREATE OR REPLACE FUNCTION validate_module3_shadow_candidate_fingerprint()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.candidate_id IS NOT NULL AND NOT EXISTS (
        SELECT 1
        FROM audience_cohort_candidates candidate
        WHERE candidate.tenant_id = NEW.tenant_id
          AND candidate.candidate_id = NEW.candidate_id
          AND candidate.candidate_version = NEW.candidate_version
          AND candidate.candidate_fingerprint = NEW.candidate_fingerprint
    ) THEN
        RAISE EXCEPTION
            'Module 3.6 candidate fingerprint does not match immutable candidate';
    END IF;

    RETURN NEW;
END;
$$;


DROP TRIGGER IF EXISTS trg_module3_shadow_candidate_validate
ON audience_module3_shadow_candidate_observations;

CREATE TRIGGER trg_module3_shadow_candidate_validate
BEFORE INSERT ON audience_module3_shadow_candidate_observations
FOR EACH ROW
EXECUTE FUNCTION validate_module3_shadow_candidate_fingerprint();


CREATE OR REPLACE FUNCTION prevent_module3_shadow_evidence_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION
        'Module 3.6 shadow evidence is immutable; create a new fingerprinted observation';
END;
$$;


DROP TRIGGER IF EXISTS trg_module3_shadow_runs_immutable
ON audience_module3_shadow_observation_runs;

CREATE TRIGGER trg_module3_shadow_runs_immutable
BEFORE UPDATE OR DELETE ON audience_module3_shadow_observation_runs
FOR EACH ROW
EXECUTE FUNCTION prevent_module3_shadow_evidence_mutation();


DROP TRIGGER IF EXISTS trg_module3_shadow_candidates_immutable
ON audience_module3_shadow_candidate_observations;

CREATE TRIGGER trg_module3_shadow_candidates_immutable
BEFORE UPDATE OR DELETE ON audience_module3_shadow_candidate_observations
FOR EACH ROW
EXECUTE FUNCTION prevent_module3_shadow_evidence_mutation();


CREATE INDEX IF NOT EXISTS idx_module3_shadow_runs_proposal
ON audience_module3_shadow_observation_runs (
    tenant_id,
    proposal_id,
    created_at DESC
);


CREATE INDEX IF NOT EXISTS idx_module3_shadow_candidates_alignment
ON audience_module3_shadow_candidate_observations (
    tenant_id,
    alignment_status,
    proposal_id,
    proposal_rank
);


ALTER TABLE audience_module3_shadow_observation_runs
ENABLE ROW LEVEL SECURITY;

ALTER TABLE audience_module3_shadow_observation_runs
FORCE ROW LEVEL SECURITY;

ALTER TABLE audience_module3_shadow_candidate_observations
ENABLE ROW LEVEL SECURITY;

ALTER TABLE audience_module3_shadow_candidate_observations
FORCE ROW LEVEL SECURITY;


DROP POLICY IF EXISTS audience_module3_shadow_runs_tenant_policy
ON audience_module3_shadow_observation_runs;

CREATE POLICY audience_module3_shadow_runs_tenant_policy
ON audience_module3_shadow_observation_runs
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));


DROP POLICY IF EXISTS audience_module3_shadow_candidates_tenant_policy
ON audience_module3_shadow_candidate_observations;

CREATE POLICY audience_module3_shadow_candidates_tenant_policy
ON audience_module3_shadow_candidate_observations
USING (tenant_id = current_setting('app.tenant_id', true))
WITH CHECK (tenant_id = current_setting('app.tenant_id', true));


REVOKE ALL ON audience_module3_shadow_observation_runs FROM PUBLIC;
REVOKE ALL ON audience_module3_shadow_candidate_observations FROM PUBLIC;
