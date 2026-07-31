CREATE TABLE IF NOT EXISTS public.punk_ai_audience_proposals (
    tenant_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_fingerprint CHAR(64) NOT NULL CHECK (
        request_fingerprint ~ '^[0-9a-f]{64}$'
    ),
    contract_version TEXT NOT NULL,
    feature_set_id TEXT NOT NULL,
    feature_set_version INTEGER NOT NULL CHECK (
        feature_set_version >= 1
    ),
    execution_mode TEXT NOT NULL CHECK (
        execution_mode IN (
            'historical_preview',
            'offline_evaluation',
            'production'
        )
    ),
    status TEXT NOT NULL,
    approval_status TEXT NOT NULL,
    activation_eligible BOOLEAN NOT NULL DEFAULT FALSE,
    safe_export_eligible BOOLEAN NOT NULL DEFAULT FALSE,
    downstream_export_enabled BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        downstream_export_enabled = FALSE
    ),
    response_document JSONB NOT NULL CHECK (
        jsonb_typeof(response_document) = 'object'
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (
        tenant_id,
        proposal_id
    ),
    UNIQUE (
        tenant_id,
        idempotency_key
    ),
    FOREIGN KEY (
        tenant_id,
        feature_set_id,
        feature_set_version
    )
    REFERENCES public.audience_feature_sets (
        tenant_id,
        feature_set_id,
        version
    ),
    CHECK (
        execution_mode = 'production'
        OR activation_eligible = FALSE
    ),
    CHECK (
        activation_eligible = TRUE
        OR safe_export_eligible = FALSE
    )
);

CREATE INDEX IF NOT EXISTS idx_punk_ai_audience_proposals_lookup
ON public.punk_ai_audience_proposals (
    tenant_id,
    campaign_id,
    created_at DESC
);

ALTER TABLE public.punk_ai_audience_proposals
ENABLE ROW LEVEL SECURITY;

ALTER TABLE public.punk_ai_audience_proposals
FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS punk_ai_audience_proposals_tenant_policy
ON public.punk_ai_audience_proposals;

CREATE POLICY punk_ai_audience_proposals_tenant_policy
ON public.punk_ai_audience_proposals
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

CREATE OR REPLACE FUNCTION public.reject_punk_ai_audience_proposal_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION
        'Punk AI audience proposal records are immutable';
END;
$$;

DROP TRIGGER IF EXISTS trg_reject_punk_ai_audience_proposal_mutation
ON public.punk_ai_audience_proposals;

CREATE TRIGGER trg_reject_punk_ai_audience_proposal_mutation
BEFORE UPDATE OR DELETE
ON public.punk_ai_audience_proposals
FOR EACH ROW
EXECUTE FUNCTION public.reject_punk_ai_audience_proposal_mutation();
