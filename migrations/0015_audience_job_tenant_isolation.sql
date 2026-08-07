CREATE TABLE IF NOT EXISTS public.audience_jobs (
    tenant_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    progress JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB,
    error TEXT,
    PRIMARY KEY (tenant_id, job_id)
);

ALTER TABLE public.audience_jobs
ADD COLUMN IF NOT EXISTS tenant_id TEXT;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM public.audience_jobs
        WHERE tenant_id IS NULL
           OR btrim(tenant_id) = ''
    ) THEN
        RAISE EXCEPTION
            'Legacy audience_jobs rows require explicit tenant assignment before migration 0015';
    END IF;
END;
$$;

ALTER TABLE public.audience_jobs
ALTER COLUMN tenant_id SET NOT NULL;

ALTER TABLE public.audience_jobs
DROP CONSTRAINT IF EXISTS audience_jobs_tenant_id_format;

ALTER TABLE public.audience_jobs
ADD CONSTRAINT audience_jobs_tenant_id_format CHECK (
    tenant_id ~ '^[a-z0-9][a-z0-9_.-]{0,127}$'
);

DO $$
DECLARE
    primary_key_name TEXT;
BEGIN
    SELECT constraint_name
    INTO primary_key_name
    FROM information_schema.table_constraints
    WHERE table_schema = 'public'
      AND table_name = 'audience_jobs'
      AND constraint_type = 'PRIMARY KEY';

    IF primary_key_name IS NOT NULL THEN
        EXECUTE format(
            'ALTER TABLE public.audience_jobs DROP CONSTRAINT %I',
            primary_key_name
        );
    END IF;
END;
$$;

ALTER TABLE public.audience_jobs
ADD CONSTRAINT audience_jobs_pkey PRIMARY KEY (
    tenant_id,
    job_id
);

DROP INDEX IF EXISTS public.idx_audience_jobs_status;

CREATE INDEX IF NOT EXISTS idx_audience_jobs_tenant_status
ON public.audience_jobs (
    tenant_id,
    status,
    updated_at DESC
);

CREATE OR REPLACE FUNCTION public.prevent_audience_job_identity_change()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.tenant_id IS DISTINCT FROM OLD.tenant_id
       OR NEW.job_id IS DISTINCT FROM OLD.job_id
    THEN
        RAISE EXCEPTION
            'Audience job tenant and job identity are immutable';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_audience_job_identity_immutable
ON public.audience_jobs;

CREATE TRIGGER trg_audience_job_identity_immutable
BEFORE UPDATE ON public.audience_jobs
FOR EACH ROW
EXECUTE FUNCTION public.prevent_audience_job_identity_change();

ALTER TABLE public.audience_jobs
ENABLE ROW LEVEL SECURITY;

ALTER TABLE public.audience_jobs
FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS audience_jobs_tenant_policy
ON public.audience_jobs;

CREATE POLICY audience_jobs_tenant_policy
ON public.audience_jobs
USING (
    tenant_id = current_setting('app.tenant_id', true)
)
WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
);

REVOKE ALL ON public.audience_jobs FROM PUBLIC;
