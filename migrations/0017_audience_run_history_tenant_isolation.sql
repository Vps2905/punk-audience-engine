-- Tenant-isolate run history, normalized cohorts, approval/audit records,
-- and privacy-budget state.
--
-- Existing ownership is intentionally not guessed. Operators must assign
-- tenant_id on audience_run_history (and any orphan privacy-ledger rows)
-- before rerunning this migration. Child run records are then derived only
-- from their explicitly assigned parent run.

ALTER TABLE IF EXISTS public.audience_run_history
ADD COLUMN IF NOT EXISTS tenant_id TEXT;

ALTER TABLE IF EXISTS public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS tenant_id TEXT;

ALTER TABLE IF EXISTS public.audience_run_artifacts
ADD COLUMN IF NOT EXISTS tenant_id TEXT;

ALTER TABLE IF EXISTS public.audience_run_warnings
ADD COLUMN IF NOT EXISTS tenant_id TEXT;

ALTER TABLE IF EXISTS public.audience_run_events
ADD COLUMN IF NOT EXISTS tenant_id TEXT;

ALTER TABLE IF EXISTS public.audience_run_approvals
ADD COLUMN IF NOT EXISTS tenant_id TEXT;

ALTER TABLE IF EXISTS public.audience_privacy_budget_ledger
ADD COLUMN IF NOT EXISTS tenant_id TEXT;

-- Reconcile older schema-reference deployments with the normalized runtime
-- contract before tenant keys and foreign keys are created.
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS cohort_id TEXT;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS export_cohort_id TEXT;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS audience_name TEXT;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS location_name TEXT;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS primary_poi_type TEXT;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS created_day_part TEXT;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS lookback_bucket TEXT;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS management_quality_score DOUBLE PRECISION;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS approval_status TEXT;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS privacy_mode TEXT;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS data_safety_status TEXT;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS risk_decision TEXT;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS risk_level TEXT;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS cohort_payload JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE public.audience_run_cohorts
ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE public.audience_run_artifacts
ADD COLUMN IF NOT EXISTS artifact_payload JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE public.audience_run_artifacts
ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE public.audience_run_warnings
ADD COLUMN IF NOT EXISTS warning_payload JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE public.audience_run_warnings
ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE public.audience_run_approvals
ADD COLUMN IF NOT EXISTS decision TEXT;
ALTER TABLE public.audience_run_approvals
ADD COLUMN IF NOT EXISTS action TEXT;
ALTER TABLE public.audience_run_approvals
ADD COLUMN IF NOT EXISTS previous_status TEXT;
ALTER TABLE public.audience_run_approvals
ADD COLUMN IF NOT EXISTS new_status TEXT;
ALTER TABLE public.audience_run_approvals
ADD COLUMN IF NOT EXISTS details JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE public.audience_run_approvals
ADD COLUMN IF NOT EXISTS privacy_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE public.audience_run_approvals
ADD COLUMN IF NOT EXISTS artifacts_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM public.audience_run_history
        WHERE tenant_id IS NULL OR btrim(tenant_id) = ''
    ) THEN
        RAISE EXCEPTION
            'Legacy audience_run_history rows require explicit tenant assignment before migration 0017';
    END IF;
END;
$$;

UPDATE public.audience_run_cohorts child
SET tenant_id = parent.tenant_id
FROM public.audience_run_history parent
WHERE child.tenant_id IS NULL
  AND child.run_id = parent.run_id;

UPDATE public.audience_run_artifacts child
SET tenant_id = parent.tenant_id
FROM public.audience_run_history parent
WHERE child.tenant_id IS NULL
  AND child.run_id = parent.run_id;

UPDATE public.audience_run_warnings child
SET tenant_id = parent.tenant_id
FROM public.audience_run_history parent
WHERE child.tenant_id IS NULL
  AND child.run_id = parent.run_id;

UPDATE public.audience_run_events child
SET tenant_id = parent.tenant_id
FROM public.audience_run_history parent
WHERE child.tenant_id IS NULL
  AND child.run_id = parent.run_id;

UPDATE public.audience_run_approvals child
SET tenant_id = parent.tenant_id
FROM public.audience_run_history parent
WHERE child.tenant_id IS NULL
  AND child.run_id = parent.run_id;

UPDATE public.audience_privacy_budget_ledger ledger
SET tenant_id = parent.tenant_id
FROM public.audience_run_history parent
WHERE ledger.tenant_id IS NULL
  AND ledger.run_id = parent.run_id;

UPDATE public.audience_run_cohorts
SET export_cohort_id = COALESCE(
    NULLIF(btrim(export_cohort_id), ''),
    NULLIF(btrim(cohort_id), ''),
    'legacy_' || md5(tenant_id || ':' || run_id || ':' || id::text)
);

UPDATE public.audience_run_cohorts
SET metadata = cohort_payload
WHERE metadata = '{}'::jsonb
  AND cohort_payload <> '{}'::jsonb;

UPDATE public.audience_run_artifacts
SET metadata = artifact_payload
WHERE metadata = '{}'::jsonb
  AND artifact_payload <> '{}'::jsonb;

UPDATE public.audience_run_warnings
SET metadata = warning_payload
WHERE metadata = '{}'::jsonb
  AND warning_payload <> '{}'::jsonb;

UPDATE public.audience_run_approvals
SET action = COALESCE(
        NULLIF(btrim(action), ''),
        NULLIF(btrim(decision), ''),
        'legacy_recorded'
    ),
    new_status = COALESCE(
        new_status,
        NULLIF(btrim(decision), '')
    ),
    privacy_snapshot = CASE
        WHEN privacy_snapshot = '{}'::jsonb THEN details
        ELSE privacy_snapshot
    END;

DO $$
DECLARE
    table_name TEXT;
    unassigned_count BIGINT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'audience_run_history',
        'audience_run_cohorts',
        'audience_run_artifacts',
        'audience_run_warnings',
        'audience_run_events',
        'audience_run_approvals',
        'audience_privacy_budget_ledger'
    ]
    LOOP
        EXECUTE format(
            'SELECT count(*) FROM public.%I WHERE tenant_id IS NULL OR btrim(tenant_id) = ''''',
            table_name
        ) INTO unassigned_count;

        IF unassigned_count > 0 THEN
            RAISE EXCEPTION
                'Legacy rows in % require explicit tenant assignment before migration 0017',
                table_name;
        END IF;

        EXECUTE format(
            'SELECT count(*) FROM public.%I WHERE tenant_id !~ ''^[a-z0-9][a-z0-9_.-]{0,127}$''',
            table_name
        ) INTO unassigned_count;

        IF unassigned_count > 0 THEN
            RAISE EXCEPTION
                'Invalid or non-canonical tenant identifiers exist in %',
                table_name;
        END IF;

        EXECUTE format(
            'ALTER TABLE public.%I ALTER COLUMN tenant_id SET NOT NULL',
            table_name
        );

        EXECUTE format(
            'ALTER TABLE public.%I DROP CONSTRAINT IF EXISTS %I',
            table_name,
            table_name || '_tenant_id_format'
        );

        EXECUTE format(
            'ALTER TABLE public.%I ADD CONSTRAINT %I CHECK (tenant_id ~ ''^[a-z0-9][a-z0-9_.-]{0,127}$'')',
            table_name,
            table_name || '_tenant_id_format'
        );
    END LOOP;
END;
$$;

ALTER TABLE public.audience_run_history
DROP CONSTRAINT IF EXISTS audience_run_history_run_id_key;

ALTER TABLE public.audience_run_history
DROP CONSTRAINT IF EXISTS audience_run_history_tenant_run_key;

ALTER TABLE public.audience_run_history
ADD CONSTRAINT audience_run_history_tenant_run_key
UNIQUE (tenant_id, run_id);

DELETE FROM public.audience_run_cohorts older
USING public.audience_run_cohorts newer
WHERE older.tenant_id = newer.tenant_id
  AND older.run_id = newer.run_id
  AND older.export_cohort_id = newer.export_cohort_id
  AND older.id < newer.id;

ALTER TABLE public.audience_run_cohorts
ALTER COLUMN export_cohort_id SET NOT NULL;

ALTER TABLE public.audience_run_approvals
ALTER COLUMN action SET NOT NULL;

ALTER TABLE public.audience_run_approvals
ALTER COLUMN decision DROP NOT NULL;

DROP INDEX IF EXISTS public.uq_audience_run_cohorts_run_export;

ALTER TABLE public.audience_run_cohorts
DROP CONSTRAINT IF EXISTS
audience_run_cohorts_run_id_export_cohort_id_key;

CREATE UNIQUE INDEX IF NOT EXISTS
uq_audience_run_cohorts_tenant_run_export
ON public.audience_run_cohorts (
    tenant_id,
    run_id,
    export_cohort_id
);

ALTER TABLE public.audience_run_cohorts
DROP CONSTRAINT IF EXISTS audience_run_cohorts_tenant_run_fkey;

ALTER TABLE public.audience_run_cohorts
ADD CONSTRAINT audience_run_cohorts_tenant_run_fkey
FOREIGN KEY (tenant_id, run_id)
REFERENCES public.audience_run_history (tenant_id, run_id)
NOT VALID;

ALTER TABLE public.audience_run_artifacts
DROP CONSTRAINT IF EXISTS audience_run_artifacts_tenant_run_fkey;

ALTER TABLE public.audience_run_artifacts
ADD CONSTRAINT audience_run_artifacts_tenant_run_fkey
FOREIGN KEY (tenant_id, run_id)
REFERENCES public.audience_run_history (tenant_id, run_id)
NOT VALID;

ALTER TABLE public.audience_run_warnings
DROP CONSTRAINT IF EXISTS audience_run_warnings_tenant_run_fkey;

ALTER TABLE public.audience_run_warnings
ADD CONSTRAINT audience_run_warnings_tenant_run_fkey
FOREIGN KEY (tenant_id, run_id)
REFERENCES public.audience_run_history (tenant_id, run_id)
NOT VALID;

ALTER TABLE public.audience_run_events
DROP CONSTRAINT IF EXISTS audience_run_events_tenant_run_fkey;

ALTER TABLE public.audience_run_events
ADD CONSTRAINT audience_run_events_tenant_run_fkey
FOREIGN KEY (tenant_id, run_id)
REFERENCES public.audience_run_history (tenant_id, run_id)
NOT VALID;

ALTER TABLE public.audience_run_approvals
DROP CONSTRAINT IF EXISTS audience_run_approvals_tenant_run_fkey;

ALTER TABLE public.audience_run_approvals
ADD CONSTRAINT audience_run_approvals_tenant_run_fkey
FOREIGN KEY (tenant_id, run_id)
REFERENCES public.audience_run_history (tenant_id, run_id)
NOT VALID;

ALTER TABLE public.audience_run_cohorts
VALIDATE CONSTRAINT audience_run_cohorts_tenant_run_fkey;
ALTER TABLE public.audience_run_artifacts
VALIDATE CONSTRAINT audience_run_artifacts_tenant_run_fkey;
ALTER TABLE public.audience_run_warnings
VALIDATE CONSTRAINT audience_run_warnings_tenant_run_fkey;
ALTER TABLE public.audience_run_events
VALIDATE CONSTRAINT audience_run_events_tenant_run_fkey;
ALTER TABLE public.audience_run_approvals
VALIDATE CONSTRAINT audience_run_approvals_tenant_run_fkey;

CREATE INDEX IF NOT EXISTS idx_audience_run_history_tenant_created
ON public.audience_run_history (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audience_run_cohorts_tenant_run
ON public.audience_run_cohorts (tenant_id, run_id);

CREATE INDEX IF NOT EXISTS idx_audience_run_artifacts_tenant_run
ON public.audience_run_artifacts (tenant_id, run_id);

CREATE INDEX IF NOT EXISTS idx_audience_run_warnings_tenant_run
ON public.audience_run_warnings (tenant_id, run_id);

CREATE INDEX IF NOT EXISTS idx_audience_run_events_tenant_run
ON public.audience_run_events (tenant_id, run_id);

CREATE INDEX IF NOT EXISTS idx_audience_run_approvals_tenant_run
ON public.audience_run_approvals (tenant_id, run_id);

CREATE INDEX IF NOT EXISTS idx_audience_privacy_budget_tenant_scope
ON public.audience_privacy_budget_ledger (tenant_id, budget_scope);

CREATE INDEX IF NOT EXISTS idx_audience_privacy_budget_tenant_run
ON public.audience_privacy_budget_ledger (tenant_id, run_id);

CREATE OR REPLACE FUNCTION public.prevent_audience_run_ownership_change()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.tenant_id IS DISTINCT FROM OLD.tenant_id
       OR NEW.run_id IS DISTINCT FROM OLD.run_id
    THEN
        RAISE EXCEPTION 'Audience run tenant and run identity are immutable';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.prevent_audience_cohort_identity_change()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.tenant_id IS DISTINCT FROM OLD.tenant_id
       OR NEW.run_id IS DISTINCT FROM OLD.run_id
       OR NEW.export_cohort_id IS DISTINCT FROM OLD.export_cohort_id
    THEN
        RAISE EXCEPTION 'Audience cohort tenant and cohort identity are immutable';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.prevent_audience_append_only_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'Audience audit and privacy-ledger records are append-only';
END;
$$;

DROP TRIGGER IF EXISTS trg_audience_run_history_identity_immutable
ON public.audience_run_history;
CREATE TRIGGER trg_audience_run_history_identity_immutable
BEFORE UPDATE ON public.audience_run_history
FOR EACH ROW EXECUTE FUNCTION public.prevent_audience_run_ownership_change();

DROP TRIGGER IF EXISTS trg_audience_run_cohorts_identity_immutable
ON public.audience_run_cohorts;
CREATE TRIGGER trg_audience_run_cohorts_identity_immutable
BEFORE UPDATE ON public.audience_run_cohorts
FOR EACH ROW EXECUTE FUNCTION public.prevent_audience_cohort_identity_change();

DROP TRIGGER IF EXISTS trg_audience_run_artifacts_identity_immutable
ON public.audience_run_artifacts;
CREATE TRIGGER trg_audience_run_artifacts_identity_immutable
BEFORE UPDATE ON public.audience_run_artifacts
FOR EACH ROW EXECUTE FUNCTION public.prevent_audience_run_ownership_change();

DROP TRIGGER IF EXISTS trg_audience_run_warnings_identity_immutable
ON public.audience_run_warnings;
CREATE TRIGGER trg_audience_run_warnings_identity_immutable
BEFORE UPDATE ON public.audience_run_warnings
FOR EACH ROW EXECUTE FUNCTION public.prevent_audience_run_ownership_change();

DROP TRIGGER IF EXISTS trg_audience_run_events_append_only
ON public.audience_run_events;
CREATE TRIGGER trg_audience_run_events_append_only
BEFORE UPDATE OR DELETE ON public.audience_run_events
FOR EACH ROW EXECUTE FUNCTION public.prevent_audience_append_only_mutation();

DROP TRIGGER IF EXISTS trg_audience_run_approvals_append_only
ON public.audience_run_approvals;
CREATE TRIGGER trg_audience_run_approvals_append_only
BEFORE UPDATE OR DELETE ON public.audience_run_approvals
FOR EACH ROW EXECUTE FUNCTION public.prevent_audience_append_only_mutation();

DROP TRIGGER IF EXISTS trg_audience_privacy_budget_append_only
ON public.audience_privacy_budget_ledger;
CREATE TRIGGER trg_audience_privacy_budget_append_only
BEFORE UPDATE OR DELETE ON public.audience_privacy_budget_ledger
FOR EACH ROW EXECUTE FUNCTION public.prevent_audience_append_only_mutation();

DO $$
DECLARE
    table_name TEXT;
    policy_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'audience_run_history',
        'audience_run_cohorts',
        'audience_run_artifacts',
        'audience_run_warnings',
        'audience_run_events',
        'audience_run_approvals',
        'audience_privacy_budget_ledger'
    ]
    LOOP
        policy_name := table_name || '_tenant_policy';
        EXECUTE format(
            'ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY',
            table_name
        );
        EXECUTE format(
            'ALTER TABLE public.%I FORCE ROW LEVEL SECURITY',
            table_name
        );
        EXECUTE format(
            'DROP POLICY IF EXISTS %I ON public.%I',
            policy_name,
            table_name
        );
        EXECUTE format(
            'CREATE POLICY %I ON public.%I USING (tenant_id = current_setting(''app.tenant_id'', true)) WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))',
            policy_name,
            table_name
        );
        EXECUTE format(
            'REVOKE ALL ON public.%I FROM PUBLIC',
            table_name
        );
    END LOOP;
END;
$$;
