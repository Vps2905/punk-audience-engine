# Audience Job Tenant Isolation Runbook

This change makes every asynchronous Audience Intelligence job tenant-owned.
The API propagates the verified tenant identity into background execution, the
local development store uses tenant-specific directories, and PostgreSQL
enforces the same boundary with a composite key and row-level security (RLS).

## Deployment order

1. Confirm the application role can call `set_config` for `app.tenant_id` and
   has only the table privileges required by the service.
2. If the existing table has no `tenant_id` column, add it in the controlled
   migration window:

   ```sql
   ALTER TABLE public.audience_jobs
   ADD COLUMN IF NOT EXISTS tenant_id TEXT;
   ```

3. Inspect legacy job ownership before applying migration `0015`:

   ```sql
   SELECT count(*) AS unassigned_jobs
   FROM public.audience_jobs
   WHERE tenant_id IS NULL OR btrim(tenant_id) = '';
   ```

4. Assign every legacy row to its verified owner using an approved ownership
   mapping. Store tenant identifiers in canonical lowercase. Do not use a
   shared, guessed, or default tenant.
5. Apply `migrations/0015_audience_job_tenant_isolation.sql`.
6. Deploy the application patch. Keep
   `AUDIENCE_JOB_STORE_BACKEND=postgres` in production.

Migration `0015` intentionally stops when any existing row lacks tenant
ownership. This is a safety gate, not a recoverable warning.

## Verification

In separate transactions, set a tenant context before querying:

```sql
BEGIN;
SELECT set_config('app.tenant_id', 'verified-tenant-a', true);
SELECT tenant_id, job_id, status
FROM public.audience_jobs
ORDER BY updated_at DESC
LIMIT 20;
ROLLBACK;
```

Repeat with another verified tenant. No rows from the first tenant should be
visible. Also confirm:

```sql
SELECT relrowsecurity, relforcerowsecurity
FROM pg_class
WHERE oid = 'public.audience_jobs'::regclass;
```

Both values must be `true`. API checks should return `404` when one tenant uses
another tenant's job identifier for status, result, or approval routes.

## Local development behavior

Local jobs are stored at:

```text
data/audience_jobs/<tenant-id>/<job-id>.json
```

The store does not read the former global `<job-id>.json` path. Any legacy
local job must be explicitly moved into its verified tenant directory.

## Rollback

Roll back application code only after stopping workers started by the new
version. Do not disable RLS or remove `tenant_id` as an incident shortcut.
Restore the prior schema only from a reviewed backup and only after preserving
the tenant ownership mapping introduced by this migration.
