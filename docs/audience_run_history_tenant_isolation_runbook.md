# Audience run-history tenant-isolation runbook

Migration `0017_audience_run_history_tenant_isolation.sql` makes run history,
cohorts, artifacts, warnings, audit events, approvals, and privacy-budget records
tenant-owned. It deliberately has no fallback or default tenant.

## Before the migration

1. Back up the database and test the migration on a production-shaped copy.
2. Stop audience run creation, approval, rejection, Meta seed generation, and
   privacy-budget spending for the migration window.
3. Add and populate `tenant_id` on `public.audience_run_history` from an
   authoritative ownership source. Values must be lowercase and match
   `^[a-z0-9][a-z0-9_.-]{0,127}$`.
4. Assign any privacy-ledger row whose `run_id` has no parent history row from
   the same authoritative source. Do not assign these rows to a shared,
   global, or guessed tenant.
5. Confirm that each legacy `run_id` maps to exactly one tenant. The migration
   derives child run-table ownership only from that explicit parent mapping.

Useful preflight checks:

```sql
SELECT run_id, count(DISTINCT tenant_id)
FROM public.audience_run_history
GROUP BY run_id
HAVING count(DISTINCT tenant_id) <> 1;

SELECT count(*)
FROM public.audience_run_history
WHERE tenant_id IS NULL OR btrim(tenant_id) = '';

SELECT count(*)
FROM public.audience_privacy_budget_ledger ledger
LEFT JOIN public.audience_run_history history
  ON history.run_id = ledger.run_id
WHERE ledger.tenant_id IS NULL
  AND history.run_id IS NULL;
```

## Apply

Run the migration with the deployment role in a transaction. A failed explicit
ownership check must stop the deployment; correct the ownership source and
rerun it. Do not edit the migration to add a default tenant.

```bash
psql "$DATABASE_URL" \
  -v ON_ERROR_STOP=1 \
  -f migrations/0017_audience_run_history_tenant_isolation.sql
```

The application role must be subject to row-level security and must not have
`BYPASSRLS`. Each application transaction sets its authenticated tenant with:

```sql
SELECT set_config('app.tenant_id', 'tenant-a', true);
```

## Validate before resuming traffic

- Confirm all seven tables have `tenant_id NOT NULL` and canonical-format
  constraints.
- Confirm `(tenant_id, run_id)` is unique on run history and is used by child
  foreign keys.
- Confirm RLS is both enabled and forced on all seven tables.
- Confirm the application role has no `BYPASSRLS` and `PUBLIC` has no table
  privileges.
- With two test tenants, reuse the same `run_id`, cohort ID, and budget scope.
  Each tenant must see only its own run, audit trail, and budget spend.
- An unset or incorrect `app.tenant_id` must return no rows and reject writes.
- Updates/deletes against events, approvals, and the privacy ledger must fail;
  those records are append-only.

## Recovery

If validation fails, keep traffic stopped and restore the pre-migration backup.
Do not disable forced RLS as a temporary workaround. Ownership, audit, and
privacy-budget isolation are one release boundary and must be rolled back or
forward together.
