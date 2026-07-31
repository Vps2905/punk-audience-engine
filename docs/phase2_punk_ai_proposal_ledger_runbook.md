# Phase 2 Punk AI Proposal Ledger

## Purpose

The proposal ledger makes the Punk AI audience-proposal boundary durable
without enabling activation or export.

For each tenant, campaign and idempotency key, it records:

- a SHA-256 fingerprint of the normalized request;
- the immutable privacy-safe proposal response;
- the exact feature-set version and contract version;
- the safety, approval and activation eligibility decision.

It does not store embeddings, lineage, provider source references, database
credentials, raw identifiers, device observations or individual-level rows.

An identical retry returns the recorded response. Reusing the same
tenant/idempotency key with changed inputs returns HTTP `409`.

## Migration

Migration `0005_punk_ai_audience_proposal_ledger.sql` requires verified
migration `0004` on the separate Punk-owned feature database.

Apply it only through the dedicated operator command:

```bash
PYTHONPATH=. python scripts/apply_phase2_proposal_ledger_migration.py \
  --confirm-punk-owned-target
```

The migration creates:

- the immutable `punk_ai_audience_proposals` table;
- tenant-scoped forced PostgreSQL row-level security;
- a tenant/idempotency uniqueness boundary;
- an update/delete rejection trigger;
- a tenant/campaign lookup index;
- hard checks that a proposal record cannot enable downstream export.

The command does not modify the historical source, feature rows, activation
state or destination systems.

## Proposal runtime identity

Never use the feature reader, feature writer or migration administrator to
write proposal records. Configure a separate runtime identity:

```dotenv
PHASE2_PROPOSAL_RUNTIME_USER=punk_proposal_runtime
PHASE2_PROPOSAL_RUNTIME_PASSWORD=
AUDIENCE_PROPOSAL_DATABASE_URL=
```

Provision it after migration `0005`:

```bash
PYTHONPATH=. python scripts/provision_phase2_proposal_runtime_role.py \
  --tenant-id <tenant-id> \
  --feature-set-id <verified-feature-set-id> \
  --feature-set-version <verified-version> \
  --confirm-punk-owned-target
```

The provisioner verifies that the runtime identity:

- is not a superuser and cannot bypass RLS;
- cannot create database or schema objects;
- cannot read or write feature tables;
- cannot read the migration ledger;
- can only `SELECT` and `INSERT` proposal records;
- cannot update or delete immutable proposals;
- can see its rollback-only verification row for the requested tenant;
- sees zero rows after switching to an isolation-probe tenant.

The verification row is rolled back. It never becomes a stored proposal.

## Runtime behavior

The API requires both explicit runtime URLs:

- `AUDIENCE_FEATURE_DATABASE_URL`: feature reader;
- `AUDIENCE_PROPOSAL_DATABASE_URL`: proposal runtime.

Neither URL falls back to the historical source or migration identity.

`POST /api/audience-intelligence/punk-ai/v1/proposals` performs:

1. API-key and signed-tenant verification;
2. terminal privacy and export-action eligibility checks;
3. canonical request fingerprinting;
4. tenant-scoped idempotency lookup;
5. structured eligibility filtering and hybrid retrieval when no record
   exists;
6. privacy-safe response validation;
7. immutable insert or atomic conflict/replay resolution.

Raw-identifier disclosure requests and export-action-only requests preserve
their specific blocked reason, return zero candidates, skip semantic ranking
and cannot enable downstream export.

Historical proposals continue to return:

- `approval_status: blocked_historical_source`;
- `activation_eligible: false`;
- `safe_export_eligible: false`;
- `downstream_export_enabled: false`.

Proposal persistence is not approval and cannot authorize activation.
