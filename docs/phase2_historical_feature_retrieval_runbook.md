# Phase 2: Historical Feature Retrieval and Punk AI Contract

## Purpose

Phase 2 makes the existing privacy-safe July data useful for internal
Audience Intelligence development without presenting it as current campaign
data.

The historical path is:

```text
Existing privacy-safe Postgres vector snapshot
    -> canonical feature adapter
    -> versioned feature registry
    -> true pgvector + lexical indexes
    -> structured eligibility filters
    -> reciprocal-rank fusion
    -> Punk AI audience proposal
    -> historical warning
    -> export blocked
```

When fresh provider data returns, Phase 1 canonical output can populate the
same feature contract. Punk AI does not need a different proposal API.

## Safety contract

Historical feature sets:

- are eligible for internal retrieval;
- retain the actual source timestamp and freshness status;
- are never eligible for activation;
- cannot be relabelled as production by the import command;
- cannot enable downstream export;
- contain privacy-safe cohort metadata only;
- do not read or store raw MAIDs, observations, coordinates, email, phone,
  database credentials, or API keys.

Semantic similarity cannot override tenant, privacy, rights, structured
location/category/daypart, freshness, data mode, approval, or export gates.

## Components

- `PostgresCanonicalFeatureAdapterService`: converts an existing safe vector
  snapshot to a versioned canonical feature set.
- `LegacyPostgresSafeVectorReader`: reads only the existing safe vector model,
  embedding, and metadata tables plus the latest source timestamp.
- `HistoricalVectorSnapshotInventoryService`: reports aggregate completeness
  evidence for legacy snapshots without returning embeddings or metadata rows.
- `PgvectorAudienceFeatureStore`: stores feature sets and 384-dimensional
  embeddings in PostgreSQL with pgvector.
- `AudienceFeatureProposalService`: produces reviewable historical or
  production proposals while preserving terminal safety decisions.
- `POST /api/audience-intelligence/punk-ai/v1/proposals`: versioned Punk AI
  integration boundary.
- `GET /api/audience-intelligence/punk-ai/v1/feature-sets/current`: safe
  feature-set freshness and eligibility status.
- `scripts/index_historical_audience_features.py`: operator-only historical
  indexing command.

## Database ownership

`ECHO_DATABASE_URL` is the read-only source used to locate the existing
privacy-safe vector snapshot and latest source timestamp.

The Phase 2 feature database uses three separate identities:

- `AUDIENCE_FEATURE_MIGRATION_DATABASE_URL`: operator-only schema identity;
- `AUDIENCE_FEATURE_WRITER_DATABASE_URL`: privacy-safe feature import writer;
- `AUDIENCE_FEATURE_DATABASE_URL`: read-only Punk AI/API identity.

All three resolve to the same Punk-owned feature database, which must remain
separate from the historical source. None may fall back to
`ECHO_DATABASE_URL`.

The Punk AI proposal ledger adds a fourth, narrowly scoped runtime identity.
It can read and insert immutable proposal records but cannot read or write
feature tables. See `docs/phase2_punk_ai_proposal_ledger_runbook.md`.

Do not run Phase 2 migrations against a provider-owned read-only database.

## Historical snapshot inventory

Run the aggregate, read-only inventory before creating or importing a Phase 2
feature set:

```bash
PYTHONPATH=. python scripts/inventory_historical_vector_snapshots.py
```

The inventory validates each legacy model against its stored vectors without
returning embeddings, trait text, metadata rows or raw identifiers. It reports:

- declared and stored vector counts;
- model and stored embedding dimensions;
- vector-index continuity;
- the latest compatible snapshot;
- the largest compatible snapshot.

`latest_compatible_snapshot_job_id` and
`largest_compatible_snapshot_job_id` are evidence for operator review. The
service performs no automatic snapshot selection. A newer small snapshot must
not silently replace a larger complete historical snapshot.

After reviewing lineage and purpose, retain the exact approved `job_id` for the
historical import command. Import now requires that identifier explicitly.

## Isolated local pgvector target

The internal Phase 2 target is separate from the read-only historical source.
The Compose file is not part of the default application stack and starts only
when an operator explicitly runs it.

Keep all database passwords in the untracked local `.env` or a deployment
secret manager. Do not put their values in source control, chat messages, logs
or screenshots.

Start only the isolated feature database:

```bash
docker compose \
  -f docker-compose.phase2-pgvector.yml \
  up -d phase2-pgvector
```

The pinned official pgvector image listens on loopback port `55432` by default
and persists data in its own named volume. Configure
`AUDIENCE_FEATURE_MIGRATION_DATABASE_URL` privately so it points to that
database. Never reuse `ECHO_DATABASE_URL` as an implicit feature target.

Stopping the container without the `--volumes` option preserves its database:

```bash
docker compose \
  -f docker-compose.phase2-pgvector.yml \
  stop phase2-pgvector
```

After the container is healthy and the target is confirmed as Punk-owned,
rerun the read-only preflight. Do not apply migration `0004` until its result is
`ready_for_approved_migration` and explicit migration approval is recorded.

## Read-only database preflight

Before applying any migration, configure the historical source and a separate,
explicit feature target through the deployment secret manager. Then run:

```bash
PYTHONPATH=. python scripts/preflight_phase2_database.py
```

This command:

- opens both databases in read-only transactions;
- never falls back from the feature target to the historical source;
- verifies the legacy safe vector snapshot and its 384 dimensions;
- checks whether pgvector is available and installed;
- checks only the privileges needed to assess migration readiness;
- verifies existing Phase 2 tables, vector type and forced RLS when present;
- does not print database URLs, hosts, usernames, passwords or raw data;
- never creates extensions, tables, indexes or feature rows.

The initial run must remain blocked until the operator confirms that the
configured Phase 2 target is a Punk-owned operational database. After that
ownership check, rerun the still-read-only command with:

```bash
PYTHONPATH=. python scripts/preflight_phase2_database.py \
  --confirm-punk-owned-target
```

The flag records an operator assertion in the report only. It does not apply a
migration or index historical data. A result of
`ready_for_approved_migration` means the prerequisites were observed; applying
the migration still requires explicit change approval.

## Migration

Migration `0004_versioned_audience_features_pgvector.sql` creates:

- the pgvector extension;
- versioned `audience_feature_sets`;
- tenant-scoped `audience_feature_vectors`;
- a generated lexical `tsvector`;
- a GIN lexical index;
- an HNSW cosine vector index;
- hard activation checks for freshness and production data mode;
- forced PostgreSQL row-level security using `app.tenant_id`.

The database migration role must be allowed to install or use the `vector`
extension. Application roles do not need extension-creation privileges.

Never apply a migration automatically from an API request or worker startup.

Use only the dedicated Phase 2 runner after the read-only preflight reports
`ready_for_approved_migration` and explicit approval is recorded:

```bash
PYTHONPATH=. python scripts/apply_phase2_feature_migration.py \
  --confirm-punk-owned-target
```

The runner:

- accepts only migration `0004_versioned_audience_features_pgvector.sql`;
- refuses an implicit or same-database feature target;
- takes a PostgreSQL transaction-level advisory lock;
- records and verifies the migration SHA-256 checksum;
- is idempotent when the same migration was already applied;
- verifies pgvector, `vector(384)`, critical indexes, forced RLS and tenant
  policies after commit;
- does not import a historical snapshot or perform activation/export.

Do not use `scripts/apply_database_migrations.py` for the Phase 2 feature
database.

## Least-privilege roles

Never connect Punk AI using the migration identity. PostgreSQL superusers and
roles with `BYPASSRLS` bypass row-level security, so a superuser query is not a
tenant-isolation test.

Configure unique reader and writer passwords privately, then provision the
roles using the migration identity:

```bash
PYTHONPATH=. python scripts/provision_phase2_feature_roles.py \
  --tenant-id punk_internal \
  --expected-feature-count 86 \
  --confirm-punk-owned-target
```

The provisioner is idempotent and verifies:

- reader and writer are `NOSUPERUSER`, `NOBYPASSRLS`, `NOCREATEDB`,
  `NOCREATEROLE`, `NOINHERIT` and `NOREPLICATION`;
- neither role can create schema/database objects or read the migration ledger;
- the reader has `SELECT` only;
- the writer has `SELECT`, `INSERT` and `UPDATE`, but not `DELETE`;
- the requested tenant can see the expected features;
- an isolation-probe tenant sees zero features through RLS.

After verification, set `AUDIENCE_FEATURE_DATABASE_URL` to the reader identity
and `AUDIENCE_FEATURE_WRITER_DATABASE_URL` to the writer identity. Retain
`AUDIENCE_FEATURE_MIGRATION_DATABASE_URL` only in the operator environment, not
the application deployment.

## Required non-secret configuration

```dotenv
PHASE2_FEATURES_ENABLED=true
PGVECTOR_DIMENSION=384
PHASE2_RETRIEVAL_BACKEND=pgvector
REQUIRE_PUNK_AI_TENANT_SIGNATURE=true
```

Required secrets and database URLs must be supplied through the deployment
secret manager, not committed files:

```dotenv
ECHO_DATABASE_URL=
AUDIENCE_FEATURE_DATABASE_URL=
AUDIENCE_FEATURE_WRITER_DATABASE_URL=
AUDIENCE_FEATURE_MIGRATION_DATABASE_URL=
AUDIENCE_PROPOSAL_DATABASE_URL=
PHASE2_FEATURE_READER_PASSWORD=
PHASE2_FEATURE_WRITER_PASSWORD=
PHASE2_PROPOSAL_RUNTIME_PASSWORD=
PUNK_AI_TENANT_AUTH_SECRET=
AUDIENCE_API_KEY=
```

## Historical import

After migration approval and application, index the exact reviewed safe vector
snapshot:

```bash
PYTHONPATH=. python scripts/index_historical_audience_features.py \
  --tenant-id trusted-tenant \
  --purpose internal-audience-evaluation \
  --rights-policy-id historical-internal-use \
  --privacy-policy-version privacy-policy-v1 \
  --job-id exact-reviewed-snapshot-id
```

The exact snapshot is mandatory. The command fails closed when
`AUDIENCE_FEATURE_WRITER_DATABASE_URL` is absent and never falls back to either
the reader identity or historical source. It prints a safe receipt and never
prints URLs, credentials, raw source values, or embeddings.

The command is idempotent for the same source fingerprint.

## ANN recall benchmark

HNSW behavior must be compared with an exact-search baseline before production
acceptance:

```bash
PYTHONPATH=. python scripts/benchmark_pgvector_recall.py \
  --tenant-id trusted-tenant \
  --query "high-quality evening restaurant audience" \
  --locations montreal \
  --categories restaurant \
  --dayparts evening \
  --top-k 25
```

The command disables index scans for the exact comparison and reports
`recall_at_k`. It performs no activation or export.

## Punk AI request

```json
{
  "tenant_id": "trusted-tenant",
  "campaign_id": "campaign-id",
  "objective": "store_visits",
  "audience_intent": "high-quality evening restaurant audience",
  "locations": ["montreal"],
  "categories": ["restaurant"],
  "dayparts": ["evening"],
  "budget": {
    "currency": "CAD",
    "daily": 100
  },
  "exclusions": [],
  "destination": "meta",
  "idempotency_key": "unique-request-key",
  "execution_mode": "historical_preview",
  "top_k": 10
}
```

Required headers:

```text
X-Audience-API-Key: deployment-managed-key
X-Audience-Tenant-Id: trusted-tenant
X-Audience-Tenant-Signature: HMAC-SHA256 tenant signature
```

The tenant signature is an HMAC-SHA256 of the lower-cased tenant identifier
using `PUNK_AI_TENANT_AUTH_SECRET`. Punk AI and Audience Intelligence should
receive that secret through their deployment secret managers.

## Historical response behavior

The response can include ranked candidate cohorts, quality, provisional
offline retrieval confidence, source timestamp, and freshness.

It must also contain:

```json
{
  "status": "historical_preview_ready",
  "approval_status": "blocked_historical_source",
  "activation_eligible": false,
  "safe_export_eligible": false,
  "downstream_export_enabled": false
}
```

Confidence is explicitly marked `provisional_offline_evidence`. Campaign-lift
calibration must remain false until controlled outcome data exists.

## Structured filter behavior

The query sequence is:

```text
tenant and feature-set version
    -> privacy eligibility
    -> data mode and freshness
    -> requested location/category/daypart
    -> exclusions
    -> pgvector retrieval
    -> lexical retrieval
    -> reciprocal-rank fusion
    -> evidence confidence
```

Location filtering uses token containment so a city can match a more specific
subarea while a requested subarea cannot fall back to a broader city.
Categories and dayparts use normalized exact taxonomy values. Category
hierarchies should come from a future versioned taxonomy registry rather than
prompt-specific code.

## Verification before activation

Before any production-mode feature set can be enabled:

1. provider rights and permitted purpose must be approved;
2. the source must be fresh;
3. the privacy policy version must be current;
4. tenant isolation and RLS tests must pass against PostgreSQL;
5. exact-search versus HNSW recall must be benchmarked;
6. Punk AI tenant signatures must be required;
7. human approval and the separate safe-export workflow must be connected;
8. destination receipt reconciliation must be proven.

The historical July feature set is not eligible for this activation path.
