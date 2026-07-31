# Module 1 provider control-plane acceptance

Module 1 uses a dedicated Punk-owned PostgreSQL database for provider
contracts, immutable object claims, distributed privacy releases, privacy
windows, canonical publication state, rights requests, and scale evidence. It
must never use the historical Echo database as its writable control store.

Schema installation is operator-only. API and worker runtimes use a tenant-
pinned database role with no superuser, schema-create, delete, truncate, or
tenant-mapping privileges. Tenant identity is derived from the authenticated
database role rather than a caller-controlled session setting.

The controlled acceptance command uses two identifier-free, entity-disjoint
mock S3 object references. It exercises durable object claims, exact-version
checksum staging, privacy-budget reservation, incomplete-window blocking,
candidate canonical state, candidate-to-active publication after both
partitions are ready, duplicate notification handling, and terminal-result
idempotency. It never calls AWS and does not certify IAM, S3, SQS, KMS, Glue,
Step Functions, throughput, activation, or export.

## Local setup

Generate credentials in the terminal without displaying them, then start the
dedicated database:

```bash
export MODULE1_CONTROL_PASSWORD="$(openssl rand -hex 24)"
export MODULE1_CONTROL_DATABASE="punk_provider_control"
export MODULE1_CONTROL_ADMIN_USER="punk_provider_admin"
export MODULE1_CONTROL_PORT="55433"

docker compose -f docker-compose.module1-control.yml up -d
```

Configure the operator URL, run the read-only preflight, then apply only the
approved Module 1 migrations:

```bash
export PROVIDER_INGESTION_MIGRATION_DATABASE_URL="postgresql://${MODULE1_CONTROL_ADMIN_USER}:${MODULE1_CONTROL_PASSWORD}@127.0.0.1:${MODULE1_CONTROL_PORT}/${MODULE1_CONTROL_DATABASE}"

PYTHONPATH=. python scripts/preflight_module1_provider_database.py \
  --confirm-punk-owned-target

PYTHONPATH=. python scripts/apply_module1_provider_migrations.py \
  --confirm-punk-owned-target
```

Provision the tenant-pinned runtime identity:

```bash
export MODULE1_PROVIDER_RUNTIME_USER="punk_provider_runtime"
export MODULE1_PROVIDER_RUNTIME_PASSWORD="$(openssl rand -hex 24)"

PYTHONPATH=. python scripts/provision_module1_provider_runtime_role.py \
  --tenant-id punk_internal \
  --confirm-punk-owned-target

export PROVIDER_INGESTION_DATABASE_URL="postgresql://${MODULE1_PROVIDER_RUNTIME_USER}:${MODULE1_PROVIDER_RUNTIME_PASSWORD}@127.0.0.1:${MODULE1_CONTROL_PORT}/${MODULE1_CONTROL_DATABASE}"
```

Run the durable offline acceptance flow:

```bash
PYTHONPATH=. python \
  scripts/run_module1_provider_control_plane_acceptance.py \
  --tenant-id punk_internal \
  --confirm-offline-control-plane-only
```

Expected result: the first partition remains blocked in an open privacy
window; the second completes the window; both canonical candidates become
active together; one epsilon unit is charged for the entity-disjoint window;
and duplicate notifications/results replay idempotently.

## Remaining cloud certification

Before production launch, run the same contracts against the actual provider
AWS environment and collect evidence for cross-account AssumeRole, prefix
restrictions, object versions and SHA-256, KMS permissions, S3-to-SQS delivery,
visibility timeout, retry/backoff, DLQ redrive, Glue/Step Functions execution,
late corrections, deletion propagation, load/soak behavior, and alarms. The
offline acceptance result must not be represented as that certification.
