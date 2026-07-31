# Phase 1: Trustworthy Provider Data

## Purpose

Phase 1 is the fail-closed boundary between provider-owned S3 objects and
Punk AI's privacy-safe canonical feature layer. It does not require a live
provider feed to be deployed and tested. When a correctly configured provider
starts delivering objects, the worker processes them automatically.

PostgreSQL stores contracts, processing state, replay approvals, privacy jobs,
and lineage. S3 stores immutable provider input and privacy-safe canonical
output. PostgreSQL is not replaced by S3.

## Delivery path

1. A provider uploads an immutable, versioned CSV or JSONL object.
2. S3 sends one object-created event per SQS message.
3. The worker reads object metadata using `HeadObject`.
4. The registry resolves the active dataset contract by bucket, longest
   matching prefix, and schema version.
5. The object fingerprint is claimed in PostgreSQL.
6. Size, content type, encryption, rights, purpose, version, checksum,
   manifest row count, and schema are validated.
7. Raw entity identifiers are HMAC-SHA256 tokenized with managed key material,
   suppressed for applied deletion/opt-out requests, contribution-bounded,
   aggregated, filtered by k-anonymity, and protected with replay-stable
   secret-seeded DP noise.
8. The output schema is checked again for raw identifiers, coordinates, and
   secret-like values.
9. Only aggregated/DP-safe JSONL is written to the canonical S3 bucket.
10. Terminal outcomes are acknowledged. Transient failures are retried with
    bounded exponential backoff. Exhausted or malformed messages are moved to
    the DLQ before the source message is acknowledged.

## Required S3 object metadata

Provider objects must supply these `x-amz-meta-*` values:

```text
punk-schema-version
punk-purpose
punk-rights-policy-id
punk-checksum-sha256
punk-row-count
punk-event-time-start
punk-event-time-end
punk-delivery-window-id
punk-partition-index
punk-partition-count
punk-partition-algorithm
punk-partition-complete
punk-delivery-type
```

The checksum must be the lowercase hexadecimal SHA-256 of the complete object.
For multipart uploads, do not use an ETag or composite checksum as a full
object checksum.

## Provider contract

A contract is non-secret JSON. It defines tenant/provider/dataset identity,
allowed bucket and prefix, input format, required columns, privacy thresholds,
rights policy, allowed purposes, maximum object size/rows, delivery interval,
and freshness SLA.

Register a version:

```bash
PYTHONPATH=. python scripts/register_provider_contract.py \
  --contract /secure-ops/provider-contract.json \
  --actor operator-name
```

Contract versions are immutable. Register a new schema version for changes.
Deactivate an obsolete version:

```bash
PYTHONPATH=. python scripts/register_provider_contract.py \
  --deactivate-key tenant:provider:dataset:v1 \
  --actor operator-name
```

Contract files must never contain credentials. AWS access comes from the
workload identity or short-lived AssumeRole sessions.

## Worker

Apply migrations before the worker starts:

```bash
PYTHONPATH=. python scripts/apply_database_migrations.py
```

Migrations through `0009_provider_privacy_windows_and_rights.sql` are
required for distributed privacy composition, correction handling, canonical
publication state, data-rights propagation, and durable scale evidence.

Run:

```bash
PYTHONPATH=. python scripts/run_provider_ingestion_worker.py
```

The worker supports S3 notifications directly, SNS-wrapped S3 notifications,
and EventBridge S3 object-created events. Each message must contain exactly one
S3 record so acknowledgement remains atomic.

## Retry and DLQ behavior

- `completed`, `blocked`, and `quarantined` are terminal and acknowledged.
- S3/database/network timeouts and gateway infrastructure failures retry.
- Missing contracts retry until the configured receive limit, allowing safe
  recovery when registration and delivery race.
- Malformed or unsupported events go directly to the DLQ.
- A DLQ transfer must succeed before the source message is deleted.
- Duplicate events reuse the existing fingerprint and cannot duplicate a
  canonical output.
- Stale `validating` or `processing` records are changed to a retryable failed
  state after the configured worker recovery interval.

## Controlled replay

Replay requires a named requester, reason, and approval reference:

```bash
PYTHONPATH=. python scripts/replay_provider_ingestion.py \
  --ingestion-id provider_ingest_xxx \
  --requested-by operator-name \
  --reason "Retry after provider outage" \
  --approval-reference approval-ticket
```

Failed, blocked, or quarantined records return to the same immutable
fingerprint boundary. Completed records run in idempotency-verification mode
and cannot create a second canonical output.

## Monitoring

Authenticated operational endpoints:

```text
GET /api/audience-intelligence/provider-ingestion/status
GET /api/audience-intelligence/provider-ingestion/runs/{ingestion_id}
GET /api/audience-intelligence/provider-ingestion/privacy-windows/{window_key}
GET /api/audience-intelligence/provider-ingestion/scale-readiness
POST /api/audience-intelligence/provider-ingestion/data-rights
POST /api/audience-intelligence/provider-ingestion/data-rights/{request_id}/apply
```

The status report returns active datasets, fresh/delayed/stale/never-received
state, queue depth, DLQ depth, and safe processing counts. It never returns
database URLs, AWS credentials, external IDs, KMS material, or raw payloads.

Audience Intelligence must treat `delayed`, `stale`, and `never_received`
datasets as ineligible until fresh canonical data is available.

## Deployment gate

Phase 1 code is deployable only after:

- database migration succeeds;
- workload identities and least-privilege bucket/queue policies are installed;
- source and canonical bucket versioning and encryption are enabled;
- the canonical writer cannot read unrelated provider prefixes;
- SQS redrive and retention policies are reviewed;
- a synthetic provider object completes the entire path;
- duplicate, restart, checksum, quarantine, replay, and restore drills pass;
- privacy/security review approves the provider contract and purpose.

No deployment may claim billion-event readiness until measured staging or
preproduction evidence is recorded:

```bash
PYTHONPATH=. python scripts/evaluate_provider_scale_acceptance.py \
  --evidence /secure-ops/provider-scale-evidence.json \
  --tenant-id tenant_identifier \
  --record \
  --confirm-measured-preproduction-evidence
```
