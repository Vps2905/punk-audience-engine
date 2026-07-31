# Provider Scale Admission Boundary

## Purpose

The API/SQS worker must never try to hold an unbounded provider object or a
billion-event delivery in memory. This boundary keeps the existing verified
privacy pipeline for small, explicitly bounded objects and dispatches large
objects to a distributed processing workflow.

This is a data-plane safety boundary. It does not change audience semantics,
quality scoring, freshness, approval, or export eligibility.

## Execution decision

Each immutable provider contract declares:

- `execution_mode`: `auto`, `in_process`, or `distributed`;
- `max_in_process_object_bytes`;
- `max_in_process_rows`;
- the existing absolute `max_object_bytes` and `max_rows_per_object`.

`auto` is the production default:

1. Validate provider, bucket, prefix, version, checksum, encryption, schema,
   rights policy, and purpose.
2. Route within-limit objects through the existing privacy pipeline.
3. Route objects above either in-process limit to distributed processing.
4. If distributed processing is not configured, block before reading the
   object.

An explicit `in_process` contract fails closed when an object exceeds its
registered limits. It never silently broadens the memory boundary.

## Distributed dispatch

The worker submits only immutable S3 references and policy metadata to an AWS
Step Functions Standard workflow. It does not submit raw rows, identifiers,
credentials, or secret material.

The Step Functions execution name is derived from the immutable object
fingerprint. Duplicate S3/SQS delivery therefore resolves to the same
execution. The SQS message is acknowledged only after a durable distributed
dispatch receipt is recorded.

State progression:

`received -> validating -> dispatching -> dispatched`

If dispatch fails:

`dispatching -> failed`

The existing bounded retry and DLQ policies then apply.

## Implemented distributed workflow

The repository now includes the deployable data-plane implementation:

- `scripts/provider_distributed_control_plane_lambda.py`;
- `scripts/provider_distributed_privacy_glue_job.py`;
- `infrastructure/stepfunctions/provider_distributed_processing.asl.json`;
- `infrastructure/cloudformation/provider_distributed_data_plane.yaml`;
- `migrations/0007_provider_distributed_privacy_releases.sql`.

The control plane:

1. atomically reserves privacy budget under a PostgreSQL advisory lock;
2. copies the exact provider S3 object version into a Punk-owned encrypted,
   short-retention staging key;
3. accepts the staged object only when S3 reports the declared full-object
   SHA-256 checksum;
4. transitions the durable ingestion record to `processing`;
5. runs Glue through the Step Functions `.sync` integration;
6. validates the bounded terminal result contract;
7. completes or blocks the ingestion record idempotently;
8. deletes the raw staging object on every handled terminal path.

The Glue/Spark job:

1. reads CSV, JSONL, or Parquet without collecting event rows on the driver;
2. validates required columns, row bounds, manifest row count, and timestamps;
3. uses AWS Glue HMAC-SHA256 tokenization backed by Secrets Manager;
4. removes the raw entity column immediately after tokenization;
5. bounds one entity contribution per cohort per UTC day;
6. aggregates by cohort and privacy window;
7. applies the registered k-anonymity threshold;
8. derives deterministic per-release Gaussian noise from a second HMAC secret;
9. removes exact bounded counts and all identifier material;
10. writes encrypted, attempt-scoped Parquet partitions;
11. inventories the immutable Parquet attempt into an encrypted canonical
    manifest and records its SHA-256;
12. keeps every partition as a non-serving candidate until all expected
    entity-disjoint partitions in the privacy window are complete;
13. activates the complete window atomically, or revokes it on correction,
    deletion, or opt-out.

The provider key is never read directly by Spark. Exact-version staging closes
the race in which a provider could replace the current key between metadata
validation and distributed processing.

Parquet is deliberately distributed-only. A Parquet contract cannot force the
in-process decoder.

## Configuration

```text
PROVIDER_DISTRIBUTED_PROCESSING_ENABLED=true
PROVIDER_DISTRIBUTED_STATE_MACHINE_ARN=<standard-workflow-arn>
PROVIDER_DISTRIBUTED_EXECUTION_PREFIX=punk-provider
PROVIDER_DISTRIBUTED_CONTROL_PLANE_LAMBDA_ARN=<lambda-arn>
PROVIDER_DISTRIBUTED_GLUE_JOB_NAME=<glue-job-name>
PROVIDER_DISTRIBUTED_STAGING_BUCKET=<punk-owned-staging-bucket>
PROVIDER_DISTRIBUTED_STAGING_PREFIX=provider-distributed-staging
PROVIDER_DISTRIBUTED_STAGING_ENCRYPTION=aws:kms
PROVIDER_DISTRIBUTED_STAGING_KMS_KEY_ID=<kms-key-reference>
PROVIDER_INGESTION_DATABASE_SECRET_ARN=<secret-arn>
PROVIDER_DISTRIBUTED_TOKENIZATION_SECRET_ARN=<secret-arn>
PROVIDER_DISTRIBUTED_DP_SEED_SECRET_ARN=<secret-arn>
```

These values are deployment identifiers and secret references, not secret
values. Credentials are supplied only by workload roles and Secrets Manager.
Database URLs, HMAC keys, KMS material, and provider credentials must not be
placed in provider contracts, Step Functions input, Glue job-run arguments,
queue messages, ingestion records, logs, or API responses.

Apply migrations `0006`, `0007`, and `0009` only through the approved operator change
process after backup and restore verification. Do not enable distributed
processing before the state machine, Lambda, Glue job, IAM policies, VPC
connectivity, KMS grants, secret access, staging lifecycle, and alarms exist.

Each retry gets a new Step Functions execution suffix while duplicate delivery
within the same dispatch attempt resolves to the same Standard execution.
Every allowed retry reserves another privacy release. Failed releases remain
charged conservatively.

## Billion-event interpretation

One billion events per day is approximately 11,574 events/second on average.
The provider should deliver partitioned immutable objects to S3. S3 events
notify SQS per object, while the distributed workflow processes and reduces
partitions. PostgreSQL remains the control plane; it must not become the raw
event lake.

This implementation removes the architectural need to embed or hold billions
of raw MAID rows in PostgreSQL or the API process. Providers must partition a
day into immutable bounded objects; Glue reduces those objects to safe daily
cohort features. The serving/vector layer embeds cohort features, never every
raw event or every MAID.

The code does not by itself prove one-billion-event throughput. Production
approval still requires measured tests for:

- sustained and peak partition arrival rate;
- Glue worker sizing, shuffle volume, skew, and cost;
- multi-object daily reconciliation;
- duplicate and out-of-order events;
- late corrections and deletion/opt-out propagation;
- schema evolution and corrupt Parquet;
- Step Functions redrive and Lambda timeout recovery;
- KMS, S3, Secrets Manager, RDS, and service-quota failure;
- canonical partition compaction and downstream feature-build latency;
- proof that no raw or hashed identifier reaches canonical/serving output.

The command `scripts/evaluate_provider_scale_acceptance.py --record` stores
immutable, tenant-scoped measured evidence. The authenticated
`/provider-ingestion/scale-readiness` endpoint remains fail-closed until a
recorded run meets volume, throughput, latency, recovery, privacy, integrity,
rights-propagation, and cost gates.

Until those acceptance gates pass, distributed processing is implemented but
not throughput-certified for one billion events per day.
