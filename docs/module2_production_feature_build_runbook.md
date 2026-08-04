# Module 2 production feature-build runbook

## Purpose

This layer converts privacy-safe canonical cohort rows produced by Module 1
into immutable, searchable feature vectors. It never embeds raw MAIDs, device
IDs, exact movement paths, or raw observations.

The production boundary is:

1. Module 1 publishes a versioned canonical object in S3.
2. A source manifest proves tenant, provider, dataset, object version,
   SHA-256 checksum, row count, privacy controls, rights, purpose, and
   freshness.
3. The feature worker validates every canonical row and rejects identifier
   fields or provenance conflicts.
4. An exact immutable embedding-model revision must already have a passed
   Punk benchmark and named approval.
5. Document embeddings are generated in bounded batches with no silent model
   fallback.
6. A durable idempotent build claim prevents duplicate embedding or
   publication after retries.
7. The feature set and feature identities are content-addressed.
8. PostgreSQL/pgvector publishes the set with insert-only conflict handling.
9. The verified receipt is saved with the terminal build state for exact
   replay.
10. RLS limits every registry, build, feature-set, and vector query to the
   active tenant.
11. Query embeddings use the exact model revision, prefix, dimension, and
   normalization recorded in feature lineage.
12. Retrieval may use the result, but activation remains controlled by
    freshness, rights, privacy, approval, and export policies downstream.

## What this migration changes

`0008_production_feature_build_registry.sql` adds:

- a tenant-scoped approved embedding-model registry;
- a tenant-scoped durable feature-build job ledger;
- immutable approved-model records;
- immutable published feature sets and feature vectors;
- forced row-level security and tenant policies.

The general migration runner excludes this migration. Only the dedicated
operator command can apply it, and that command requires explicit confirmation
of a separate Punk-owned feature database. It does not modify the historical
provider database, build features, activate an audience, or export data.

## Pre-deployment gates

Before enabling production feature builds:

- the canonical S3 bucket and provider permissions are configured;
- Module 1 privacy outputs contain no raw identifiers;
- migration 0004 is already verified on the separate feature database;
- migration 0008 has been reviewed and approved;
- the feature reader and writer roles are reprovisioned after migration 0008;
- a Punk-specific retrieval benchmark has passed;
- the exact model revision is immutable and reproducibly downloadable;
- production build workers have bounded CPU/GPU, memory, timeout, and retry
  settings;
- freshness and rights policies are enabled;
- staging replay and failure-recovery tests pass.

## Controlled migration

Do not run this during ordinary application startup.

```bash
PYTHONPATH=. python scripts/apply_production_feature_build_migration.py \
  --confirm-punk-owned-target
```

Expected safety fields include:

- `source_database_modified: false`
- `features_built: false`
- `activation_or_export_performed: false`
- `credentials_exposed: false`

After the migration, reprovision the existing least-privilege feature roles so
the feature writer can read approved model records and maintain build-job
state. The reader receives no model-registry or build-ledger privileges.

## Model approval

Model approval is now evidence-driven. A manually authored JSON file containing
`"passed": true` is not accepted. The benchmark runner calculates the decision
and binds it to:

- the exact model name, immutable revision, prefixes and dimension;
- an immutable benchmark dataset fingerprint;
- privacy and evaluation-rights status;
- the versioned `punk-global-embedding-policy-v2` policy;
- ranking, constraint-preservation and hard-negative metrics;
- unsupported-location rejection and multilingual consistency;
- p50, p95 and p99 query latency, peak memory and declared query cost;
- minimum case, document, language and adversarial-coverage counts;
- a content fingerprint covering the complete report.

The checked-in dataset and catalog files are schema examples only and are
intentionally too small to pass production coverage. Follow
`module2_embedding_benchmark_dataset_runbook.md` to export safe historical
trait documents, add reviewed safe coverage, author human-reviewed gold
labels, and compile the immutable dataset. Build a reviewed Punk benchmark
with at least 100 documents, 100 cases, five languages, and the
policy-required geographic, category, daypart, hard-negative,
unsupported-location and multilingual coverage. Version 2 also enforces
minimum distinct location, category and daypart values, minimum cases per
language, and multi-constraint cases. Use curated synthetic or privacy-safe
aggregated examples only.

```bash
PYTHONPATH=. python scripts/run_production_embedding_benchmark.py \
  --dataset "<immutable-reviewed-dataset.json>" \
  --output "/tmp/punk-embedding-benchmark-report.json" \
  --model-name "<model-repository>" \
  --model-revision "<immutable-commit-revision>" \
  --embedding-dimension 384 \
  --document-prefix "<document-prefix>" \
  --query-prefix "<query-prefix>" \
  --batch-size 64 \
  --top-k 10 \
  --rejection-similarity-threshold 0.78 \
  --cost-per-1000-queries-usd "<measured-cost>"
```

The command exits after writing evidence whether the model passes or fails. A
failed report remains useful for comparison but cannot be registered. Public
leaderboard results alone are insufficient.

Before model inference begins, the runner now calculates the theoretical
maximum standard `precision_at_k` allowed by the reviewed relevant-document
labels. It fails closed when the selected `top_k` can never meet the production
policy. For example, a benchmark with one relevant document per supported case
has a maximum precision of `0.10` at `top_k=10`, so it cannot satisfy a `0.60`
precision floor. Such a single-label benchmark must be run explicitly as a
top-one engineering evaluation, or the reviewed case catalog must be expanded
with enough genuinely relevant documents for the intended retrieval depth.
Do not weaken the policy or fabricate extra relevant labels to clear this gate.

```bash
PYTHONPATH=. python scripts/register_production_embedding_model.py \
  --tenant-id "<tenant>" \
  --backend "<sentence_transformers-or-external_embedding_service>" \
  --model-name "<model-repository-or-service-id>" \
  --model-revision "<immutable-revision>" \
  --embedding-dimension 384 \
  --approved-by "<named-approver>" \
  --benchmark-dataset "<immutable-reviewed-dataset.json>" \
  --benchmark-report "<reviewed-report.json>" \
  --confirm-punk-owned-target \
  --confirm-benchmark-accepted
```

The command does not print database credentials or external model-service
secrets. An approved record is immutable; changing model weights, prefixes,
normalization, or benchmark evidence requires a new revision and fingerprint.
Registration independently recalculates all threshold decisions and verifies
the report fingerprint. It fails closed if a metric, threshold, dataset claim
or model field was edited after the benchmark.

## Runtime configuration

Production feature builds remain disabled by default.

Required configuration is documented in `.env.example`. Runtime feature
workers use `AUDIENCE_FEATURE_WRITER_DATABASE_URL`, never the migration
identity. External embedding-service credentials must be referenced through a
secret manager rather than stored in source code or logs.

## Bounded partition build

The distributed scheduler should invoke one worker per immutable canonical
JSONL partition. The request file contains no credentials and must include the
exact S3 version, byte length, full SHA-256 checksum, row count, privacy and
rights provenance, and approved model specification.

```bash
PYTHONPATH=. python scripts/build_production_audience_features.py \
  --request "<reviewed-feature-build-request.json>" \
  --confirm-privacy-safe-canonical-input
```

The worker refuses raw provider data, mutable model versions, unsupported
formats, missing object sizes, mismatched checksums or row counts, and
unapproved models. Repeated delivery of the same request returns the durable
receipt without repeating embedding or publication.

## Failure behaviour

The build fails closed when:

- the canonical object version, checksum, row count, or provenance differs;
- required privacy controls are missing;
- production rights are not permitted;
- a mapped field names an identifier column;
- rows contain blocked raw-identifier fields;
- the model revision is mutable, unregistered, unbenchmarked, or unapproved;
- embedding dimension, shape, finiteness, or normalization is invalid;
- feature identities collide;
- an existing immutable feature record conflicts with the replay;
- the publication receipt is incomplete or inconsistent.

No failure path should silently switch models, broaden audience constraints, or
enable activation.

## Remaining deployment evidence

Code and unit tests do not by themselves prove global production scale. Before
launch, run a distributed canonical-object replay with representative feature
volume, benchmark p50/p95/p99 latency and cost, test worker interruption and
resume, verify ANN recall against exact search, and validate fresh provider
data. Historical data through 8 July 2026 remains retrieval-only and
activation-ineligible.
