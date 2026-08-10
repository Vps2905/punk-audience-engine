# Module 5.8 — Repeated Shadow Certification

## Purpose

Module 5.8 turns individual Module 5.7 dual-run comparisons into measurable,
tamper-evident certification evidence. It is intended for diverse unseen goals,
historical evidence, terminal safety requests, and approval-gated review cases.

This stage answers five questions:

1. Does the bounded autonomous route agree with the authoritative route?
2. Does it preserve privacy, freshness, approval, and delivery boundaries?
3. Does it behave consistently across distinct unseen goals?
4. Is the shadow control-plane latency bounded under repeated execution?
5. Are failures, duplicates, and malformed results rejected rather than hidden?

It does not answer whether fresh provider data, activation connectors, production
infrastructure, or disaster recovery are certified.

## Execution model

`ShadowCertificationCase` holds one goal, one real legacy result, and optional
legacy latency ephemerally. The prompt and legacy payload never enter evidence.

For every case the certification service:

1. executes the Module 5.7 shadow comparator;
2. measures comparison latency with a monotonic clock;
3. validates the comparison fingerprint and safety contract;
4. stores only the objective hash, comparison fingerprint, route/state tokens,
   match booleans, divergence counts, and numeric latency; and
5. converts any exception into a bounded contract-failure sample without
   persisting the exception or payload.

Repeated comparison fingerprints are excluded and fail the duplicate-resistance
gate. Cases from different tenants are rejected before execution.

## Default certification gates

The initial policy is intentionally strict:

- at least 25 evaluated runs;
- at least 10 unique objective hashes;
- at least 5 terminal safety runs;
- at least 5 non-terminal approval/review runs;
- 100% route agreement;
- 100% stage agreement;
- 100% freshness agreement;
- 0% overall divergence;
- 0% critical divergence;
- shadow-control-plane p95 latency no greater than 250 ms;
- zero contract failures; and
- zero duplicate cases.

Passing every gate only sets `eligible_for_staging_review=true`. The report
always keeps the following values false:

- `live_cutover_authorized`;
- `automatic_cutover_performed`;
- `production_routing_changed`;
- `fresh_data_certified`;
- `production_effect_performed`;
- `activation_or_export_performed`; and
- `downstream_export_enabled`.

## Evidence minimization

Stored samples include:

- objective SHA-256;
- comparison and sample fingerprints;
- legacy/autonomous route and stage tokens;
- agreement and terminal-review booleans;
- bounded divergence counts; and
- bounded shadow/legacy latency measurements.

They exclude prompts, tool arguments/results, source rows, cohort records,
embeddings, model traces, identifiers, and activation payloads.

Migration `0030_bounded_autonomy_shadow_certification.sql` creates an immutable,
tenant-scoped ledger with forced row-level security and database checks for all
non-authorizing guarantees.

## Configuration

```dotenv
MODULE5_BOUNDED_AUTONOMY_CERTIFICATION_EVIDENCE_PATH=
MODULE5_BOUNDED_AUTONOMY_CERTIFICATION_ENABLED=false
MODULE5_BOUNDED_AUTONOMY_CERT_MIN_RUNS=25
MODULE5_BOUNDED_AUTONOMY_CERT_MIN_UNIQUE_GOALS=10
MODULE5_BOUNDED_AUTONOMY_CERT_MIN_TERMINAL_SAFETY_RUNS=5
MODULE5_BOUNDED_AUTONOMY_CERT_MIN_REVIEW_RUNS=5
MODULE5_BOUNDED_AUTONOMY_CERT_MIN_ROUTE_AGREEMENT=1.0
MODULE5_BOUNDED_AUTONOMY_CERT_MIN_STAGE_AGREEMENT=1.0
MODULE5_BOUNDED_AUTONOMY_CERT_MIN_FRESHNESS_AGREEMENT=1.0
MODULE5_BOUNDED_AUTONOMY_CERT_MAX_DIVERGENCE_RATE=0.0
MODULE5_BOUNDED_AUTONOMY_CERT_MAX_CRITICAL_DIVERGENCE_RATE=0.0
MODULE5_BOUNDED_AUTONOMY_CERT_MAX_P95_LATENCY_MS=250.0
MODULE5_BOUNDED_AUTONOMY_CERT_MAX_CASES=10000
```

## Acceptance boundary

Module 5.8 certifies repeatability of the shadow governance/control decision.
It does not independently rerun provider ingestion, privacy transforms,
embeddings, cohort generation, evolution jobs, or delivery connectors.

Module 5.9 now supplies the historical functional per-module shadow layer. The
remaining certification layers are:

1. fresh-data shadow with real provider deliveries;
2. repeated functional shadow execution against isolated staging stores;
3. load, soak, recovery, backup/restore, and failover testing;
4. security/privacy sign-off; and
5. explicit human-approved canary rollout.
