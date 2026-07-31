# Module 2 Production Embedding Benchmark Gate

## Outcome

The production embedding registry no longer trusts a self-declared
`"passed": true` field. Approval requires a complete report generated for the
exact immutable model specification and verified against the versioned global
benchmark policy.

## Added

- `app/models/embedding_benchmark_contracts.py`
  - canonical benchmark documents, queries and dataset identity;
  - privacy and evaluation-rights boundary;
  - versioned production threshold and coverage policy.
- `app/services/production_embedding_benchmark_service.py`
  - pinned query/document encoding;
  - Recall@K, Precision@K, nDCG@K and MRR;
  - geographic, category and daypart fidelity;
  - hard-negative and unsupported-location rejection;
  - multilingual consistency;
  - p50/p95/p99 latency, peak memory and cost evidence;
  - deterministic pass/fail evaluation and report fingerprint;
  - independent report validation before registration.
- `scripts/run_production_embedding_benchmark.py`
  - atomic report generation without credentials or activation.
- `samples/embedding_benchmark_dataset.example.json`
  - non-production schema example.

## Strengthened

- `ProductionEmbeddingModelRegistrationService` validates the complete report,
  its model binding, policy, thresholds, coverage and fingerprint before any
  approved model record is inserted.
- Every dataset now carries approved authoring evidence and the exact
  fingerprints of its reviewed case catalog and safe document catalogs.
- Unsupported-location abstentions are evaluated independently from
  geographic fidelity on supported requests.
- Weaker threshold overrides, insufficient global coverage, modified evidence,
  unapproved evaluation rights and raw-identifier benchmark claims fail closed.

## Unchanged safety boundaries

- No raw MAIDs or device identifiers are embedded.
- No audience activation or export is performed.
- Historical data remains non-activatable.
- Approved models and published feature artifacts remain immutable.
- Tenant row-level security and least-privilege reader/writer roles remain in
  force.
