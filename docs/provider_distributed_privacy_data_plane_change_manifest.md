# Distributed Provider Privacy Data Plane

## Scope

This change converts the existing distributed-admission decision into an
executable, fail-closed AWS data plane for provider objects that are too large
for the API worker process.

It adds:

- immutable S3 `VersionId` staging with SHA-256 verification;
- KMS-encrypted, short-lived raw staging;
- Step Functions orchestration;
- a Glue/Spark privacy transformation job;
- immediate HMAC tokenization and removal of raw entity identifiers;
- daily contribution bounding;
- aggregate k-anonymity enforcement;
- deterministic per-release Gaussian differential-privacy noise;
- an atomic cumulative privacy-budget ledger;
- immutable, attempt-scoped canonical Parquet output;
- durable completion and failure reconciliation;
- controlled retry identity and duplicate callback handling;
- production environment validation;
- least-privilege Lambda, Glue, and Step Functions infrastructure.

Raw MAIDs or device identifiers are never embedded or published. Only
privacy-safe cohort features can proceed to the feature and retrieval layers.

## Safety boundaries

- Migrations `0006` and `0007` are not applied by installing these files.
- No AWS resource is deployed by installing these files.
- Existing historical data remains stale and ineligible for activation.
- Every distributed retry consumes a new privacy release.
- Failed releases remain charged conservatively.
- Missing, changed, or unverifiable object versions fail closed.
- Semantic logic cannot override privacy, freshness, approval, or export
  controls.

## Verification completed in the review environment

- Provider, privacy, environment, and status tests: `96 passed`.
- Python compilation: passed.
- Step Functions JSON validation: passed.
- A broad non-semantic repository run passed before the final infrastructure
  hardening. The complete repository suite must still be rerun in the project
  virtual environment because the isolated review environment does not contain
  `sentence-transformers`.

## Remaining production gates

This is a production implementation slice, not proof of billion-event
throughput. Before production enablement:

1. run the complete repository suite in the project virtual environment;
2. review and approve migrations `0006` and `0007`;
3. package the Lambda and Glue Python dependencies;
4. deploy into a non-production AWS account;
5. run duplicate, corrupt, delayed, out-of-order, and retry fault tests;
6. benchmark representative Parquet object sizes and sustained daily volume;
7. add cross-object daily contribution consolidation for provider deliveries
   that split the same privacy window across many objects;
8. validate privacy accounting and policy with the privacy owner;
9. prove replay, recovery, observability, cost, and SLO targets;
10. keep activation disabled until fresh provider data and rights checks pass.
