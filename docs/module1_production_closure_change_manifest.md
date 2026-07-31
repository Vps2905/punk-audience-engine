# Module 1 Production Closure

## Scope

This change completes the repository-side production boundary for provider
ingestion and privacy. It does not manufacture live provider data and it does
not claim billion-event throughput without measured cloud evidence.

## Implemented

- Entity-hash partition contracts for cross-object contribution bounding.
- Exact S3 object version, full checksum, schema, manifest, rights, purpose,
  encryption, row-count, and event-window validation.
- Standard Step Functions plus Glue/Spark distributed processing assets.
- HMAC-SHA256 tokenization with managed secret references; legacy public salts
  are rejected in production.
- Applied deletion/opt-out suppression before aggregation in both bounded and
  distributed paths.
- K-anonymity, contribution bounding, deterministic Gaussian DP, a durable
  privacy budget, parallel composition across disjoint entity partitions, and
  sequential charging for corrections of the same partition.
- Privacy-window registration, completeness checks, correction handling,
  sealing, revocation, and tenant-scoped row-level security.
- Encrypted Parquet canonical output with an immutable object inventory and
  SHA-256 manifest verification before publication.
- Candidate-to-active publication barrier; incomplete windows cannot become a
  serving source.
- Tenant-safe provider status, run, privacy-window, data-rights, and scale
  readiness APIs.
- Durable, immutable measured scale-acceptance evidence with fail-closed
  volume, throughput, latency, recovery, privacy, integrity, rights SLA, and
  cost gates.
- Monitoring for fresh/delayed/stale feeds, queue/DLQ state, and incomplete or
  stale privacy windows.

## Operator sequence

1. Review and apply migration `0009_provider_privacy_windows_and_rights.sql`.
2. Deploy the CloudFormation, Step Functions, Lambda, and Glue assets with
   least-privilege workload identities and VPC access to the provider control
   database.
3. Register an entity-partitioned provider contract.
4. Run corrupt, duplicate, delayed, correction, deletion, restart, DLQ,
   replay, KMS, database, and backup/restore drills.
5. Run the full measured staging load and record scale evidence.
6. Keep audience activation blocked until live data is fresh and every
   production acceptance gate is satisfied.

## External acceptance still required

Repository tests validate behavior and failure boundaries. AWS quotas, Glue
worker sizing, data skew, network capacity, KMS throughput, actual cost,
one-billion-event latency, security review, provider rights, and backup/restore
must be proven in the target staging/production accounts. Those are deployment
and operational gates, not claims that source code can prove locally.
