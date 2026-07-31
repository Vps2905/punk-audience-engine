# Module 1 production control-plane change manifest

This increment closes the writable-database and durable distributed-release
boundary for Module 1.

Included controls:

- a separate Punk-owned local provider control database;
- explicit operator-only preflight and migrations;
- no fallback from provider writes to the historical Echo database;
- tenant identity pinned to the authenticated PostgreSQL runtime role;
- forced row-level security across provider control tables;
- least-privilege runtime role provisioning;
- no runtime PostgreSQL schema creation;
- stable HMAC-derived historical DP replay noise;
- generic non-coordinate geography normalization and stable geo keys;
- fail-closed handling for ambiguous sensitive health taxonomy;
- durable two-partition privacy-budget and publication acceptance;
- duplicate object notification and terminal result replay verification.

The acceptance flow is intentionally offline and identifier-free. It verifies
application control-plane behavior but does not replace AWS integration,
security, throughput, chaos, recovery, or observability certification.
