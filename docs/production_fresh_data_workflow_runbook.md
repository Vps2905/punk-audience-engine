# Production fresh-data workflow runbook

## Scope

Migration `0014` and the fresh-data workflow service provide the durable bridge
from a completed Module 1 ingestion to Module 2 feature publication and Module 3
governed candidate/overlap review.

The workflow accepts only an exact privacy-safe canonical S3 object whose
checksum, object version, byte size, row count, tenant, provider, and dataset
match the completed Module 1 ingestion receipt.

## Safety boundary

The terminal success state is `awaiting_review`. The workflow does not:

- read, store, or return raw identifiers;
- persist Module 3 candidate rows;
- calculate membership intersections or overlap percentages;
- sum cohort sizes or claim unique reach;
- generate lookalikes;
- create Punk AI proposals;
- enable production routing;
- activate or export an audience.

## Durable execution

Each request has a content-addressed workflow ID and request fingerprint.
Worker leases prevent concurrent execution. An expired lease can be reacquired,
and every stage is replay-safe:

1. Module 1 ingestion validation;
2. canonical object integrity validation;
3. Module 2 idempotent feature build;
4. exact safe feature snapshot read without embeddings;
5. deterministic Module 3 candidate generation;
6. deterministic duplicate/overlap analysis;
7. `awaiting_review` result receipt.

Every transition creates an append-only audit event. Workflow identity, safety
flags, and request manifests are immutable.

## Deployment order

1. Apply migrations through `0014` using the migration/admin role.
2. Rerun Phase 2 feature-role provisioning so the writer receives only the
   required workflow-table privileges.
3. Keep `FRESH_DATA_WORKFLOW_ENABLED=false` until the worker is deployed.
4. Configure provider, reader, writer, and migration database URLs.
5. Register and approve the pinned embedding model.
6. Deploy the queue/event worker with a stable worker identity.
7. Enable the workflow in staging and submit one fresh provider delivery.
8. Verify workflow events, feature receipts, candidate/overlap evidence, and
   zero activation/export side effects.
9. Complete load, failure-recovery, privacy, security, and DR certification.

## Operational recovery

A worker crash leaves a bounded lease. After expiry, another worker can claim
the same request. Content-addressed Module 2 publication and deterministic
Module 3 reports make replay safe. Terminal `blocked`, `quarantined`, and
`failed` workflows require an operator-reviewed new request or controlled retry
policy; they are never silently reactivated.

## Automatic trigger boundary

`ProductionFreshDataTriggerService` is the production-safe bridge from completed
Module 1 receipts into the durable fresh-data workflow. It is deliberately
bounded and fail-closed:

- only `completed` ingestion receipts are eligible;
- the immutable provider contract supplies cohort-column semantics;
- canonical checksum, object version, source timestamp, row count, privacy
  controls, rights policy, and rights status are mandatory;
- the default execution mode is `offline_evaluation` until an operator
  explicitly configures `production`;
- the embedding model revision is immutable;
- a bad receipt is isolated without preventing other completed receipts from
  being examined;
- workflow idempotency remains enforced by the durable workflow fingerprint;
- lookalike generation, activation, routing, and export remain disabled.

A continuously running queue/poll worker and its deployment supervision are a
separate operational step; this service is the deterministic trigger primitive
used by that worker.
