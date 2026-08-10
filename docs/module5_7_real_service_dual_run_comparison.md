# Module 5.7 — Real-Service Adapters and Dual-Run Comparison

## Purpose

Module 5.7 connects the bounded autonomy kernel to facts produced by the real
Audience Intelligence orchestrator. It does not execute the legacy pipeline a
second time and does not duplicate provider reads, database writes, audience
preparation, approval, activation, or export.

The existing orchestrator remains authoritative. After it completes, a strict
adapter converts its result into minimized facts. The bounded kernel uses those
facts to independently compute a shadow recommendation. A comparator records
whether the two control paths agree.

## Governing boundary

> Observe real services; never let a shadow decision cause a real effect.

The dual-run path is disabled by default and cannot:

- change the legacy supervisor route;
- authorize an audience;
- mutate an audience or lifecycle state;
- invoke an activation connector;
- expose identifiers, rows, prompts, model reasoning, or tool payloads; or
- perform automatic cutover.

## Adapter flow

```text
Existing orchestrator result (authoritative)
                  |
LegacyOrchestratorObservationAdapter
                  |
privacy-minimized canonical observation
                  |
RealServiceCapabilityAdapterFactory
                  |
Module 5.6 bounded planner/executor/critic
                  |
independent shadow decision
                  |
ProductionBoundedAutonomyShadowComparisonService
                  |
immutable divergence evidence + human review eligibility
```

The real-service adapters expose only bounded counts and state tokens:

- Module 1: whether source evidence was evaluated and its row count;
- Module 2: vector and ranked-match counts;
- Module 3: selected-cohort and prepared-candidate counts;
- Module 4: review-only evolution state with mutation disabled; and
- Module 5: independently recomputed safety/freshness/coverage route.

No prompt words, city names, categories, data rows, identifiers, embeddings, or
activation payloads enter the comparison evidence.

## Divergence gates

The default policy requires exact agreement for the initial certification
window. It compares:

- supervisor route and stage;
- terminal/non-terminal behavior;
- freshness state and whether source access was evaluated;
- source, vector, ranked-match, selected-cohort, and candidate counts;
- downstream-delivery state;
- identifier-return state; and
- production-effect state.

Any critical mismatch makes the report
`engineering_preview_blocked`. A clean report is only
`eligible_for_human_review`; it never performs cutover.

## Integration

`AudienceSupervisorAgent` can attach the minimized comparison after the
authoritative result when all of the following are true:

1. `MODULE5_BOUNDED_AUTONOMY_DUAL_RUN_ENABLED=true`;
2. an authenticated `tenant_id` is supplied to the run; and
3. the result satisfies the comparison contracts.

Missing tenant context or a contract failure produces a shadow-only blocked
status and leaves the authoritative result unchanged.

## Evidence storage

Migration `0029_bounded_autonomy_dual_run_comparison.sql` creates an immutable,
tenant-scoped ledger with forced row-level security. Database checks keep
automatic cutover, routing changes, approval, mutation, production effects,
activation, and export disabled.

## Configuration

```dotenv
MODULE5_BOUNDED_AUTONOMY_COMPARISON_EVIDENCE_PATH=
MODULE5_BOUNDED_AUTONOMY_DUAL_RUN_ENABLED=false
MODULE5_BOUNDED_AUTONOMY_MAX_SOURCE_ROW_DELTA=0
MODULE5_BOUNDED_AUTONOMY_MAX_VECTOR_COUNT_DELTA=0
MODULE5_BOUNDED_AUTONOMY_MAX_RANKED_MATCH_DELTA=0
MODULE5_BOUNDED_AUTONOMY_MAX_SELECTED_COHORT_DELTA=0
MODULE5_BOUNDED_AUTONOMY_MAX_PREPARED_CANDIDATE_DELTA=0
```

Thresholds remain zero until a measured staging calibration justifies a
documented, reviewed tolerance.

## Acceptance boundary

This increment certifies real-result adaptation and deterministic dual-run
comparison. It does not certify live cutover. Cutover still requires repeated
unseen-goal agreement, fresh-data shadow evidence, load/soak results, recovery
tests, security review, and explicit human authorization.
