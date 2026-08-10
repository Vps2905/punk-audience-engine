# Module 5.9 — Functional Shadow Execution

## Purpose

Module 5.9 moves bounded autonomy from control-decision comparison into a real,
read-only service execution path. The autonomous planner can select and order
approved capabilities across Modules 1–5, while every capability remains bound
to an existing privacy, lineage, review, and no-effects contract.

This stage answers whether the bounded autonomous path can execute the real safe
module services against an isolated historical feature snapshot and reproduce
the authoritative route, state, freshness, retrieval, cohort, and approval
signals within explicit latency and resource limits.

It does not certify fresh provider data, production databases, activation
connectors, production traffic, or unattended decisions.

## Real functional path

For a non-terminal shadow goal, the functional adapter invokes:

| Stage | Real service boundary | Allowed result |
|---|---|---|
| Module 1 | Rollback-only production feature snapshot reader and aggregate privacy metadata validation | Privacy-safe feature metadata in memory |
| Module 2 | Governed multilingual canonicalization and dual-model candidate retrieval | Review-only retrieval evidence |
| Module 3 | Cohort candidate, overlap/deduplication, and lookalike services | Aggregate candidate evidence only |
| Module 4 | Evolution snapshot, optional drift detection, and recommendation services | Historical review recommendations only |
| Module 5 | Deterministic supervisor decision and delivery review | Manual-review decision with delivery disabled |

The existing bounded autonomy planner and executor choose the approved route,
validate capability inputs and outcomes, enforce attempt/replan budgets, and
stop downstream tasks after a failed dependency. This is bounded autonomy: it
can plan and execute within the allowlist, but it cannot invent a capability,
expand permissions, approve itself, or change a production route.

Module 5.10 adds a second gate at execution time. Each real handler invocation
requires a short-lived, tenant-bound bounded-agent principal with the matching
scope and capability allowlist entry. A denial blocks the handler before source
access. Authorization evidence stores only policy, principal/context and
decision fingerprints plus aggregate allow/deny counts; it stores no credential
or authentication-token material.

## Terminal safety first

Raw-identifier requests, approval-bypass attempts, and unsupported direct
export requests are evaluated before any source adapter is called. A terminal
decision records only a minimized reason code, route, stage, latency, and
fingerprint. Source access, semantic retrieval, cohort work, and evolution work
are not evaluated.

## Functional comparison

The report compares the historical authoritative observation with the
functional shadow result for:

- supervisor route and stage;
- freshness and whether source data was evaluated;
- vector, ranked-match, and selected-cohort counts;
- terminal and approval-required decisions;
- downstream export remaining disabled;
- raw identifiers remaining absent; and
- production effects remaining absent.

Legacy raw-source row counts are not compared with privacy-safe feature row
counts because those values have different semantics. Persisted legacy
candidate counts are also not compared with the functional path because the
functional path intentionally performs no candidate persistence. These are
reported as explicit non-comparisons rather than false equivalence.

## Evidence minimization

Evidence stores only:

- opaque tenant/request/run and feature-set lineage;
- certification and bounded-run fingerprints;
- stage IDs, module IDs, status tokens, reason codes, metrics, and fingerprints;
- minimized legacy and functional summary signals;
- comparison booleans and bounded divergence codes; and
- latency and Python-allocation gate results.

Prompts, query text, tool arguments, tool results, source rows, selected
candidates, embeddings, vectors, raw identifiers, exception text, and complete
service outputs are excluded.

Migration `0031_bounded_autonomy_functional_shadow.sql` provides an immutable,
tenant-scoped ledger with forced row-level security and database checks that
keep shadow-only, read-only, no-cutover, no-write, no-activation, and
manual-approval guarantees authoritative.

## Resource gates

Each service stage is measured with a monotonic clock. The run also records a
bounded peak Python allocation measurement using `tracemalloc`.

`tracemalloc` does not represent total process RSS, native model allocations,
GPU memory, database resource use, or network saturation. Those require the
separate staging load, soak, concurrency, and infrastructure observability
certification. Module 5.9 must not be presented as that certification.

## Configuration

```dotenv
MODULE5_BOUNDED_AUTONOMY_FUNCTIONAL_SHADOW_EVIDENCE_PATH=
MODULE5_BOUNDED_AUTONOMY_FUNCTIONAL_SHADOW_ENABLED=false
MODULE5_FUNCTIONAL_SHADOW_MAX_SAFE_FEATURE_ROWS=5000
MODULE5_FUNCTIONAL_SHADOW_MAX_VECTOR_COUNT_DELTA=0
MODULE5_FUNCTIONAL_SHADOW_MAX_RANKED_MATCH_DELTA=0
MODULE5_FUNCTIONAL_SHADOW_MAX_SELECTED_COHORT_DELTA=0
MODULE5_FUNCTIONAL_SHADOW_MAX_TOTAL_LATENCY_MS=15000
MODULE5_FUNCTIONAL_SHADOW_MAX_STAGE_LATENCY_MS=10000
MODULE5_FUNCTIONAL_SHADOW_MAX_PEAK_PYTHON_BYTES=1073741824
MODULE5_AGENT_CAPABILITY_AUTHORIZATION_REQUIRED=true
```

The execution flag defaults to false. When it is enabled, the Module 5 status
service fails closed unless it can validate a ready Module 5.8 certification,
matching tenant and certification lineage, zero functional divergences, passed
resource gates, and `live_cutover_authorized=false`.

## Acceptance boundary

`module5_bounded_autonomy_functional_shadow_ready=true` means one validated
historical functional-shadow report is suitable for staging review. It always
coexists with:

- `fresh_data_certified=false`;
- `live_cutover_authorized=false`;
- `automatic_cutover_performed=false`;
- `production_routing_changed=false`;
- `database_write_performed=false`;
- `activation_or_export_performed=false`; and
- `manual_approval_required=true`.

Remaining production certification requires repeated functional runs in an
isolated staging account, fresh provider shadow delivery, external load and
recovery testing, security/privacy sign-off, and an explicitly approved canary.
