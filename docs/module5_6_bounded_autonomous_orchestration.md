# Module 5.6 — Bounded Autonomous Orchestration

## Purpose

This increment introduces a production-oriented autonomy kernel for the five
Audience Intelligence modules. It plans from typed outcomes and capability
contracts rather than matching prompts, cities, categories, or demonstration
phrases.

The governing principle is:

> Hardcode safety invariants; never hardcode business intelligence.

The kernel is shadow-only. It cannot authorize or perform a production effect,
audience mutation, approval, activation, or export.

## Architecture

1. `AutonomyGoal` records tenant, execution mode, budgets, constraints, and
   requested outcomes. The objective is available ephemerally to agents but is
   represented in persisted evidence only by its SHA-256 digest.
2. `CapabilityRegistry` describes the inputs, outputs, risk, modes, retry
   categories, alternatives, and Module 1-5 ownership of each capability.
3. `BoundedAutonomyPlanner` resolves the minimal dependency closure for the
   requested outcomes. Selection is deterministic and independent of prompt
   words.
4. `BoundedAutonomyExecutor` runs typed handlers, preserves only minimized
   metrics/reason codes, bounds attempts, and blocks undeclared or
   identifier-shaped output.
5. An authoritative policy preflight and evidence critic cannot be replaced by
   a model-provided handler.
6. `ProductionBoundedAutonomyService` may replan once to a declared alternative
   capability. It never invents an undeclared source or tool.

## Default five-module capability graph

```text
Module 5 policy preflight
        |
Module 1 source discovery
        |
Module 1 privacy-safe aggregation
        |
Module 2 semantic retrieval
        |
Module 3 cohort strategy
        |
Module 4 evolution review
        |
Module 5 governed recommendation / delivery review
        |
Module 5 evidence critic
```

This is not a fixed pipeline. A `coverage_assessment` goal selects only policy,
source discovery, and criticism. A `governed_recommendation` selects the full
dependency closure. New outcomes and capabilities can be registered without
editing prompt-routing code.

## Failure and replan rules

- Only declared error categories can be retried.
- Every capability has a maximum attempt count.
- Total attempts and replans are bounded by `AutonomyBudget`.
- Replanning excludes the failed capability and selects a declared producer of
  the same outcome.
- Missing capabilities, dependency cycles, unknown outcomes, exhausted budgets,
  and production-effect-only routes fail closed.
- The final critic verifies that every requested outcome exists before the run
  can be recorded as complete.

## Evidence minimization

Persisted reports contain:

- goal and plan fingerprints;
- requested outcome names;
- selected capability identifiers;
- task states, attempts, safe reason codes, and bounded numeric metrics;
- replan lineage; and
- immutable safety assertions.

Persisted reports do not contain:

- prompt/objective text;
- tool arguments or results;
- model reasoning traces;
- source data rows;
- identifiers; or
- activation payloads.

Migration `0028_bounded_autonomous_orchestration_evidence.sql` creates a
tenant-scoped immutable ledger with forced row-level security. The table has
database checks that keep production effects, mutation, approval, activation,
and export false.

## Feature configuration

The kernel remains disabled by default:

```dotenv
MODULE5_BOUNDED_AUTONOMY_SHADOW_ENABLED=false
MODULE5_BOUNDED_AUTONOMY_EVIDENCE_PATH=
MODULE5_BOUNDED_AUTONOMY_MAX_TASKS=16
MODULE5_BOUNDED_AUTONOMY_MAX_TOTAL_ATTEMPTS=24
MODULE5_BOUNDED_AUTONOMY_MAX_REPLANS=1
MODULE5_BOUNDED_AUTONOMY_MAX_PARALLEL_TASKS=4
```

Enabling shadow mode does not enable the existing autonomous supervisor graph,
production routing, autonomous mutation, activation, or export.

## Acceptance boundary

This increment certifies the autonomy kernel and generalized planning behavior.
It does not yet replace the live audience orchestrator with per-module tool
handlers. That cutover requires adapter contracts, shadow comparison against
the existing pipeline, unseen-goal evaluation, load testing, and explicit human
approval.
