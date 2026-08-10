# Module 5.11 — Scale, latency and recovery certification

## Outcome

Module 5.11 adds a measured engineering gate around the authorized functional
agent path created in Modules 5.9 and 5.10. It executes a caller-supplied safe
workload adapter under real threads, measures the execution rather than trusting
declared latency, and produces tamper-evident evidence for staging review.
Module 5.12 additionally supplies the concrete historical-Postgres adapter.

This module does **not** claim that AWS, the vendor feed, PostgreSQL failover,
backup restoration, disaster recovery, or a delivery connector has been
certified. Those claims remain false until their corresponding exercises run in
the real staging environment.

## What is measured

- declared and actually observed safe work units and invocation count;
- wall-clock work units and invocations per second;
- observed maximum concurrency;
- invocation p50, p95 and p99 latency;
- per-stage p50, p95 and p99 latency supplied by the real adapter;
- vector-query count and measured vector queries per second;
- database connection-wait p99 and maximum observed queue depth;
- peak Python allocation during the exercise;
- attempt, recovery, blocked and unrecovered-failure counts;
- deterministic duplicate replay without duplicate side effects;
- timeout containment;
- worker-restart recovery;
- transient database retry and recovery;
- queue backpressure containment; and
- circuit-breaker containment.

The production-default engineering policy requires at least 10,000 observed
work
units, 16 invocations, four-way observed concurrency, all eight recovery
scenarios, no unrecovered failure, and bounded latency/memory. Lower policies
are useful only for unit tests; they must not be represented as production
evidence.

`MODULE5_SCALE_RECOVERY_REQUIRE_OBSERVED_WORK_UNITS` and
`MODULE5_SCALE_RECOVERY_REQUIRE_HISTORICAL_PIPELINE` default to `true` for the
runtime service. A runner that only repeats a declared row estimate cannot pass
those gates.

## Required lineage

The certification request must bind to:

1. a validated Module 5.9 functional-shadow fingerprint; and
2. a validated Module 5.10 agent-security certification fingerprint whose own
   lineage points to that exact functional-shadow report.

Tenant identity must match across every report. Recomputed summaries, modified
observations, missing cases, changed fingerprints, or unsafe result fields make
validation fail closed.

## Workload adapter contract

Integrate the harness with the staging service through
`ScaleRecoveryWorkloadRunner.execute`. Each invocation receives a scenario and
must return `ScaleRecoveryInvocationResult`. The harness owns timing and
concurrency measurement. The adapter must return only minimized aggregate
evidence and a SHA-256 result fingerprint—never prompts, tool arguments,
credentials, raw identifiers, source rows, vectors, or audience membership.

Recommended adapter behavior:

- `baseline_throughput`: invoke the ordinary read-only functional path;
- `concurrent_execution`: invoke the same safe path from independent workers;
- `duplicate_replay`: reuse one idempotency key and prove identical output;
- `timeout_containment`: exercise a deadline and return `deadline_exceeded`;
- `worker_restart_recovery`: terminate/restart a disposable worker and resume;
- `transient_database_recovery`: inject a retryable database interruption;
- `backpressure_containment`: saturate the bounded queue and record pressure;
- `circuit_breaker_containment`: trip the breaker and return
  `circuit_breaker_open` without calling downstream work.

Runner exceptions are converted to the fixed code `workload_runner_exception`.
Exception text is not stored.

The historical adapter deliberately returns
`fault_exercise_not_configured` for fault scenarios when no staging-owned fault
controller is supplied. A timeout, database outage, worker restart, queue
saturation or breaker event must never be manufactured as passing evidence.

## Staging procedure

1. Deploy the same image and migration set intended for production.
2. Apply migration
   `0033_module5_scale_latency_recovery_certification.sql` with the migration
   role.
3. Configure a read-only, tenant-scoped functional workload adapter.
4. Use fresh privacy-safe or synthetic aggregate data. Keep every delivery and
   production-effect flag disabled.
5. Run the eight required scenarios with at least the production-default
   workload and concurrency.
6. Validate the returned report with
   `ProductionModule5ScaleRecoveryCertificationService.validate_report`.
7. Store the exact validated JSON as immutable evidence and configure
   `MODULE5_SCALE_RECOVERY_CERTIFICATION_EVIDENCE_PATH`.
8. Set `MODULE5_SCALE_RECOVERY_CERTIFICATION_ENABLED=true` only after the file
   is present and reviewed.
9. Verify Module 5 status reports
   `module_5_11_scale_latency_recovery_certification: true`.

`MODULE5_SCALE_RECOVERY_PRODUCTION_CUTOVER_ENABLED` must remain `false`. Setting
it to true is treated as an unsafe release-affecting configuration.

## Evidence boundaries

Even a passing report keeps all of these values false:

- live cutover authorized;
- fresh data certified;
- external distributed load certified;
- database failover certified;
- backup/restore certified;
- disaster recovery certified;
- automatic approval performed; and
- activation or export performed.

Therefore Module 5.11 completion means the in-process engineering harness and
its safe recovery semantics are ready for staging review. Production readiness
still requires the same scenarios on deployed infrastructure plus provider,
database, security, and delivery certification.
