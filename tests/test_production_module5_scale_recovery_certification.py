import hashlib
import json
import threading

import pytest

from app.models.production_module3_cohort_contracts import stable_fingerprint
from app.models.production_module5_scale_recovery_contracts import (
    ScaleRecoveryCertificationRequest,
    ScaleRecoveryExerciseCase,
    ScaleRecoveryInvocationResult,
    ScaleRecoveryPolicy,
)
from app.services.production_module5_scale_recovery_certification_service import (
    ProductionModule5ScaleRecoveryCertificationService,
)
from app.services.production_module5_status_service import (
    ProductionModule5StatusService,
)


TENANT = "punk_internal"
FUNCTIONAL_FINGERPRINT = "1" * 64
AUTHORIZATION_FINGERPRINT = "2" * 64


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def functional_report():
    return {
        "status": "engineering_preview_ready",
        "request": {"tenant_id": TENANT},
        "functional_shadow_report_fingerprint": FUNCTIONAL_FINGERPRINT,
    }


def authorization_report():
    return {
        "status": "engineering_preview_ready",
        "request": {"tenant_id": TENANT},
        "lineage": {
            "source_functional_shadow_report_fingerprint": (
                FUNCTIONAL_FINGERPRINT
            )
        },
        "review": {"live_cutover_authorized": False},
        "agent_security_certification_fingerprint": AUTHORIZATION_FINGERPRINT,
    }


def request():
    return ScaleRecoveryCertificationRequest(
        tenant_id=TENANT,
        certification_id="module5-scale-recovery-1",
        evaluation_epoch_seconds=1_786_320_000,
        source_functional_shadow_report_fingerprint=FUNCTIONAL_FINGERPRINT,
        source_agent_security_certification_fingerprint=(
            AUTHORIZATION_FINGERPRINT
        ),
    )


def cases():
    return [
        ScaleRecoveryExerciseCase(
            case_id="baseline",
            scenario="baseline_throughput",
            invocation_count=1,
            work_units_per_invocation=10,
            requested_concurrency=1,
        ),
        ScaleRecoveryExerciseCase(
            case_id="concurrent",
            scenario="concurrent_execution",
            invocation_count=4,
            work_units_per_invocation=10,
            requested_concurrency=4,
        ),
        ScaleRecoveryExerciseCase(
            case_id="duplicate",
            scenario="duplicate_replay",
            invocation_count=2,
            work_units_per_invocation=10,
            requested_concurrency=2,
        ),
        *[
            ScaleRecoveryExerciseCase(
                case_id=scenario,
                scenario=scenario,
                invocation_count=1,
                work_units_per_invocation=10,
                requested_concurrency=1,
            )
            for scenario in (
                "timeout_containment",
                "worker_restart_recovery",
                "transient_database_recovery",
                "backpressure_containment",
                "circuit_breaker_containment",
            )
        ],
    ]


class SafeScenarioRunner:
    def __init__(self):
        self.barrier = threading.Barrier(4)

    def execute(self, *, case, invocation_index):
        scenario = case.scenario
        fingerprint = digest(f"{case.case_id}:safe-output")
        key = f"{case.case_id}-{invocation_index}"
        if scenario == "concurrent_execution":
            self.barrier.wait(timeout=2)
        if scenario == "duplicate_replay":
            key = "duplicate-idempotency-key"
        if scenario == "timeout_containment":
            return ScaleRecoveryInvocationResult(
                status="blocked",
                result_fingerprint=None,
                idempotency_key=key,
                error_code="deadline_exceeded",
            )
        if scenario in {
            "worker_restart_recovery",
            "transient_database_recovery",
        }:
            return ScaleRecoveryInvocationResult(
                status="recovered",
                result_fingerprint=fingerprint,
                idempotency_key=key,
                attempt_count=2,
                recovery_count=1,
                database_connection_wait_ms=(
                    5.0 if scenario == "transient_database_recovery" else 0.0
                ),
            )
        if scenario == "backpressure_containment":
            return ScaleRecoveryInvocationResult(
                status="completed",
                result_fingerprint=fingerprint,
                idempotency_key=key,
                backpressure_observed=True,
                queue_depth_observed=8,
            )
        if scenario == "circuit_breaker_containment":
            return ScaleRecoveryInvocationResult(
                status="blocked",
                result_fingerprint=None,
                idempotency_key=key,
                error_code="circuit_breaker_open",
                circuit_breaker_opened=True,
            )
        return ScaleRecoveryInvocationResult(
            status="completed",
            result_fingerprint=fingerprint,
            idempotency_key=key,
            stage_latencies_ms={
                "semantic_retrieval": 2.0,
                "cohort_strategy": 1.0,
            },
            vector_query_count=1,
        )


def service():
    return ProductionModule5ScaleRecoveryCertificationService(
        policy=ScaleRecoveryPolicy(
            minimum_total_work_units=100,
            minimum_invocation_count=12,
            minimum_observed_concurrency=4,
            minimum_work_units_per_second=0,
            maximum_p95_latency_ms=1000,
            maximum_p99_latency_ms=1000,
            maximum_peak_python_bytes=1_073_741_824,
            maximum_total_invocations=100,
            maximum_total_work_units=10_000,
        ),
        functional_shadow_validator=lambda value: dict(value),
        agent_security_validator=lambda value: dict(value),
    )


def run_report(runner=None):
    return service().run(
        request=request(),
        functional_shadow_report=functional_report(),
        agent_security_certification_report=authorization_report(),
        cases=cases(),
        runner=runner or SafeScenarioRunner(),
    )


def test_measures_concurrency_latency_throughput_and_recovery_scenarios():
    report = run_report()

    assert report["status"] == "engineering_preview_ready"
    assert report["summary"]["total_work_units"] == 120
    assert report["summary"]["invocation_count"] == 12
    assert report["summary"]["maximum_observed_concurrency"] >= 4
    assert report["summary"]["latency_p50_ms"] >= 0
    assert report["summary"]["latency_p95_ms"] >= 0
    assert report["summary"]["latency_p99_ms"] >= 0
    assert report["summary"]["work_units_per_second"] > 0
    assert report["summary"]["vector_query_count"] > 0
    assert report["summary"]["vector_queries_per_second"] > 0
    assert report["summary"]["database_connection_wait_p99_ms"] == 5.0
    assert report["summary"]["maximum_queue_depth_observed"] == 8
    assert "semantic_retrieval" in report["summary"][
        "stage_latency_percentiles"
    ]
    assert all(
        value["passed"] is True
        for value in report["scenario_results"].values()
    )
    assert report["review"]["eligible_for_staging_review"] is True
    assert report["review"]["live_cutover_authorized"] is False
    assert report["review"]["external_distributed_load_certified"] is False
    assert report["safety"]["activation_or_export_performed"] is False


def test_duplicate_replay_requires_the_same_output_fingerprint():
    class DivergentDuplicateRunner(SafeScenarioRunner):
        def execute(self, *, case, invocation_index):
            result = super().execute(case=case, invocation_index=invocation_index)
            if case.scenario != "duplicate_replay":
                return result
            return ScaleRecoveryInvocationResult(
                status="completed",
                result_fingerprint=digest(f"changed-{invocation_index}"),
                idempotency_key="duplicate-idempotency-key",
            )

    report = run_report(DivergentDuplicateRunner())

    assert report["status"] == "engineering_preview_blocked"
    assert report["scenario_results"]["duplicate_replay"]["passed"] is False
    assert report["certification_gates"]["required_recovery_scenarios"][
        "passed"
    ] is False


def test_runner_exception_is_minimized_and_blocks_certification():
    class ExceptionRunner(SafeScenarioRunner):
        def execute(self, *, case, invocation_index):
            if case.scenario == "baseline_throughput":
                raise RuntimeError("secret database host and credential")
            return super().execute(case=case, invocation_index=invocation_index)

    report = run_report(ExceptionRunner())
    encoded = json.dumps(report)

    assert report["status"] == "engineering_preview_blocked"
    assert report["summary"]["failed_invocation_count"] == 1
    assert "secret database host" not in encoded
    assert "workload_runner_exception" in encoded


def test_recomputed_summary_or_gate_tampering_is_rejected():
    report = run_report()
    tampered = json.loads(json.dumps(report))
    tampered["summary"]["latency_p99_ms"] = 0
    tampered.pop("scale_recovery_certification_fingerprint")
    tampered["scale_recovery_certification_fingerprint"] = stable_fingerprint(
        tampered
    )

    with pytest.raises(ValueError, match="summary is inconsistent"):
        service().validate_report(tampered)


def test_unexpected_observation_payload_is_rejected():
    report = run_report()
    tampered = json.loads(json.dumps(report))
    tampered["observations"][0]["internal_connection_string"] = "not-allowed"
    observation = tampered["observations"][0]
    observation.pop("observation_fingerprint")
    observation["observation_fingerprint"] = stable_fingerprint(observation)
    tampered.pop("scale_recovery_certification_fingerprint")
    tampered["scale_recovery_certification_fingerprint"] = stable_fingerprint(
        tampered
    )

    with pytest.raises(ValueError, match="unsupported fields"):
        service().validate_report(tampered)


def test_wrong_tenant_or_fingerprint_lineage_is_rejected():
    wrong = authorization_report()
    wrong["request"]["tenant_id"] = "other_tenant"

    with pytest.raises(ValueError, match="tenant lineage mismatch"):
        service().run(
            request=request(),
            functional_shadow_report=functional_report(),
            agent_security_certification_report=wrong,
            cases=cases(),
            runner=SafeScenarioRunner(),
        )


def test_unsafe_runner_result_cannot_be_constructed():
    with pytest.raises(ValueError, match="Unsafe scale invocation field"):
        ScaleRecoveryInvocationResult(
            status="completed",
            result_fingerprint=digest("unsafe"),
            idempotency_key="unsafe-result",
            production_effect_performed=True,
        )


def test_production_default_requires_at_least_ten_thousand_work_units():
    policy = ScaleRecoveryPolicy()
    assert policy.minimum_total_work_units >= 10_000
    assert policy.minimum_observed_concurrency >= 4
    assert policy.minimum_invocation_count >= 16

    production_policy = ProductionModule5ScaleRecoveryCertificationService(
        environment={}
    ).policy
    assert production_policy.require_observed_work_units is True
    assert production_policy.require_historical_pipeline is True


def test_module5_status_fails_closed_without_enabled_scale_evidence():
    status = ProductionModule5StatusService(
        environment={"MODULE5_SCALE_RECOVERY_CERTIFICATION_ENABLED": "true"}
    ).status()

    assert status["status"] == "unsafe_configuration_scale_recovery_not_ready"
    assert status["module5_scale_latency_recovery_ready"] is False
    assert status["components"][
        "module_5_11_scale_latency_recovery_certification"
    ] is False


def test_module5_status_blocks_scale_production_cutover_flag():
    status = ProductionModule5StatusService(
        environment={"MODULE5_SCALE_RECOVERY_PRODUCTION_CUTOVER_ENABLED": "true"}
    ).status()

    assert status["status"] == (
        "unsafe_configuration_release_affecting_feature_blocked"
    )
    assert status["live_production_certified"] is False


def test_scale_recovery_migration_is_immutable_tenant_scoped_and_no_effects():
    sql = open(
        "migrations/0033_module5_scale_latency_recovery_certification.sql",
        encoding="utf-8",
    ).read().lower()

    assert "enable row level security" in sql
    assert "force row level security" in sql
    assert "current_setting('app.tenant_id'" in sql
    assert "prevent_update_delete" in sql
    assert "activation" not in sql
    assert "export" not in sql
