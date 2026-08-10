import json

import pandas as pd

from app.agents.embedding_feature_store_agent import EmbeddingFeatureStoreAgent
from app.core.audit_logger import AuditLogger
from app.core.certification_evaluation_context import (
    certification_evaluation_context,
)
from app.models.production_module5_historical_scale_contracts import (
    HistoricalScaleWorkloadConfiguration,
    build_historical_scale_cases,
)
from app.models.production_module5_scale_recovery_contracts import (
    ScaleRecoveryCertificationRequest,
    ScaleRecoveryPolicy,
)
from app.services.production_module5_historical_scale_runner import (
    ProductionHistoricalScaleWorkloadRunner,
)
from app.services.production_module5_scale_recovery_certification_service import (
    ProductionModule5ScaleRecoveryCertificationService,
)

OBJECTIVE = "Build a privacy-safe evening restaurant audience for Montreal."
FUNCTIONAL = "1" * 64
AUTHORIZATION = "2" * 64


def configuration(expected_rows=220):
    return HistoricalScaleWorkloadConfiguration(
        tenant_id="punk_internal",
        workload_id="historical-scale-evaluation",
        objective=OBJECTIVE,
        expected_source_rows=expected_rows,
    )


class FakeHistoricalAgent:
    def __init__(self, calls):
        self.calls = calls

    def run(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "status": "completed",
            "run_id": f"volatile-run-{len(self.calls)}",
            "prompt": kwargs["prompt"],
            "source_mode": "postgres_safe_derived",
            "source_rows": 220,
            "freshness_status": "stale",
            "privacy_cohorts": 90,
            "prompt_selected_cohorts": 7,
            "approval_status": "blocked_stale_source",
            "downstream_export_enabled": False,
            "certification_evaluation": True,
            "v2_autonomous": {
                "embedding_manifest": {
                    "vector_count": 90,
                    "vector_dimension": 384,
                },
                "ranked_match_count": 90,
            },
            "safe_export": {"downstream_export_enabled": False},
            "privacy_guarantees": {
                "raw_maids_exported": False,
                "hashed_identifiers_exported": False,
                "raw_observations_exported": False,
                "raw_lat_lng_exported": False,
                "raw_email_exported": False,
                "raw_phone_exported": False,
                "individual_user_data_exported": False,
                "approval_required": True,
            },
        }


def runner(calls=None, environment=None):
    calls = calls if calls is not None else []
    return ProductionHistoricalScaleWorkloadRunner(
        configuration=configuration(),
        agent_factory=lambda: FakeHistoricalAgent(calls),
        environment=environment or {},
    )


def test_real_pipeline_adapter_is_non_persistent_and_counts_observed_rows():
    calls = []
    active = runner(calls)
    case = build_historical_scale_cases(
        profile="smoke",
        expected_source_rows=220,
    )[0]

    result = active.execute(case=case, invocation_index=0)

    assert result.status == "completed"
    assert result.observed_work_units == 220
    assert result.workload_source == "historical_postgres_pipeline"
    assert result.vector_query_count == 1
    assert calls[0]["source"] == "postgres"
    assert calls[0]["persist_artifacts"] is False
    assert calls[0]["certification_evaluation"] is True
    assert calls[0]["approval_required"] is True


def test_duplicate_replay_ignores_volatile_run_identifiers():
    active = runner([])
    case = build_historical_scale_cases(
        profile="smoke",
        expected_source_rows=220,
    )[2]

    first = active.execute(case=case, invocation_index=0)
    second = active.execute(case=case, invocation_index=1)

    assert first.idempotency_key == second.idempotency_key
    assert first.result_fingerprint == second.result_fingerprint


def test_fault_scenarios_fail_closed_without_real_fault_driver():
    active = runner([])
    timeout_case = build_historical_scale_cases(
        profile="smoke",
        expected_source_rows=220,
    )[3]

    result = active.execute(case=timeout_case, invocation_index=0)

    assert result.status == "blocked"
    assert result.error_code == "fault_exercise_not_configured"
    assert result.observed_work_units == 0
    assert result.workload_source == "infrastructure_fault_driver"


def test_release_effect_flag_blocks_before_pipeline_execution():
    calls = []
    active = runner(
        calls,
        environment={"MODULE5_PRODUCTION_ROUTING_ENABLED": "true"},
    )
    case = build_historical_scale_cases(
        profile="smoke",
        expected_source_rows=220,
    )[0]

    result = active.execute(case=case, invocation_index=0)

    assert result.status == "blocked"
    assert result.error_code == "release_effect_flag_enabled"
    assert calls == []


def test_planner_targets_ten_thousand_and_one_hundred_thousand_rows():
    ten_thousand = build_historical_scale_cases(
        profile="10k",
        expected_source_rows=220,
        concurrency=4,
    )
    one_hundred_thousand = build_historical_scale_cases(
        profile="100k",
        expected_source_rows=220,
        concurrency=8,
    )

    ten_k_real_calls = sum(case.invocation_count for case in ten_thousand[:3])
    hundred_k_real_calls = sum(
        case.invocation_count for case in one_hundred_thousand[:3]
    )
    assert ten_k_real_calls * 220 >= 10_000
    assert hundred_k_real_calls * 220 >= 100_000
    assert {case.scenario for case in ten_thousand} == {
        "baseline_throughput",
        "concurrent_execution",
        "duplicate_replay",
        "timeout_containment",
        "worker_restart_recovery",
        "transient_database_recovery",
        "backpressure_containment",
        "circuit_breaker_containment",
    }


def test_scale_report_uses_observed_rows_and_does_not_store_objective():
    cases = build_historical_scale_cases(
        profile="smoke",
        expected_source_rows=220,
        concurrency=1,
    )
    service = ProductionModule5ScaleRecoveryCertificationService(
        policy=ScaleRecoveryPolicy(
            minimum_total_work_units=1,
            minimum_invocation_count=1,
            minimum_observed_concurrency=1,
            minimum_work_units_per_second=0,
            maximum_p95_latency_ms=10_000,
            maximum_p99_latency_ms=10_000,
            maximum_peak_python_bytes=1_073_741_824,
            maximum_total_invocations=100,
            maximum_total_work_units=100_000,
            require_observed_work_units=True,
            require_historical_pipeline=True,
        ),
        functional_shadow_validator=lambda value: dict(value),
        agent_security_validator=lambda value: dict(value),
    )
    report = service.run(
        request=ScaleRecoveryCertificationRequest(
            tenant_id="punk_internal",
            certification_id="historical-scale-test",
            evaluation_epoch_seconds=1,
            source_functional_shadow_report_fingerprint=FUNCTIONAL,
            source_agent_security_certification_fingerprint=AUTHORIZATION,
        ),
        functional_shadow_report={
            "status": "engineering_preview_ready",
            "request": {"tenant_id": "punk_internal"},
            "functional_shadow_report_fingerprint": FUNCTIONAL,
        },
        agent_security_certification_report={
            "status": "engineering_preview_ready",
            "request": {"tenant_id": "punk_internal"},
            "lineage": {
                "source_functional_shadow_report_fingerprint": FUNCTIONAL
            },
            "review": {"live_cutover_authorized": False},
            "agent_security_certification_fingerprint": AUTHORIZATION,
        },
        cases=cases,
        runner=runner([]),
    )

    assert report["summary"]["declared_total_work_units"] > (
        report["summary"]["total_work_units"]
    )
    assert report["certification_gates"]["observed_work_units"]["passed"]
    assert report["certification_gates"]["historical_pipeline"]["passed"]
    assert report["certification_gates"]["required_recovery_scenarios"][
        "passed"
    ] is False
    assert OBJECTIVE not in json.dumps(report)


def test_in_memory_embedding_creates_no_artifact_directory(tmp_path):
    target = tmp_path / "must-not-exist"
    cohorts = pd.DataFrame(
        [
            {
                "location_name": "montreal",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
                "lookback_bucket": "0_30d",
                "sessions": 1000,
                "total_maid_volume": 1200,
                "noisy_maid_volume": 1199,
                "total_observations": 1500,
                "quality_score": 0.8,
                "privacy_status": "passed",
                "trait_text": "montreal restaurant evening",
            }
        ]
    )

    manifest, metadata, vectors = EmbeddingFeatureStoreAgent().build_in_memory(
        cohorts=cohorts,
        run_id="memory-evaluation",
    )

    assert manifest["storage_backend"] == "memory"
    assert len(metadata) == 1
    assert vectors.shape[0] == 1
    assert target.exists() is False


def test_certification_context_suppresses_legacy_local_audit_writes(tmp_path):
    audit_path = tmp_path / "audit" / "events.jsonl"

    with certification_evaluation_context():
        record = AuditLogger(audit_path).log("scale_evaluation", {"safe": True})

    assert record["storage_backend"] == "run_history_jsonb"
    assert audit_path.exists() is False
