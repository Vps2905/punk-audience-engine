from app.models.provider_scale_acceptance_contracts import (
    ProviderScaleAcceptanceEvidence,
)
from app.services.provider_scale_acceptance_service import (
    ProviderScaleAcceptanceService,
)


def _evidence(**overrides):
    values = {
        "evidence_id": "scale_run_1",
        "environment": "preproduction",
        "total_events": 1_000_000_000,
        "duration_seconds": 86_400,
        "average_events_per_second": 11_575,
        "peak_events_per_second": 35_000,
        "queue_age_p99_seconds": 600,
        "end_to_end_p99_seconds": 3_600,
        "rights_propagation_seconds": 10_000,
        "cost_usd": 500,
        "duplicate_side_effect_count": 0,
        "raw_identifier_output_count": 0,
        "privacy_partition_violation_count": 0,
        "unrecovered_failure_count": 0,
        "canonical_checksum_failure_count": 0,
        "stale_data_activated_count": 0,
        "source_replay_verified": True,
        "worker_restart_verified": True,
        "backup_restore_verified": True,
    }
    values.update(overrides)
    return ProviderScaleAcceptanceEvidence(**values)


def test_measured_billion_event_evidence_passes_all_gates():
    report = ProviderScaleAcceptanceService().evaluate(_evidence())
    assert report["production_scale_certified"] is True
    assert report["blockers"] == []
    assert report["activation_or_export_performed"] is False


def test_identifier_leak_or_unrecovered_failure_blocks_certification():
    report = ProviderScaleAcceptanceService().evaluate(
        _evidence(
            raw_identifier_output_count=1,
            unrecovered_failure_count=1,
        )
    )
    assert report["production_scale_certified"] is False
    assert "raw_identifier_output_count" in report["blockers"]
    assert "unrecovered_failure_count" in report["blockers"]


def test_internally_inconsistent_throughput_evidence_is_rejected():
    report = ProviderScaleAcceptanceService().evaluate(
        _evidence(average_events_per_second=99_999)
    )
    assert report["production_scale_certified"] is False
    assert (
        "average_throughput_measurement_inconsistent"
        in report["blockers"]
    )


def test_scale_evidence_is_immutable_durable_and_tenant_scoped(tmp_path):
    service = ProviderScaleAcceptanceService(
        database_url=f"sqlite:///{tmp_path / 'scale.db'}"
    )
    first = service.record(tenant_id="tenant_a", evidence=_evidence())
    replay = service.record(tenant_id="tenant_a", evidence=_evidence())
    latest = service.latest(tenant_id="tenant_a")
    other = service.latest(tenant_id="tenant_b")

    assert first["recorded"] is True
    assert first["replayed"] is False
    assert replay["replayed"] is True
    assert latest["production_scale_certified"] is True
    assert other["production_scale_certified"] is False

    try:
        service.record(
            tenant_id="tenant_a",
            evidence=_evidence(total_events=10),
        )
    except ValueError as exc:
        assert "cannot be reused" in str(exc)
    else:
        raise AssertionError("Changed immutable evidence was accepted")
