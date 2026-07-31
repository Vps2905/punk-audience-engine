from app.services.module1_provider_control_plane_acceptance_service import (
    Module1ProviderControlPlaneAcceptanceService,
)


def test_identifier_free_two_partition_acceptance_flow(tmp_path):
    report = Module1ProviderControlPlaneAcceptanceService(
        database_url=f"sqlite:///{tmp_path / 'control-plane.db'}"
    ).run(tenant_id="tenant_a")

    assert report["status"] == "module1_control_plane_acceptance_completed"
    assert report["window_status_after_first_partition"] == "open"
    assert report["window_status_after_all_partitions"] == "sealed"
    assert report["canonical_publication_status"] == "active"
    assert report["privacy_epsilon_charged_for_disjoint_window"] == 1.0
    assert report["duplicate_notification_idempotent"] is True
    assert report["terminal_result_replay_idempotent"] is True
    assert report["raw_identifiers_read"] is False
    assert report["activation_or_export_performed"] is False


def test_acceptance_requires_tenant(tmp_path):
    service = Module1ProviderControlPlaneAcceptanceService(
        database_url=f"sqlite:///{tmp_path / 'control-plane.db'}"
    )

    try:
        service.run(tenant_id="")
    except ValueError as exc:
        assert "tenant_id" in str(exc)
    else:
        raise AssertionError("missing tenant_id was accepted")
