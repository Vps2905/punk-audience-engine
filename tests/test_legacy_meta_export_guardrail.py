import pytest
from fastapi import HTTPException

from app.api import export_routes
from app.services import meta_export_service


def test_legacy_meta_route_disabled_without_local_storage(
    monkeypatch,
):
    monkeypatch.setattr(
        export_routes,
        "local_file_storage_allowed",
        lambda: False,
    )

    with pytest.raises(HTTPException) as exc_info:
        export_routes.export_meta(
            "cohort_1",
            seed_limit=10,
            approval_status="pending_approval",
        )

    assert exc_info.value.status_code == 404


def test_legacy_meta_route_rejects_caller_approval(
    monkeypatch,
):
    monkeypatch.setattr(
        export_routes,
        "local_file_storage_allowed",
        lambda: True,
    )

    with pytest.raises(HTTPException) as exc_info:
        export_routes.export_meta(
            "cohort_1",
            seed_limit=10,
            approval_status="approved",
        )

    assert exc_info.value.status_code == 400


def test_legacy_service_checks_guard_before_mkdir(
    tmp_path,
    monkeypatch,
):
    export_dir = tmp_path / "exports"

    monkeypatch.setattr(
        meta_export_service,
        "EXPORT_DIR",
        export_dir,
    )

    def blocked(_feature_name):
        raise RuntimeError("local storage blocked")

    monkeypatch.setattr(
        meta_export_service,
        "require_local_file_storage_allowed",
        blocked,
    )

    with pytest.raises(RuntimeError):
        meta_export_service.generate_meta_safe_export(
            cohort_id="cohort_1",
        )

    assert not export_dir.exists()
