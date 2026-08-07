from __future__ import annotations

from pathlib import Path

import pytest

from app.core.audience_job_store import AudienceJobStore


def test_audience_job_store_create_and_update(tmp_path: Path):
    store = AudienceJobStore(root_dir=tmp_path / "jobs")
    tenant_id = "tenant-a"

    record = store.create_job(
        {
            "prompt": "restaurant evening audience",
            "source": "postgres",
        },
        tenant_id=tenant_id,
    )

    assert record["tenant_id"] == tenant_id
    assert record["status"] == "queued"
    assert record["job_id"].startswith("job_")

    loaded = store.get(
        record["job_id"],
        tenant_id=tenant_id,
    )
    assert loaded["payload"]["prompt"] == "restaurant evening audience"
    assert (
        tmp_path
        / "jobs"
        / tenant_id
        / f"{record['job_id']}.json"
    ).is_file()

    updated = store.update_status(
        record["job_id"],
        tenant_id=tenant_id,
        status="running",
        stage="privacy",
        message="Running privacy layer.",
    )

    assert updated["status"] == "running"
    assert updated["progress"]["stage"] == "privacy"

    completed = store.update_status(
        record["job_id"],
        tenant_id=tenant_id,
        status="completed",
        stage="completed",
        message="Done.",
        result={"run_dir": "data/prompt_runs/example"},
    )

    assert completed["status"] == "completed"
    assert completed["result"]["run_dir"] == "data/prompt_runs/example"


def test_audience_job_store_hides_cross_tenant_jobs(tmp_path: Path):
    store = AudienceJobStore(root_dir=tmp_path / "jobs")
    record = store.create_job(
        {"prompt": "privacy-safe global audience"},
        tenant_id="tenant-a",
    )

    with pytest.raises(FileNotFoundError):
        store.get(record["job_id"], tenant_id="tenant-b")


def test_audience_job_store_rejects_tenant_reassignment(tmp_path: Path):
    store = AudienceJobStore(root_dir=tmp_path / "jobs")
    record = store.create_job(
        {"prompt": "privacy-safe global audience"},
        tenant_id="tenant-a",
    )

    with pytest.raises(ValueError) as exc_info:
        store.save(record, tenant_id="tenant-b")

    assert "ownership cannot be changed" in str(exc_info.value)


def test_audience_job_store_does_not_fallback_to_global_legacy_file(
    tmp_path: Path,
):
    root = tmp_path / "jobs"
    root.mkdir()
    (root / "job_legacy.json").write_text(
        '{"job_id": "job_legacy", "tenant_id": "tenant-a"}',
        encoding="utf-8",
    )
    store = AudienceJobStore(root_dir=root)

    with pytest.raises(FileNotFoundError):
        store.get("job_legacy", tenant_id="tenant-a")
