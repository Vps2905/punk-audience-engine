from __future__ import annotations

from pathlib import Path

from app.core.audience_job_store import AudienceJobStore


def test_audience_job_store_create_and_update(tmp_path: Path):
    store = AudienceJobStore(root_dir=tmp_path / "jobs")

    record = store.create_job(
        {
            "prompt": "restaurant evening audience",
            "source": "postgres",
        }
    )

    assert record["status"] == "queued"
    assert record["job_id"].startswith("job_")

    loaded = store.get(record["job_id"])
    assert loaded["payload"]["prompt"] == "restaurant evening audience"

    updated = store.update_status(
        record["job_id"],
        status="running",
        stage="privacy",
        message="Running privacy layer.",
    )

    assert updated["status"] == "running"
    assert updated["progress"]["stage"] == "privacy"

    completed = store.update_status(
        record["job_id"],
        status="completed",
        stage="completed",
        message="Done.",
        result={"run_dir": "data/prompt_runs/example"},
    )

    assert completed["status"] == "completed"
    assert completed["result"]["run_dir"] == "data/prompt_runs/example"
