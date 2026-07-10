from pathlib import Path

import pytest

from app.services.ingestion_job_service import (
    IngestionJobCreate,
    IngestionJobService,
)


def test_ingestion_job_lifecycle(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    service = IngestionJobService(database_url=db_url)

    created = service.create_job(
        IngestionJobCreate(
            source_type="csv",
            source_ref="sample.csv",
            run_id="run_1",
            actor="tester",
            metadata={"customer": "demo"},
        )
    )

    assert created["enabled"] is True
    assert created["status"] == "queued"
    assert created["job_id"].startswith("ingest_")

    job_id = created["job_id"]

    running = service.mark_running(job_id)
    assert running["status"] == "running"
    assert running["job"]["status"] == "running"

    completed = service.mark_completed(
        job_id=job_id,
        input_rows=100,
        output_rows=80,
        dropped_rows=20,
        metadata_update={"lineage_recorded": True},
    )

    assert completed["status"] == "completed"
    assert completed["job"]["status"] == "completed"
    assert completed["job"]["input_rows"] == 100
    assert completed["job"]["output_rows"] == 80
    assert completed["job"]["dropped_rows"] == 20
    assert completed["job"]["metadata"]["customer"] == "demo"
    assert completed["job"]["metadata"]["lineage_recorded"] is True

    fetched = service.get_job(job_id)
    assert fetched["status"] == "ok"
    assert fetched["job"]["job_id"] == job_id
    assert fetched["job"]["status"] == "completed"


def test_ingestion_job_failed_and_blocked_states(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    service = IngestionJobService(database_url=db_url)

    failed_job = service.create_job(IngestionJobCreate(source_type="api"))
    failed = service.mark_failed(
        job_id=failed_job["job_id"],
        error_message="source API timeout",
    )

    assert failed["status"] == "failed"
    assert failed["job"]["error_message"] == "source API timeout"

    blocked_job = service.create_job(IngestionJobCreate(source_type="csv"))
    blocked = service.mark_blocked(
        job_id=blocked_job["job_id"],
        reason="k-anonymity threshold failed",
    )

    assert blocked["status"] == "blocked"
    assert blocked["job"]["error_message"] == "k-anonymity threshold failed"


def test_ingestion_job_validation(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    service = IngestionJobService(database_url=db_url)

    with pytest.raises(ValueError, match="source_type is required"):
        service.create_job(IngestionJobCreate(source_type=""))

    with pytest.raises(ValueError, match="job_id is required"):
        service.get_job("")

    with pytest.raises(ValueError, match="input_rows must be >= 0"):
        service.mark_completed(
            job_id="missing",
            input_rows=-1,
            output_rows=0,
        )


def test_ingestion_job_not_found(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    service = IngestionJobService(database_url=db_url)

    result = service.mark_running("missing_job")

    assert result["enabled"] is True
    assert result["status"] == "not_found"
    assert result["job_id"] == "missing_job"
