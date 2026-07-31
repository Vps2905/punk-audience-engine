from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from app.models.provider_ingestion_contracts import ProviderObjectDescriptor
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)


def _descriptor():
    return ProviderObjectDescriptor(
        tenant_id="tenant_a",
        provider_id="provider_a",
        dataset_id="dataset_a",
        bucket="provider-landing",
        key="delivery/object.csv",
        version_id="version-1",
        size_bytes=10,
        content_type="text/csv",
        server_side_encryption="AES256",
    )


def test_state_claim_is_idempotent(tmp_path: Path):
    service = ProviderIngestionStateService(
        database_url=f"sqlite:///{tmp_path / 'state.db'}"
    )

    first = service.claim_object(_descriptor())
    second = service.claim_object(_descriptor())

    assert first["claimed"] is True
    assert first["duplicate"] is False
    assert second["claimed"] is False
    assert second["duplicate"] is True
    assert first["record"]["ingestion_id"] == second["record"]["ingestion_id"]


def test_state_enforces_terminal_transitions(tmp_path: Path):
    service = ProviderIngestionStateService(
        database_url=f"sqlite:///{tmp_path / 'state.db'}"
    )
    ingestion_id = service.claim_object(_descriptor())["record"]["ingestion_id"]

    service.transition(ingestion_id, status="validating")
    service.transition(
        ingestion_id,
        status="processing",
        input_rows=10,
        increment_attempt=True,
    )
    completed = service.transition(
        ingestion_id,
        status="completed",
        output_rows=1,
        canonical_ref="s3://canonical/output.jsonl",
    )

    assert completed["attempt_count"] == 1
    assert completed["status"] == "completed"
    with pytest.raises(ValueError, match="completed -> failed"):
        service.transition(ingestion_id, status="failed")


def test_state_recovers_stale_processing_after_worker_restart(tmp_path: Path):
    db_path = tmp_path / "state.db"
    db_url = f"sqlite:///{db_path}"
    service = ProviderIngestionStateService(database_url=db_url)
    ingestion_id = service.claim_object(_descriptor())["record"]["ingestion_id"]
    service.transition(ingestion_id, status="validating")
    service.transition(ingestion_id, status="processing")

    with create_engine(db_url).begin() as conn:
        conn.execute(
            text(
                """
                UPDATE provider_ingestion_objects
                SET updated_at = :updated_at
                WHERE ingestion_id = :ingestion_id
                """
            ),
            {
                "ingestion_id": ingestion_id,
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
        )

    recovered = service.recover_stale_in_progress(stale_after_seconds=1)

    assert recovered == 1
    record = service.get(ingestion_id)
    assert record["status"] == "failed"
    assert record["reason_code"] == "worker_execution_stale"


def test_state_supports_durable_distributed_dispatch(tmp_path: Path):
    service = ProviderIngestionStateService(
        database_url=f"sqlite:///{tmp_path / 'state.db'}"
    )
    ingestion_id = service.claim_object(_descriptor())["record"]["ingestion_id"]

    service.transition(ingestion_id, status="validating")
    service.transition(ingestion_id, status="dispatching")
    dispatched = service.transition(
        ingestion_id,
        status="dispatched",
        metadata_update={
            "execution_mode": "distributed",
            "distributed_job_id": "job-1",
        },
    )

    assert dispatched["status"] == "dispatched"
    assert dispatched["completed_at"] is None
    assert dispatched["metadata"]["distributed_job_id"] == "job-1"
