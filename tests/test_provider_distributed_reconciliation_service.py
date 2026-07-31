from sqlalchemy import create_engine, text

from app.services.provider_distributed_reconciliation_service import (
    ProviderDistributedReconciliationService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from tests.test_provider_distributed_data_plane_service import _request


class FakeStepFunctions:
    def __init__(self, status):
        self.status = status
        self.calls = []

    def describe_execution(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": self.status}


def _distributed_processing_state(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'state.db'}"
    state = ProviderIngestionStateService(database_url=database_url)
    initial = _request()
    claim = state.claim_object(initial.descriptor)
    ingestion_id = claim["record"]["ingestion_id"]
    state.transition(ingestion_id, status="validating")
    state.transition(
        ingestion_id,
        status="dispatching",
        metadata_update={"execution_mode": "distributed"},
    )
    state.transition(
        ingestion_id,
        status="dispatched",
        metadata_update={
            "distributed_job_id": (
                "arn:aws:states:test:123:execution:provider:test"
            )
        },
    )
    state.transition(
        ingestion_id,
        status="processing",
        privacy_job_id="provider_privacy_test",
    )
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE provider_ingestion_objects
                SET updated_at = :updated_at
                WHERE ingestion_id = :ingestion_id
                """
            ),
            {
                "ingestion_id": ingestion_id,
                "updated_at": "2020-01-01T00:00:00+00:00",
            },
        )
    return state, ingestion_id


def test_generic_worker_recovery_does_not_kill_active_distributed_job(
    tmp_path,
):
    state, ingestion_id = _distributed_processing_state(tmp_path)

    recovered = state.recover_stale_in_progress(
        stale_after_seconds=1
    )

    assert recovered == 0
    assert state.get(ingestion_id)["status"] == "processing"


def test_reconciliation_leaves_running_execution_active(tmp_path):
    state, ingestion_id = _distributed_processing_state(tmp_path)
    client = FakeStepFunctions("RUNNING")
    service = ProviderDistributedReconciliationService(
        state_service=state,
        step_functions_client=client,
    )

    report = service.reconcile(stale_after_seconds=1)

    assert report["still_running"] == 1
    assert report["failed_closed"] == 0
    assert state.get(ingestion_id)["status"] == "processing"


def test_reconciliation_fails_closed_when_terminal_callback_is_missing(
    tmp_path,
):
    state, ingestion_id = _distributed_processing_state(tmp_path)
    client = FakeStepFunctions("SUCCEEDED")
    service = ProviderDistributedReconciliationService(
        state_service=state,
        step_functions_client=client,
    )

    report = service.reconcile(stale_after_seconds=1)
    record = state.get(ingestion_id)

    assert report["failed_closed"] == 1
    assert record["status"] == "failed"
    assert record["reason_code"] == (
        "distributed_completion_callback_missing"
    )
