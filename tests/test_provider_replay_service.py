from pathlib import Path

import pytest

from app.models.provider_ingestion_contracts import ProviderObjectDescriptor
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_replay_service import ProviderReplayService


class FakeQueue:
    def __init__(self):
        self.messages = []

    def send_message(self, *, body):
        self.messages.append(body)
        return f"message-{len(self.messages)}"


def _record(state, *, status):
    descriptor = ProviderObjectDescriptor(
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
    ingestion_id = state.claim_object(descriptor)["record"]["ingestion_id"]
    state.transition(ingestion_id, status="validating")
    if status in {"processing", "completed", "blocked"}:
        state.transition(ingestion_id, status="processing")
    if status == "completed":
        state.transition(
            ingestion_id,
            status="completed",
            canonical_ref="s3://canonical-safe/output.jsonl",
        )
    elif status == "blocked":
        state.transition(ingestion_id, status="blocked")
    elif status == "failed":
        state.transition(ingestion_id, status="failed")
    return ingestion_id


def test_failed_replay_requires_approval_and_requeues_same_object(tmp_path: Path):
    state = ProviderIngestionStateService(
        database_url=f"sqlite:///{tmp_path / 'replay.db'}"
    )
    ingestion_id = _record(state, status="failed")
    queue = FakeQueue()

    result = ProviderReplayService(
        state_service=state,
        queue=queue,
    ).enqueue(
        ingestion_id,
        requested_by="operator-a",
        reason="Retry after temporary provider outage",
        approval_reference="approval-123",
    )

    assert result["status"] == "queued"
    assert result["replay_mode"] == "terminal_failure_retry"
    assert len(queue.messages) == 1
    assert state.get(ingestion_id)["status"] == "failed"
    assert (
        state.get(ingestion_id)["reason_code"]
        == "controlled_replay_requested"
    )


def test_completed_replay_is_idempotency_verification_only(tmp_path: Path):
    state = ProviderIngestionStateService(
        database_url=f"sqlite:///{tmp_path / 'replay.db'}"
    )
    ingestion_id = _record(state, status="completed")
    queue = FakeQueue()

    result = ProviderReplayService(
        state_service=state,
        queue=queue,
    ).enqueue(
        ingestion_id,
        requested_by="operator-a",
        reason="Historical replay verification",
        approval_reference="approval-456",
    )

    assert result["replay_mode"] == "idempotency_verification"
    assert state.get(ingestion_id)["status"] == "completed"


def test_replay_fails_closed_without_approval_reference(tmp_path: Path):
    state = ProviderIngestionStateService(
        database_url=f"sqlite:///{tmp_path / 'replay.db'}"
    )
    ingestion_id = _record(state, status="failed")

    with pytest.raises(ValueError, match="approval_reference"):
        ProviderReplayService(
            state_service=state,
            queue=FakeQueue(),
        ).enqueue(
            ingestion_id,
            requested_by="operator-a",
            reason="Retry",
            approval_reference="",
        )
