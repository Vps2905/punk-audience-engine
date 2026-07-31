import hashlib
import json
from pathlib import Path

from app.models.provider_ingestion_contracts import (
    CanonicalObjectTarget,
    ProviderDatasetContract,
    ProviderObjectHead,
    ProviderObjectManifest,
)
from app.models.provider_queue_contracts import (
    ProviderQueueMessage,
    ProviderWorkerConfig,
)
from app.services.privacy_ingestion_pipeline_service import (
    PrivacyIngestionPipelineService,
)
from app.services.provider_contract_registry_service import (
    ProviderContractRegistryService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_ingestion_worker_service import (
    ProviderIngestionWorkerService,
)
from app.services.provider_object_ingestion_gateway_service import (
    ProviderObjectIngestionGatewayService,
)


def _payload() -> bytes:
    return (
        b"signal_id,event_time,region,affinity,time_window\n"
        b"signal-1,2026-07-24T10:00:00Z,region-a,category-a,window-a\n"
        b"signal-2,2026-07-24T10:05:00Z,region-a,category-a,window-a\n"
        b"signal-3,2026-07-24T10:10:00Z,region-a,category-a,window-a\n"
    )


def _contract():
    return ProviderDatasetContract(
        tenant_id="tenant_a",
        provider_id="provider_a",
        dataset_id="signals_a",
        schema_version="v1",
        data_format="csv",
        allowed_bucket="provider-landing",
        allowed_prefix="delivery/",
        entity_id_column="signal_id",
        timestamp_column="event_time",
        cohort_columns=("region", "affinity", "time_window"),
        min_cohort_size=2,
    )


def _message(message_id="message-1", receive_count=1, body=None):
    event = {
        "Records": [
            {
                "eventTime": "2026-07-24T10:15:00Z",
                "eventName": "ObjectCreated:Put",
                "responseElements": {"x-amz-request-id": message_id},
                "s3": {
                    "bucket": {"name": "provider-landing"},
                    "object": {
                        "key": "delivery%2Fobject.csv",
                        "versionId": "version-1",
                    },
                },
            }
        ]
    }
    return ProviderQueueMessage(
        message_id=message_id,
        receipt_handle=f"receipt-{message_id}",
        body=body if body is not None else json.dumps(event),
        receive_count=receive_count,
    )


class FakeQueue:
    def __init__(self, messages=None):
        self.messages = list(messages or [])
        self.deleted = []
        self.visibility = []
        self.dead_letters = []
        self.sent = []

    def receive_messages(self, **kwargs):
        del kwargs
        messages = self.messages
        self.messages = []
        return messages

    def delete_message(self, receipt_handle):
        self.deleted.append(receipt_handle)

    def change_visibility(
        self,
        receipt_handle,
        *,
        visibility_timeout_seconds,
    ):
        self.visibility.append(
            (receipt_handle, visibility_timeout_seconds)
        )

    def send_to_dlq(self, **kwargs):
        self.dead_letters.append(kwargs)

    def send_message(self, *, body):
        self.sent.append(body)
        return f"sent-{len(self.sent)}"

    def attributes(self):
        return {"available": 0, "in_flight": 0, "delayed": 0, "dlq": 0}


class TransientConnectionError(RuntimeError):
    pass


class FakeObjectStore:
    def __init__(self, payload, *, read_failures=0):
        self.payload = payload
        self.read_failures = read_failures
        self.read_count = 0
        self.write_count = 0
        self.writes = []

    def head_object(self, event):
        return ProviderObjectHead(
            size_bytes=len(self.payload),
            content_type="text/csv",
            version_id=event.version_id,
            checksum_sha256=hashlib.sha256(self.payload).hexdigest(),
            server_side_encryption="AES256",
            last_modified="2026-07-24T10:15:00+00:00",
            manifest=ProviderObjectManifest(
                schema_version="v1",
                purpose="audience_intelligence",
                rights_policy_id="audience_intelligence_default",
                checksum_sha256=hashlib.sha256(self.payload).hexdigest(),
                row_count=3,
                event_time_start="2026-07-24T10:00:00+00:00",
                event_time_end="2026-07-24T10:10:00+00:00",
            ),
        )

    def read_object(self, descriptor, *, max_bytes):
        del descriptor
        self.read_count += 1
        assert len(self.payload) <= max_bytes
        if self.read_failures:
            self.read_failures -= 1
            raise TransientConnectionError("temporary")
        return self.payload

    def write_canonical(self, **kwargs):
        self.write_count += 1
        self.writes.append(kwargs)
        return {
            "source_ref": (
                f"s3://{kwargs['target'].bucket}/{kwargs['key']}"
            ),
            "version_id": f"canonical-{self.write_count}",
        }


def _worker(
    tmp_path: Path,
    *,
    queue,
    store,
    register_contract=True,
    max_receive_count=3,
    gateway_override=None,
):
    db_url = f"sqlite:///{tmp_path / 'worker.db'}"
    registry = ProviderContractRegistryService(database_url=db_url)
    if register_contract:
        registry.register(_contract(), actor="test")
    state = ProviderIngestionStateService(database_url=db_url)
    gateway = gateway_override or ProviderObjectIngestionGatewayService(
        state_service=state,
        object_store=store,
        privacy_pipeline=PrivacyIngestionPipelineService(
            database_url=db_url
        ),
    )
    return ProviderIngestionWorkerService(
        queue=queue,
        registry=registry,
        state_service=state,
        object_store=store,
        gateway=gateway,
        canonical_target=CanonicalObjectTarget(
            bucket="canonical-safe",
            prefix="canonical/",
        ),
        config=ProviderWorkerConfig(
            max_messages=5,
            wait_time_seconds=0,
            visibility_timeout_seconds=1,
            max_receive_count=max_receive_count,
            retry_base_seconds=2,
            retry_max_seconds=10,
            stale_processing_seconds=1,
        ),
    )


def test_worker_processes_and_acknowledges_safe_provider_object(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "test-hash-salt")
    monkeypatch.setenv("AUDIENCE_DP_SEED_SECRET", "test-dp-secret")
    queue = FakeQueue([_message()])
    store = FakeObjectStore(_payload())
    worker = _worker(tmp_path, queue=queue, store=store)

    result = worker.run_once()

    assert result["acknowledged"] == 1
    assert result["dead_lettered"] == 0
    assert store.write_count == 1
    assert queue.deleted == ["receipt-message-1"]


def test_duplicate_s3_messages_create_one_canonical_output(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "test-hash-salt")
    monkeypatch.setenv("AUDIENCE_DP_SEED_SECRET", "test-dp-secret")
    queue = FakeQueue(
        [
            _message("message-1"),
            _message("message-2"),
        ]
    )
    store = FakeObjectStore(_payload())
    worker = _worker(tmp_path, queue=queue, store=store)

    result = worker.run_once()

    assert result["acknowledged"] == 2
    assert store.read_count == 1
    assert store.write_count == 1


def test_transient_gateway_failure_retries_then_reenters_failed_run(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "test-hash-salt")
    monkeypatch.setenv("AUDIENCE_DP_SEED_SECRET", "test-dp-secret")
    queue = FakeQueue()
    store = FakeObjectStore(_payload(), read_failures=1)
    worker = _worker(tmp_path, queue=queue, store=store)

    first = worker.process_message(_message(receive_count=1))
    second = worker.process_message(_message(receive_count=2))

    assert first["queue_action"] == "retry_scheduled"
    assert first["retry_delay_seconds"] == 2
    assert second["queue_action"] == "acknowledged"
    assert store.write_count == 1


def test_missing_contract_retries_then_dead_letters_at_limit(tmp_path):
    queue = FakeQueue()
    store = FakeObjectStore(_payload())
    worker = _worker(
        tmp_path,
        queue=queue,
        store=store,
        register_contract=False,
        max_receive_count=2,
    )

    first = worker.process_message(_message(receive_count=1))
    final = worker.process_message(_message(receive_count=2))

    assert first["queue_action"] == "retry_scheduled"
    assert final["queue_action"] == "dead_lettered"
    assert final["reason_code"] == "provider_contract_not_available"
    assert len(queue.dead_letters) == 1


def test_malformed_event_goes_directly_to_dlq(tmp_path):
    queue = FakeQueue()
    store = FakeObjectStore(_payload())
    worker = _worker(tmp_path, queue=queue, store=store)

    result = worker.process_message(
        _message(body="not-json", receive_count=1)
    )

    assert result["queue_action"] == "dead_lettered"
    assert result["reason_code"] == "invalid_queue_message_json"
    assert len(queue.dead_letters) == 1


class DispatchedGateway:
    def ingest_object(self, **kwargs):
        del kwargs
        return {
            "ingestion_id": "provider_ingest_dispatched",
            "status": "dispatched",
            "reason_code": None,
            "duplicate": False,
        }


def test_worker_acknowledges_durable_distributed_dispatch(tmp_path):
    queue = FakeQueue()
    store = FakeObjectStore(_payload())
    worker = _worker(
        tmp_path,
        queue=queue,
        store=store,
        gateway_override=DispatchedGateway(),
    )

    result = worker.process_message(_message())

    assert result["queue_action"] == "acknowledged"
    assert result["ingestion_status"] == "dispatched"
    assert queue.deleted == ["receipt-message-1"]
