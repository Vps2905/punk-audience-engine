import hashlib
import json
from pathlib import Path

from app.models.provider_ingestion_contracts import (
    CanonicalObjectTarget,
    ProviderDatasetContract,
    ProviderObjectDescriptor,
    ProviderObjectManifest,
)
from app.services.privacy_ingestion_pipeline_service import (
    PrivacyIngestionPipelineService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_object_ingestion_gateway_service import (
    ProviderObjectIngestionGatewayService,
)
from app.models.provider_scale_contracts import (
    ProviderDistributedJobReceipt,
)


class FakeObjectStore:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.read_count = 0
        self.write_count = 0
        self.writes = []

    def read_object(self, descriptor, *, max_bytes):
        self.read_count += 1
        assert len(self.payload) <= max_bytes
        return self.payload

    def write_canonical(
        self,
        *,
        target,
        key,
        payload,
        content_type,
        metadata=None,
    ):
        self.write_count += 1
        self.writes.append(
            {
                "target": target,
                "key": key,
                "payload": payload,
                "content_type": content_type,
                "metadata": metadata,
            }
        )
        return {
            "source_ref": f"s3://{target.bucket}/{key}",
            "version_id": "canonical-version-1",
        }


def _payload() -> bytes:
    return (
        "signal_id,event_time,region,affinity,time_window\n"
        "signal-1,2026-07-10T10:00:00Z,region-a,category-a,window-a\n"
        "signal-2,2026-07-10T10:05:00Z,region-a,category-a,window-a\n"
        "signal-3,2026-07-10T10:10:00Z,region-a,category-a,window-a\n"
    ).encode("utf-8")


def _contract(*, min_cohort_size=2, **overrides):
    values = {
        "tenant_id": "tenant_a",
        "provider_id": "provider_a",
        "dataset_id": "signals_a",
        "schema_version": "v1",
        "data_format": "csv",
        "allowed_bucket": "provider-landing",
        "allowed_prefix": "delivery/",
        "entity_id_column": "signal_id",
        "timestamp_column": "event_time",
        "cohort_columns": ("region", "affinity", "time_window"),
        "min_cohort_size": min_cohort_size,
    }
    values.update(overrides)
    return ProviderDatasetContract(
        **values,
    )


def _descriptor(payload: bytes, *, checksum=None):
    return ProviderObjectDescriptor(
        tenant_id="tenant_a",
        provider_id="provider_a",
        dataset_id="signals_a",
        bucket="provider-landing",
        key="delivery/object.csv",
        version_id="version-1",
        size_bytes=len(payload),
        content_type="text/csv",
        checksum_sha256=checksum or hashlib.sha256(payload).hexdigest(),
        server_side_encryption="AES256",
    )


def _manifest(payload: bytes, *, checksum=None, row_count=3):
    return ProviderObjectManifest(
        schema_version="v1",
        purpose="audience_intelligence",
        rights_policy_id="audience_intelligence_default",
        checksum_sha256=checksum or hashlib.sha256(payload).hexdigest(),
        row_count=row_count,
    )


def _gateway(
    tmp_path: Path,
    store,
    privacy_pipeline=None,
    distributed_launcher=None,
):
    db_url = f"sqlite:///{tmp_path / 'gateway.db'}"
    return ProviderObjectIngestionGatewayService(
        state_service=ProviderIngestionStateService(database_url=db_url),
        object_store=store,
        privacy_pipeline=privacy_pipeline
        or PrivacyIngestionPipelineService(database_url=db_url),
        distributed_launcher=distributed_launcher,
    )


def _ingest(
    gateway,
    payload,
    *,
    contract=None,
    descriptor=None,
    manifest=None,
    retry_failed=False,
):
    return gateway.ingest_object(
        contract=contract or _contract(),
        descriptor=descriptor or _descriptor(payload),
        manifest=manifest or _manifest(payload),
        canonical_target=CanonicalObjectTarget(
            bucket="canonical-safe",
            prefix="canonical/",
        ),
        retry_failed=retry_failed,
    )


def test_valid_provider_object_becomes_privacy_safe_canonical_output(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "gateway-test-salt")
    payload = _payload()
    store = FakeObjectStore(payload)
    gateway = _gateway(tmp_path, store)

    result = _ingest(gateway, payload)

    assert result["status"] == "completed"
    assert result["duplicate"] is False
    assert result["privacy_pipeline_started"] is True
    assert result["input_rows"] == 3
    assert result["output_rows"] == 1
    assert result["canonical_ref"].startswith("s3://canonical-safe/")
    assert store.read_count == 1
    assert store.write_count == 1

    canonical_text = store.writes[0]["payload"].decode("utf-8")
    canonical_row = json.loads(canonical_text)
    assert canonical_row["bounded_count"] == 3
    assert "dp_noisy_count" in canonical_row
    assert "signal_id" not in canonical_text
    assert "signal-1" not in canonical_text


def test_duplicate_notification_does_not_repeat_privacy_or_write(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "gateway-test-salt")
    payload = _payload()
    store = FakeObjectStore(payload)
    gateway = _gateway(tmp_path, store)

    first = _ingest(gateway, payload)
    second = _ingest(gateway, payload)

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert second["duplicate"] is True
    assert second["privacy_pipeline_started"] is False
    assert first["ingestion_id"] == second["ingestion_id"]
    assert store.read_count == 1
    assert store.write_count == 1


def test_checksum_tampering_is_quarantined_before_privacy_processing(
    tmp_path: Path,
):
    payload = _payload()
    declared_checksum = hashlib.sha256(b"different payload").hexdigest()
    store = FakeObjectStore(payload)
    gateway = _gateway(tmp_path, store)

    result = _ingest(
        gateway,
        payload,
        descriptor=_descriptor(payload, checksum=declared_checksum),
        manifest=_manifest(payload, checksum=declared_checksum),
    )

    assert result["status"] == "quarantined"
    assert result["reason_code"] == "checksum_verification_failed"
    assert result["privacy_pipeline_started"] is False
    assert store.read_count == 1
    assert store.write_count == 0


def test_contract_violation_is_quarantined_before_object_read(tmp_path: Path):
    payload = _payload()
    store = FakeObjectStore(payload)
    gateway = _gateway(tmp_path, store)
    descriptor = _descriptor(payload)
    descriptor = ProviderObjectDescriptor(
        **{
            **descriptor.to_safe_dict(),
            "bucket": "unregistered-bucket",
        }
    )

    result = _ingest(gateway, payload, descriptor=descriptor)

    assert result["status"] == "quarantined"
    assert result["reason_code"] == "bucket_not_allowed"
    assert result["privacy_pipeline_started"] is False
    assert store.read_count == 0
    assert store.write_count == 0


def test_k_anonymity_failure_is_blocked_without_canonical_write(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "gateway-test-salt")
    payload = _payload()
    store = FakeObjectStore(payload)
    gateway = _gateway(tmp_path, store)

    result = _ingest(
        gateway,
        payload,
        contract=_contract(min_cohort_size=1000),
    )

    assert result["status"] == "blocked"
    assert result["reason_code"] == "k_anonymity_threshold_not_met"
    assert result["privacy_pipeline_started"] is True
    assert result["output_rows"] == 0
    assert store.write_count == 0


class UnsafePrivacyPipeline:
    def process_events(self, events, **kwargs):
        return {
            "status": "completed",
            "job_id": "privacy-job",
            "safe_feature_rows": [
                {
                    "cohort": "unsafe",
                    "device_id": "must-not-cross-boundary",
                    "bounded_count": 1000,
                }
            ],
            "privacy_controls": [],
        }


def test_raw_identifier_leakage_is_quarantined(tmp_path: Path):
    payload = _payload()
    store = FakeObjectStore(payload)
    gateway = _gateway(
        tmp_path,
        store,
        privacy_pipeline=UnsafePrivacyPipeline(),
    )

    result = _ingest(gateway, payload)

    assert result["status"] == "quarantined"
    assert result["reason_code"] == "unsafe_canonical_output"
    assert result["privacy_pipeline_started"] is True
    assert store.write_count == 0


def test_same_immutable_object_has_replay_stable_dp_output(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "gateway-test-salt")
    monkeypatch.setenv("AUDIENCE_DP_SEED_SECRET", "gateway-dp-secret")
    payload = _payload()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_store = FakeObjectStore(payload)
    second_store = FakeObjectStore(payload)

    first = _ingest(_gateway(first_dir, first_store), payload)
    second = _ingest(_gateway(second_dir, second_store), payload)

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert (
        first_store.writes[0]["payload"]
        == second_store.writes[0]["payload"]
    )


class FakeDistributedLauncher:
    def __init__(self, failures=0):
        self.failures = failures
        self.requests = []

    def submit_or_get(self, request):
        self.requests.append(request)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("temporary distributed control-plane failure")
        return ProviderDistributedJobReceipt(
            job_id="distributed-job-1",
            backend="test_distributed_backend",
        )


def _distributed_contract():
    return _contract(
        max_object_bytes=1024 * 1024,
        max_rows_per_object=10_000,
        max_in_process_object_bytes=1,
        max_in_process_rows=1,
    )


def test_large_object_is_dispatched_without_entering_worker_memory(
    tmp_path: Path,
):
    payload = _payload()
    store = FakeObjectStore(payload)
    launcher = FakeDistributedLauncher()
    gateway = _gateway(
        tmp_path,
        store,
        distributed_launcher=launcher,
    )

    result = _ingest(
        gateway,
        payload,
        contract=_distributed_contract(),
    )

    assert result["status"] == "dispatched"
    assert result["execution_mode"] == "distributed"
    assert result["distributed_job_id"] == "distributed-job-1"
    assert result["privacy_pipeline_started"] is False
    assert store.read_count == 0
    assert store.write_count == 0
    assert len(launcher.requests) == 1


def test_duplicate_large_object_is_not_dispatched_twice(tmp_path: Path):
    payload = _payload()
    store = FakeObjectStore(payload)
    launcher = FakeDistributedLauncher()
    gateway = _gateway(
        tmp_path,
        store,
        distributed_launcher=launcher,
    )

    first = _ingest(
        gateway,
        payload,
        contract=_distributed_contract(),
    )
    second = _ingest(
        gateway,
        payload,
        contract=_distributed_contract(),
    )

    assert first["status"] == "dispatched"
    assert second["status"] == "dispatched"
    assert second["duplicate"] is True
    assert len(launcher.requests) == 1
    assert store.read_count == 0


def test_large_object_blocks_when_distributed_backend_is_not_configured(
    tmp_path: Path,
):
    payload = _payload()
    store = FakeObjectStore(payload)
    gateway = _gateway(tmp_path, store)

    result = _ingest(
        gateway,
        payload,
        contract=_distributed_contract(),
    )

    assert result["status"] == "blocked"
    assert result["reason_code"] == "distributed_processing_not_configured"
    assert result["execution_mode"] == "distributed"
    assert store.read_count == 0
    assert store.write_count == 0


def test_failed_distributed_dispatch_retries_idempotently(tmp_path: Path):
    payload = _payload()
    store = FakeObjectStore(payload)
    launcher = FakeDistributedLauncher(failures=1)
    gateway = _gateway(
        tmp_path,
        store,
        distributed_launcher=launcher,
    )

    first = _ingest(
        gateway,
        payload,
        contract=_distributed_contract(),
    )
    second = _ingest(
        gateway,
        payload,
        contract=_distributed_contract(),
        retry_failed=True,
    )

    assert first["status"] == "failed"
    assert first["reason_code"] == "distributed_dispatch_failed"
    assert second["status"] == "dispatched"
    assert second["distributed_job_id"] == "distributed-job-1"
    assert len(launcher.requests) == 2
    assert store.read_count == 0
