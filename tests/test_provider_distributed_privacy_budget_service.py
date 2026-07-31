from app.services.provider_distributed_privacy_budget_service import (
    ProviderDistributedPrivacyBudgetService,
)
from app.models.provider_ingestion_contracts import (
    ProviderDatasetContract,
    ProviderObjectManifest,
)
from app.models.provider_scale_contracts import ProviderDistributedJobRequest
from tests.test_provider_distributed_data_plane_service import _request


def test_privacy_budget_reservation_is_idempotent(tmp_path):
    service = ProviderDistributedPrivacyBudgetService(
        database_url=f"sqlite:///{tmp_path / 'budget.db'}"
    )
    request = _request(epsilon=0.6, max_budget=1.0)

    first = service.reserve(request)
    second = service.reserve(request)

    assert first["decision"] == "allowed"
    assert first["replayed"] is False
    assert second["decision"] == "allowed"
    assert second["replayed"] is True
    assert second["release_id"] == first["release_id"]
    assert second["budget_after"] == 0.6


def test_concurrent_scope_budget_is_reserved_before_processing(tmp_path):
    service = ProviderDistributedPrivacyBudgetService(
        database_url=f"sqlite:///{tmp_path / 'budget.db'}"
    )
    first = _request(
        ingestion_id="provider_ingest_1",
        key="delivery/one.parquet",
        epsilon=0.6,
        max_budget=1.0,
    )
    second = _request(
        ingestion_id="provider_ingest_2",
        key="delivery/two.parquet",
        epsilon=0.6,
        max_budget=1.0,
    )

    allowed = service.reserve(first)
    blocked = service.reserve(second)

    assert allowed["decision"] == "allowed"
    assert blocked["decision"] == "blocked"
    assert blocked["reason_code"] == "privacy_budget_exceeded"
    assert blocked["budget_before"] == 0.6
    assert blocked["budget_after"] == 0.6


def test_failed_release_remains_conservatively_charged(tmp_path):
    service = ProviderDistributedPrivacyBudgetService(
        database_url=f"sqlite:///{tmp_path / 'budget.db'}"
    )
    first = _request(
        ingestion_id="provider_ingest_1",
        key="delivery/one.parquet",
        epsilon=0.6,
        max_budget=1.0,
    )
    second = _request(
        ingestion_id="provider_ingest_2",
        key="delivery/two.parquet",
        epsilon=0.6,
        max_budget=1.0,
    )

    reservation = service.reserve(first)
    terminal = service.mark_terminal(
        reservation["release_id"],
        status="failed",
        reason_code="distributed_execution_failed",
    )
    blocked = service.reserve(second)

    assert terminal["status"] == "failed"
    assert blocked["decision"] == "blocked"


def _partitioned_request(index: int, *, name: str | None = None):
    object_name = name or f"partition-{index}"
    base = _request(
        ingestion_id=f"provider_ingest_{object_name}",
        key=f"delivery/{object_name}.parquet",
        epsilon=0.6,
        max_budget=1.0,
    )
    contract = ProviderDatasetContract(
        **{
            **base.contract.to_safe_dict(),
            "distributed_partition_strategy": "entity_hash_v1",
            "require_complete_privacy_partitions": True,
        }
    )
    manifest = ProviderObjectManifest(
        **{
            **base.manifest.to_safe_dict(),
            "event_time_start": "2026-07-30T00:00:00Z",
            "event_time_end": "2026-07-31T00:00:00Z",
            "delivery_window_id": "window_20260730",
            "partition_index": index,
            "partition_count": 2,
            "partition_algorithm": "spark_xxhash64_v1",
            "partition_complete": True,
        }
    )
    return ProviderDistributedJobRequest(
        **{
            **base.__dict__,
            "contract": contract,
            "manifest": manifest,
        }
    )


def test_disjoint_window_partitions_use_parallel_dp_composition(tmp_path):
    service = ProviderDistributedPrivacyBudgetService(
        database_url=f"sqlite:///{tmp_path / 'budget.db'}"
    )

    first = service.reserve(_partitioned_request(0))
    second = service.reserve(_partitioned_request(1))

    assert first["decision"] == "allowed"
    assert first["charged_epsilon"] == 0.6
    assert second["decision"] == "allowed"
    assert second["charged_epsilon"] == 0.0
    assert second["budget_after"] == 0.6
    assert second["composition_group"] == first["composition_group"]


def test_correction_of_same_partition_is_sequentially_charged(tmp_path):
    service = ProviderDistributedPrivacyBudgetService(
        database_url=f"sqlite:///{tmp_path / 'correction-budget.db'}"
    )
    first = _partitioned_request(0)
    correction_base = _partitioned_request(0, name="correction-0")
    correction_manifest = ProviderObjectManifest(
        **{
            **correction_base.manifest.to_safe_dict(),
            "delivery_type": "correction",
            "supersedes_fingerprint": first.fingerprint,
        }
    )
    correction_contract = ProviderDatasetContract(
        **{
            **correction_base.contract.to_safe_dict(),
            "max_cumulative_epsilon": 2.0,
        }
    )
    correction = ProviderDistributedJobRequest(
        **{
            **correction_base.__dict__,
            "ingestion_id": "provider_ingest_correction",
            "contract": correction_contract,
            "manifest": correction_manifest,
        }
    )

    initial = service.reserve(first)
    corrected = service.reserve(correction)

    assert initial["charged_epsilon"] == 0.6
    assert corrected["charged_epsilon"] == 0.6
    assert corrected["budget_after"] == 1.2
