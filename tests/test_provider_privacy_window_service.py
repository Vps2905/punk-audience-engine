import pytest

from app.models.provider_privacy_window_contracts import (
    ProviderPrivacyPartitionRegistration,
)
from app.services.provider_privacy_window_service import (
    ProviderPrivacyWindowService,
)


def _registration(index: int, *, fingerprint: str | None = None, **overrides):
    values = {
        "tenant_id": "tenant_a",
        "provider_id": "provider_a",
        "dataset_id": "signals_a",
        "schema_version": "v1",
        "delivery_window_id": "window_20260730",
        "event_time_start": "2026-07-30T00:00:00Z",
        "event_time_end": "2026-07-31T00:00:00Z",
        "partition_index": index,
        "partition_count": 2,
        "partition_algorithm": "spark_xxhash64_v1",
        "ingestion_id": f"ingestion_{index}",
        "fingerprint": fingerprint or str(index + 1) * 64,
        "row_count": 1_000_000,
    }
    values.update(overrides)
    return ProviderPrivacyPartitionRegistration(**values)


def test_window_seals_only_after_every_partition_is_ready(tmp_path):
    service = ProviderPrivacyWindowService(
        database_url=f"sqlite:///{tmp_path / 'windows.db'}"
    )
    first = _registration(0)
    second = _registration(1)

    service.register_partition(first)
    service.register_partition(second)
    service.mark_partition_ready(
        tenant_id="tenant_a",
        fingerprint=first.fingerprint,
        canonical_ref="s3://safe/window/0/",
        canonical_checksum_sha256="a" * 64,
        privacy_release_id="release_0",
        output_rows=10,
    )

    with pytest.raises(ValueError, match="incomplete"):
        service.seal_window(first.window_key)

    service.mark_partition_ready(
        tenant_id="tenant_a",
        fingerprint=second.fingerprint,
        canonical_ref="s3://safe/window/1/",
        canonical_checksum_sha256="b" * 64,
        privacy_release_id="release_1",
        output_rows=12,
    )
    sealed = service.seal_window(first.window_key)

    assert sealed["status"] == "sealed"
    assert sealed["ready_partition_count"] == 2
    assert sealed["output_rows"] == 22


def test_duplicate_partition_is_idempotent_and_changed_snapshot_is_blocked(
    tmp_path,
):
    service = ProviderPrivacyWindowService(
        database_url=f"sqlite:///{tmp_path / 'windows.db'}"
    )
    first = _registration(0)
    service.register_partition(first)
    replay = service.register_partition(first)
    assert replay["replayed"] is True

    with pytest.raises(ValueError, match="exact correction"):
        service.register_partition(
            _registration(0, fingerprint="f" * 64)
        )


def test_publication_health_reports_unsealed_windows(tmp_path):
    service = ProviderPrivacyWindowService(
        database_url=f"sqlite:///{tmp_path / 'health.db'}"
    )
    service.register_partition(_registration(0))

    report = service.publication_health(
        tenant_id="tenant_a",
        stale_after_seconds=7_200,
    )

    assert report["open_window_count"] == 1
    assert report["stale_open_window_count"] == 0
    assert report["windows"][0]["ready_partition_count"] == 0


def test_exact_correction_supersedes_open_partition(tmp_path):
    service = ProviderPrivacyWindowService(
        database_url=f"sqlite:///{tmp_path / 'windows.db'}"
    )
    first = _registration(0)
    service.register_partition(first)
    corrected = _registration(
        0,
        fingerprint="f" * 64,
        ingestion_id="ingestion_corrected",
        delivery_type="correction",
        supersedes_fingerprint=first.fingerprint,
    )
    state = service.register_partition(corrected)
    assert state["registered_partition_count"] == 1
    assert state["replayed"] is False


def test_exact_correction_reopens_and_reseals_published_window(tmp_path):
    service = ProviderPrivacyWindowService(
        database_url=f"sqlite:///{tmp_path / 'sealed-correction.db'}"
    )
    first = _registration(0, partition_count=1)
    service.register_partition(first)
    service.mark_partition_ready(
        tenant_id="tenant_a",
        fingerprint=first.fingerprint,
        canonical_ref="s3://safe/window/original/",
        canonical_checksum_sha256="a" * 64,
        privacy_release_id="release_original",
        output_rows=10,
    )
    service.seal_window(first.window_key)

    correction = _registration(
        0,
        fingerprint="f" * 64,
        partition_count=1,
        ingestion_id="ingestion_correction",
        delivery_type="correction",
        supersedes_fingerprint=first.fingerprint,
    )
    reopened = service.register_partition(correction)
    assert reopened["status"] == "open"
    assert reopened["canonical_publication_status"] == "blocked"

    service.mark_partition_ready(
        tenant_id="tenant_a",
        fingerprint=correction.fingerprint,
        canonical_ref="s3://safe/window/corrected/",
        canonical_checksum_sha256="f" * 64,
        privacy_release_id="release_correction",
        output_rows=11,
    )
    resealed = service.seal_window(first.window_key)
    assert resealed["status"] == "sealed"
    assert resealed["output_rows"] == 11
