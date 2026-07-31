from app.models.provider_privacy_window_contracts import (
    ProviderDataRightsRequest,
    ProviderPrivacyPartitionRegistration,
)
from app.services.provider_data_rights_service import (
    ProviderDataRightsService,
)
from app.services.provider_privacy_window_service import (
    ProviderPrivacyWindowService,
)


def test_data_rights_request_revokes_released_windows_without_exposing_token(
    tmp_path,
):
    database_url = f"sqlite:///{tmp_path / 'rights.db'}"
    windows = ProviderPrivacyWindowService(database_url=database_url)
    partition = ProviderPrivacyPartitionRegistration(
        tenant_id="tenant_a",
        provider_id="provider_a",
        dataset_id="signals_a",
        schema_version="v1",
        delivery_window_id="window_20260730",
        event_time_start="2026-07-30T00:00:00Z",
        event_time_end="2026-07-31T00:00:00Z",
        partition_index=0,
        partition_count=1,
        partition_algorithm="spark_xxhash64_v1",
        ingestion_id="ingestion_0",
        fingerprint="a" * 64,
    )
    windows.register_partition(partition)
    windows.mark_partition_ready(
        tenant_id="tenant_a",
        fingerprint=partition.fingerprint,
        canonical_ref="s3://safe/window/0/",
        canonical_checksum_sha256="a" * 64,
        privacy_release_id="release_0",
        output_rows=10,
    )
    windows.seal_window(partition.window_key)

    service = ProviderDataRightsService(
        database_url=database_url,
        window_service=windows,
    )
    request = ProviderDataRightsRequest(
        request_id="rights_1",
        tenant_id="tenant_a",
        provider_id="provider_a",
        dataset_id="signals_a",
        request_type="delete",
        subject_token_sha256="b" * 64,
        requested_at="2026-07-31T10:00:00Z",
    )
    submitted = service.submit(request)
    applied = service.apply("rights_1", tenant_id="tenant_a")

    assert submitted["subject_token_exposed"] is False
    assert applied["status"] == "applied"
    assert applied["affected_window_count"] == 1
    assert applied["activation_blocked"] is True
    assert windows.get_window(partition.window_key)["status"] == "revoked"
