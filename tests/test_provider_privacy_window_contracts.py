import pytest

from app.models.provider_ingestion_contracts import (
    ProviderContractValidator,
    ProviderDatasetContract,
    ProviderObjectDescriptor,
    ProviderObjectManifest,
    ProviderObjectValidationError,
)


def _contract(**overrides):
    values = {
        "tenant_id": "tenant_a",
        "provider_id": "provider_a",
        "dataset_id": "signals_a",
        "schema_version": "v1",
        "data_format": "parquet",
        "allowed_bucket": "provider-landing",
        "distributed_partition_strategy": "entity_hash_v1",
        "require_complete_privacy_partitions": True,
    }
    values.update(overrides)
    return ProviderDatasetContract(**values)


def _descriptor():
    return ProviderObjectDescriptor(
        tenant_id="tenant_a",
        provider_id="provider_a",
        dataset_id="signals_a",
        bucket="provider-landing",
        key="window/part-000.parquet",
        size_bytes=100,
        content_type="application/x-parquet",
        version_id="version-1",
        checksum_sha256="a" * 64,
        server_side_encryption="aws:kms",
    )


def _manifest(**overrides):
    values = {
        "schema_version": "v1",
        "purpose": "audience_intelligence",
        "rights_policy_id": "audience_intelligence_default",
        "checksum_sha256": "a" * 64,
        "row_count": 100,
        "event_time_start": "2026-07-30T00:00:00Z",
        "event_time_end": "2026-07-31T00:00:00Z",
        "delivery_window_id": "window_20260730",
        "partition_index": 0,
        "partition_count": 16,
        "partition_algorithm": "spark_xxhash64_v1",
        "partition_complete": True,
    }
    values.update(overrides)
    return ProviderObjectManifest(**values)


def test_complete_entity_partition_contract_is_accepted():
    ProviderContractValidator().validate(
        contract=_contract(),
        descriptor=_descriptor(),
        manifest=_manifest(),
    )


def test_partitioned_contract_fails_closed_without_window_manifest():
    with pytest.raises(ProviderObjectValidationError) as exc_info:
        ProviderContractValidator().validate(
            contract=_contract(),
            descriptor=_descriptor(),
            manifest=ProviderObjectManifest(
                schema_version="v1",
                purpose="audience_intelligence",
                rights_policy_id="audience_intelligence_default",
                checksum_sha256="a" * 64,
            ),
        )
    assert exc_info.value.reason_code == (
        "privacy_partition_manifest_required"
    )


def test_correction_requires_exact_superseded_fingerprint():
    with pytest.raises(ValueError, match="supersedes_fingerprint"):
        _manifest(delivery_type="correction")


def test_partial_partition_manifest_is_rejected():
    with pytest.raises(ValueError, match="requires window"):
        ProviderObjectManifest(
            schema_version="v1",
            purpose="audience_intelligence",
            rights_policy_id="audience_intelligence_default",
            partition_index=0,
        )

