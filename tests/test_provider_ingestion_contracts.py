import hashlib

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
        "dataset_id": "visitation_signals",
        "schema_version": "v1",
        "data_format": "csv",
        "allowed_bucket": "provider-landing",
        "allowed_prefix": "delivery/",
    }
    values.update(overrides)
    return ProviderDatasetContract(**values)


def _descriptor(payload: bytes, **overrides):
    values = {
        "tenant_id": "tenant_a",
        "provider_id": "provider_a",
        "dataset_id": "visitation_signals",
        "bucket": "provider-landing",
        "key": "delivery/object.csv",
        "size_bytes": len(payload),
        "content_type": "text/csv",
        "version_id": "version-1",
        "checksum_sha256": hashlib.sha256(payload).hexdigest(),
        "server_side_encryption": "AES256",
    }
    values.update(overrides)
    return ProviderObjectDescriptor(**values)


def _manifest(payload: bytes, **overrides):
    values = {
        "schema_version": "v1",
        "purpose": "audience_intelligence",
        "rights_policy_id": "audience_intelligence_default",
        "checksum_sha256": hashlib.sha256(payload).hexdigest(),
        "row_count": 1,
    }
    values.update(overrides)
    return ProviderObjectManifest(**values)


@pytest.mark.parametrize(
    ("descriptor_overrides", "manifest_overrides", "reason_code"),
    [
        ({"bucket": "unregistered-bucket"}, {}, "bucket_not_allowed"),
        ({"key": "outside/object.csv"}, {}, "key_prefix_not_allowed"),
        ({"server_side_encryption": None}, {}, "encryption_required"),
        ({}, {"schema_version": "v2"}, "schema_version_mismatch"),
        ({}, {"rights_policy_id": "unapproved"}, "rights_policy_mismatch"),
        ({}, {"purpose": "unapproved"}, "purpose_not_allowed"),
    ],
)
def test_contract_validator_fails_closed(
    descriptor_overrides,
    manifest_overrides,
    reason_code,
):
    payload = b"value\n1\n"
    with pytest.raises(ProviderObjectValidationError) as exc_info:
        ProviderContractValidator().validate(
            contract=_contract(),
            descriptor=_descriptor(payload, **descriptor_overrides),
            manifest=_manifest(payload, **manifest_overrides),
        )

    assert exc_info.value.reason_code == reason_code


def test_object_fingerprint_changes_with_immutable_version():
    payload = b"value\n1\n"
    first = _descriptor(payload, version_id="version-1")
    second = _descriptor(payload, version_id="version-2")

    assert first.fingerprint != second.fingerprint
    assert len(first.fingerprint) == 64


def test_contract_rejects_non_object_gateway_connection():
    with pytest.raises(ValueError, match="connection_type=s3"):
        _contract(connection_type="database")


def test_contract_rejects_invalid_scale_execution_limits():
    with pytest.raises(
        ValueError,
        match="max_in_process_object_bytes",
    ):
        _contract(
            max_object_bytes=100,
            max_in_process_object_bytes=101,
        )

    with pytest.raises(ValueError, match="execution_mode"):
        _contract(execution_mode="unbounded")
