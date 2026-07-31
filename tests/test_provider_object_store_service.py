import base64
import hashlib
from datetime import datetime, timezone
from io import BytesIO

import pytest

from app.models.provider_ingestion_contracts import (
    CanonicalObjectTarget,
    ProviderObjectDescriptor,
)
from app.models.provider_queue_contracts import ProviderS3ObjectEvent
from app.services.provider_object_store_service import (
    S3ProviderObjectStore,
)


class FakeS3Client:
    def __init__(self, payload=b"provider-data"):
        self.payload = payload
        self.get_requests = []
        self.head_requests = []
        self.put_requests = []

    def get_object(self, **request):
        self.get_requests.append(request)
        return {"Body": BytesIO(self.payload)}

    def put_object(self, **request):
        self.put_requests.append(request)
        return {"VersionId": "canonical-version"}

    def head_object(self, **request):
        self.head_requests.append(request)
        return {
            "ContentLength": len(self.payload),
            "ContentType": "text/csv",
            "VersionId": "provider-version",
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(self.payload).digest()
            ).decode("ascii"),
            "ServerSideEncryption": "AES256",
            "LastModified": datetime(
                2026,
                7,
                24,
                10,
                0,
                tzinfo=timezone.utc,
            ),
            "Metadata": {
                "punk-schema-version": "v1",
                "punk-purpose": "audience_intelligence",
                "punk-rights-policy-id": "audience_intelligence_default",
                "punk-row-count": "3",
                "punk-event-time-start": "2026-07-24T09:00:00+00:00",
                "punk-event-time-end": "2026-07-24T10:00:00+00:00",
                "punk-delivery-window-id": "window-2026-07-24",
                "punk-partition-index": "0",
                "punk-partition-count": "4",
                "punk-partition-algorithm": "spark_xxhash64_v1",
                "punk-partition-complete": "true",
                "punk-delivery-type": "snapshot",
            },
        }


def _descriptor():
    return ProviderObjectDescriptor(
        tenant_id="tenant_a",
        provider_id="provider_a",
        dataset_id="dataset_a",
        bucket="provider-landing",
        key="delivery/object.csv",
        size_bytes=13,
        content_type="text/csv",
        version_id="provider-version",
        server_side_encryption="AES256",
    )


def test_s3_adapter_reads_exact_immutable_version_with_bound():
    client = FakeS3Client()
    store = S3ProviderObjectStore(client=client)

    payload = store.read_object(_descriptor(), max_bytes=100)

    assert payload == b"provider-data"
    assert client.get_requests == [
        {
            "Bucket": "provider-landing",
            "Key": "delivery/object.csv",
            "VersionId": "provider-version",
        }
    ]


def test_s3_adapter_rejects_response_over_read_limit():
    store = S3ProviderObjectStore(client=FakeS3Client(payload=b"too-large"))

    with pytest.raises(RuntimeError, match="configured read limit"):
        store.read_object(_descriptor(), max_bytes=3)


def test_s3_adapter_builds_manifest_from_safe_object_metadata():
    client = FakeS3Client()
    store = S3ProviderObjectStore(client=client)

    head = store.head_object(
        ProviderS3ObjectEvent(
            bucket="provider-landing",
            key="delivery/object.csv",
            version_id="provider-version",
        )
    )

    assert head.manifest.schema_version == "v1"
    assert head.manifest.row_count == 3
    assert head.manifest.delivery_window_id == "window-2026-07-24"
    assert head.manifest.partition_index == 0
    assert head.manifest.partition_count == 4
    assert head.manifest.partition_complete is True
    assert head.checksum_sha256 == hashlib.sha256(
        b"provider-data"
    ).hexdigest()
    assert client.head_requests[0]["ChecksumMode"] == "ENABLED"


def test_s3_adapter_writes_encrypted_canonical_object():
    client = FakeS3Client()
    store = S3ProviderObjectStore(client=client)

    receipt = store.write_canonical(
        target=CanonicalObjectTarget(
            bucket="canonical-safe",
            prefix="canonical/",
        ),
        key="canonical/tenant/provider/output.jsonl",
        payload=b'{"safe":true}\n',
        content_type="application/x-ndjson",
        metadata={"source-fingerprint": "fingerprint"},
    )

    assert receipt["source_ref"].startswith("s3://canonical-safe/")
    assert receipt["version_id"] == "canonical-version"
    assert client.put_requests[0]["ServerSideEncryption"] == "AES256"
    assert "SSEKMSKeyId" not in client.put_requests[0]


def test_s3_adapter_requires_kms_key_reference_for_kms_output():
    store = S3ProviderObjectStore(client=FakeS3Client())

    with pytest.raises(ValueError, match="KMS key reference"):
        store.write_canonical(
            target=CanonicalObjectTarget(
                bucket="canonical-safe",
                server_side_encryption="aws:kms",
            ),
            key="canonical/output.jsonl",
            payload=b'{"safe":true}\n',
            content_type="application/x-ndjson",
        )
