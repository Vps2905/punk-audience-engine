from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from app.services.production_canonical_feature_source_service import (
    ProductionCanonicalFeatureSourceService,
)
from tests.test_production_feature_embedding_service import _rows, _source


def _payload(rows=None):
    records = rows or _rows()
    return (
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in records
        )
        + "\n"
    ).encode("utf-8")


class FakeObjectStore:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def read_object(self, descriptor, *, max_bytes):
        self.calls.append((descriptor, max_bytes))
        return self.payload


def _manifest(payload):
    return replace(
        _source(
            source_ref=(
                "s3://canonical-bucket/tenant/provider/"
                "partition-00001.jsonl"
            ),
            source_version="immutable-object-version",
        ),
        source_fingerprint=hashlib.sha256(payload).hexdigest(),
        source_size_bytes=len(payload),
    )


def test_exact_versioned_canonical_s3_partition_is_verified_and_read():
    payload = _payload()
    store = FakeObjectStore(payload)
    rows = ProductionCanonicalFeatureSourceService(
        object_store=store
    ).read(_manifest(payload))

    assert rows == _rows()
    descriptor, max_bytes = store.calls[0]
    assert descriptor.bucket == "canonical-bucket"
    assert descriptor.key.endswith("partition-00001.jsonl")
    assert descriptor.version_id == "immutable-object-version"
    assert descriptor.checksum_sha256 == hashlib.sha256(
        payload
    ).hexdigest()
    assert max_bytes == 512 * 1024 * 1024


def test_source_reader_requires_exact_size_and_checksum():
    payload = _payload()
    service = ProductionCanonicalFeatureSourceService(
        object_store=FakeObjectStore(payload + b" ")
    )
    with pytest.raises(ValueError, match="byte length"):
        service.read(_manifest(payload))

    wrong_checksum = replace(
        _manifest(payload),
        source_fingerprint="f" * 64,
    )
    service = ProductionCanonicalFeatureSourceService(
        object_store=FakeObjectStore(payload)
    )
    with pytest.raises(ValueError, match="checksum"):
        service.read(wrong_checksum)


def test_source_reader_requires_declared_object_size():
    service = ProductionCanonicalFeatureSourceService(
        object_store=FakeObjectStore(_payload())
    )

    with pytest.raises(ValueError, match="source_size_bytes"):
        service.read(_source())


def test_source_reader_rejects_row_count_mismatch():
    payload = _payload(_rows()[:1])
    manifest = replace(
        _manifest(payload),
        expected_row_count=2,
    )

    with pytest.raises(ValueError, match="row count"):
        ProductionCanonicalFeatureSourceService(
            object_store=FakeObjectStore(payload)
        ).read(manifest)


def test_source_manifest_rejects_unbounded_or_unsupported_input():
    with pytest.raises(ValueError, match="JSONL"):
        _source(data_format="parquet")
    with pytest.raises(ValueError, match="bounded"):
        _source(
            source_size_bytes=200,
            max_object_bytes=100,
        )
