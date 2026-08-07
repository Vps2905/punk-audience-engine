from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from app.models.production_feature_build_contracts import (
    CanonicalFeatureSourceManifest,
)
from app.models.provider_ingestion_contracts import ProviderObjectDescriptor
from app.services.provider_object_store_service import ProviderObjectStore
from app.services.provider_payload_decoder_service import (
    ProviderPayloadDecoderService,
)


class ProductionCanonicalFeatureSourceService:
    """
    Read one exact privacy-safe canonical S3 object partition.

    Distributed execution invokes this boundary per bounded object partition.
    It validates version, byte length, SHA-256 and row count before feature
    generation. Raw provider objects are not accepted by this service.
    """

    def __init__(
        self,
        *,
        object_store: ProviderObjectStore,
        decoder: ProviderPayloadDecoderService | None = None,
    ) -> None:
        self._object_store = object_store
        self._decoder = decoder or ProviderPayloadDecoderService()

    def read(
        self,
        manifest: CanonicalFeatureSourceManifest,
    ) -> Sequence[dict[str, Any]]:
        if manifest.source_size_bytes is None:
            raise ValueError(
                "Exact canonical source_size_bytes is required for S3 reads."
            )
        bucket, key = self._parse_s3_ref(manifest.source_ref)
        descriptor = ProviderObjectDescriptor(
            tenant_id=manifest.tenant_id,
            provider_id=manifest.provider_id,
            dataset_id=manifest.dataset_id,
            bucket=bucket,
            key=key,
            size_bytes=manifest.source_size_bytes,
            content_type="application/x-ndjson",
            version_id=(
                manifest.source_version
                if manifest.source_version_kind == "s3_version_id"
                else None
            ),
            checksum_sha256=manifest.source_fingerprint,
        )
        payload = self._object_store.read_object(
            descriptor,
            max_bytes=int(manifest.max_object_bytes),
        )
        if len(payload) != manifest.source_size_bytes:
            raise ValueError(
                "Canonical source byte length does not match its manifest."
            )
        actual_checksum = hashlib.sha256(payload).hexdigest()
        if actual_checksum != manifest.source_fingerprint:
            raise ValueError(
                "Canonical source failed SHA-256 checksum verification."
            )
        rows = self._decoder.decode(
            payload,
            data_format=manifest.data_format,
            max_rows=manifest.expected_row_count,
        )
        if len(rows) != manifest.expected_row_count:
            raise ValueError(
                "Canonical source row count does not match its manifest."
            )
        return rows

    def _parse_s3_ref(self, source_ref: str) -> tuple[str, str]:
        parsed = urlsplit(source_ref)
        bucket = parsed.netloc.strip()
        key = parsed.path.lstrip("/").strip()
        if (
            parsed.scheme != "s3"
            or not bucket
            or not key
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Canonical source_ref must identify one exact S3 object."
            )
        return bucket, key
