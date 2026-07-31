from __future__ import annotations

import base64
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol

from app.models.provider_ingestion_contracts import (
    CanonicalObjectTarget,
    ProviderObjectHead,
    ProviderObjectDescriptor,
    ProviderObjectManifest,
)
from app.models.provider_queue_contracts import ProviderS3ObjectEvent


class ProviderObjectStore(Protocol):
    def head_object(
        self,
        event: ProviderS3ObjectEvent,
    ) -> ProviderObjectHead:
        """Read safe S3 metadata needed to resolve and validate a contract."""

    def read_object(
        self,
        descriptor: ProviderObjectDescriptor,
        *,
        max_bytes: int,
    ) -> bytes:
        """Read one immutable provider object version."""

    def write_canonical(
        self,
        *,
        target: CanonicalObjectTarget,
        key: str,
        payload: bytes,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Write privacy-safe canonical output and return a non-secret receipt."""


@dataclass(frozen=True)
class S3ClientConfig:
    region_name: Optional[str] = None
    endpoint_url: Optional[str] = None
    role_arn: Optional[str] = None
    role_session_name: str = "punk-audience-provider-ingestion"
    external_id: Optional[str] = field(default=None, repr=False)


class S3ProviderObjectStore:
    """
    S3 adapter with optional short-lived AssumeRole credentials.

    Tests inject a compatible client. Production can inject a tenant/provider
    scoped client factory so long-lived credentials never enter a contract,
    request, log, or ingestion record.
    """

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        canonical_client: Optional[Any] = None,
        client_config: Optional[S3ClientConfig] = None,
        canonical_client_config: Optional[S3ClientConfig] = None,
        client_factory: Optional[Callable[[ProviderObjectDescriptor], Any]] = None,
    ) -> None:
        self._client = client
        self._canonical_client = canonical_client
        self._client_config = client_config or S3ClientConfig()
        self._canonical_client_config = canonical_client_config
        self._client_factory = client_factory

    def read_object(
        self,
        descriptor: ProviderObjectDescriptor,
        *,
        max_bytes: int,
    ) -> bytes:
        client = (
            self._client_factory(descriptor)
            if self._client_factory is not None
            else self._client_or_create()
        )
        request: Dict[str, Any] = {
            "Bucket": descriptor.bucket,
            "Key": descriptor.key,
        }
        if descriptor.version_id:
            request["VersionId"] = descriptor.version_id

        response = client.get_object(**request)
        body = response.get("Body")
        if body is None:
            raise RuntimeError("The provider object response did not contain a body.")
        payload = body.read(max_bytes + 1)
        if not isinstance(payload, bytes):
            raise RuntimeError("The provider object body was not returned as bytes.")
        if len(payload) > max_bytes:
            raise RuntimeError(
                "The provider object exceeded the configured read limit."
            )
        return payload

    def head_object(
        self,
        event: ProviderS3ObjectEvent,
    ) -> ProviderObjectHead:
        client = self._canonical_client_or_create()
        request: Dict[str, Any] = {
            "Bucket": event.bucket,
            "Key": event.key,
            "ChecksumMode": "ENABLED",
        }
        if event.version_id:
            request["VersionId"] = event.version_id

        response = client.head_object(**request)
        metadata = {
            str(key).strip().lower(): str(value).strip()
            for key, value in (response.get("Metadata") or {}).items()
        }
        checksum = metadata.get("punk-checksum-sha256")
        checksum_type = str(response.get("ChecksumType") or "").upper()
        if (
            not checksum
            and response.get("ChecksumSHA256")
            and checksum_type in {"", "FULL_OBJECT"}
        ):
            checksum = self._base64_checksum_to_hex(
                str(response["ChecksumSHA256"])
            )

        row_count_value = metadata.get("punk-row-count")
        try:
            row_count = (
                int(row_count_value)
                if row_count_value not in {None, ""}
                else None
            )
        except ValueError as exc:
            raise ValueError(
                "Provider object metadata contains an invalid row count."
            ) from exc

        def _metadata_int(name: str) -> Optional[int]:
            value = metadata.get(name)
            if value in {None, ""}:
                return None
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError(
                    f"Provider object metadata contains invalid {name}."
                ) from exc

        partition_complete_value = metadata.get(
            "punk-partition-complete"
        )
        partition_complete = None
        if partition_complete_value not in {None, ""}:
            normalized_complete = partition_complete_value.lower()
            if normalized_complete not in {"true", "false"}:
                raise ValueError(
                    "Provider object metadata contains invalid "
                    "punk-partition-complete."
                )
            partition_complete = normalized_complete == "true"

        last_modified = response.get("LastModified")
        if isinstance(last_modified, datetime):
            last_modified = last_modified.isoformat()
        elif last_modified is not None:
            last_modified = str(last_modified)

        return ProviderObjectHead(
            size_bytes=int(response.get("ContentLength") or 0),
            content_type=str(
                response.get("ContentType") or "application/octet-stream"
            ),
            version_id=str(
                response.get("VersionId") or event.version_id or ""
            )
            or None,
            checksum_sha256=checksum,
            server_side_encryption=response.get("ServerSideEncryption"),
            last_modified=last_modified,
            manifest=ProviderObjectManifest(
                schema_version=metadata.get("punk-schema-version", ""),
                purpose=metadata.get("punk-purpose", ""),
                rights_policy_id=metadata.get("punk-rights-policy-id", ""),
                checksum_sha256=checksum,
                row_count=row_count,
                event_time_start=metadata.get("punk-event-time-start"),
                event_time_end=metadata.get("punk-event-time-end"),
                delivery_window_id=metadata.get(
                    "punk-delivery-window-id"
                ),
                partition_index=_metadata_int("punk-partition-index"),
                partition_count=_metadata_int("punk-partition-count"),
                partition_algorithm=metadata.get(
                    "punk-partition-algorithm"
                ),
                partition_complete=partition_complete,
                delivery_type=metadata.get(
                    "punk-delivery-type", "snapshot"
                ),
                supersedes_fingerprint=metadata.get(
                    "punk-supersedes-fingerprint"
                ),
            ),
        )

    def write_canonical(
        self,
        *,
        target: CanonicalObjectTarget,
        key: str,
        payload: bytes,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        client = self._client_or_create()
        request: Dict[str, Any] = {
            "Bucket": target.bucket,
            "Key": key,
            "Body": payload,
            "ContentType": content_type,
            "ServerSideEncryption": target.server_side_encryption,
            "Metadata": {
                str(name): str(value) for name, value in (metadata or {}).items()
            },
        }
        if target.server_side_encryption in {"aws:kms", "aws:kms:dsse"}:
            if not target.kms_key_id:
                raise ValueError(
                    "A KMS key reference is required for KMS-encrypted "
                    "canonical output."
                )
            request["SSEKMSKeyId"] = target.kms_key_id

        response = client.put_object(**request)
        return {
            "source_ref": f"s3://{target.bucket}/{key.lstrip('/')}",
            "version_id": response.get("VersionId"),
            "server_side_encryption": target.server_side_encryption,
        }

    def _client_or_create(self):
        if self._client is not None:
            return self._client
        self._client = self._create_client(self._client_config)
        return self._client

    def _canonical_client_or_create(self):
        if self._canonical_client is not None:
            return self._canonical_client
        if self._canonical_client_config is None:
            return self._client_or_create()
        self._canonical_client = self._create_client(
            self._canonical_client_config
        )
        return self._canonical_client

    def _create_client(self, config: S3ClientConfig):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required when an S3 client is not injected."
            ) from exc

        session = boto3.Session(region_name=config.region_name)
        client_kwargs: Dict[str, Any] = {}
        if config.region_name:
            client_kwargs["region_name"] = config.region_name
        if config.endpoint_url:
            client_kwargs["endpoint_url"] = config.endpoint_url

        if not config.role_arn:
            return session.client("s3", **client_kwargs)

        sts = session.client("sts", region_name=config.region_name)
        assume_request: Dict[str, Any] = {
            "RoleArn": config.role_arn,
            "RoleSessionName": config.role_session_name,
        }
        if config.external_id:
            assume_request["ExternalId"] = config.external_id
        response = sts.assume_role(**assume_request)
        credentials = response["Credentials"]

        return session.client(
            "s3",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            **client_kwargs,
        )

    def _base64_checksum_to_hex(self, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except ValueError as exc:
            raise ValueError(
                "Provider object metadata contains an invalid SHA-256 checksum."
            ) from exc
        if len(decoded) != 32:
            raise ValueError(
                "Provider object metadata contains an invalid SHA-256 checksum."
            )
        return decoded.hex()
