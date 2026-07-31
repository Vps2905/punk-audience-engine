from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Sequence


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[A-Fa-f0-9]{64}$")
_SUPPORTED_S3_ENCRYPTION = {"AES256", "aws:kms", "aws:kms:dsse"}


def _normalized_identifier(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must contain only letters, numbers, '.', '_' or '-'."
        )
    return normalized


def _normalized_prefix(value: str) -> str:
    normalized = str(value or "").strip().lstrip("/")
    if normalized and not normalized.endswith("/"):
        normalized += "/"
    if ".." in normalized.split("/"):
        raise ValueError("S3 prefixes must not contain parent-directory segments.")
    return normalized


def _normalized_column(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or len(normalized) > 128 or "\x00" in normalized:
        raise ValueError(f"{field_name} contains an invalid column name.")
    return normalized


def _normalized_sha256(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(
            "SHA-256 checksum must contain exactly 64 hexadecimal characters."
        )
    return normalized


@dataclass(frozen=True)
class ProviderDatasetContract:
    """
    Versioned, provider-configurable input contract.

    The contract contains no credentials. Connection secrets are supplied by
    the deployment through a secret manager or injected object-store client.
    """

    tenant_id: str
    provider_id: str
    dataset_id: str
    schema_version: str
    data_format: str
    connection_type: str = "s3"
    allowed_bucket: Optional[str] = None
    allowed_prefix: str = ""
    allowed_content_types: Sequence[str] = ()
    entity_id_column: str = "entity_id"
    timestamp_column: str = "created_at"
    cohort_columns: Sequence[str] = (
        "location_name",
        "primary_poi_type",
        "created_day_part",
    )
    min_cohort_size: int = 1000
    epsilon: float = 1.0
    max_cumulative_epsilon: float = 5.0
    delta: float = 1e-5
    sensitivity: float = 1.0
    mechanism: str = "gaussian"
    max_object_bytes: int = 512 * 1024 * 1024
    max_rows_per_object: int = 5_000_000
    execution_mode: str = "auto"
    max_in_process_object_bytes: Optional[int] = None
    max_in_process_rows: Optional[int] = None
    require_checksum: bool = True
    require_encryption: bool = True
    rights_policy_id: str = "audience_intelligence_default"
    allowed_purposes: Sequence[str] = ("audience_intelligence",)
    expected_delivery_interval_minutes: int = 1440
    freshness_sla_minutes: int = 2880
    privacy_window_minutes: int = 1440
    distributed_partition_strategy: str = "single_object"
    require_complete_privacy_partitions: bool = False
    raw_retention_days: int = 30

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _normalized_identifier(self.tenant_id, "tenant_id"),
        )
        object.__setattr__(
            self,
            "provider_id",
            _normalized_identifier(self.provider_id, "provider_id"),
        )
        object.__setattr__(
            self,
            "dataset_id",
            _normalized_identifier(self.dataset_id, "dataset_id"),
        )
        object.__setattr__(
            self,
            "schema_version",
            _normalized_identifier(self.schema_version, "schema_version"),
        )
        object.__setattr__(
            self,
            "allowed_prefix",
            _normalized_prefix(self.allowed_prefix),
        )

        data_format = str(self.data_format or "").strip().lower()
        if data_format not in {"csv", "jsonl", "parquet"}:
            raise ValueError("data_format must be one of: csv, jsonl, parquet")
        object.__setattr__(self, "data_format", data_format)

        content_types = tuple(
            str(value or "").split(";", 1)[0].strip().lower()
            for value in self.allowed_content_types
            if str(value or "").strip()
        )
        if not content_types:
            if data_format == "csv":
                content_types = (
                    "text/csv",
                    "application/csv",
                    "application/octet-stream",
                )
            elif data_format == "jsonl":
                content_types = (
                    "application/x-ndjson",
                    "application/jsonl",
                    "application/octet-stream",
                )
            else:
                content_types = (
                    "application/vnd.apache.parquet",
                    "application/x-parquet",
                    "application/octet-stream",
                )
        object.__setattr__(
            self,
            "allowed_content_types",
            tuple(dict.fromkeys(content_types)),
        )

        connection_type = str(self.connection_type or "").strip().lower()
        if connection_type != "s3":
            raise ValueError(
                "The provider object gateway currently supports connection_type=s3."
            )
        object.__setattr__(self, "connection_type", connection_type)

        allowed_bucket = str(self.allowed_bucket or "").strip()
        if not allowed_bucket:
            raise ValueError("allowed_bucket is required for an S3 provider contract.")
        object.__setattr__(self, "allowed_bucket", allowed_bucket)
        object.__setattr__(
            self,
            "entity_id_column",
            _normalized_column(self.entity_id_column, "entity_id_column"),
        )
        object.__setattr__(
            self,
            "timestamp_column",
            _normalized_column(self.timestamp_column, "timestamp_column"),
        )
        cohort_columns = tuple(
            _normalized_column(value, "cohort_columns") for value in self.cohort_columns
        )
        if len(set(cohort_columns)) != len(cohort_columns):
            raise ValueError("cohort_columns must not contain duplicates")
        if self.entity_id_column in cohort_columns:
            raise ValueError("entity_id_column must not be a cohort column")
        object.__setattr__(self, "cohort_columns", cohort_columns)

        purposes = tuple(
            str(value or "").strip()
            for value in self.allowed_purposes
            if str(value or "").strip()
        )
        object.__setattr__(self, "allowed_purposes", tuple(dict.fromkeys(purposes)))
        object.__setattr__(
            self,
            "rights_policy_id",
            _normalized_identifier(self.rights_policy_id, "rights_policy_id"),
        )
        if self.min_cohort_size < 1:
            raise ValueError("min_cohort_size must be >= 1")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if self.max_cumulative_epsilon <= 0:
            raise ValueError("max_cumulative_epsilon must be > 0")
        if self.epsilon > self.max_cumulative_epsilon:
            raise ValueError(
                "epsilon must not exceed max_cumulative_epsilon"
            )
        if not 0 < self.delta < 1:
            raise ValueError("delta must be between 0 and 1")
        if self.sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")
        if str(self.mechanism or "").strip().lower() != "gaussian":
            raise ValueError("mechanism must be gaussian")
        object.__setattr__(self, "mechanism", "gaussian")
        if self.max_object_bytes < 1:
            raise ValueError("max_object_bytes must be >= 1")
        if self.max_rows_per_object < 1:
            raise ValueError("max_rows_per_object must be >= 1")
        execution_mode = str(self.execution_mode or "").strip().lower()
        if execution_mode not in {"auto", "in_process", "distributed"}:
            raise ValueError(
                "execution_mode must be one of: auto, in_process, distributed"
            )
        object.__setattr__(self, "execution_mode", execution_mode)
        max_in_process_object_bytes = (
            min(self.max_object_bytes, 16 * 1024 * 1024)
            if self.max_in_process_object_bytes is None
            else int(self.max_in_process_object_bytes)
        )
        if max_in_process_object_bytes < 1:
            raise ValueError("max_in_process_object_bytes must be >= 1")
        if max_in_process_object_bytes > self.max_object_bytes:
            raise ValueError(
                "max_in_process_object_bytes must not exceed max_object_bytes"
            )
        object.__setattr__(
            self,
            "max_in_process_object_bytes",
            max_in_process_object_bytes,
        )
        max_in_process_rows = (
            min(self.max_rows_per_object, 100_000)
            if self.max_in_process_rows is None
            else int(self.max_in_process_rows)
        )
        if max_in_process_rows < 1:
            raise ValueError("max_in_process_rows must be >= 1")
        if max_in_process_rows > self.max_rows_per_object:
            raise ValueError(
                "max_in_process_rows must not exceed max_rows_per_object"
            )
        object.__setattr__(
            self,
            "max_in_process_rows",
            max_in_process_rows,
        )
        if self.expected_delivery_interval_minutes < 1:
            raise ValueError("expected_delivery_interval_minutes must be >= 1")
        if self.freshness_sla_minutes < self.expected_delivery_interval_minutes:
            raise ValueError(
                "freshness_sla_minutes must be >= expected_delivery_interval_minutes"
            )
        if int(self.privacy_window_minutes) < 1:
            raise ValueError("privacy_window_minutes must be >= 1")
        object.__setattr__(
            self,
            "privacy_window_minutes",
            int(self.privacy_window_minutes),
        )
        partition_strategy = str(
            self.distributed_partition_strategy or ""
        ).strip().lower()
        if partition_strategy not in {
            "single_object",
            "entity_hash_v1",
        }:
            raise ValueError(
                "distributed_partition_strategy must be single_object or "
                "entity_hash_v1"
            )
        object.__setattr__(
            self,
            "distributed_partition_strategy",
            partition_strategy,
        )
        if (
            self.require_complete_privacy_partitions
            and partition_strategy != "entity_hash_v1"
        ):
            raise ValueError(
                "require_complete_privacy_partitions requires "
                "distributed_partition_strategy=entity_hash_v1"
            )
        if int(self.raw_retention_days) < 1:
            raise ValueError("raw_retention_days must be >= 1")
        object.__setattr__(
            self,
            "raw_retention_days",
            int(self.raw_retention_days),
        )
        if not self.cohort_columns:
            raise ValueError("cohort_columns is required")
        if not self.allowed_purposes:
            raise ValueError("allowed_purposes is required")

    @property
    def contract_key(self) -> str:
        return ":".join(
            (
                self.tenant_id,
                self.provider_id,
                self.dataset_id,
                self.schema_version,
            )
        )

    def to_safe_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["allowed_content_types"] = list(self.allowed_content_types)
        payload["cohort_columns"] = list(self.cohort_columns)
        payload["allowed_purposes"] = list(self.allowed_purposes)
        return payload


@dataclass(frozen=True)
class ProviderObjectDescriptor:
    tenant_id: str
    provider_id: str
    dataset_id: str
    bucket: str
    key: str
    size_bytes: int
    content_type: str
    version_id: Optional[str] = None
    checksum_sha256: Optional[str] = None
    server_side_encryption: Optional[str] = None
    event_id: Optional[str] = None
    last_modified: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _normalized_identifier(self.tenant_id, "tenant_id"),
        )
        object.__setattr__(
            self,
            "provider_id",
            _normalized_identifier(self.provider_id, "provider_id"),
        )
        object.__setattr__(
            self,
            "dataset_id",
            _normalized_identifier(self.dataset_id, "dataset_id"),
        )
        object.__setattr__(
            self,
            "checksum_sha256",
            _normalized_sha256(self.checksum_sha256),
        )
        bucket = str(self.bucket or "").strip()
        key = str(self.key or "").strip().lstrip("/")
        if not bucket:
            raise ValueError("bucket is required")
        if not key:
            raise ValueError("key is required")
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "key", key)
        if int(self.size_bytes) < 0:
            raise ValueError("size_bytes must be >= 0")

    @property
    def source_ref(self) -> str:
        return f"s3://{self.bucket}/{self.key.lstrip('/')}"

    @property
    def fingerprint(self) -> str:
        """
        Stable object identity used before any downstream side effects.

        S3 ETags are intentionally not used as universal content checksums.
        """
        stable_parts = (
            self.tenant_id,
            self.provider_id,
            self.dataset_id,
            self.bucket,
            self.key,
            self.version_id or "",
            self.checksum_sha256 or "",
        )
        return hashlib.sha256("\x1f".join(stable_parts).encode("utf-8")).hexdigest()

    def to_safe_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderObjectManifest:
    schema_version: str
    purpose: str
    rights_policy_id: str
    checksum_sha256: Optional[str] = None
    row_count: Optional[int] = None
    event_time_start: Optional[str] = None
    event_time_end: Optional[str] = None
    delivery_window_id: Optional[str] = None
    partition_index: Optional[int] = None
    partition_count: Optional[int] = None
    partition_algorithm: Optional[str] = None
    partition_complete: Optional[bool] = None
    delivery_type: str = "snapshot"
    supersedes_fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _normalized_identifier(self.schema_version, "schema_version"),
        )
        object.__setattr__(
            self,
            "checksum_sha256",
            _normalized_sha256(self.checksum_sha256),
        )
        if not str(self.purpose or "").strip():
            raise ValueError("purpose is required")
        if not str(self.rights_policy_id or "").strip():
            raise ValueError("rights_policy_id is required")
        object.__setattr__(self, "purpose", str(self.purpose).strip())
        object.__setattr__(
            self,
            "rights_policy_id",
            _normalized_identifier(self.rights_policy_id, "rights_policy_id"),
        )
        if self.row_count is not None and int(self.row_count) < 0:
            raise ValueError("row_count must be >= 0")
        start = self._parse_timestamp(self.event_time_start, "event_time_start")
        end = self._parse_timestamp(self.event_time_end, "event_time_end")
        if start and end and start > end:
            raise ValueError("event_time_start must not be after event_time_end")
        delivery_type = str(self.delivery_type or "").strip().lower()
        if delivery_type not in {"snapshot", "correction"}:
            raise ValueError("delivery_type must be snapshot or correction")
        object.__setattr__(self, "delivery_type", delivery_type)
        object.__setattr__(
            self,
            "supersedes_fingerprint",
            _normalized_sha256(self.supersedes_fingerprint),
        )
        if delivery_type == "correction" and not self.supersedes_fingerprint:
            raise ValueError(
                "A correction manifest requires supersedes_fingerprint"
            )
        if delivery_type == "snapshot" and self.supersedes_fingerprint:
            raise ValueError(
                "A snapshot manifest must not supersede another fingerprint"
            )

        partition_values = (
            self.delivery_window_id,
            self.partition_index,
            self.partition_count,
            self.partition_algorithm,
            self.partition_complete,
        )
        if any(value is not None for value in partition_values):
            if any(value is None for value in partition_values):
                raise ValueError(
                    "A privacy partition requires window, index, count, "
                    "algorithm, and completion fields"
                )
            window_id = _normalized_identifier(
                str(self.delivery_window_id),
                "delivery_window_id",
            )
            partition_index = int(self.partition_index)
            partition_count = int(self.partition_count)
            if not 1 <= partition_count <= 100_000:
                raise ValueError(
                    "partition_count must be between 1 and 100000"
                )
            if not 0 <= partition_index < partition_count:
                raise ValueError(
                    "partition_index must be within partition_count"
                )
            algorithm = str(self.partition_algorithm).strip().lower()
            if algorithm != "spark_xxhash64_v1":
                raise ValueError(
                    "partition_algorithm must be spark_xxhash64_v1"
                )
            if self.partition_complete is not True:
                raise ValueError(
                    "partition_complete must be true for release processing"
                )
            if start is None or end is None:
                raise ValueError(
                    "Privacy partitions require event-time boundaries"
                )
            object.__setattr__(self, "delivery_window_id", window_id)
            object.__setattr__(self, "partition_index", partition_index)
            object.__setattr__(self, "partition_count", partition_count)
            object.__setattr__(self, "partition_algorithm", algorithm)

    def to_safe_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def _parse_timestamp(
        self,
        value: Optional[str],
        field_name: str,
    ) -> Optional[datetime]:
        if value is None:
            return None
        normalized = str(value).strip()
        try:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError(f"{field_name} must include a timezone")
        return parsed


@dataclass(frozen=True)
class ProviderObjectHead:
    size_bytes: int
    content_type: str
    version_id: Optional[str]
    checksum_sha256: Optional[str]
    server_side_encryption: Optional[str]
    last_modified: Optional[str]
    manifest: ProviderObjectManifest

    def __post_init__(self) -> None:
        if int(self.size_bytes) < 0:
            raise ValueError("size_bytes must be >= 0")
        object.__setattr__(
            self,
            "checksum_sha256",
            _normalized_sha256(self.checksum_sha256),
        )


@dataclass(frozen=True)
class CanonicalObjectTarget:
    bucket: str
    prefix: str = "canonical"
    server_side_encryption: str = "AES256"
    kms_key_id: Optional[str] = None

    def __post_init__(self) -> None:
        bucket = str(self.bucket or "").strip()
        if not bucket:
            raise ValueError("Canonical target bucket is required.")
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "prefix", _normalized_prefix(self.prefix))
        if self.server_side_encryption not in _SUPPORTED_S3_ENCRYPTION:
            raise ValueError(
                "Canonical output requires an approved server-side encryption mode."
            )
        if (
            self.server_side_encryption in {"aws:kms", "aws:kms:dsse"}
            and not self.kms_key_id
        ):
            raise ValueError(
                "A KMS key reference is required for KMS-encrypted canonical output."
            )


class ProviderObjectValidationError(ValueError):
    def __init__(self, reason_code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.reason_code = reason_code
        self.safe_message = safe_message


class ProviderContractValidator:
    def validate(
        self,
        *,
        contract: ProviderDatasetContract,
        descriptor: ProviderObjectDescriptor,
        manifest: ProviderObjectManifest,
    ) -> None:
        if (
            descriptor.tenant_id != contract.tenant_id
            or descriptor.provider_id != contract.provider_id
            or descriptor.dataset_id != contract.dataset_id
        ):
            raise ProviderObjectValidationError(
                "provider_identity_mismatch",
                "The provider object identity does not match the registered "
                "dataset contract.",
            )

        if descriptor.bucket != contract.allowed_bucket:
            raise ProviderObjectValidationError(
                "bucket_not_allowed",
                "The provider object bucket is not allowed by the dataset contract.",
            )

        normalized_key = descriptor.key.lstrip("/")
        if contract.allowed_prefix and not normalized_key.startswith(
            contract.allowed_prefix
        ):
            raise ProviderObjectValidationError(
                "key_prefix_not_allowed",
                "The provider object key is outside the allowed dataset prefix.",
            )

        if descriptor.size_bytes <= 0:
            raise ProviderObjectValidationError(
                "object_empty",
                "The provider object is empty.",
            )
        if descriptor.size_bytes > contract.max_object_bytes:
            raise ProviderObjectValidationError(
                "object_too_large",
                "The provider object exceeds the configured maximum size.",
            )

        content_type = (
            str(descriptor.content_type or "").split(";", 1)[0].strip().lower()
        )
        allowed_content_types = {
            str(value).split(";", 1)[0].strip().lower()
            for value in contract.allowed_content_types
        }
        if content_type not in allowed_content_types:
            raise ProviderObjectValidationError(
                "unsupported_content_type",
                "The provider object content type is not allowed by the "
                "dataset contract.",
            )

        if (
            contract.require_encryption
            and descriptor.server_side_encryption not in _SUPPORTED_S3_ENCRYPTION
        ):
            raise ProviderObjectValidationError(
                "encryption_required",
                "The provider object is not protected by an approved "
                "server-side encryption mode.",
            )

        if manifest.schema_version != contract.schema_version:
            raise ProviderObjectValidationError(
                "schema_version_mismatch",
                "The provider manifest schema version does not match the "
                "registered contract.",
            )

        if manifest.rights_policy_id != contract.rights_policy_id:
            raise ProviderObjectValidationError(
                "rights_policy_mismatch",
                "The provider manifest rights policy does not match the "
                "registered contract.",
            )

        if manifest.purpose not in set(contract.allowed_purposes):
            raise ProviderObjectValidationError(
                "purpose_not_allowed",
                "The provider manifest purpose is not allowed by the "
                "registered contract.",
            )

        if (
            contract.distributed_partition_strategy == "entity_hash_v1"
            or contract.require_complete_privacy_partitions
        ):
            if manifest.delivery_window_id is None:
                raise ProviderObjectValidationError(
                    "privacy_partition_manifest_required",
                    "The distributed dataset requires a complete privacy "
                    "partition manifest.",
                )
            if manifest.partition_algorithm != "spark_xxhash64_v1":
                raise ProviderObjectValidationError(
                    "privacy_partition_algorithm_mismatch",
                    "The privacy partition algorithm does not match the "
                    "registered dataset contract.",
                )
            start = manifest._parse_timestamp(
                manifest.event_time_start,
                "event_time_start",
            )
            end = manifest._parse_timestamp(
                manifest.event_time_end,
                "event_time_end",
            )
            if start is None or end is None:
                raise ProviderObjectValidationError(
                    "privacy_window_boundaries_required",
                    "The distributed dataset requires event-time boundaries.",
                )
            duration_minutes = (end - start).total_seconds() / 60.0
            if duration_minutes > contract.privacy_window_minutes:
                raise ProviderObjectValidationError(
                    "privacy_window_exceeds_contract",
                    "The provider privacy window exceeds the registered "
                    "maximum duration.",
                )

        descriptor_checksum = descriptor.checksum_sha256
        manifest_checksum = manifest.checksum_sha256
        if (
            descriptor_checksum
            and manifest_checksum
            and descriptor_checksum != manifest_checksum
        ):
            raise ProviderObjectValidationError(
                "checksum_declaration_mismatch",
                "The object descriptor and manifest declare different "
                "SHA-256 checksums.",
            )
        if contract.require_checksum and not (descriptor_checksum or manifest_checksum):
            raise ProviderObjectValidationError(
                "checksum_required",
                "A SHA-256 checksum is required by the registered dataset contract.",
            )

        if not descriptor.version_id and not (descriptor_checksum or manifest_checksum):
            raise ProviderObjectValidationError(
                "stable_object_identity_required",
                "A provider object version or SHA-256 checksum is required "
                "for idempotency.",
            )
