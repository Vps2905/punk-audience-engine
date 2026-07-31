from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from app.models.audience_feature_contracts import (
    SAFE_PRIVACY_STATUSES,
    normalize_taxonomy_value,
    parse_utc_datetime,
    stable_digest,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_BLOCKED_COLUMN_PARTS = {
    "maid",
    "device_id",
    "email",
    "phone",
    "latitude",
    "longitude",
    "raw_observation",
}
_REQUIRED_PRIVACY_CONTROLS = {
    "daily_contribution_bounding",
    "k_anonymity",
    "no_identifier_output",
}


def _required_slug(value: Any, label: str) -> str:
    normalized = normalize_taxonomy_value(value)
    if not normalized:
        raise ValueError(f"{label} is required.")
    return normalized


def _column_name(value: Any, label: str) -> str:
    normalized = str(value or "").strip()
    if not _COLUMN_RE.fullmatch(normalized):
        raise ValueError(f"{label} must be a safe canonical column name.")
    lowered = normalized.lower()
    if any(token in lowered for token in _BLOCKED_COLUMN_PARTS):
        raise ValueError(f"{label} cannot reference an identifier column.")
    return normalized


@dataclass(frozen=True)
class EmbeddingModelSpec:
    backend: str
    model_name: str
    model_revision: str
    dimension: int = 384
    normalize_embeddings: bool = True
    document_prefix: str = ""
    query_prefix: str = ""

    def __post_init__(self) -> None:
        backend = normalize_taxonomy_value(self.backend)
        if backend not in {
            "sentence_transformers",
            "external_embedding_service",
        }:
            raise ValueError(
                "Production embeddings require sentence_transformers or "
                "external_embedding_service."
            )
        model_name = str(self.model_name or "").strip()
        revision = str(self.model_revision or "").strip()
        if not model_name:
            raise ValueError("model_name is required.")
        if not revision or revision.lower() in {"latest", "main", "master"}:
            raise ValueError(
                "An immutable embedding model revision is required."
            )
        if int(self.dimension) != 384:
            raise ValueError(
                "The current pgvector schema requires 384-dimensional embeddings."
            )
        for label, prefix in {
            "document_prefix": self.document_prefix,
            "query_prefix": self.query_prefix,
        }.items():
            if len(str(prefix)) > 200:
                raise ValueError(f"{label} is too long.")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "model_revision", revision)

    @property
    def fingerprint(self) -> str:
        return stable_digest(asdict(self))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "model_fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class CanonicalFeatureSourceManifest:
    tenant_id: str
    provider_id: str
    dataset_id: str
    schema_version: str
    source_ref: str
    source_version: str
    source_fingerprint: str
    source_latest_at: datetime
    expected_row_count: int
    data_use_mode: str
    privacy_status: str
    privacy_policy_version: str
    privacy_controls: tuple[str, ...]
    rights_status: str
    rights_policy_id: str
    purpose: str
    location_column: str
    category_column: str
    cohort_size_column: str = "dp_noisy_count"
    daypart_column: str | None = None
    lookback_bucket_column: str | None = None
    privacy_window_column: str | None = "privacy_window"
    contract_version: str = "canonical-feature-source-v1"
    data_format: str = "jsonl"
    source_size_bytes: int | None = None
    max_object_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        for label in (
            "tenant_id",
            "provider_id",
            "dataset_id",
            "schema_version",
            "privacy_policy_version",
            "rights_policy_id",
            "purpose",
        ):
            object.__setattr__(
                self,
                label,
                _required_slug(getattr(self, label), label),
            )
        source_ref = str(self.source_ref or "").strip()
        if not source_ref.startswith("s3://"):
            raise ValueError(
                "Production canonical feature input must be an s3:// reference."
            )
        object.__setattr__(self, "source_ref", source_ref)
        source_version = str(self.source_version or "").strip()
        if not source_version:
            raise ValueError("source_version is required.")
        object.__setattr__(self, "source_version", source_version)
        fingerprint = str(self.source_fingerprint or "").strip().lower()
        if not _SHA256_RE.fullmatch(fingerprint):
            raise ValueError(
                "source_fingerprint must be a full lowercase SHA-256 digest."
            )
        object.__setattr__(self, "source_fingerprint", fingerprint)
        data_format = normalize_taxonomy_value(self.data_format)
        if data_format != "jsonl":
            raise ValueError(
                "Production canonical feature input currently requires JSONL."
            )
        object.__setattr__(self, "data_format", data_format)
        if self.source_size_bytes is not None:
            source_size_bytes = int(self.source_size_bytes)
            if source_size_bytes < 1:
                raise ValueError("source_size_bytes must be at least 1.")
            object.__setattr__(
                self,
                "source_size_bytes",
                source_size_bytes,
            )
        if not 1 <= int(self.max_object_bytes) <= 5 * 1024 * 1024 * 1024:
            raise ValueError(
                "max_object_bytes must be between 1 and 5 GiB."
            )
        if (
            self.source_size_bytes is not None
            and self.source_size_bytes > int(self.max_object_bytes)
        ):
            raise ValueError(
                "source_size_bytes exceeds the bounded object limit."
            )
        source_latest_at = parse_utc_datetime(self.source_latest_at)
        if source_latest_at is None:
            raise ValueError("source_latest_at must be a valid UTC timestamp.")
        object.__setattr__(self, "source_latest_at", source_latest_at)
        if int(self.expected_row_count) < 1:
            raise ValueError("expected_row_count must be at least 1.")

        mode = normalize_taxonomy_value(self.data_use_mode)
        if mode not in {
            "historical_preview",
            "offline_evaluation",
            "production",
        }:
            raise ValueError("Unsupported canonical feature data_use_mode.")
        object.__setattr__(self, "data_use_mode", mode)

        privacy_status = normalize_taxonomy_value(self.privacy_status)
        if privacy_status not in SAFE_PRIVACY_STATUSES:
            raise ValueError("Canonical feature input is not privacy-safe.")
        object.__setattr__(self, "privacy_status", privacy_status)
        controls = tuple(
            dict.fromkeys(
                _required_slug(value, "privacy_control")
                for value in self.privacy_controls
            )
        )
        missing_controls = _REQUIRED_PRIVACY_CONTROLS.difference(controls)
        if missing_controls:
            raise ValueError(
                "Canonical feature input is missing required privacy controls: "
                + ", ".join(sorted(missing_controls))
            )
        object.__setattr__(self, "privacy_controls", controls)

        rights_status = normalize_taxonomy_value(self.rights_status)
        if rights_status not in {
            "permitted",
            "historical_internal_only",
            "offline_evaluation_only",
        }:
            raise ValueError("Unsupported rights_status.")
        if mode == "production" and rights_status != "permitted":
            raise ValueError(
                "Production feature input requires permitted data rights."
            )
        object.__setattr__(self, "rights_status", rights_status)

        mapped_columns = {
            "location_column": _column_name(
                self.location_column,
                "location_column",
            ),
            "category_column": _column_name(
                self.category_column,
                "category_column",
            ),
            "cohort_size_column": _column_name(
                self.cohort_size_column,
                "cohort_size_column",
            ),
        }
        for label in (
            "daypart_column",
            "lookback_bucket_column",
            "privacy_window_column",
        ):
            value = getattr(self, label)
            mapped_columns[label] = (
                _column_name(value, label) if value else None
            )
        nonempty = [value for value in mapped_columns.values() if value]
        if len(nonempty) != len(set(nonempty)):
            raise ValueError("Canonical feature column mappings must be unique.")
        for label, value in mapped_columns.items():
            object.__setattr__(self, label, value)

    def to_safe_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["source_latest_at"] = self.source_latest_at.isoformat()
        record["privacy_controls"] = list(self.privacy_controls)
        return record


@dataclass(frozen=True)
class ProductionFeatureBuildRequest:
    source: CanonicalFeatureSourceManifest
    model: EmbeddingModelSpec
    batch_size: int = 256
    max_features: int = 1_000_000
    requested_by: str = "feature_build_worker"
    activation_requested: bool = False
    metadata_fields: tuple[str, ...] = field(default_factory=tuple)
    contract_version: str = "production-feature-build-v1"

    def __post_init__(self) -> None:
        if not 1 <= int(self.batch_size) <= 4096:
            raise ValueError("batch_size must be between 1 and 4096.")
        if not 1 <= int(self.max_features) <= 10_000_000:
            raise ValueError("max_features must be between 1 and 10000000.")
        if self.activation_requested:
            raise ValueError(
                "Module 2 feature builds cannot request audience activation."
            )
        requested_by = _required_slug(self.requested_by, "requested_by")
        object.__setattr__(self, "requested_by", requested_by)
        safe_metadata_fields = tuple(
            dict.fromkeys(
                _column_name(value, "metadata_field")
                for value in self.metadata_fields
            )
        )
        object.__setattr__(self, "metadata_fields", safe_metadata_fields)

    @property
    def request_fingerprint(self) -> str:
        return stable_digest(
            {
                "contract_version": self.contract_version,
                "source": self.source.to_safe_dict(),
                "model": self.model.to_safe_dict(),
                "metadata_fields": list(self.metadata_fields),
            }
        )

    @property
    def build_id(self) -> str:
        return "feature_build_" + self.request_fingerprint[:32]

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "build_id": self.build_id,
            "request_fingerprint": self.request_fingerprint,
            "source": self.source.to_safe_dict(),
            "model": self.model.to_safe_dict(),
            "batch_size": int(self.batch_size),
            "max_features": int(self.max_features),
            "requested_by": self.requested_by,
            "activation_requested": False,
            "metadata_fields": list(self.metadata_fields),
        }
