from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


def _identifier(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not _ID_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} is invalid")
    return normalized


def _sha256(value: str, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return normalized


def _utc(value: str, field_name: str) -> str:
    try:
        parsed = datetime.fromisoformat(
            str(value or "").strip().replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class ProviderPrivacyPartitionRegistration:
    tenant_id: str
    provider_id: str
    dataset_id: str
    schema_version: str
    delivery_window_id: str
    event_time_start: str
    event_time_end: str
    partition_index: int
    partition_count: int
    partition_algorithm: str
    ingestion_id: str
    fingerprint: str
    row_count: Optional[int] = None
    delivery_type: str = "snapshot"
    supersedes_fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in (
            "tenant_id",
            "provider_id",
            "dataset_id",
            "schema_version",
            "delivery_window_id",
            "ingestion_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "fingerprint",
            _sha256(self.fingerprint, "fingerprint"),
        )
        if self.supersedes_fingerprint is not None:
            object.__setattr__(
                self,
                "supersedes_fingerprint",
                _sha256(
                    self.supersedes_fingerprint,
                    "supersedes_fingerprint",
                ),
            )
        start = _utc(self.event_time_start, "event_time_start")
        end = _utc(self.event_time_end, "event_time_end")
        if datetime.fromisoformat(start) >= datetime.fromisoformat(end):
            raise ValueError("event_time_start must be before event_time_end")
        object.__setattr__(self, "event_time_start", start)
        object.__setattr__(self, "event_time_end", end)
        count = int(self.partition_count)
        index = int(self.partition_index)
        if not 1 <= count <= 100_000:
            raise ValueError("partition_count must be between 1 and 100000")
        if not 0 <= index < count:
            raise ValueError("partition_index is outside partition_count")
        object.__setattr__(self, "partition_count", count)
        object.__setattr__(self, "partition_index", index)
        if str(self.partition_algorithm).strip().lower() != "spark_xxhash64_v1":
            raise ValueError(
                "partition_algorithm must be spark_xxhash64_v1"
            )
        object.__setattr__(
            self,
            "partition_algorithm",
            "spark_xxhash64_v1",
        )
        delivery_type = str(self.delivery_type or "").strip().lower()
        if delivery_type not in {"snapshot", "correction"}:
            raise ValueError("delivery_type must be snapshot or correction")
        if delivery_type == "correction" and not self.supersedes_fingerprint:
            raise ValueError(
                "Correction partitions require supersedes_fingerprint"
            )
        if delivery_type == "snapshot" and self.supersedes_fingerprint:
            raise ValueError(
                "Snapshot partitions cannot supersede another fingerprint"
            )
        object.__setattr__(self, "delivery_type", delivery_type)
        if self.row_count is not None and int(self.row_count) < 0:
            raise ValueError("row_count must be >= 0")

    @property
    def window_key(self) -> str:
        return ":".join(
            (
                self.tenant_id,
                self.provider_id,
                self.dataset_id,
                self.schema_version,
                self.delivery_window_id,
            )
        )

    def to_safe_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderDataRightsRequest:
    request_id: str
    tenant_id: str
    provider_id: str
    dataset_id: str
    request_type: str
    subject_token_sha256: str
    requested_at: str
    event_time_start: Optional[str] = None
    event_time_end: Optional[str] = None
    source_request_ref: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in (
            "request_id",
            "tenant_id",
            "provider_id",
            "dataset_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field_name),
            )
        request_type = str(self.request_type or "").strip().lower()
        if request_type not in {"delete", "opt_out"}:
            raise ValueError("request_type must be delete or opt_out")
        object.__setattr__(self, "request_type", request_type)
        object.__setattr__(
            self,
            "subject_token_sha256",
            _sha256(self.subject_token_sha256, "subject_token_sha256"),
        )
        object.__setattr__(
            self,
            "requested_at",
            _utc(self.requested_at, "requested_at"),
        )
        if (self.event_time_start is None) != (self.event_time_end is None):
            raise ValueError(
                "event_time_start and event_time_end must be supplied together"
            )
        if self.event_time_start is not None:
            start = _utc(self.event_time_start, "event_time_start")
            end = _utc(self.event_time_end, "event_time_end")
            if datetime.fromisoformat(start) >= datetime.fromisoformat(end):
                raise ValueError(
                    "event_time_start must be before event_time_end"
                )
            object.__setattr__(self, "event_time_start", start)
            object.__setattr__(self, "event_time_end", end)
        if self.source_request_ref is not None:
            normalized = str(self.source_request_ref).strip()
            if not normalized or len(normalized) > 512:
                raise ValueError("source_request_ref is invalid")
            object.__setattr__(self, "source_request_ref", normalized)

    def to_safe_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["subject_token_sha256"] = "redacted"
        return payload

