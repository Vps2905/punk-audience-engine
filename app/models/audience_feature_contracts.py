from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

SAFE_PRIVACY_STATUSES = {
    "passed",
    "privacy_safe",
    "safe",
}

NON_ACTIVATABLE_DATA_MODES = {
    "historical_preview",
    "offline_evaluation",
}


def normalize_taxonomy_value(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text)).strip("_")


def parse_utc_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


@dataclass(frozen=True)
class CanonicalAudienceFeature:
    tenant_id: str
    feature_set_id: str
    feature_set_version: int
    feature_id: str
    location_name: str
    primary_poi_type: str
    created_day_part: str
    lookback_bucket: str | None
    cohort_size: int
    quality_score: float
    privacy_status: str
    rights_status: str
    purpose: str
    source_latest_at: datetime | None
    freshness_status: str
    data_use_mode: str
    eligible_for_retrieval: bool
    eligible_for_activation: bool
    trait_text: str
    embedding: Sequence[float]
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        if self.source_latest_at is not None:
            record["source_latest_at"] = self.source_latest_at.isoformat()
        record["embedding"] = [float(value) for value in self.embedding]
        return record


@dataclass(frozen=True)
class CanonicalFeatureSet:
    tenant_id: str
    feature_set_id: str
    version: int
    status: str
    source_mode: str
    data_use_mode: str
    source_ref: str
    source_version: str
    source_fingerprint: str
    source_latest_at: datetime | None
    freshness_status: str
    stale_after_hours: int
    model_backend: str
    model_name: str
    model_version: str
    embedding_dimension: int
    privacy_policy_version: str
    rights_policy_id: str
    purpose: str
    eligible_for_retrieval: bool
    eligible_for_activation: bool
    feature_count: int
    lineage: dict[str, Any]
    features: Sequence[CanonicalAudienceFeature]

    def to_record(self, *, include_features: bool = False) -> dict[str, Any]:
        record = asdict(self)
        if self.source_latest_at is not None:
            record["source_latest_at"] = self.source_latest_at.isoformat()
        if not include_features:
            record.pop("features", None)
        return record


@dataclass(frozen=True)
class LegacySafeVectorSnapshot:
    job_id: str
    model_info: dict[str, Any]
    vector_count: int
    vector_dimension: int
    indexed_at: datetime | None
    latest_source_timestamp: datetime | None
    rows: Sequence[dict[str, Any]]


def validate_embedding(
    values: Iterable[Any],
    *,
    expected_dimension: int,
) -> list[float]:
    vector = [_finite_float(value, default=float("nan")) for value in values]
    if len(vector) != expected_dimension:
        raise ValueError(
            "Embedding dimension mismatch. "
            f"Expected {expected_dimension}, received {len(vector)}."
        )
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding contains non-finite values.")
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude <= 0:
        raise ValueError("Embedding must have a non-zero magnitude.")
    return vector
