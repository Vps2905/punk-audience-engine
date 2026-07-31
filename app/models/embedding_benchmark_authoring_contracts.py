from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timezone
from typing import Any

from app.models.audience_feature_contracts import (
    normalize_taxonomy_value,
    parse_utc_datetime,
    stable_digest,
)
from app.models.embedding_benchmark_contracts import (
    EmbeddingBenchmarkCase,
    EmbeddingBenchmarkDocument,
)

SAFE_CATALOG_SOURCE_TYPES = {
    "pgvector_privacy_safe_features",
    "curated_synthetic_features",
    "curated_aggregated_features",
}


def _required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required.")
    return text


def _sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef"
        for character in digest
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return digest


def _json_safe_mapping(
    payload: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    try:
        return json.loads(
            json.dumps(dict(payload), allow_nan=False)
        )
    except (TypeError, ValueError):
        raise ValueError(
            f"{label} must contain finite JSON values."
        ) from None


@dataclass(frozen=True)
class EmbeddingBenchmarkDocumentCatalog:
    catalog_id: str
    catalog_version: str
    source_type: str
    source_fingerprint: str
    privacy_status: str
    rights_status: str
    contains_raw_identifiers: bool
    documents: tuple[EmbeddingBenchmarkDocument, ...]
    lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "catalog_id",
            _required_text(self.catalog_id, "catalog_id"),
        )
        object.__setattr__(
            self,
            "catalog_version",
            _required_text(self.catalog_version, "catalog_version"),
        )
        source_type = normalize_taxonomy_value(self.source_type)
        if source_type not in SAFE_CATALOG_SOURCE_TYPES:
            raise ValueError("Unsupported benchmark catalog source_type.")
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(
            self,
            "source_fingerprint",
            _sha256(self.source_fingerprint, "source_fingerprint"),
        )
        privacy_status = normalize_taxonomy_value(self.privacy_status)
        if privacy_status not in {"safe", "passed", "privacy_safe"}:
            raise ValueError("Benchmark document catalog is not privacy-safe.")
        object.__setattr__(self, "privacy_status", privacy_status)
        rights_status = normalize_taxonomy_value(self.rights_status)
        if rights_status not in {
            "permitted",
            "offline_evaluation_only",
            "synthetic_evaluation",
        }:
            raise ValueError("Benchmark catalog rights are not permitted.")
        object.__setattr__(self, "rights_status", rights_status)
        if self.contains_raw_identifiers:
            raise ValueError(
                "Benchmark document catalogs cannot contain raw identifiers."
            )
        object.__setattr__(self, "contains_raw_identifiers", False)
        if not self.documents:
            raise ValueError("Benchmark document catalog cannot be empty.")
        document_ids = [value.document_id for value in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError(
                "Benchmark document catalog IDs must be unique."
            )
        object.__setattr__(
            self,
            "lineage",
            _json_safe_mapping(self.lineage, "catalog lineage"),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> EmbeddingBenchmarkDocumentCatalog:
        if not isinstance(payload, Mapping):
            raise TypeError("Benchmark document catalog must be an object.")
        documents_payload = payload.get("documents")
        if not isinstance(documents_payload, list):
            raise TypeError("Benchmark catalog documents must be an array.")
        documents = tuple(
            EmbeddingBenchmarkDocument(
                document_id=value.get("document_id", ""),
                text=value.get("text", ""),
                location=value.get("location", ""),
                category=value.get("category", ""),
                daypart=value.get("daypart", ""),
            )
            for value in documents_payload
            if isinstance(value, Mapping)
        )
        if len(documents) != len(documents_payload):
            raise ValueError("Every benchmark document must be an object.")
        return cls(
            catalog_id=payload.get("catalog_id", ""),
            catalog_version=payload.get("catalog_version", ""),
            source_type=payload.get("source_type", ""),
            source_fingerprint=payload.get("source_fingerprint", ""),
            privacy_status=payload.get("privacy_status", ""),
            rights_status=payload.get("rights_status", ""),
            contains_raw_identifiers=bool(
                payload.get("contains_raw_identifiers", True)
            ),
            documents=documents,
            lineage=(
                payload.get("lineage")
                if isinstance(payload.get("lineage"), Mapping)
                else {}
            ),
        )

    @classmethod
    def from_json(
        cls,
        content: str,
    ) -> EmbeddingBenchmarkDocumentCatalog:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(
                "Benchmark document catalog is not valid JSON."
            ) from None
        return cls.from_mapping(payload)

    @property
    def fingerprint(self) -> str:
        return stable_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "source_type": self.source_type,
            "source_fingerprint": self.source_fingerprint,
            "privacy_status": self.privacy_status,
            "rights_status": self.rights_status,
            "contains_raw_identifiers": False,
            "documents": [
                value.to_dict()
                for value in sorted(
                    self.documents,
                    key=lambda item: item.document_id,
                )
            ],
            "lineage": dict(self.lineage),
        }


@dataclass(frozen=True)
class EmbeddingBenchmarkCaseCatalog:
    catalog_id: str
    catalog_version: str
    review_status: str
    reviewed_by: str
    reviewed_at: Any
    cases: tuple[EmbeddingBenchmarkCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "catalog_id",
            _required_text(self.catalog_id, "case catalog_id"),
        )
        object.__setattr__(
            self,
            "catalog_version",
            _required_text(
                self.catalog_version,
                "case catalog_version",
            ),
        )
        if normalize_taxonomy_value(self.review_status) != "approved":
            raise ValueError("Benchmark case catalog review is not approved.")
        object.__setattr__(self, "review_status", "approved")
        reviewer = normalize_taxonomy_value(self.reviewed_by)
        if not reviewer:
            raise ValueError("Benchmark cases require a named reviewer.")
        object.__setattr__(self, "reviewed_by", reviewer)
        reviewed_at = parse_utc_datetime(self.reviewed_at)
        if reviewed_at is None:
            raise ValueError("Benchmark cases require reviewed_at.")
        object.__setattr__(
            self,
            "reviewed_at",
            reviewed_at.astimezone(timezone.utc),
        )
        if not self.cases:
            raise ValueError("Benchmark case catalog cannot be empty.")
        case_ids = [value.case_id for value in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Benchmark case IDs must be unique.")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> EmbeddingBenchmarkCaseCatalog:
        if not isinstance(payload, Mapping):
            raise TypeError("Benchmark case catalog must be an object.")
        cases_payload = payload.get("cases")
        if not isinstance(cases_payload, list):
            raise TypeError("Benchmark catalog cases must be an array.")
        cases = tuple(
            EmbeddingBenchmarkCase.from_mapping(value)
            for value in cases_payload
            if isinstance(value, Mapping)
        )
        if len(cases) != len(cases_payload):
            raise ValueError("Every benchmark case must be an object.")
        return cls(
            catalog_id=payload.get("catalog_id", ""),
            catalog_version=payload.get("catalog_version", ""),
            review_status=payload.get("review_status", ""),
            reviewed_by=payload.get("reviewed_by", ""),
            reviewed_at=payload.get("reviewed_at"),
            cases=cases,
        )

    @classmethod
    def from_json(
        cls,
        content: str,
    ) -> EmbeddingBenchmarkCaseCatalog:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(
                "Benchmark case catalog is not valid JSON."
            ) from None
        return cls.from_mapping(payload)

    @property
    def fingerprint(self) -> str:
        return stable_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "review_status": self.review_status,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat(),
            "cases": [
                value.to_dict()
                for value in sorted(
                    self.cases,
                    key=lambda item: item.case_id,
                )
            ],
        }
