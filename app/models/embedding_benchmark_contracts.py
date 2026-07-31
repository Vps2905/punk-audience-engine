from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import timezone
from typing import Any

from app.models.audience_feature_contracts import (
    normalize_taxonomy_value,
    parse_utc_datetime,
    stable_digest,
)

BENCHMARK_REPORT_SCHEMA_VERSION = "punk-embedding-benchmark-report-v1"
PRODUCTION_BENCHMARK_POLICY_ID = "punk-global-embedding-policy-v2"

MINIMUM_METRIC_THRESHOLDS: dict[str, float] = {
    "recall_at_k": 0.85,
    "precision_at_k": 0.60,
    "ndcg_at_k": 0.85,
    "mrr": 0.90,
    "geographic_constraint_accuracy": 0.95,
    "category_constraint_accuracy": 0.95,
    "daypart_accuracy": 0.95,
    "hard_negative_rejection_rate": 0.98,
    "multilingual_consistency": 0.90,
}

MAXIMUM_METRIC_THRESHOLDS: dict[str, float] = {
    "unsupported_location_false_match_rate": 0.01,
    "p50_latency_ms": 250.0,
    "p95_latency_ms": 500.0,
    "p99_latency_ms": 1000.0,
    "peak_memory_mb": 8192.0,
    "cost_per_1000_queries_usd": 5.0,
}

MINIMUM_COVERAGE_THRESHOLDS: dict[str, int] = {
    "case_count": 100,
    "document_count": 100,
    "language_count": 5,
    "geographic_case_count": 20,
    "category_case_count": 20,
    "daypart_case_count": 20,
    "hard_negative_case_count": 20,
    "unsupported_location_case_count": 20,
    "multilingual_group_count": 10,
    "location_value_count": 10,
    "category_value_count": 10,
    "daypart_value_count": 4,
    "minimum_cases_per_language": 10,
    "constraint_intersection_case_count": 20,
}


def _required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required.")
    return text


def _normalized_values(values: Any, label: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{label} must be an array.")
    normalized = tuple(
        dict.fromkeys(
            normalize_taxonomy_value(value)
            for value in values
            if normalize_taxonomy_value(value)
        )
    )
    return normalized


def _finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be numeric.") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    return parsed


def _sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef"
        for character in digest
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return digest


def _authoring_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("authoring_manifest must be an object.")
    safe = json.loads(json.dumps(dict(value), allow_nan=False))
    required = {
        "schema_version",
        "builder_version",
        "review_status",
        "reviewed_by",
        "reviewed_at",
        "case_catalog_fingerprint",
        "document_catalog_fingerprints",
    }
    if required.difference(safe):
        raise ValueError(
            "authoring_manifest is missing required review or lineage fields."
        )
    if safe["schema_version"] != "punk-embedding-benchmark-authoring-v1":
        raise ValueError("Unsupported benchmark authoring schema.")
    if safe["builder_version"] != "production-dataset-builder-v1":
        raise ValueError("Unsupported benchmark dataset builder version.")
    if normalize_taxonomy_value(safe["review_status"]) != "approved":
        raise ValueError("Benchmark gold labels require approved review.")
    if not normalize_taxonomy_value(safe["reviewed_by"]):
        raise ValueError("Benchmark gold labels require a named reviewer.")
    reviewed_at = parse_utc_datetime(safe["reviewed_at"])
    if reviewed_at is None:
        raise ValueError("Benchmark reviewed_at must be a UTC timestamp.")
    safe["review_status"] = "approved"
    safe["reviewed_by"] = normalize_taxonomy_value(safe["reviewed_by"])
    safe["reviewed_at"] = reviewed_at.astimezone(
        timezone.utc
    ).isoformat()
    safe["case_catalog_fingerprint"] = _sha256(
        safe["case_catalog_fingerprint"],
        "case_catalog_fingerprint",
    )
    catalog_fingerprints = safe["document_catalog_fingerprints"]
    if not isinstance(catalog_fingerprints, list) or not catalog_fingerprints:
        raise ValueError(
            "authoring_manifest requires document catalog fingerprints."
        )
    safe["document_catalog_fingerprints"] = sorted(
        {
            _sha256(value, "document_catalog_fingerprint")
            for value in catalog_fingerprints
        }
    )
    return safe


@dataclass(frozen=True)
class EmbeddingBenchmarkDocument:
    document_id: str
    text: str
    location: str
    category: str
    daypart: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_id",
            _required_text(self.document_id, "document_id"),
        )
        object.__setattr__(self, "text", _required_text(self.text, "text"))
        for field_name in ("location", "category", "daypart"):
            normalized = normalize_taxonomy_value(getattr(self, field_name))
            if not normalized:
                raise ValueError(
                    f"Benchmark document {field_name} is required."
                )
            object.__setattr__(self, field_name, normalized)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingBenchmarkCase:
    case_id: str
    query: str
    language: str
    relevant_document_ids: tuple[str, ...]
    hard_negative_document_ids: tuple[str, ...] = ()
    expected_locations: tuple[str, ...] = ()
    expected_categories: tuple[str, ...] = ()
    expected_dayparts: tuple[str, ...] = ()
    semantic_group_id: str | None = None
    unsupported_location: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_id",
            _required_text(self.case_id, "case_id"),
        )
        object.__setattr__(
            self,
            "query",
            _required_text(self.query, "query"),
        )
        language = normalize_taxonomy_value(self.language)
        if not language:
            raise ValueError("language is required.")
        object.__setattr__(self, "language", language)

        relevant = tuple(
            dict.fromkeys(
                _required_text(value, "relevant_document_id")
                for value in self.relevant_document_ids
            )
        )
        hard_negatives = tuple(
            dict.fromkeys(
                _required_text(value, "hard_negative_document_id")
                for value in self.hard_negative_document_ids
            )
        )
        if not self.unsupported_location and not relevant:
            raise ValueError(
                "Supported benchmark cases require relevant documents."
            )
        if set(relevant).intersection(hard_negatives):
            raise ValueError(
                "Relevant and hard-negative document IDs cannot overlap."
            )
        object.__setattr__(self, "relevant_document_ids", relevant)
        object.__setattr__(
            self,
            "hard_negative_document_ids",
            hard_negatives,
        )
        object.__setattr__(
            self,
            "expected_locations",
            _normalized_values(
                self.expected_locations,
                "expected_locations",
            ),
        )
        object.__setattr__(
            self,
            "expected_categories",
            _normalized_values(
                self.expected_categories,
                "expected_categories",
            ),
        )
        object.__setattr__(
            self,
            "expected_dayparts",
            _normalized_values(
                self.expected_dayparts,
                "expected_dayparts",
            ),
        )
        group_id = str(self.semantic_group_id or "").strip() or None
        object.__setattr__(self, "semantic_group_id", group_id)
        object.__setattr__(
            self,
            "unsupported_location",
            bool(self.unsupported_location),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> EmbeddingBenchmarkCase:
        return cls(
            case_id=payload.get("case_id", ""),
            query=payload.get("query", ""),
            language=payload.get("language", ""),
            relevant_document_ids=tuple(
                payload.get("relevant_document_ids") or ()
            ),
            hard_negative_document_ids=tuple(
                payload.get("hard_negative_document_ids") or ()
            ),
            expected_locations=tuple(
                payload.get("expected_locations") or ()
            ),
            expected_categories=tuple(
                payload.get("expected_categories") or ()
            ),
            expected_dayparts=tuple(
                payload.get("expected_dayparts") or ()
            ),
            semantic_group_id=payload.get("semantic_group_id"),
            unsupported_location=bool(
                payload.get("unsupported_location", False)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        for key in (
            "relevant_document_ids",
            "hard_negative_document_ids",
            "expected_locations",
            "expected_categories",
            "expected_dayparts",
        ):
            record[key] = list(record[key])
        return record


@dataclass(frozen=True)
class EmbeddingBenchmarkDataset:
    benchmark_id: str
    dataset_version: str
    privacy_status: str
    rights_status: str
    contains_raw_identifiers: bool
    authoring_manifest: Mapping[str, Any]
    documents: tuple[EmbeddingBenchmarkDocument, ...]
    cases: tuple[EmbeddingBenchmarkCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "benchmark_id",
            _required_text(self.benchmark_id, "benchmark_id"),
        )
        object.__setattr__(
            self,
            "dataset_version",
            _required_text(self.dataset_version, "dataset_version"),
        )
        privacy_status = normalize_taxonomy_value(self.privacy_status)
        if privacy_status not in {"safe", "passed", "privacy_safe"}:
            raise ValueError(
                "Embedding benchmarks require privacy-safe evaluation data."
            )
        object.__setattr__(self, "privacy_status", privacy_status)
        rights_status = normalize_taxonomy_value(self.rights_status)
        if rights_status not in {
            "permitted",
            "offline_evaluation_only",
            "synthetic_evaluation",
        }:
            raise ValueError(
                "Embedding benchmark evaluation rights are not permitted."
            )
        object.__setattr__(self, "rights_status", rights_status)
        if self.contains_raw_identifiers:
            raise ValueError(
                "Embedding benchmarks cannot contain raw identifiers."
            )
        object.__setattr__(
            self,
            "contains_raw_identifiers",
            False,
        )
        object.__setattr__(
            self,
            "authoring_manifest",
            _authoring_manifest(self.authoring_manifest),
        )
        if not self.documents or not self.cases:
            raise ValueError(
                "Benchmark datasets require documents and cases."
            )
        document_ids = [value.document_id for value in self.documents]
        case_ids = [value.case_id for value in self.cases]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Benchmark document IDs must be unique.")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Benchmark case IDs must be unique.")

        known_ids = set(document_ids)
        for case in self.cases:
            referenced = set(case.relevant_document_ids).union(
                case.hard_negative_document_ids
            )
            missing = sorted(referenced.difference(known_ids))
            if missing:
                raise ValueError(
                    f"Benchmark case {case.case_id} references unknown "
                    "documents."
                )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> EmbeddingBenchmarkDataset:
        if not isinstance(payload, Mapping):
            raise TypeError("Benchmark dataset must be an object.")
        documents_payload = payload.get("documents")
        cases_payload = payload.get("cases")
        if not isinstance(documents_payload, list):
            raise TypeError("Benchmark documents must be an array.")
        if not isinstance(cases_payload, list):
            raise TypeError("Benchmark cases must be an array.")
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
        cases = tuple(
            EmbeddingBenchmarkCase.from_mapping(value)
            for value in cases_payload
            if isinstance(value, Mapping)
        )
        if len(documents) != len(documents_payload):
            raise ValueError("Every benchmark document must be an object.")
        if len(cases) != len(cases_payload):
            raise ValueError("Every benchmark case must be an object.")
        return cls(
            benchmark_id=payload.get("benchmark_id", ""),
            dataset_version=payload.get("dataset_version", ""),
            privacy_status=payload.get("privacy_status", ""),
            rights_status=payload.get("rights_status", ""),
            contains_raw_identifiers=bool(
                payload.get("contains_raw_identifiers", True)
            ),
            authoring_manifest=payload.get("authoring_manifest"),
            documents=documents,
            cases=cases,
        )

    @classmethod
    def from_json(cls, content: str) -> EmbeddingBenchmarkDataset:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Benchmark dataset is not valid JSON.") from None
        return cls.from_mapping(payload)

    @property
    def fingerprint(self) -> str:
        return stable_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "dataset_version": self.dataset_version,
            "privacy_status": self.privacy_status,
            "rights_status": self.rights_status,
            "contains_raw_identifiers": False,
            "authoring_manifest": dict(self.authoring_manifest),
            "documents": [
                value.to_dict()
                for value in sorted(
                    self.documents,
                    key=lambda item: item.document_id,
                )
            ],
            "cases": [
                value.to_dict()
                for value in sorted(
                    self.cases,
                    key=lambda item: item.case_id,
                )
            ],
        }


@dataclass(frozen=True)
class ProductionEmbeddingBenchmarkPolicy:
    policy_id: str = PRODUCTION_BENCHMARK_POLICY_ID
    minimum_metric_thresholds: Mapping[str, float] = field(
        default_factory=lambda: dict(MINIMUM_METRIC_THRESHOLDS)
    )
    maximum_metric_thresholds: Mapping[str, float] = field(
        default_factory=lambda: dict(MAXIMUM_METRIC_THRESHOLDS)
    )
    minimum_coverage_thresholds: Mapping[str, int] = field(
        default_factory=lambda: dict(MINIMUM_COVERAGE_THRESHOLDS)
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_id",
            _required_text(self.policy_id, "policy_id"),
        )
        object.__setattr__(
            self,
            "minimum_metric_thresholds",
            dict(self.minimum_metric_thresholds),
        )
        object.__setattr__(
            self,
            "maximum_metric_thresholds",
            dict(self.maximum_metric_thresholds),
        )
        object.__setattr__(
            self,
            "minimum_coverage_thresholds",
            dict(self.minimum_coverage_thresholds),
        )

    @property
    def default_thresholds(self) -> dict[str, float | int]:
        return {
            **self.minimum_metric_thresholds,
            **self.maximum_metric_thresholds,
            **self.minimum_coverage_thresholds,
        }

    def resolve_thresholds(
        self,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, float | int]:
        resolved = self.default_thresholds
        for key, value in dict(overrides or {}).items():
            if key not in resolved:
                raise ValueError(f"Unsupported benchmark threshold: {key}")
            if key in self.minimum_coverage_thresholds:
                parsed: float | int = int(value)
            else:
                parsed = _finite_float(value, f"threshold {key}")
            resolved[key] = parsed

        for key, policy_floor in self.minimum_metric_thresholds.items():
            if float(resolved[key]) < float(policy_floor):
                raise ValueError(
                    f"Benchmark threshold {key} is weaker than "
                    f"{self.policy_id}."
                )
        for key, policy_ceiling in self.maximum_metric_thresholds.items():
            if float(resolved[key]) > float(policy_ceiling):
                raise ValueError(
                    f"Benchmark threshold {key} is weaker than "
                    f"{self.policy_id}."
                )
        for key, policy_floor in self.minimum_coverage_thresholds.items():
            if int(resolved[key]) < int(policy_floor):
                raise ValueError(
                    f"Benchmark coverage threshold {key} is weaker than "
                    f"{self.policy_id}."
                )
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "minimum_metric_thresholds": dict(
                self.minimum_metric_thresholds
            ),
            "maximum_metric_thresholds": dict(
                self.maximum_metric_thresholds
            ),
            "minimum_coverage_thresholds": dict(
                self.minimum_coverage_thresholds
            ),
        }
