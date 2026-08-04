from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from app.models.audience_feature_contracts import stable_digest


@dataclass(frozen=True)
class GovernedRetrievalBenchmarkPolicy:
    """Engineering acceptance policy for the offline Module 2.1 evaluation."""

    min_location_accuracy: float = 0.95
    min_category_accuracy: float = 0.95
    min_daypart_accuracy: float = 0.95
    min_full_canonicalization_accuracy: float = 0.95
    min_base_candidate_recall: float = 0.95
    min_final_candidate_recall: float = 0.99
    min_structured_semantic_top1_accuracy: float = 0.95
    min_unsupported_rejection_rate: float = 0.99
    min_clarification_accuracy: float = 0.95
    min_per_language_success_rate: float = 0.90
    max_unsafe_unsupported_ready_rate: float = 0.01
    max_p95_latency_ms: float = 1000.0
    max_p99_latency_ms: float = 2000.0
    target_unsupported_false_match_rate: float = 0.01
    confidence_level: float = 0.95
    contract_version: str = "production-governed-retrieval-benchmark-policy-v1"

    def __post_init__(self) -> None:
        rate_fields = (
            "min_location_accuracy",
            "min_category_accuracy",
            "min_daypart_accuracy",
            "min_full_canonicalization_accuracy",
            "min_base_candidate_recall",
            "min_final_candidate_recall",
            "min_structured_semantic_top1_accuracy",
            "min_unsupported_rejection_rate",
            "min_clarification_accuracy",
            "min_per_language_success_rate",
            "max_unsafe_unsupported_ready_rate",
            "target_unsupported_false_match_rate",
            "confidence_level",
        )
        for field_name in rate_fields:
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1.")
            object.__setattr__(self, field_name, value)
        for field_name in ("max_p95_latency_ms", "max_p99_latency_ms"):
            value = float(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive.")
            object.__setattr__(self, field_name, value)
        if self.max_p99_latency_ms < self.max_p95_latency_ms:
            raise ValueError("p99 latency limit cannot be below p95.")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1.")

    @property
    def fingerprint(self) -> str:
        return stable_digest(asdict(self))

    def to_safe_dict(self) -> dict[str, Any]:
        return {**asdict(self), "policy_fingerprint": self.fingerprint}


@dataclass(frozen=True)
class GovernedRetrievalBenchmarkDatasetIdentity:
    benchmark_id: str
    dataset_version: str
    dataset_fingerprint: str
    document_count: int
    case_count: int
    supported_case_count: int
    unsupported_case_count: int

    def __post_init__(self) -> None:
        benchmark_id = " ".join(str(self.benchmark_id or "").split())
        dataset_version = " ".join(str(self.dataset_version or "").split())
        fingerprint = str(self.dataset_fingerprint or "").strip().lower()
        if not benchmark_id or not dataset_version:
            raise ValueError("benchmark_id and dataset_version are required.")
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("dataset_fingerprint must be a lowercase SHA-256 digest.")
        for field_name in (
            "document_count",
            "case_count",
            "supported_case_count",
            "unsupported_case_count",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative.")
        if int(self.document_count) < 1 or int(self.case_count) < 1:
            raise ValueError("Benchmark datasets require documents and cases.")
        if int(self.supported_case_count) + int(self.unsupported_case_count) != int(
            self.case_count
        ):
            raise ValueError("Supported and unsupported counts must equal case_count.")
        object.__setattr__(self, "benchmark_id", benchmark_id)
        object.__setattr__(self, "dataset_version", dataset_version)
        object.__setattr__(self, "dataset_fingerprint", fingerprint)

    def to_safe_dict(self) -> dict[str, Any]:
        return asdict(self)
