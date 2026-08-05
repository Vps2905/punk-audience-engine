from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Sequence

from app.models.audience_feature_contracts import (
    normalize_taxonomy_value,
    parse_utc_datetime,
    stable_digest,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _text(value: Any, label: str) -> str:
    clean = " ".join(str(value or "").split())
    if not clean:
        raise ValueError(f"{label} is required.")
    return clean


def _slug(value: Any, label: str) -> str:
    clean = normalize_taxonomy_value(value)
    if not clean:
        raise ValueError(f"{label} is required.")
    return clean


def _sha256(value: Any, label: str) -> str:
    clean = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(clean):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return clean


def _utc(value: Any, label: str) -> datetime:
    parsed = parse_utc_datetime(value)
    if parsed is None:
        raise ValueError(f"{label} must be a UTC timestamp.")
    return parsed.astimezone(timezone.utc)


def _unique_slugs(values: Sequence[Any], label: str) -> tuple[str, ...]:
    clean = tuple(dict.fromkeys(_slug(value, label) for value in values))
    if not clean:
        raise ValueError(f"{label} cannot be empty.")
    return clean


@dataclass(frozen=True)
class NativeLanguageReviewDecision:
    language: str
    reviewer_id: str
    reviewer_native_language_confirmed: bool
    taxonomy_fingerprint: str
    language_pack_sha256: str
    reviewed_alias_count: int
    decision: Literal["approved", "rejected"]
    conflict_count: int
    unresolved_conflict_count: int
    reviewed_at: datetime
    machine_translation_auto_approved: bool = False
    raw_identifiers_reviewed: bool = False
    notes_fingerprint: str | None = None
    contract_version: str = "module2-native-language-review-decision-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "language", _slug(self.language, "language"))
        object.__setattr__(self, "reviewer_id", _slug(self.reviewer_id, "reviewer_id"))
        object.__setattr__(
            self,
            "taxonomy_fingerprint",
            _sha256(self.taxonomy_fingerprint, "taxonomy_fingerprint"),
        )
        object.__setattr__(
            self,
            "language_pack_sha256",
            _sha256(self.language_pack_sha256, "language_pack_sha256"),
        )
        if int(self.reviewed_alias_count) < 1:
            raise ValueError("reviewed_alias_count must be at least 1.")
        if int(self.conflict_count) < 0 or int(self.unresolved_conflict_count) < 0:
            raise ValueError("Review conflict counts cannot be negative.")
        if int(self.unresolved_conflict_count) > int(self.conflict_count):
            raise ValueError("Unresolved conflicts cannot exceed total conflicts.")
        if self.decision == "approved":
            if not self.reviewer_native_language_confirmed:
                raise ValueError("Approval requires native-language confirmation.")
            if int(self.unresolved_conflict_count) != 0:
                raise ValueError("Approval cannot contain unresolved conflicts.")
            if self.machine_translation_auto_approved:
                raise ValueError("Machine translation cannot be auto-approved.")
            if self.raw_identifiers_reviewed:
                raise ValueError("Native-language review cannot include raw identifiers.")
        object.__setattr__(self, "reviewed_at", _utc(self.reviewed_at, "reviewed_at"))
        if self.notes_fingerprint is not None:
            object.__setattr__(
                self,
                "notes_fingerprint",
                _sha256(self.notes_fingerprint, "notes_fingerprint"),
            )

    def to_safe_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["reviewed_at"] = self.reviewed_at.isoformat()
        return record


@dataclass(frozen=True)
class NativeLanguageReviewManifest:
    manifest_id: str
    manifest_version: str
    taxonomy_fingerprint: str
    language_pack_sha256: str
    required_languages: tuple[str, ...]
    decisions: tuple[NativeLanguageReviewDecision, ...]
    release_owner: str
    review_status: Literal["pending", "approved", "rejected"]
    created_at: datetime
    finalized_at: datetime | None = None
    contains_raw_queries: bool = False
    contains_raw_identifiers: bool = False
    contract_version: str = "module2-native-language-review-manifest-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_id", _slug(self.manifest_id, "manifest_id"))
        object.__setattr__(self, "manifest_version", _text(self.manifest_version, "manifest_version"))
        object.__setattr__(
            self,
            "taxonomy_fingerprint",
            _sha256(self.taxonomy_fingerprint, "taxonomy_fingerprint"),
        )
        object.__setattr__(
            self,
            "language_pack_sha256",
            _sha256(self.language_pack_sha256, "language_pack_sha256"),
        )
        languages = _unique_slugs(self.required_languages, "required_language")
        decisions = tuple(self.decisions)
        decision_languages = [decision.language for decision in decisions]
        if len(decision_languages) != len(set(decision_languages)):
            raise ValueError("Each language may have only one final review decision.")
        if set(decision_languages).difference(languages):
            raise ValueError("Review decisions include an unrequired language.")
        for decision in decisions:
            if decision.taxonomy_fingerprint != self.taxonomy_fingerprint:
                raise ValueError("Review decision taxonomy fingerprint mismatch.")
            if decision.language_pack_sha256 != self.language_pack_sha256:
                raise ValueError("Review decision language-pack checksum mismatch.")
        release_owner = _slug(self.release_owner, "release_owner")
        if any(decision.reviewer_id == release_owner for decision in decisions):
            raise ValueError("Release owner cannot self-approve native-language review.")
        created_at = _utc(self.created_at, "created_at")
        finalized_at = (
            _utc(self.finalized_at, "finalized_at")
            if self.finalized_at is not None
            else None
        )
        if finalized_at is not None and finalized_at < created_at:
            raise ValueError("finalized_at cannot precede created_at.")
        if self.contains_raw_queries or self.contains_raw_identifiers:
            raise ValueError("Certification review manifests must be privacy-safe.")
        if self.review_status == "approved":
            if finalized_at is None:
                raise ValueError("Approved review manifests require finalized_at.")
            if set(decision_languages) != set(languages):
                raise ValueError("Approved review manifests require every language.")
            if any(decision.decision != "approved" for decision in decisions):
                raise ValueError("Approved manifest contains a rejected language.")
        if self.review_status == "rejected" and not any(
            decision.decision == "rejected" for decision in decisions
        ):
            raise ValueError("Rejected manifest requires a rejected decision.")
        object.__setattr__(self, "required_languages", languages)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "release_owner", release_owner)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "finalized_at", finalized_at)

    @property
    def fingerprint(self) -> str:
        return stable_digest(self.to_safe_dict(include_fingerprint=False))

    @property
    def approved(self) -> bool:
        return self.review_status == "approved"

    def to_safe_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        record = {
            "contract_version": self.contract_version,
            "manifest_id": self.manifest_id,
            "manifest_version": self.manifest_version,
            "taxonomy_fingerprint": self.taxonomy_fingerprint,
            "language_pack_sha256": self.language_pack_sha256,
            "required_languages": list(self.required_languages),
            "decisions": [decision.to_safe_dict() for decision in self.decisions],
            "release_owner": self.release_owner,
            "review_status": self.review_status,
            "created_at": self.created_at.isoformat(),
            "finalized_at": self.finalized_at.isoformat() if self.finalized_at else None,
            "contains_raw_queries": False,
            "contains_raw_identifiers": False,
        }
        if include_fingerprint:
            record["manifest_fingerprint"] = self.fingerprint
        return record

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "NativeLanguageReviewManifest":
        decisions = tuple(
            NativeLanguageReviewDecision(**dict(item))
            for item in payload.get("decisions") or ()
        )
        instance = cls(
            manifest_id=payload.get("manifest_id", ""),
            manifest_version=payload.get("manifest_version", ""),
            taxonomy_fingerprint=payload.get("taxonomy_fingerprint", ""),
            language_pack_sha256=payload.get("language_pack_sha256", ""),
            required_languages=tuple(payload.get("required_languages") or ()),
            decisions=decisions,
            release_owner=payload.get("release_owner", ""),
            review_status=payload.get("review_status", "pending"),
            created_at=payload.get("created_at"),
            finalized_at=payload.get("finalized_at"),
            contains_raw_queries=bool(payload.get("contains_raw_queries", False)),
            contains_raw_identifiers=bool(payload.get("contains_raw_identifiers", False)),
        )
        declared = str(payload.get("manifest_fingerprint") or "").strip()
        if declared and declared != instance.fingerprint:
            raise ValueError("Declared manifest_fingerprint does not match content.")
        return instance


@dataclass(frozen=True)
class UnsupportedCalibrationObservation:
    case_id: str
    language: str
    location_token_fingerprint: str
    taxonomy_fingerprint: str
    expected_rejection: bool
    observed_retrieval_ready: bool
    observed_reason_code: str
    reviewed_by: str
    reviewed_at: datetime
    raw_query_stored: bool = False
    raw_identifiers_stored: bool = False
    contract_version: str = "module2-unsupported-calibration-observation-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _slug(self.case_id, "case_id"))
        object.__setattr__(self, "language", _slug(self.language, "language"))
        object.__setattr__(
            self,
            "location_token_fingerprint",
            _sha256(self.location_token_fingerprint, "location_token_fingerprint"),
        )
        object.__setattr__(
            self,
            "taxonomy_fingerprint",
            _sha256(self.taxonomy_fingerprint, "taxonomy_fingerprint"),
        )
        if not self.expected_rejection:
            raise ValueError("Unsupported calibration cases must expect rejection.")
        object.__setattr__(
            self,
            "observed_reason_code",
            _slug(self.observed_reason_code, "observed_reason_code"),
        )
        object.__setattr__(self, "reviewed_by", _slug(self.reviewed_by, "reviewed_by"))
        object.__setattr__(self, "reviewed_at", _utc(self.reviewed_at, "reviewed_at"))
        if self.raw_query_stored or self.raw_identifiers_stored:
            raise ValueError("Calibration evidence cannot store raw queries or identifiers.")

    @property
    def failure(self) -> bool:
        return bool(self.observed_retrieval_ready)

    def to_safe_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["reviewed_at"] = self.reviewed_at.isoformat()
        return record


@dataclass(frozen=True)
class UnsupportedCalibrationDataset:
    dataset_id: str
    dataset_version: str
    taxonomy_fingerprint: str
    observations: tuple[UnsupportedCalibrationObservation, ...]
    review_status: Literal["pending", "approved", "rejected"]
    reviewed_by: str
    reviewed_at: datetime
    contains_raw_queries: bool = False
    contains_raw_identifiers: bool = False
    contract_version: str = "module2-unsupported-calibration-dataset-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _slug(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "dataset_version", _text(self.dataset_version, "dataset_version"))
        object.__setattr__(
            self,
            "taxonomy_fingerprint",
            _sha256(self.taxonomy_fingerprint, "taxonomy_fingerprint"),
        )
        observations = tuple(self.observations)
        if not observations:
            raise ValueError("Calibration dataset requires observations.")
        case_ids = [observation.case_id for observation in observations]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Calibration case IDs must be unique.")
        if any(
            observation.taxonomy_fingerprint != self.taxonomy_fingerprint
            for observation in observations
        ):
            raise ValueError("Calibration observation taxonomy mismatch.")
        if self.contains_raw_queries or self.contains_raw_identifiers:
            raise ValueError("Calibration datasets must be privacy-safe.")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "reviewed_by", _slug(self.reviewed_by, "reviewed_by"))
        object.__setattr__(self, "reviewed_at", _utc(self.reviewed_at, "reviewed_at"))

    @property
    def fingerprint(self) -> str:
        return stable_digest(self.to_safe_dict(include_fingerprint=False))

    def to_safe_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        record = {
            "contract_version": self.contract_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "taxonomy_fingerprint": self.taxonomy_fingerprint,
            "observations": [value.to_safe_dict() for value in self.observations],
            "review_status": self.review_status,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat(),
            "contains_raw_queries": False,
            "contains_raw_identifiers": False,
        }
        if include_fingerprint:
            record["dataset_fingerprint"] = self.fingerprint
        return record

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "UnsupportedCalibrationDataset":
        observations = tuple(
            UnsupportedCalibrationObservation(**dict(item))
            for item in payload.get("observations") or ()
        )
        instance = cls(
            dataset_id=payload.get("dataset_id", ""),
            dataset_version=payload.get("dataset_version", ""),
            taxonomy_fingerprint=payload.get("taxonomy_fingerprint", ""),
            observations=observations,
            review_status=payload.get("review_status", "pending"),
            reviewed_by=payload.get("reviewed_by", ""),
            reviewed_at=payload.get("reviewed_at"),
            contains_raw_queries=bool(payload.get("contains_raw_queries", False)),
            contains_raw_identifiers=bool(payload.get("contains_raw_identifiers", False)),
        )
        declared = str(payload.get("dataset_fingerprint") or "").strip()
        if declared and declared != instance.fingerprint:
            raise ValueError("Declared dataset_fingerprint does not match content.")
        return instance


@dataclass(frozen=True)
class Module2CertificationPolicy:
    target_unsupported_false_match_rate: float = 0.01
    confidence_level: float = 0.95
    required_languages: tuple[str, ...] = ("en", "fr", "es", "hi", "bn")
    minimum_cases_per_language: int = 20
    required_external_gates: tuple[str, ...] = (
        "staging_deployment",
        "fresh_provider_data",
        "privacy_review",
        "scale_and_recovery",
        "cost_and_capacity",
    )
    contract_version: str = "module2-certification-policy-v1"

    def __post_init__(self) -> None:
        target = float(self.target_unsupported_false_match_rate)
        confidence = float(self.confidence_level)
        if not 0.0 < target < 1.0:
            raise ValueError("target_unsupported_false_match_rate must be between 0 and 1.")
        if not 0.5 < confidence < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1.")
        if int(self.minimum_cases_per_language) < 1:
            raise ValueError("minimum_cases_per_language must be positive.")
        object.__setattr__(self, "target_unsupported_false_match_rate", target)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "required_languages", _unique_slugs(self.required_languages, "required_language"))
        object.__setattr__(self, "required_external_gates", _unique_slugs(self.required_external_gates, "external_gate"))

    @property
    def minimum_zero_failure_cases(self) -> int:
        return int(
            math.ceil(
                math.log(1.0 - self.confidence_level)
                / math.log(1.0 - self.target_unsupported_false_match_rate)
            )
        )

    @property
    def fingerprint(self) -> str:
        return stable_digest(self.to_safe_dict(include_fingerprint=False))

    def to_safe_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        record = asdict(self)
        record["required_languages"] = list(self.required_languages)
        record["required_external_gates"] = list(self.required_external_gates)
        record["minimum_zero_failure_cases"] = self.minimum_zero_failure_cases
        if include_fingerprint:
            record["policy_fingerprint"] = self.fingerprint
        return record


IndexStatus = Literal[
    "candidate",
    "building",
    "built",
    "validated",
    "shadow",
    "active",
    "retired",
    "failed",
]


@dataclass(frozen=True)
class Module2IndexManifest:
    tenant_id: str
    index_id: str
    index_version: int
    taxonomy_fingerprint: str
    primary_model_fingerprint: str
    complementary_model_fingerprint: str
    canonicalizer_model_fingerprint: str
    feature_set_id: str
    feature_set_version: int
    source_fingerprint: str
    data_use_mode: Literal["historical_preview", "offline_evaluation", "production"]
    embedding_dimension: int
    expected_document_count: int
    built_by: str
    created_at: datetime
    activation_requested: bool = False
    downstream_export_enabled: bool = False
    contract_version: str = "module2-governed-index-manifest-v1"

    def __post_init__(self) -> None:
        for name in ("tenant_id", "index_id", "feature_set_id", "built_by"):
            object.__setattr__(self, name, _slug(getattr(self, name), name))
        for name in (
            "taxonomy_fingerprint",
            "primary_model_fingerprint",
            "complementary_model_fingerprint",
            "canonicalizer_model_fingerprint",
            "source_fingerprint",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if int(self.index_version) < 1 or int(self.feature_set_version) < 1:
            raise ValueError("Index and feature-set versions must be positive.")
        if int(self.embedding_dimension) != 384:
            raise ValueError("Module 2 currently requires 384-dimensional indexes.")
        if int(self.expected_document_count) < 1:
            raise ValueError("expected_document_count must be at least 1.")
        if self.activation_requested or self.downstream_export_enabled:
            raise ValueError("Index manifests cannot request activation or export.")
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))

    @property
    def fingerprint(self) -> str:
        return stable_digest(self.to_safe_dict(include_fingerprint=False))

    def to_safe_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        record = asdict(self)
        record["created_at"] = self.created_at.isoformat()
        if include_fingerprint:
            record["manifest_fingerprint"] = self.fingerprint
        return record


@dataclass(frozen=True)
class Module2IndexValidationEvidence:
    manifest_fingerprint: str
    validated_by: str
    validated_at: datetime
    observed_document_count: int
    embedding_dimension: int
    missing_vector_count: int
    invalid_vector_count: int
    duplicate_document_count: int
    tenant_isolation_passed: bool
    model_binding_passed: bool
    taxonomy_binding_passed: bool
    source_binding_passed: bool
    recall_at_10: float
    p95_latency_ms: float
    checksum_sha256: str
    contract_version: str = "module2-governed-index-validation-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_fingerprint", _sha256(self.manifest_fingerprint, "manifest_fingerprint"))
        object.__setattr__(self, "validated_by", _slug(self.validated_by, "validated_by"))
        object.__setattr__(self, "validated_at", _utc(self.validated_at, "validated_at"))
        object.__setattr__(self, "checksum_sha256", _sha256(self.checksum_sha256, "checksum_sha256"))
        for name in (
            "observed_document_count",
            "missing_vector_count",
            "invalid_vector_count",
            "duplicate_document_count",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative.")
        if int(self.embedding_dimension) != 384:
            raise ValueError("Index validation requires 384 dimensions.")
        recall = float(self.recall_at_10)
        latency = float(self.p95_latency_ms)
        if not 0.0 <= recall <= 1.0 or latency <= 0:
            raise ValueError("Index recall and latency values are invalid.")
        object.__setattr__(self, "recall_at_10", recall)
        object.__setattr__(self, "p95_latency_ms", latency)

    @property
    def passed(self) -> bool:
        return (
            self.observed_document_count > 0
            and self.missing_vector_count == 0
            and self.invalid_vector_count == 0
            and self.duplicate_document_count == 0
            and self.tenant_isolation_passed
            and self.model_binding_passed
            and self.taxonomy_binding_passed
            and self.source_binding_passed
            and self.recall_at_10 >= 0.95
            and self.p95_latency_ms <= 500.0
        )

    def to_safe_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["validated_at"] = self.validated_at.isoformat()
        record["passed"] = self.passed
        return record


@dataclass(frozen=True)
class Module2ShadowPolicy:
    minimum_samples: int = 1000
    minimum_agreement_rate: float = 0.95
    maximum_safety_divergence_rate: float = 0.0
    maximum_candidate_error_rate: float = 0.005
    maximum_p95_latency_overhead_ms: float = 250.0
    contract_version: str = "module2-shadow-serving-policy-v2"

    def __post_init__(self) -> None:
        if int(self.minimum_samples) < 1:
            raise ValueError("minimum_samples must be positive.")
        for name in (
            "minimum_agreement_rate",
            "maximum_safety_divergence_rate",
            "maximum_candidate_error_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
            object.__setattr__(self, name, value)
        if float(self.maximum_p95_latency_overhead_ms) < 0:
            raise ValueError("maximum_p95_latency_overhead_ms cannot be negative.")

    @property
    def fingerprint(self) -> str:
        return stable_digest(asdict(self))


@dataclass(frozen=True)
class Module2ShadowObservation:
    tenant_id: str
    request_fingerprint: str
    observed_at: datetime
    incumbent_signature: str
    candidate_signature: str
    incumbent_status: str
    candidate_status: str
    incumbent_latency_ms: float
    candidate_latency_ms: float
    candidate_error: bool
    safety_divergence: bool
    agreement: bool
    raw_query_stored: bool = False
    raw_identifiers_stored: bool = False
    contract_version: str = "module2-shadow-observation-v2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _slug(self.tenant_id, "tenant_id"))
        for name in ("request_fingerprint", "incumbent_signature", "candidate_signature"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "observed_at", _utc(self.observed_at, "observed_at"))
        object.__setattr__(self, "incumbent_status", _slug(self.incumbent_status, "incumbent_status"))
        object.__setattr__(self, "candidate_status", _slug(self.candidate_status, "candidate_status"))
        if float(self.incumbent_latency_ms) < 0 or float(self.candidate_latency_ms) < 0:
            raise ValueError("Shadow latency cannot be negative.")
        if self.raw_query_stored or self.raw_identifiers_stored:
            raise ValueError("Shadow observations cannot store raw queries or identifiers.")

    @property
    def latency_overhead_ms(self) -> float:
        return float(self.candidate_latency_ms) - float(self.incumbent_latency_ms)

    def to_safe_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["observed_at"] = self.observed_at.isoformat()
        record["latency_overhead_ms"] = round(self.latency_overhead_ms, 6)
        return record
