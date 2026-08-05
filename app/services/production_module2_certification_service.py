from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

from app.models.audience_feature_contracts import stable_digest
from app.models.production_module2_completion_contracts import (
    Module2CertificationPolicy,
    NativeLanguageReviewManifest,
    UnsupportedCalibrationDataset,
)
from app.services.production_governed_retrieval_benchmark_service import (
    validate_governed_retrieval_benchmark_report,
)

_FORBIDDEN_KEYS = {
    "query",
    "query_text",
    "raw_query",
    "document_text",
    "embedding",
    "embeddings",
    "maid",
    "maids",
    "device_id",
    "email",
    "phone",
    "latitude",
    "longitude",
}


def minimum_zero_failure_sample_size(
    *,
    target_rate: float,
    confidence_level: float,
) -> int:
    target = float(target_rate)
    confidence = float(confidence_level)
    if not 0.0 < target < 1.0:
        raise ValueError("target_rate must be between 0 and 1.")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence_level must be between 0.5 and 1.")
    return int(math.ceil(math.log(1.0 - confidence) / math.log(1.0 - target)))


def zero_failure_upper_confidence_bound(
    *,
    sample_count: int,
    confidence_level: float,
) -> float:
    count = int(sample_count)
    confidence = float(confidence_level)
    if count < 1:
        return 1.0
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence_level must be between 0.5 and 1.")
    alpha = 1.0 - confidence
    return 1.0 - math.pow(alpha, 1.0 / count)


class ProductionModule2CertificationService:
    """Combine engineering, human-review, and calibration evidence safely.

    The service never performs model registration, index activation, production
    routing, proposal creation, audience activation, or export.
    """

    def __init__(
        self,
        *,
        policy: Module2CertificationPolicy | None = None,
    ) -> None:
        self._policy = policy or Module2CertificationPolicy()

    def plan_native_language_review(
        self,
        *,
        taxonomy_payload: Mapping[str, Any],
        language_pack_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        taxonomy_fingerprint = str(
            taxonomy_payload.get("taxonomy_fingerprint") or ""
        ).strip().lower()
        language_pack_sha256 = str(
            taxonomy_payload.get("lineage", {}).get(
                "source_language_pack_sha256"
            )
            or ""
        ).strip().lower()
        languages = language_pack_payload.get("languages") or {}
        if not isinstance(languages, Mapping):
            raise ValueError("language pack languages must be an object.")
        required = self._policy.required_languages
        missing = sorted(set(required).difference(languages))
        if missing:
            raise ValueError(
                "Language pack is missing required review languages: "
                + ", ".join(missing)
            )
        worksheets = []
        for language in required:
            payload = languages[language]
            category_labels = dict(payload.get("category_labels") or {})
            daypart_labels = dict(payload.get("daypart_labels") or {})
            alias_count = len(category_labels) + len(daypart_labels)
            if alias_count < 1:
                raise ValueError(
                    f"Language {language} has no reviewable aliases."
                )
            worksheets.append(
                {
                    "language": language,
                    "review_status": "pending_native_human_review",
                    "reviewer_id": None,
                    "reviewer_native_language_confirmed": False,
                    "reviewed_alias_count": alias_count,
                    "category_label_count": len(category_labels),
                    "daypart_label_count": len(daypart_labels),
                    "conflict_count": None,
                    "unresolved_conflict_count": None,
                    "decision": None,
                    "reviewed_at": None,
                    "machine_translation_auto_approved": False,
                    "raw_identifiers_reviewed": False,
                }
            )
        plan = {
            "contract_version": "module2-native-language-review-plan-v1",
            "taxonomy_fingerprint": taxonomy_fingerprint,
            "language_pack_sha256": language_pack_sha256,
            "required_languages": list(required),
            "worksheets": worksheets,
            "release_owner": None,
            "review_status": "pending",
            "production_certification_ready": False,
            "model_registration_performed": False,
            "production_routing_enabled": False,
            "activation_or_export_performed": False,
        }
        plan["plan_fingerprint"] = stable_digest(plan)
        return plan

    def evaluate(
        self,
        *,
        benchmark_report: Mapping[str, Any],
        native_review_manifest: NativeLanguageReviewManifest,
        calibration_dataset: UnsupportedCalibrationDataset,
        external_gates: Mapping[str, bool] | None = None,
        model_registration_approved: bool = False,
        index_release_ready: bool = False,
        shadow_release_ready: bool = False,
    ) -> dict[str, Any]:
        validate_governed_retrieval_benchmark_report(benchmark_report)
        safe_benchmark = _json_safe(benchmark_report)
        benchmark_taxonomy = str(
            safe_benchmark.get("taxonomy", {}).get("taxonomy_fingerprint")
            or ""
        )
        if benchmark_taxonomy != native_review_manifest.taxonomy_fingerprint:
            raise ValueError("Benchmark and native-review taxonomy mismatch.")
        if benchmark_taxonomy != calibration_dataset.taxonomy_fingerprint:
            raise ValueError("Benchmark and calibration taxonomy mismatch.")
        if set(native_review_manifest.required_languages) != set(
            self._policy.required_languages
        ):
            raise ValueError("Native-review language coverage does not match policy.")

        observations = calibration_dataset.observations
        sample_count = len(observations)
        failures = sum(observation.failure for observation in observations)
        counts = Counter(observation.language for observation in observations)
        minimum_cases = minimum_zero_failure_sample_size(
            target_rate=self._policy.target_unsupported_false_match_rate,
            confidence_level=self._policy.confidence_level,
        )
        upper_bound = (
            zero_failure_upper_confidence_bound(
                sample_count=sample_count,
                confidence_level=self._policy.confidence_level,
            )
            if failures == 0
            else 1.0
        )
        per_language_complete = all(
            counts.get(language, 0) >= self._policy.minimum_cases_per_language
            for language in self._policy.required_languages
        )
        calibration_passed = (
            calibration_dataset.review_status == "approved"
            and failures == 0
            and sample_count >= minimum_cases
            and upper_bound
            <= self._policy.target_unsupported_false_match_rate
            and per_language_complete
        )
        native_review_passed = native_review_manifest.approved
        engineering_passed = bool(safe_benchmark.get("engineering_passed"))
        module2_evidence_ready = (
            engineering_passed
            and native_review_passed
            and calibration_passed
        )

        supplied_external = {
            str(key): bool(value)
            for key, value in (external_gates or {}).items()
        }
        external_gate_status = {
            gate: bool(supplied_external.get(gate, False))
            for gate in self._policy.required_external_gates
        }
        external_gates_passed = all(external_gate_status.values())
        production_certification_ready = (
            module2_evidence_ready
            and bool(model_registration_approved)
            and bool(index_release_ready)
            and bool(shadow_release_ready)
            and external_gates_passed
        )

        reason_codes: list[str] = []
        if not engineering_passed:
            reason_codes.append("engineering_benchmark_not_passed")
        if not native_review_passed:
            reason_codes.append("native_language_review_pending")
        if calibration_dataset.review_status != "approved":
            reason_codes.append("unsupported_calibration_review_pending")
        if sample_count < minimum_cases:
            reason_codes.append("unsupported_calibration_sample_insufficient")
        if failures:
            reason_codes.append("unsupported_calibration_failures_observed")
        if not per_language_complete:
            reason_codes.append("unsupported_calibration_language_coverage_incomplete")
        if not model_registration_approved:
            reason_codes.append("model_registration_approval_pending")
        if not index_release_ready:
            reason_codes.append("governed_index_release_pending")
        if not shadow_release_ready:
            reason_codes.append("shadow_release_evidence_pending")
        for gate, passed in external_gate_status.items():
            if not passed:
                reason_codes.append(f"external_gate_pending_{gate}")

        report = {
            "contract_version": "module2-production-certification-report-v1",
            "status": (
                "production_certification_ready"
                if production_certification_ready
                else "certification_blocked"
            ),
            "engineering_passed": engineering_passed,
            "module2_evidence_ready": module2_evidence_ready,
            "production_certification_ready": production_certification_ready,
            "benchmark_report_fingerprint": safe_benchmark[
                "report_fingerprint"
            ],
            "taxonomy_fingerprint": benchmark_taxonomy,
            "native_review": {
                "passed": native_review_passed,
                "manifest_fingerprint": native_review_manifest.fingerprint,
                "required_languages": list(
                    native_review_manifest.required_languages
                ),
            },
            "unsupported_calibration": {
                "passed": calibration_passed,
                "dataset_fingerprint": calibration_dataset.fingerprint,
                "sample_count": sample_count,
                "failure_count": failures,
                "minimum_zero_failure_cases": minimum_cases,
                "target_false_match_rate": (
                    self._policy.target_unsupported_false_match_rate
                ),
                "confidence_level": self._policy.confidence_level,
                "zero_failure_upper_confidence_bound": round(
                    upper_bound,
                    12,
                ),
                "minimum_cases_per_language": (
                    self._policy.minimum_cases_per_language
                ),
                "per_language_counts": {
                    language: counts.get(language, 0)
                    for language in self._policy.required_languages
                },
                "per_language_complete": per_language_complete,
            },
            "release_dependencies": {
                "model_registration_approved": bool(
                    model_registration_approved
                ),
                "index_release_ready": bool(index_release_ready),
                "shadow_release_ready": bool(shadow_release_ready),
                "external_gates": external_gate_status,
                "external_gates_passed": external_gates_passed,
            },
            "reason_codes": reason_codes,
            "policy": self._policy.to_safe_dict(),
            "raw_query_text_stored": False,
            "document_text_stored": False,
            "embeddings_stored": False,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
            "model_registration_performed": False,
            "production_routing_enabled": False,
            "automatic_proposal_creation_enabled": False,
            "activation_or_export_performed": False,
            "downstream_export_enabled": False,
        }
        report["report_fingerprint"] = stable_digest(report)
        return validate_module2_certification_report(report)


def validate_module2_certification_report(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    safe = _json_safe(payload)
    declared = str(safe.pop("report_fingerprint", "")).strip().lower()
    expected = stable_digest(safe)
    if declared != expected:
        raise ValueError("Module 2 certification report fingerprint mismatch.")
    for path, value in _walk(safe):
        terminal = path.rsplit(".", 1)[-1].lower()
        if terminal in _FORBIDDEN_KEYS and value not in (False, None, [], {}):
            raise ValueError(
                f"Module 2 certification report contains forbidden field {path}."
            )
    for flag in (
        "model_registration_performed",
        "production_routing_enabled",
        "automatic_proposal_creation_enabled",
        "activation_or_export_performed",
        "downstream_export_enabled",
    ):
        if safe.get(flag) is not False:
            raise ValueError(f"Certification evaluation must keep {flag} disabled.")
    safe["report_fingerprint"] = declared
    return safe


def _walk(value: Any, prefix: str = ""):
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def _json_safe(value: Mapping[str, Any]) -> dict[str, Any]:
    import json

    return json.loads(json.dumps(dict(value), allow_nan=False))
