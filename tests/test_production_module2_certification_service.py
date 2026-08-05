from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.models.audience_feature_contracts import stable_digest
from app.models.production_module2_completion_contracts import (
    NativeLanguageReviewDecision,
    NativeLanguageReviewManifest,
    UnsupportedCalibrationDataset,
    UnsupportedCalibrationObservation,
)
from app.services.production_module2_certification_service import (
    ProductionModule2CertificationService,
    minimum_zero_failure_sample_size,
    zero_failure_upper_confidence_bound,
)
from tests.test_production_governed_retrieval_benchmark_service import (
    _dataset,
    _service,
)

NOW = datetime(2026, 8, 5, tzinfo=timezone.utc)
LANGUAGES = ("en", "fr", "es", "hi", "bn")


def _review_manifest(taxonomy_fingerprint: str):
    pack_sha = "a" * 64
    decisions = tuple(
        NativeLanguageReviewDecision(
            language=language,
            reviewer_id=f"native_reviewer_{language}",
            reviewer_native_language_confirmed=True,
            taxonomy_fingerprint=taxonomy_fingerprint,
            language_pack_sha256=pack_sha,
            reviewed_alias_count=43,
            decision="approved",
            conflict_count=2,
            unresolved_conflict_count=0,
            reviewed_at=NOW,
            notes_fingerprint=stable_digest({"language": language}),
        )
        for language in LANGUAGES
    )
    return NativeLanguageReviewManifest(
        manifest_id="module2-native-review",
        manifest_version="v1",
        taxonomy_fingerprint=taxonomy_fingerprint,
        language_pack_sha256=pack_sha,
        required_languages=LANGUAGES,
        decisions=decisions,
        release_owner="module2_release_owner",
        review_status="approved",
        created_at=NOW,
        finalized_at=NOW,
    )


def _calibration(taxonomy_fingerprint: str, count: int = 299, failure_at=None):
    observations = []
    for index in range(count):
        language = LANGUAGES[index % len(LANGUAGES)]
        observations.append(
            UnsupportedCalibrationObservation(
                case_id=f"unsupported_{index:04d}",
                language=language,
                location_token_fingerprint=stable_digest(
                    {"location_case": index, "language": language}
                ),
                taxonomy_fingerprint=taxonomy_fingerprint,
                expected_rejection=True,
                observed_retrieval_ready=index == failure_at,
                observed_reason_code="location_requires_clarification",
                reviewed_by=f"calibration_reviewer_{language}",
                reviewed_at=NOW,
            )
        )
    return UnsupportedCalibrationDataset(
        dataset_id="module2-unsupported-calibration",
        dataset_version=f"v1-{count}",
        taxonomy_fingerprint=taxonomy_fingerprint,
        observations=tuple(observations),
        review_status="approved",
        reviewed_by="calibration_release_owner",
        reviewed_at=NOW,
    )


def test_exact_zero_failure_sample_math_matches_299_case_gate():
    assert minimum_zero_failure_sample_size(
        target_rate=0.01,
        confidence_level=0.95,
    ) == 299
    assert zero_failure_upper_confidence_bound(
        sample_count=299,
        confidence_level=0.95,
    ) <= 0.01
    assert zero_failure_upper_confidence_bound(
        sample_count=298,
        confidence_level=0.95,
    ) > 0.01


def test_certification_evidence_passes_but_release_stays_blocked_by_external_gates():
    benchmark = _service().evaluate(_dataset())
    taxonomy_fingerprint = benchmark["taxonomy"]["taxonomy_fingerprint"]

    report = ProductionModule2CertificationService().evaluate(
        benchmark_report=benchmark,
        native_review_manifest=_review_manifest(taxonomy_fingerprint),
        calibration_dataset=_calibration(taxonomy_fingerprint),
    )

    assert report["engineering_passed"] is True
    assert report["module2_evidence_ready"] is True
    assert report["production_certification_ready"] is False
    assert report["unsupported_calibration"]["sample_count"] == 299
    assert report["unsupported_calibration"]["failure_count"] == 0
    assert report["model_registration_performed"] is False
    assert report["production_routing_enabled"] is False
    assert report["activation_or_export_performed"] is False


def test_calibration_fails_with_298_cases_or_one_false_match():
    benchmark = _service().evaluate(_dataset())
    taxonomy_fingerprint = benchmark["taxonomy"]["taxonomy_fingerprint"]
    service = ProductionModule2CertificationService()

    insufficient = service.evaluate(
        benchmark_report=benchmark,
        native_review_manifest=_review_manifest(taxonomy_fingerprint),
        calibration_dataset=_calibration(taxonomy_fingerprint, 298),
    )
    assert insufficient["module2_evidence_ready"] is False
    assert "unsupported_calibration_sample_insufficient" in insufficient[
        "reason_codes"
    ]

    failed = service.evaluate(
        benchmark_report=benchmark,
        native_review_manifest=_review_manifest(taxonomy_fingerprint),
        calibration_dataset=_calibration(taxonomy_fingerprint, 299, failure_at=4),
    )
    assert failed["module2_evidence_ready"] is False
    assert failed["unsupported_calibration"]["failure_count"] == 1


def test_release_owner_cannot_self_approve_language_review():
    taxonomy = "b" * 64
    decision = NativeLanguageReviewDecision(
        language="en",
        reviewer_id="release_owner",
        reviewer_native_language_confirmed=True,
        taxonomy_fingerprint=taxonomy,
        language_pack_sha256="a" * 64,
        reviewed_alias_count=10,
        decision="approved",
        conflict_count=0,
        unresolved_conflict_count=0,
        reviewed_at=NOW,
    )
    with pytest.raises(ValueError, match="self-approve"):
        NativeLanguageReviewManifest(
            manifest_id="review",
            manifest_version="v1",
            taxonomy_fingerprint=taxonomy,
            language_pack_sha256="a" * 64,
            required_languages=("en",),
            decisions=(decision,),
            release_owner="release_owner",
            review_status="approved",
            created_at=NOW,
            finalized_at=NOW,
        )
