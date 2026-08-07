from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from app.models.production_module3_cohort_contracts import (
    Module3CohortGenerationRequest,
)
from app.models.production_module3_lookalike_contracts import (
    Module3LookalikePolicy,
    Module3LookalikeRequest,
)
from app.models.production_module3_overlap_contracts import (
    Module3OverlapDeduplicationRequest,
)
from app.services.production_module3_cohort_candidate_service import (
    ProductionModule3CohortCandidateService,
)
from app.services.production_module3_lookalike_service import (
    ProductionModule3LookalikeService,
)
from app.services.production_module3_overlap_deduplication_service import (
    ProductionModule3OverlapDeduplicationService,
)


def feature_set() -> dict:
    return {
        "tenant_id": "punk_internal",
        "feature_set_id": "feature_set_v1",
        "version": 1,
        "data_use_mode": "historical_preview",
        "source_fingerprint": "source-fingerprint",
        "freshness_status": "stale",
        "privacy_policy_version": "privacy-v1",
        "rights_policy_id": "rights-v1",
        "eligible_for_retrieval": True,
        "eligible_for_activation": False,
    }


def row(
    feature_id: str,
    *,
    location: str,
    poi: str = "restaurant",
    daypart: str = "evening",
    lookback: str = "31_90d",
    quality: float = 0.90,
) -> dict:
    return {
        "tenant_id": "punk_internal",
        "feature_set_id": "feature_set_v1",
        "feature_set_version": 1,
        "feature_id": feature_id,
        "location_name": location,
        "primary_poi_type": poi,
        "created_day_part": daypart,
        "lookback_bucket": lookback,
        "cohort_size": 2500,
        "quality_score": quality,
        "privacy_status": "passed",
        "rights_status": "historical_internal_only",
        "purpose": "internal_audience_evaluation",
        "source_latest_at": "2026-07-08T08:40:40+00:00",
        "freshness_status": "stale",
        "data_use_mode": "historical_preview",
        "eligible_for_retrieval": True,
        "eligible_for_activation": False,
    }


def candidate_batch(*rows: dict) -> dict:
    request = Module3CohortGenerationRequest(
        tenant_id="punk_internal",
        feature_set_id="feature_set_v1",
        feature_set_version=1,
        execution_mode="historical_preview",
        purpose="internal_audience_evaluation",
    )

    return ProductionModule3CohortCandidateService().generate(
        request=request,
        feature_set=feature_set(),
        feature_rows=list(rows),
    ).to_record()


def overlap_report(*rows: dict) -> dict:
    batch = candidate_batch(*rows)

    request = Module3OverlapDeduplicationRequest(
        tenant_id="punk_internal",
        batch_fingerprint=batch["batch_fingerprint"],
        execution_mode="historical_preview",
        purpose="internal_audience_evaluation",
    )

    return ProductionModule3OverlapDeduplicationService().analyze(
        request=request,
        candidate_batch=batch,
    ).to_record()


def lookalike_request(report: dict) -> Module3LookalikeRequest:
    return Module3LookalikeRequest(
        tenant_id="punk_internal",
        overlap_report_fingerprint=report["report_fingerprint"],
        execution_mode="historical_preview",
        purpose="internal_audience_evaluation",
    )


def generate(
    report: dict,
    *,
    policy: Module3LookalikePolicy | None = None,
) -> dict:
    return ProductionModule3LookalikeService(
        policy=policy
    ).generate(
        request=lookalike_request(report),
        overlap_report=report,
    ).to_record()


def test_generates_cross_location_review_pairs_without_membership_generation():
    report = overlap_report(
        row("feature-montreal", location="montreal"),
        row("feature-toronto", location="toronto"),
    )

    result = generate(report)

    assert result["status"] == "engineering_preview_ready"
    assert result["eligible_seed_count"] == 2
    assert result["generated_lookalike_candidate_count"] == 2

    for pair in result["lookalike_candidates"]:
        assert pair["review_required"] is True
        assert pair["membership_generated"] is False
        assert pair["eligible_for_activation"] is False
        assert pair["eligible_for_export"] is False
        assert 0.0 <= pair["similarity_score"] <= 1.0

    assert result["safety"]["audience_membership_read"] is False
    assert result["safety"]["membership_similarity_computed"] is False
    assert result["safety"]["audience_membership_generated"] is False
    assert result["safety"]["overlap_rate_computed"] is False
    assert result["safety"]["unique_reach_claimed"] is False
    assert result["safety"]["cohort_sizes_summed"] is False
    assert result["safety"]["activation_or_export_performed"] is False


def test_generation_is_deterministic_across_source_order():
    first = overlap_report(
        row("feature-montreal", location="montreal"),
        row("feature-toronto", location="toronto"),
        row("feature-vancouver", location="vancouver"),
    )
    second = overlap_report(
        row("feature-vancouver", location="vancouver"),
        row("feature-montreal", location="montreal"),
        row("feature-toronto", location="toronto"),
    )

    first_result = generate(first)
    second_result = generate(second)

    assert first["report_fingerprint"] == second["report_fingerprint"]
    assert first_result["report_fingerprint"] == second_result["report_fingerprint"]
    assert first_result["lookalike_candidates"] == (
        second_result["lookalike_candidates"]
    )


def test_candidates_requiring_overlap_review_are_excluded():
    report = overlap_report(
        row(
            "feature-a",
            location="montreal",
            lookback="0_30d",
        ),
        row(
            "feature-b",
            location="montreal",
            lookback="31_90d",
        ),
        row(
            "feature-c",
            location="toronto",
            lookback="31_90d",
        ),
    )

    result = generate(report)

    assert report["potential_overlap_group_count"] >= 1
    assert result["excluded_overlap_review_candidate_count"] == 2
    assert result["eligible_seed_count"] == 1
    assert result["generated_lookalike_candidate_count"] == 0


def test_different_poi_type_is_not_a_lookalike_target_by_default():
    report = overlap_report(
        row(
            "feature-restaurant",
            location="montreal",
            poi="restaurant",
        ),
        row(
            "feature-gym",
            location="toronto",
            poi="gym",
        ),
    )

    result = generate(report)

    assert result["generated_lookalike_candidate_count"] == 0


def test_low_quality_historical_candidates_remain_excluded():
    report = overlap_report(
        row(
            "feature-low-montreal",
            location="montreal",
            quality=0.10,
        ),
        row(
            "feature-low-toronto",
            location="toronto",
            quality=0.10,
        ),
    )

    result = generate(report)

    assert result["eligible_seed_count"] == 0
    assert result["excluded_policy_candidate_count"] == 2
    assert result["generated_lookalike_candidate_count"] == 0
    assert result["safety"]["activation_or_export_performed"] is False


def test_review_policy_version_and_thresholds_are_explicit():
    policy = Module3LookalikePolicy()

    assert policy.policy_version == "module3_governed_lookalike_policy_v2"
    assert policy.min_seed_quality_score == 0.60
    assert policy.min_target_quality_score == 0.60


def test_source_report_fingerprint_mismatch_fails_closed():
    report = overlap_report(
        row("feature-montreal", location="montreal"),
        row("feature-toronto", location="toronto"),
    )

    request = Module3LookalikeRequest(
        tenant_id="punk_internal",
        overlap_report_fingerprint="0" * 64,
        execution_mode="historical_preview",
        purpose="internal_audience_evaluation",
    )

    with pytest.raises(ValueError, match="fingerprint"):
        ProductionModule3LookalikeService().generate(
            request=request,
            overlap_report=report,
        )


def test_raw_identifier_field_fails_closed():
    report = overlap_report(
        row("feature-montreal", location="montreal"),
        row("feature-toronto", location="toronto"),
    )
    unsafe = deepcopy(report)
    unsafe["retained_candidates"][0]["device_id"] = "forbidden"

    with pytest.raises(ValueError, match="prohibited raw identifier"):
        ProductionModule3LookalikeService().generate(
            request=lookalike_request(report),
            overlap_report=unsafe,
        )


def test_generated_report_round_trip_validation():
    report = overlap_report(
        row("feature-montreal", location="montreal"),
        row("feature-toronto", location="toronto"),
    )

    result = generate(report)

    validated = ProductionModule3LookalikeService().validate_report(result)

    assert validated == result


def test_report_tampering_fails_fingerprint_validation():
    report = overlap_report(
        row("feature-montreal", location="montreal"),
        row("feature-toronto", location="toronto"),
    )

    result = generate(report)
    result["eligible_seed_count"] += 1

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ProductionModule3LookalikeService().validate_report(result)


def test_module3_4_migration_is_review_only_and_tenant_scoped():
    sql = Path(
        "migrations/0016_module3_governed_lookalike_evidence.sql"
    ).read_text(encoding="utf-8")

    required_fragments = (
        "raw_identifiers_stored = FALSE",
        "audience_membership_read = FALSE",
        "membership_similarity_computed = FALSE",
        "audience_membership_generated = FALSE",
        "overlap_rate_computed = FALSE",
        "unique_reach_claimed = FALSE",
        "cohort_sizes_summed = FALSE",
        "candidate_lifecycle_mutated = FALSE",
        "activation_or_export_performed = FALSE",
        "review_required = TRUE",
        "membership_generated = FALSE",
        "eligible_for_activation = FALSE",
        "eligible_for_export = FALSE",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "prevent_module3_lookalike_evidence_mutation",
        "validate_module3_lookalike_candidate_fingerprints",
        "REVOKE ALL",
    )

    for fragment in required_fragments:
        assert fragment in sql


def test_module3_status_accepts_safe_lookalike_evidence(tmp_path):
    import json

    from app.services.production_module3_status_service import (
        ProductionModule3StatusService,
    )

    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "status": "engineering_preview_ready",
                "safety": {
                    "raw_identifiers_returned": False,
                    "activation_or_export_performed": False,
                },
            }
        ),
        encoding="utf-8",
    )

    overlap_path = tmp_path / "overlap.json"
    overlap_path.write_text(
        json.dumps(
            {
                "status": "engineering_preview_ready",
                "safety": {
                    "raw_identifiers_returned": False,
                    "membership_intersection_read": False,
                    "overlap_rate_computed": False,
                    "unique_reach_claimed": False,
                    "cohort_sizes_summed": False,
                    "candidate_lifecycle_mutated": False,
                    "activation_or_export_performed": False,
                },
            }
        ),
        encoding="utf-8",
    )

    lookalike_path = tmp_path / "lookalike.json"
    lookalike_path.write_text(
        json.dumps(
            {
                "status": "engineering_preview_ready",
                "safety": {
                    "raw_identifiers_returned": False,
                    "audience_membership_read": False,
                    "membership_similarity_computed": False,
                    "audience_membership_generated": False,
                    "overlap_rate_computed": False,
                    "unique_reach_claimed": False,
                    "cohort_sizes_summed": False,
                    "candidate_lifecycle_mutated": False,
                    "activation_or_export_performed": False,
                    "downstream_export_enabled": False,
                },
            }
        ),
        encoding="utf-8",
    )

    status = ProductionModule3StatusService(
        environment={
            "MODULE3_COHORT_EVIDENCE_PATH": str(candidate_path),
            "MODULE3_OVERLAP_EVIDENCE_PATH": str(overlap_path),
            "MODULE3_LOOKALIKE_EVIDENCE_PATH": str(lookalike_path),
        }
    ).status()

    assert status["status"] == "module3_4_engineering_evidence_ready"
    assert status["module3_4_engineering_evidence_ready"] is True
    assert (
        status["components"]["module_3_4_governed_lookalike_generation"]
        is True
    )
    assert status["feature_flags"]["lookalike_generation_enabled"] is False
