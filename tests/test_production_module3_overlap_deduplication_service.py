from copy import deepcopy

import pytest

from app.models.production_module3_cohort_contracts import (
    Module3CohortGenerationRequest,
)
from app.models.production_module3_overlap_contracts import (
    Module3OverlapDeduplicationPolicy,
    Module3OverlapDeduplicationRequest,
)
from app.services.production_module3_cohort_candidate_service import (
    ProductionModule3CohortCandidateService,
)
from app.services.production_module3_overlap_deduplication_service import (
    ProductionModule3OverlapDeduplicationService,
)


def feature_set():
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


def row(feature_id, *, lookback="31_90d", location="montreal", poi="restaurant"):
    return {
        "tenant_id": "punk_internal",
        "feature_set_id": "feature_set_v1",
        "feature_set_version": 1,
        "feature_id": feature_id,
        "location_name": location,
        "primary_poi_type": poi,
        "created_day_part": "evening",
        "lookback_bucket": lookback,
        "cohort_size": 2000,
        "quality_score": 0.8,
        "privacy_status": "passed",
        "rights_status": "historical_internal_only",
        "purpose": "internal_audience_evaluation",
        "source_latest_at": "2026-07-08T08:40:40+00:00",
        "freshness_status": "stale",
        "data_use_mode": "historical_preview",
        "eligible_for_retrieval": True,
        "eligible_for_activation": False,
    }


def candidate_batch(*rows):
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


def overlap_request(batch):
    return Module3OverlapDeduplicationRequest(
        tenant_id="punk_internal",
        batch_fingerprint=batch["batch_fingerprint"],
        execution_mode="historical_preview",
        purpose="internal_audience_evaluation",
    )


def analyze(batch, *, policy=None):
    return ProductionModule3OverlapDeduplicationService(
        policy=policy
    ).analyze(
        request=overlap_request(batch),
        candidate_batch=batch,
    ).to_record()


def test_exact_repeated_candidate_is_suppressed_without_summing_size():
    batch = candidate_batch(row("feature-1"))
    batch["candidates"].append(deepcopy(batch["candidates"][0]))
    batch["generated_candidate_count"] = 2

    report = analyze(batch)

    assert report["source_candidate_count"] == 2
    assert report["retained_candidate_count"] == 1
    assert report["exact_duplicate_group_count"] == 1
    assert report["suppressed_exact_duplicate_occurrence_count"] == 1
    assert report["retained_candidates"][0]["cohort_size"] == 2000
    assert report["safety"]["cohort_sizes_summed"] is False
    assert report["safety"]["unique_reach_computed"] is False


def test_same_constraint_candidates_are_reviewed_but_not_suppressed():
    batch = candidate_batch(row("feature-a"), row("feature-b"))

    report = analyze(batch)

    constraint_groups = [
        group
        for group in report["overlap_groups"]
        if group["group_type"] == "potential_constraint_overlap"
    ]
    assert report["retained_candidate_count"] == 2
    assert report["suppressed_exact_duplicate_occurrence_count"] == 0
    assert len(constraint_groups) == 1
    assert constraint_groups[0]["candidate_count"] == 2
    assert constraint_groups[0]["review_required"] is True
    assert constraint_groups[0]["overlap_estimate_available"] is False
    assert report["safety"]["potential_overlap_candidates_suppressed"] is False


def test_different_lookbacks_create_temporal_overlap_review_only():
    batch = candidate_batch(
        row("feature-short", lookback="0_30d"),
        row("feature-long", lookback="31_90d"),
    )

    report = analyze(batch)

    temporal_groups = [
        group
        for group in report["overlap_groups"]
        if group["group_type"] == "potential_temporal_overlap"
    ]
    assert len(temporal_groups) == 1
    assert temporal_groups[0]["candidate_count"] == 2
    assert temporal_groups[0]["unique_reach_claimed"] is False
    assert report["safety"]["membership_intersection_read"] is False
    assert report["safety"]["overlap_rate_computed"] is False


def test_unrelated_candidates_are_not_grouped():
    batch = candidate_batch(
        row("montreal-restaurant"),
        row("toronto-cafe", location="toronto", poi="cafe"),
    )

    report = analyze(batch)

    assert report["potential_overlap_group_count"] == 0
    assert report["candidates_requiring_overlap_review_count"] == 0


def test_conflicting_candidate_identity_fails_closed():
    batch = candidate_batch(row("feature-1"))
    conflict = deepcopy(batch["candidates"][0])
    conflict["candidate_fingerprint"] = "f" * 64
    batch["candidates"].append(conflict)
    batch["generated_candidate_count"] = 2

    with pytest.raises(
        ValueError,
        match="Conflicting fingerprints share a candidate identity",
    ):
        analyze(batch)


def test_raw_identifier_or_release_eligible_candidate_fails_closed():
    raw_batch = candidate_batch(row("feature-1"))
    raw_batch["candidates"][0]["device_id"] = "forbidden"
    with pytest.raises(ValueError, match="prohibited raw identifier field"):
        analyze(raw_batch)

    release_batch = candidate_batch(row("feature-1"))
    release_batch["candidates"][0]["eligible_for_export"] = True
    with pytest.raises(ValueError, match="export-eligible"):
        analyze(release_batch)


def test_analysis_is_deterministic_across_candidate_order():
    batch = candidate_batch(
        row("feature-a"),
        row("feature-b"),
        row("feature-c", lookback="0_30d"),
    )
    reversed_batch = deepcopy(batch)
    reversed_batch["candidates"] = list(reversed(reversed_batch["candidates"]))

    first = analyze(batch)
    second = analyze(reversed_batch)

    assert first == second
    assert first["report_fingerprint"] == second["report_fingerprint"]


def test_input_and_group_bounds_fail_closed():
    batch = candidate_batch(row("feature-a"), row("feature-b"))
    with pytest.raises(ValueError, match="max_input_candidates"):
        analyze(
            batch,
            policy=Module3OverlapDeduplicationPolicy(max_input_candidates=1),
        )

    three_member_batch = candidate_batch(
        row("feature-a"),
        row("feature-b"),
        row("feature-c"),
    )
    with pytest.raises(ValueError, match="max_group_members"):
        analyze(
            three_member_batch,
            policy=Module3OverlapDeduplicationPolicy(max_group_members=2),
        )


def test_report_validator_detects_tampering_and_unsafe_flags():
    batch = candidate_batch(row("feature-a"), row("feature-b"))
    service = ProductionModule3OverlapDeduplicationService()
    report = service.analyze(
        request=overlap_request(batch),
        candidate_batch=batch,
    ).to_record()

    assert service.validate_report(report) == report

    tampered = deepcopy(report)
    tampered["retained_candidates"][0]["quality_score"] = 0.01
    with pytest.raises(ValueError, match="report fingerprint mismatch"):
        service.validate_report(tampered)

    unsafe = deepcopy(report)
    unsafe["safety"]["overlap_rate_computed"] = True
    with pytest.raises(ValueError, match="Unsafe Module 3.3 report safety field"):
        service.validate_report(unsafe)
