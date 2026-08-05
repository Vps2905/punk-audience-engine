from app.models.production_module3_cohort_contracts import (
    Module3CohortCandidatePolicy,
    Module3CohortGenerationRequest,
)
from app.services.production_module3_cohort_candidate_service import (
    ProductionModule3CohortCandidateService,
)


def feature_set(**overrides):
    value = {
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
    value.update(overrides)
    return value


def row(feature_id, poi="restaurant", cohort_size=2000, **overrides):
    value = {
        "tenant_id": "punk_internal",
        "feature_set_id": "feature_set_v1",
        "feature_set_version": 1,
        "feature_id": feature_id,
        "location_name": "montreal",
        "primary_poi_type": poi,
        "created_day_part": "evening",
        "lookback_bucket": "31_90d",
        "cohort_size": cohort_size,
        "quality_score": 0.8,
        "privacy_status": "passed",
        "rights_status": "historical_internal_only",
        "purpose": "internal_audience_evaluation",
        "source_latest_at": "2026-07-08T08:40:40+00:00",
        "freshness_status": "stale",
        "data_use_mode": "historical_preview",
        "eligible_for_retrieval": True,
        "eligible_for_activation": False,
        "device_id": "must-never-be-returned",
    }
    value.update(overrides)
    return value


def request():
    return Module3CohortGenerationRequest(
        tenant_id="punk_internal",
        feature_set_id="feature_set_v1",
        feature_set_version=1,
        execution_mode="historical_preview",
        purpose="internal_audience_evaluation",
    )


def test_historical_low_risk_candidate_is_explainable_and_not_activatable():
    report = ProductionModule3CohortCandidateService().generate(
        request=request(),
        feature_set=feature_set(),
        feature_rows=[row("feature-1")],
    ).to_record()

    candidate = report["candidates"][0]
    assert report["status"] == "engineering_preview_ready"
    assert candidate["lifecycle_status"] == "historical_preview_only"
    assert candidate["eligible_for_activation"] is False
    assert candidate["eligible_for_export"] is False
    assert candidate["approval_required"] is True
    assert candidate["quality_score"] <= 1.0
    assert "cohort_overlap" in candidate["metric_disclosure"][
        "unavailable_not_inferred"
    ]
    assert "device_id" not in str(report)


def test_below_k_and_unsafe_rows_are_excluded_before_candidate_creation():
    report = ProductionModule3CohortCandidateService().generate(
        request=request(),
        feature_set=feature_set(),
        feature_rows=[
            row("small", cohort_size=999),
            row("unsafe", privacy_status="failed"),
            row("ineligible", eligible_for_retrieval=False),
        ],
    ).to_record()

    assert report["generated_candidate_count"] == 0
    assert report["excluded_below_k_count"] == 1
    assert report["excluded_unsafe_privacy_count"] == 1
    assert report["excluded_ineligible_retrieval_count"] == 1


def test_sensitive_and_regulated_pois_fail_closed_or_require_review():
    report = ProductionModule3CohortCandidateService().generate(
        request=request(),
        feature_set=feature_set(),
        feature_rows=[
            row("hospital", poi="hospital"),
            row("casino", poi="casino"),
        ],
    ).to_record()

    by_poi = {item["primary_poi_type"]: item for item in report["candidates"]}
    assert by_poi["hospital"]["lifecycle_status"] == "blocked_sensitive_poi"
    assert by_poi["casino"]["lifecycle_status"] == (
        "review_required_sensitive_poi"
    )
    assert report["blocked_sensitive_count"] == 1
    assert report["review_required_sensitive_count"] == 1


def test_generation_is_deterministic_and_never_sums_source_cohort_sizes():
    service = ProductionModule3CohortCandidateService()
    rows = [
        row("feature-b", cohort_size=5000),
        row("feature-a", cohort_size=3000),
    ]
    first = service.generate(
        request=request(), feature_set=feature_set(), feature_rows=rows
    ).to_record()
    second = service.generate(
        request=request(), feature_set=feature_set(), feature_rows=list(reversed(rows))
    ).to_record()

    assert first == second
    assert first["batch_fingerprint"] == second["batch_fingerprint"]
    assert sorted(item["cohort_size"] for item in first["candidates"]) == [3000, 5000]
    assert first["safety"]["overlap_or_unique_reach_computed"] is False


def test_max_candidates_selects_highest_quality_and_reports_truncation():
    policy = Module3CohortCandidatePolicy(max_candidates=1)
    report = ProductionModule3CohortCandidateService(policy=policy).generate(
        request=request(),
        feature_set=feature_set(),
        feature_rows=[
            row("low", quality_score=0.1, cohort_size=1000),
            row("high", quality_score=0.9, cohort_size=100000),
        ],
    ).to_record()
    assert report["generated_candidate_count"] == 1
    assert report["truncated_candidate_count"] == 1
    assert report["candidates"][0]["source_feature_id"] == "high"


def test_policy_rejects_k_below_global_privacy_floor():
    try:
        Module3CohortCandidatePolicy(min_cohort_size=999)
    except ValueError as exc:
        assert "min_cohort_size >= 1000" in str(exc)
    else:
        raise AssertionError("Expected policy validation failure")
