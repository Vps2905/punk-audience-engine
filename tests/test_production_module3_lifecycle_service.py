from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.models.production_module3_cohort_contracts import (
    Module3CohortGenerationRequest,
)
from app.models.production_module3_lifecycle_contracts import (
    Module3LifecycleEvaluationRequest,
    Module3LifecyclePolicy,
)
from app.models.production_module3_lookalike_contracts import (
    Module3LookalikeRequest,
)
from app.models.production_module3_overlap_contracts import (
    Module3OverlapDeduplicationRequest,
)
from app.services.production_module3_cohort_candidate_service import (
    ProductionModule3CohortCandidateService,
)
from app.services.production_module3_lifecycle_service import (
    ProductionModule3LifecycleService,
)
from app.services.production_module3_lookalike_service import (
    ProductionModule3LookalikeService,
)
from app.services.production_module3_overlap_deduplication_service import (
    ProductionModule3OverlapDeduplicationService,
)


TENANT_ID = "punk_internal"
PURPOSE = "internal_audience_evaluation"


def feature_set(*, mode="historical_preview", version=1):
    return {
        "tenant_id": TENANT_ID,
        "feature_set_id": "feature_set_v1",
        "version": version,
        "data_use_mode": mode,
        "source_fingerprint": f"source-{version}",
        "freshness_status": (
            "fresh" if mode == "production" else "stale"
        ),
        "privacy_policy_version": "privacy-v1",
        "rights_policy_id": "rights-v1",
        "eligible_for_retrieval": True,
        "eligible_for_activation": mode == "production",
    }


def row(
    feature_id,
    *,
    location="montreal",
    poi="cafe",
    mode="historical_preview",
    version=1,
    quality=0.90,
    freshness=None,
    rights=None,
):
    return {
        "tenant_id": TENANT_ID,
        "feature_set_id": "feature_set_v1",
        "feature_set_version": version,
        "feature_id": feature_id,
        "location_name": location,
        "primary_poi_type": poi,
        "created_day_part": "evening",
        "lookback_bucket": "31_90d",
        "cohort_size": 5000,
        "quality_score": quality,
        "privacy_status": "passed",
        "rights_status": rights or (
            "permitted"
            if mode == "production"
            else "historical_internal_only"
        ),
        "purpose": PURPOSE,
        "source_latest_at": "2026-08-07T10:00:00+00:00",
        "freshness_status": freshness or (
            "fresh" if mode == "production" else "stale"
        ),
        "data_use_mode": mode,
        "eligible_for_retrieval": True,
        "eligible_for_activation": mode == "production",
    }


def evidence(*rows, mode="historical_preview", version=1):
    candidate_request = Module3CohortGenerationRequest(
        tenant_id=TENANT_ID,
        feature_set_id="feature_set_v1",
        feature_set_version=version,
        execution_mode=mode,
        purpose=PURPOSE,
    )
    batch = ProductionModule3CohortCandidateService().generate(
        request=candidate_request,
        feature_set=feature_set(mode=mode, version=version),
        feature_rows=list(rows),
    ).to_record()
    overlap = ProductionModule3OverlapDeduplicationService().analyze(
        request=Module3OverlapDeduplicationRequest(
            tenant_id=TENANT_ID,
            batch_fingerprint=batch["batch_fingerprint"],
            execution_mode=mode,
            purpose=PURPOSE,
        ),
        candidate_batch=batch,
    ).to_record()
    lookalike = ProductionModule3LookalikeService().generate(
        request=Module3LookalikeRequest(
            tenant_id=TENANT_ID,
            overlap_report_fingerprint=overlap["report_fingerprint"],
            execution_mode=mode,
            purpose=PURPOSE,
        ),
        overlap_report=overlap,
    ).to_record()
    return overlap, lookalike


def evaluate(
    overlap,
    lookalike,
    *,
    mode="historical_preview",
    previous=None,
    policy=None,
):
    return ProductionModule3LifecycleService(policy=policy).evaluate(
        request=Module3LifecycleEvaluationRequest(
            tenant_id=TENANT_ID,
            overlap_report_fingerprint=overlap["report_fingerprint"],
            lookalike_report_fingerprint=lookalike["report_fingerprint"],
            execution_mode=mode,
            purpose=PURPOSE,
        ),
        overlap_report=overlap,
        lookalike_report=lookalike,
        previous_lifecycle_report=previous,
    ).to_record()


def test_historical_candidates_remain_preview_only():
    overlap, lookalike = evidence(row("feature-1"))

    report = evaluate(overlap, lookalike)

    assert report["status"] == "engineering_preview_ready"
    assert report["historical_preview_count"] == 1
    assert report["recommendations"][0][
        "recommended_lifecycle_status"
    ] == "historical_preview_only"
    assert report["safety"]["candidate_lifecycle_mutated"] is False
    assert report["safety"]["activation_or_export_performed"] is False


def test_clean_production_candidate_enters_manual_shadow_review_only():
    overlap, lookalike = evidence(
        row(
            "feature-montreal",
            location="montreal",
            mode="production",
        ),
        row(
            "feature-toronto",
            location="toronto",
            mode="production",
        ),
        mode="production",
    )

    report = evaluate(overlap, lookalike, mode="production")

    assert report["shadow_review_pending_count"] == 2
    for recommendation in report["recommendations"]:
        assert recommendation["manual_approval_required"] is True
        assert recommendation["shadow_routing_enabled"] is False
        assert recommendation["eligible_for_activation"] is False
        assert recommendation["eligible_for_export"] is False


def test_potential_overlap_and_sensitive_poi_fail_closed():
    overlap, lookalike = evidence(
        row("overlap-a", location="montreal"),
        row("overlap-b", location="montreal"),
        row("hospital", location="toronto", poi="hospital"),
    )

    report = evaluate(overlap, lookalike)
    by_id = {
        value["candidate_id"]: value
        for value in report["recommendations"]
    }
    overlap_ids = {
        candidate_ref["candidate_id"]
        for group in overlap["overlap_groups"]
        for candidate_ref in group["candidate_refs"]
    }

    assert overlap_ids
    for candidate_id in overlap_ids:
        assert by_id[candidate_id]["recommended_lifecycle_status"] == (
            "review_required_overlap"
        )
    assert any(
        value["recommended_lifecycle_status"] == "blocked_policy"
        for value in report["recommendations"]
    )


def test_stale_rights_and_low_quality_production_evidence_are_paused_or_blocked():
    overlap, lookalike = evidence(
        row("stale", location="montreal", mode="production", freshness="stale"),
        row(
            "rights",
            location="toronto",
            mode="production",
            rights="historical_internal_only",
        ),
        row(
            "quality",
            location="vancouver",
            mode="production",
            quality=0.01,
        ),
        mode="production",
    )

    report = evaluate(overlap, lookalike, mode="production")
    statuses = {
        value["recommended_lifecycle_status"]
        for value in report["recommendations"]
    }

    assert "paused_stale_source" in statuses
    assert "blocked_rights" in statuses
    assert "paused_quality_degraded" in statuses


def test_quality_drop_monitoring_uses_prior_aggregate_evidence():
    first_overlap, first_lookalike = evidence(
        row("stable-feature", mode="production", quality=1.0),
        mode="production",
    )
    first = evaluate(first_overlap, first_lookalike, mode="production")

    second_overlap, second_lookalike = evidence(
        row("stable-feature", mode="production", quality=0.65),
        mode="production",
    )
    second = evaluate(
        second_overlap,
        second_lookalike,
        mode="production",
        previous=first,
    )

    recommendation = second["recommendations"][0]
    assert recommendation["quality_delta"] < -0.15
    assert recommendation["recommended_lifecycle_status"] == (
        "paused_quality_degraded"
    )
    assert "quality_drop_exceeds_monitoring_threshold" in (
        recommendation["reason_codes"]
    )


def test_report_validation_detects_tampering_and_unsafe_flags():
    overlap, lookalike = evidence(row("feature-1"))
    service = ProductionModule3LifecycleService()
    report = evaluate(overlap, lookalike)

    assert service.validate_report(report) == report

    tampered = deepcopy(report)
    tampered["recommendations"][0]["quality_score"] = 0.01
    with pytest.raises(ValueError, match="recommendation fingerprint mismatch"):
        service.validate_report(tampered)

    unsafe = deepcopy(report)
    unsafe["safety"]["candidate_lifecycle_mutated"] = True
    with pytest.raises(ValueError, match="Unsafe Module 3.5"):
        service.validate_report(unsafe)


def test_tenant_or_source_fingerprint_mismatch_fails_closed():
    overlap, lookalike = evidence(row("feature-1"))
    request = Module3LifecycleEvaluationRequest(
        tenant_id="another_tenant",
        overlap_report_fingerprint=overlap["report_fingerprint"],
        lookalike_report_fingerprint=lookalike["report_fingerprint"],
        execution_mode="historical_preview",
        purpose=PURPOSE,
    )

    with pytest.raises(ValueError, match="tenant"):
        ProductionModule3LifecycleService().evaluate(
            request=request,
            overlap_report=overlap,
            lookalike_report=lookalike,
        )

    wrong_source = Module3LifecycleEvaluationRequest(
        tenant_id=TENANT_ID,
        overlap_report_fingerprint="0" * 64,
        lookalike_report_fingerprint=lookalike["report_fingerprint"],
        execution_mode="historical_preview",
        purpose=PURPOSE,
    )
    with pytest.raises(ValueError, match="Overlap report fingerprint"):
        ProductionModule3LifecycleService().evaluate(
            request=wrong_source,
            overlap_report=overlap,
            lookalike_report=lookalike,
        )


def test_policy_bounds_and_input_limit_fail_closed():
    with pytest.raises(ValueError, match="max_input_candidates"):
        Module3LifecyclePolicy(max_input_candidates=0)

    overlap, lookalike = evidence(
        row("feature-a", location="montreal"),
        row("feature-b", location="toronto"),
    )
    with pytest.raises(ValueError, match="max_input_candidates"):
        evaluate(
            overlap,
            lookalike,
            policy=Module3LifecyclePolicy(max_input_candidates=1),
        )


def test_module3_5_migration_is_immutable_review_only_and_tenant_scoped():
    sql = Path(
        "migrations/0018_module3_governed_lifecycle_evidence.sql"
    ).read_text(encoding="utf-8")

    for fragment in (
        "candidate_lifecycle_mutated = FALSE",
        "manual_approval_required = TRUE",
        "shadow_routing_enabled = FALSE",
        "eligible_for_activation = FALSE",
        "eligible_for_export = FALSE",
        "activation_or_export_performed = FALSE",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "current_setting('app.tenant_id', true)",
        "prevent_module3_lifecycle_evidence_mutation",
        "REVOKE ALL",
    ):
        assert fragment in sql


def test_module3_status_accepts_safe_lifecycle_evidence(tmp_path):
    from app.services.production_module3_status_service import (
        ProductionModule3StatusService,
    )

    evidence = {
        "candidate": {
            "status": "engineering_preview_ready",
            "safety": {
                "raw_identifiers_returned": False,
                "activation_or_export_performed": False,
            },
        },
        "overlap": {
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
        },
        "lookalike": {
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
        },
        "lifecycle": {
            "status": "engineering_preview_ready",
            "safety": {
                "raw_identifiers_returned": False,
                "audience_membership_read": False,
                "membership_intersection_read": False,
                "overlap_rate_computed": False,
                "unique_reach_claimed": False,
                "cohort_sizes_summed": False,
                "candidate_lifecycle_mutated": False,
                "automatic_approval_performed": False,
                "manual_approval_required": True,
                "monitoring_required": True,
                "shadow_routing_enabled": False,
                "activation_or_export_performed": False,
                "downstream_export_enabled": False,
            },
        },
    }
    environment = {}
    for name, payload in evidence.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        environment[
            {
                "candidate": "MODULE3_COHORT_EVIDENCE_PATH",
                "overlap": "MODULE3_OVERLAP_EVIDENCE_PATH",
                "lookalike": "MODULE3_LOOKALIKE_EVIDENCE_PATH",
                "lifecycle": "MODULE3_LIFECYCLE_EVIDENCE_PATH",
            }[name]
        ] = str(path)

    status = ProductionModule3StatusService(
        environment=environment
    ).status()

    assert status["status"] == "module3_5_engineering_evidence_ready"
    assert status["module3_5_engineering_evidence_ready"] is True
    assert status["components"][
        "module_3_5_lifecycle_approval_and_monitoring"
    ] is True
