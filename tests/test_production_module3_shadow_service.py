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
)
from app.models.production_module3_lookalike_contracts import (
    Module3LookalikeRequest,
)
from app.models.production_module3_overlap_contracts import (
    Module3OverlapDeduplicationRequest,
)
from app.models.production_module3_shadow_contracts import (
    Module3ShadowObservationRequest,
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
from app.services.production_module3_shadow_service import (
    ProductionModule3ShadowService,
)


TENANT_ID = "punk_internal"
PURPOSE = "internal_audience_evaluation"


def row(
    feature_id="feature-1",
    *,
    mode="historical_preview",
    poi="cafe",
    location="montreal",
):
    return {
        "tenant_id": TENANT_ID,
        "feature_set_id": "feature_set_v1",
        "feature_set_version": 1,
        "feature_id": feature_id,
        "location_name": location,
        "primary_poi_type": poi,
        "created_day_part": "evening",
        "lookback_bucket": "31_90d",
        "cohort_size": 5000,
        "quality_score": 0.95,
        "privacy_status": "passed",
        "rights_status": (
            "permitted" if mode == "production" else "historical_internal_only"
        ),
        "purpose": PURPOSE,
        "source_latest_at": "2026-08-07T10:00:00+00:00",
        "freshness_status": "fresh" if mode == "production" else "stale",
        "data_use_mode": mode,
        "eligible_for_retrieval": True,
        "eligible_for_activation": mode == "production",
    }


def evidence(*rows, mode="historical_preview"):
    feature_set = {
        "tenant_id": TENANT_ID,
        "feature_set_id": "feature_set_v1",
        "version": 1,
        "data_use_mode": mode,
        "source_fingerprint": "source-1",
        "freshness_status": "fresh" if mode == "production" else "stale",
        "privacy_policy_version": "privacy-v1",
        "rights_policy_id": "rights-v1",
        "eligible_for_retrieval": True,
        "eligible_for_activation": mode == "production",
    }
    batch = ProductionModule3CohortCandidateService().generate(
        request=Module3CohortGenerationRequest(
            tenant_id=TENANT_ID,
            feature_set_id="feature_set_v1",
            feature_set_version=1,
            execution_mode=mode,
            purpose=PURPOSE,
        ),
        feature_set=feature_set,
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
    lifecycle = ProductionModule3LifecycleService().evaluate(
        request=Module3LifecycleEvaluationRequest(
            tenant_id=TENANT_ID,
            overlap_report_fingerprint=overlap["report_fingerprint"],
            lookalike_report_fingerprint=lookalike["report_fingerprint"],
            execution_mode=mode,
            purpose=PURPOSE,
        ),
        overlap_report=overlap,
        lookalike_report=lookalike,
    ).to_record()
    return overlap, lifecycle


def proposal(feature_id="feature-1", *, mode="historical_preview"):
    return {
        "contract_version": "2026-07-25",
        "proposal_id": "audience_proposal_1",
        "tenant_id": TENANT_ID,
        "campaign_id": "campaign-1",
        "status": (
            "proposal_ready_for_manual_approval"
            if mode == "production"
            else "historical_preview_ready"
        ),
        "execution_mode": mode,
        "approval_required": True,
        "approval_status": (
            "pending_manual_approval"
            if mode == "production"
            else "blocked_historical_source"
        ),
        "feature_set": {
            "feature_set_id": "feature_set_v1",
            "version": 1,
            "data_use_mode": mode,
        },
        "candidate_cohorts": [
            {
                "rank": 1,
                "feature_id": feature_id,
                "quality_score": 0.90,
                "confidence": 0.80,
                "activation_eligible": mode == "production",
            }
        ],
        "activation_eligible": mode == "production",
        "safe_export_eligible": mode == "production",
        "downstream_export_enabled": False,
    }


def observe(overlap, lifecycle, proposal_value, *, mode="historical_preview"):
    return ProductionModule3ShadowService().observe(
        request=Module3ShadowObservationRequest(
            tenant_id=TENANT_ID,
            proposal_id=proposal_value["proposal_id"],
            overlap_report_fingerprint=overlap["report_fingerprint"],
            lifecycle_evaluation_fingerprint=lifecycle[
                "evaluation_fingerprint"
            ],
            execution_mode=mode,
        ),
        overlap_report=overlap,
        lifecycle_report=lifecycle,
        proposal_response=proposal_value,
    ).to_record()


def test_historical_proposal_is_observed_without_routing_or_mutation():
    overlap, lifecycle = evidence(row())
    report = observe(overlap, lifecycle, proposal())

    assert report["shadow_alignment_passed"] is True
    assert report["aligned_candidate_count"] == 1
    assert report["observations"][0]["alignment_status"] == (
        "aligned_historical_preview"
    )
    assert report["safety"]["proposal_modified"] is False
    assert report["safety"]["shadow_routing_enabled"] is False
    assert report["safety"]["activation_or_export_performed"] is False


def test_production_candidate_can_only_align_to_manual_shadow_review():
    overlap, lifecycle = evidence(row(mode="production"), mode="production")
    report = observe(
        overlap,
        lifecycle,
        proposal(mode="production"),
        mode="production",
    )

    assert report["shadow_alignment_passed"] is True
    assert report["observations"][0]["alignment_status"] == (
        "aligned_manual_shadow_review"
    )
    assert report["observations"][0]["eligible_for_activation"] is False
    assert report["observations"][0]["eligible_for_export"] is False


def test_lifecycle_policy_block_fails_shadow_alignment_closed():
    overlap, lifecycle = evidence(row(poi="hospital"))
    report = observe(overlap, lifecycle, proposal())

    assert report["shadow_alignment_passed"] is False
    assert report["blocked_candidate_count"] == 1
    assert report["observations"][0]["alignment_status"] == (
        "blocked_lifecycle"
    )


def test_unmatched_proposal_feature_is_reported_without_inference():
    overlap, lifecycle = evidence(row())
    report = observe(overlap, lifecycle, proposal("unknown-feature"))

    assert report["shadow_alignment_passed"] is False
    assert report["unmatched_candidate_count"] == 1
    assert report["observations"][0]["candidate_id"] is None


def test_unsafe_or_cross_tenant_proposals_are_rejected():
    overlap, lifecycle = evidence(row())
    unsafe = proposal()
    unsafe["downstream_export_enabled"] = True
    with pytest.raises(ValueError, match="export"):
        observe(overlap, lifecycle, unsafe)

    cross_tenant = proposal()
    cross_tenant["tenant_id"] = "another_tenant"
    with pytest.raises(ValueError, match="tenant"):
        observe(overlap, lifecycle, cross_tenant)

    cross_feature_set = proposal()
    cross_feature_set["feature_set"]["version"] = 2
    with pytest.raises(ValueError, match="feature set"):
        observe(overlap, lifecycle, cross_feature_set)


def test_report_validation_detects_tampering_and_unsafe_flags():
    overlap, lifecycle = evidence(row())
    service = ProductionModule3ShadowService()
    report = observe(overlap, lifecycle, proposal())

    assert service.validate_report(report) == report

    tampered = deepcopy(report)
    tampered["observations"][0]["feature_id"] = "tampered"
    with pytest.raises(ValueError, match="fingerprint"):
        service.validate_report(tampered)

    unsafe = deepcopy(report)
    unsafe["safety"]["proposal_modified"] = True
    with pytest.raises(ValueError, match="Unsafe Module 3.6"):
        service.validate_report(unsafe)


def test_source_fingerprint_mismatch_fails_closed():
    overlap, lifecycle = evidence(row())
    request = Module3ShadowObservationRequest(
        tenant_id=TENANT_ID,
        proposal_id="audience_proposal_1",
        overlap_report_fingerprint="0" * 64,
        lifecycle_evaluation_fingerprint=lifecycle["evaluation_fingerprint"],
        execution_mode="historical_preview",
    )
    with pytest.raises(ValueError, match="overlap fingerprint"):
        ProductionModule3ShadowService().observe(
            request=request,
            overlap_report=overlap,
            lifecycle_report=lifecycle,
            proposal_response=proposal(),
        )


def test_module3_6_migration_is_immutable_observation_only_and_tenant_scoped():
    sql = Path(
        "migrations/0019_module3_punk_ai_shadow_observations.sql"
    ).read_text(encoding="utf-8")

    for fragment in (
        "proposal_created = FALSE",
        "proposal_modified = FALSE",
        "candidate_lifecycle_mutated = FALSE",
        "shadow_routing_enabled = FALSE",
        "production_routing_enabled = FALSE",
        "eligible_for_activation = FALSE",
        "eligible_for_export = FALSE",
        "activation_or_export_performed = FALSE",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "current_setting('app.tenant_id', true)",
        "prevent_module3_shadow_evidence_mutation",
        "REVOKE ALL",
    ):
        assert fragment in sql


def test_module3_status_accepts_safe_shadow_evidence(tmp_path):
    from app.services.production_module3_status_service import (
        ProductionModule3StatusService,
    )

    common_false = {
        "raw_identifiers_returned": False,
        "activation_or_export_performed": False,
    }
    evidence_values = {
        "MODULE3_COHORT_EVIDENCE_PATH": {
            "status": "engineering_preview_ready",
            "safety": dict(common_false),
        },
        "MODULE3_OVERLAP_EVIDENCE_PATH": {
            "status": "engineering_preview_ready",
            "safety": {
                **common_false,
                "membership_intersection_read": False,
                "overlap_rate_computed": False,
                "unique_reach_claimed": False,
                "cohort_sizes_summed": False,
                "candidate_lifecycle_mutated": False,
            },
        },
        "MODULE3_LOOKALIKE_EVIDENCE_PATH": {
            "status": "engineering_preview_ready",
            "safety": {
                **common_false,
                "audience_membership_read": False,
                "membership_similarity_computed": False,
                "audience_membership_generated": False,
                "overlap_rate_computed": False,
                "unique_reach_claimed": False,
                "cohort_sizes_summed": False,
                "candidate_lifecycle_mutated": False,
                "downstream_export_enabled": False,
            },
        },
        "MODULE3_LIFECYCLE_EVIDENCE_PATH": {
            "status": "engineering_preview_ready",
            "safety": {
                **common_false,
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
                "downstream_export_enabled": False,
            },
        },
        "MODULE3_SHADOW_EVIDENCE_PATH": {
            "status": "engineering_preview_ready",
            "shadow_alignment_passed": True,
            "observation_count": 1,
            "safety": {
                **common_false,
                "audience_membership_read": False,
                "proposal_created": False,
                "proposal_modified": False,
                "candidate_lifecycle_mutated": False,
                "automatic_approval_performed": False,
                "manual_approval_required": True,
                "monitoring_required": True,
                "shadow_routing_enabled": False,
                "production_routing_enabled": False,
                "downstream_export_enabled": False,
            },
        },
    }
    environment = {}
    for key, payload in evidence_values.items():
        path = tmp_path / f"{key.lower()}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        environment[key] = str(path)

    status = ProductionModule3StatusService(environment=environment).status()

    assert status["status"] == "module3_6_engineering_evidence_ready"
    assert status["module3_6_engineering_evidence_ready"] is True
    assert status["components"]["module_3_6_punk_ai_shadow_integration"] is True
