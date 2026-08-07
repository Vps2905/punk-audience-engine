from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.models.audience_feature_contracts import normalize_taxonomy_value
from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    finite_unit_interval,
    stable_fingerprint,
)
from app.models.production_module3_lifecycle_contracts import (
    GovernedLifecycleEvaluationReport,
    GovernedLifecycleRecommendation,
    Module3LifecycleEvaluationRequest,
    Module3LifecyclePolicy,
)
from app.services.production_module3_lookalike_service import (
    ProductionModule3LookalikeService,
)
from app.services.production_module3_overlap_deduplication_service import (
    ProductionModule3OverlapDeduplicationService,
)


class ProductionModule3LifecycleService:
    """Build governed lifecycle and monitoring recommendations.

    Module 3.5 produces immutable review evidence only. It does not change a
    candidate, approve shadow routing, activate an audience, or export data.
    All inputs and outputs remain aggregate-only.
    """

    _SAFE_FALSE_FIELDS = (
        "raw_identifiers_read",
        "raw_identifiers_stored",
        "raw_identifiers_returned",
        "audience_membership_read",
        "membership_intersection_read",
        "overlap_rate_computed",
        "unique_reach_claimed",
        "cohort_sizes_summed",
        "candidate_lifecycle_mutated",
        "automatic_approval_performed",
        "shadow_routing_enabled",
        "activation_or_export_performed",
        "downstream_export_enabled",
    )

    def __init__(
        self,
        *,
        policy: Module3LifecyclePolicy | None = None,
    ) -> None:
        self._policy = policy or Module3LifecyclePolicy()

    def evaluate(
        self,
        *,
        request: Module3LifecycleEvaluationRequest,
        overlap_report: Mapping[str, Any],
        lookalike_report: Mapping[str, Any],
        previous_lifecycle_report: Mapping[str, Any] | None = None,
    ) -> GovernedLifecycleEvaluationReport:
        overlap = ProductionModule3OverlapDeduplicationService().validate_report(
            overlap_report
        )
        lookalike = ProductionModule3LookalikeService().validate_report(
            lookalike_report
        )
        self._validate_source_reports(
            request=request,
            overlap=overlap,
            lookalike=lookalike,
        )

        previous = None
        if previous_lifecycle_report is not None:
            previous = self.validate_report(previous_lifecycle_report)
            if previous["request"]["tenant_id"] != request.tenant_id:
                raise ValueError(
                    "Previous lifecycle evidence tenant does not match request."
                )

        candidates = [
            dict(value)
            for value in overlap.get("retained_candidates") or []
            if isinstance(value, Mapping)
        ]
        if len(candidates) != int(overlap.get("retained_candidate_count") or 0):
            raise ValueError("Module 3.5 source candidate accounting is invalid.")
        if len(candidates) > self._policy.max_input_candidates:
            raise ValueError(
                "Lifecycle evaluation exceeds max_input_candidates."
            )

        overlap_review_ids = {
            str(candidate_ref.get("candidate_id") or "")
            for group in overlap.get("overlap_groups") or []
            if isinstance(group, Mapping)
            for candidate_ref in group.get("candidate_refs") or []
            if isinstance(candidate_ref, Mapping)
        }
        previous_quality = self._previous_quality_by_candidate(previous)

        evaluation_fingerprint = stable_fingerprint(
            {
                "request": request.to_record(),
                "policy": self._policy.to_record(),
                "source_overlap_report_fingerprint": overlap[
                    "report_fingerprint"
                ],
                "source_lookalike_report_fingerprint": lookalike[
                    "report_fingerprint"
                ],
                "previous_lifecycle_evaluation_fingerprint": (
                    previous.get("evaluation_fingerprint") if previous else None
                ),
                "source_candidate_refs": [
                    {
                        "candidate_id": value.get("candidate_id"),
                        "candidate_version": value.get("candidate_version"),
                        "candidate_fingerprint": value.get(
                            "candidate_fingerprint"
                        ),
                    }
                    for value in sorted(
                        candidates,
                        key=self._candidate_sort_key,
                    )
                ],
            }
        )

        recommendations = [
            self._recommendation(
                request=request,
                evaluation_fingerprint=evaluation_fingerprint,
                candidate=candidate,
                overlap_review_ids=overlap_review_ids,
                previous_quality=previous_quality,
            )
            for candidate in sorted(candidates, key=self._candidate_sort_key)
        ]

        statuses = [
            value.recommended_lifecycle_status
            for value in recommendations
        ]
        safety = {
            "evaluation_strategy": (
                "aggregate_candidate_policy_and_quality_monitoring_review_only"
            ),
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
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
        }

        return GovernedLifecycleEvaluationReport(
            status="engineering_preview_ready",
            request=request,
            policy=self._policy,
            evaluation_fingerprint=evaluation_fingerprint,
            source_candidate_count=len(candidates),
            recommendation_count=len(recommendations),
            shadow_review_pending_count=statuses.count(
                "shadow_review_pending"
            ),
            review_required_count=sum(
                status.startswith("review_required_") for status in statuses
            ),
            paused_count=sum(status.startswith("paused_") for status in statuses),
            blocked_count=sum(status.startswith("blocked_") for status in statuses),
            historical_preview_count=statuses.count(
                "historical_preview_only"
            ),
            recommendations=recommendations,
            safety=safety,
        )

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported Module 3.5 report status.")

        request_value = payload.get("request")
        policy_value = payload.get("policy")
        if not isinstance(request_value, Mapping) or not isinstance(
            policy_value, Mapping
        ):
            raise ValueError("Module 3.5 request and policy are required.")
        request = Module3LifecycleEvaluationRequest(**dict(request_value))
        policy = Module3LifecyclePolicy(**dict(policy_value))
        evaluation_fingerprint = str(
            payload.get("evaluation_fingerprint") or ""
        )

        recommendation_values = payload.get("recommendations")
        if not isinstance(recommendation_values, list):
            raise ValueError("Module 3.5 recommendations must be a list.")
        recommendations = [
            GovernedLifecycleRecommendation(**dict(value))
            for value in recommendation_values
            if isinstance(value, Mapping)
        ]
        if len(recommendations) != len(recommendation_values):
            raise ValueError("Module 3.5 recommendations contain invalid values.")
        for recommendation in recommendations:
            if recommendation.tenant_id != request.tenant_id:
                raise ValueError("Lifecycle recommendation tenant mismatch.")
            if recommendation.evaluation_fingerprint != evaluation_fingerprint:
                raise ValueError("Lifecycle evaluation fingerprint mismatch.")
            expected = stable_fingerprint(
                self._recommendation_identity(recommendation.to_record())
            )
            if expected != recommendation.recommendation_fingerprint:
                raise ValueError("Lifecycle recommendation fingerprint mismatch.")

        safety = payload.get("safety")
        if not isinstance(safety, Mapping):
            raise ValueError("Module 3.5 safety evidence is required.")
        for field in self._SAFE_FALSE_FIELDS:
            if safety.get(field) is not False:
                raise ValueError(
                    f"Unsafe Module 3.5 report safety field: {field}."
                )
        if safety.get("manual_approval_required") is not True:
            raise ValueError("Module 3.5 manual approval must remain required.")
        if safety.get("monitoring_required") is not True:
            raise ValueError("Module 3.5 monitoring must remain required.")

        rebuilt = GovernedLifecycleEvaluationReport(
            status="engineering_preview_ready",
            request=request,
            policy=policy,
            evaluation_fingerprint=evaluation_fingerprint,
            source_candidate_count=int(
                payload.get("source_candidate_count") or 0
            ),
            recommendation_count=int(
                payload.get("recommendation_count") or 0
            ),
            shadow_review_pending_count=int(
                payload.get("shadow_review_pending_count") or 0
            ),
            review_required_count=int(
                payload.get("review_required_count") or 0
            ),
            paused_count=int(payload.get("paused_count") or 0),
            blocked_count=int(payload.get("blocked_count") or 0),
            historical_preview_count=int(
                payload.get("historical_preview_count") or 0
            ),
            recommendations=recommendations,
            safety=dict(safety),
        ).to_record()
        self._validate_counts(rebuilt)
        if rebuilt != payload:
            raise ValueError("Module 3.5 report contains non-canonical values.")
        return rebuilt

    def _recommendation(
        self,
        *,
        request: Module3LifecycleEvaluationRequest,
        evaluation_fingerprint: str,
        candidate: Mapping[str, Any],
        overlap_review_ids: set[str],
        previous_quality: Mapping[str, float],
    ) -> GovernedLifecycleRecommendation:
        candidate_id = str(candidate.get("candidate_id") or "")
        quality = finite_unit_interval(
            candidate.get("quality_score"),
            label="candidate.quality_score",
        )
        previous = previous_quality.get(candidate_id)
        delta = round(quality - previous, 8) if previous is not None else None
        recommendation, reason_codes = self._recommended_status(
            request=request,
            candidate=candidate,
            overlap_review_ids=overlap_review_ids,
            quality=quality,
            quality_delta=delta,
        )
        identity = {
            "tenant_id": request.tenant_id,
            "evaluation_fingerprint": evaluation_fingerprint,
            "candidate_id": candidate_id,
            "candidate_version": int(candidate.get("candidate_version") or 0),
            "candidate_fingerprint": str(
                candidate.get("candidate_fingerprint") or ""
            ),
            "current_lifecycle_status": str(
                candidate.get("lifecycle_status") or "unknown"
            ),
            "recommended_lifecycle_status": recommendation,
            "quality_score": quality,
            "previous_quality_score": previous,
            "quality_delta": delta,
            "reason_codes": reason_codes,
            "manual_approval_required": True,
            "monitoring_required": True,
            "lifecycle_mutated": False,
            "shadow_routing_enabled": False,
            "eligible_for_activation": False,
            "eligible_for_export": False,
        }
        return GovernedLifecycleRecommendation(
            recommendation_fingerprint=stable_fingerprint(identity),
            **identity,
        )

    def _recommended_status(
        self,
        *,
        request: Module3LifecycleEvaluationRequest,
        candidate: Mapping[str, Any],
        overlap_review_ids: set[str],
        quality: float,
        quality_delta: float | None,
    ) -> tuple[str, list[str]]:
        candidate_id = str(candidate.get("candidate_id") or "")
        current = normalize_taxonomy_value(candidate.get("lifecycle_status"))
        sensitive = normalize_taxonomy_value(
            candidate.get("sensitive_poi_decision")
        )
        rights = normalize_taxonomy_value(candidate.get("rights_status"))
        freshness = normalize_taxonomy_value(
            candidate.get("freshness_status")
        )
        data_use_mode = normalize_taxonomy_value(
            candidate.get("data_use_mode")
        )

        if current == "blocked_sensitive_poi" or sensitive == "block_export":
            return "blocked_policy", ["sensitive_poi_policy_block"]
        if candidate_id in overlap_review_ids:
            return "review_required_overlap", [
                "potential_overlap_requires_manual_review"
            ]
        if (
            current == "review_required_sensitive_poi"
            or sensitive == "review_required"
        ):
            return "review_required_sensitive_poi", [
                "sensitive_poi_requires_manual_review"
            ]
        if request.execution_mode != "production" or data_use_mode != "production":
            return "historical_preview_only", [
                "non_production_evidence_cannot_enter_shadow_review"
            ]
        if rights != "permitted":
            return "blocked_rights", ["production_rights_not_permitted"]
        if freshness != "fresh":
            return "paused_stale_source", ["source_freshness_not_fresh"]
        if quality_delta is not None and quality_delta < -self._policy.max_quality_drop:
            return "paused_quality_degraded", [
                "quality_drop_exceeds_monitoring_threshold"
            ]
        if quality < self._policy.min_quality_for_shadow_review:
            return "paused_quality_degraded", [
                "quality_below_shadow_review_threshold"
            ]
        return "shadow_review_pending", [
            "aggregate_candidate_passed_shadow_review_entry_checks",
            "manual_approval_still_required",
        ]

    def _validate_source_reports(
        self,
        *,
        request: Module3LifecycleEvaluationRequest,
        overlap: Mapping[str, Any],
        lookalike: Mapping[str, Any],
    ) -> None:
        overlap_request = overlap.get("request") or {}
        lookalike_request = lookalike.get("request") or {}
        if overlap_request.get("tenant_id") != request.tenant_id:
            raise ValueError("Overlap evidence tenant does not match request.")
        if lookalike_request.get("tenant_id") != request.tenant_id:
            raise ValueError("Lookalike evidence tenant does not match request.")
        if (
            overlap.get("report_fingerprint")
            != request.overlap_report_fingerprint
        ):
            raise ValueError("Overlap report fingerprint does not match request.")
        if (
            lookalike.get("report_fingerprint")
            != request.lookalike_report_fingerprint
        ):
            raise ValueError("Lookalike report fingerprint does not match request.")
        if (
            lookalike_request.get("overlap_report_fingerprint")
            != overlap.get("report_fingerprint")
        ):
            raise ValueError("Lookalike evidence is not derived from overlap evidence.")
        if (
            overlap_request.get("purpose") != request.purpose
            or lookalike_request.get("purpose") != request.purpose
        ):
            raise ValueError("Module 3.5 evidence purpose mismatch.")
        assert_no_raw_identifier_fields(overlap)
        assert_no_raw_identifier_fields(lookalike)

    def _previous_quality_by_candidate(
        self,
        previous: Mapping[str, Any] | None,
    ) -> dict[str, float]:
        if not previous:
            return {}
        return {
            str(value.get("candidate_id") or ""): finite_unit_interval(
                value.get("quality_score"),
                label="previous quality_score",
            )
            for value in previous.get("recommendations") or []
            if isinstance(value, Mapping)
        }

    def _validate_counts(self, report: Mapping[str, Any]) -> None:
        statuses = [
            value.get("recommended_lifecycle_status")
            for value in report.get("recommendations") or []
        ]
        expected = {
            "recommendation_count": len(statuses),
            "shadow_review_pending_count": statuses.count(
                "shadow_review_pending"
            ),
            "review_required_count": sum(
                str(value).startswith("review_required_") for value in statuses
            ),
            "paused_count": sum(
                str(value).startswith("paused_") for value in statuses
            ),
            "blocked_count": sum(
                str(value).startswith("blocked_") for value in statuses
            ),
            "historical_preview_count": statuses.count(
                "historical_preview_only"
            ),
        }
        if int(report.get("source_candidate_count") or 0) != len(statuses):
            raise ValueError("Module 3.5 source candidate count is inconsistent.")
        for field, value in expected.items():
            if int(report.get(field) or 0) != value:
                raise ValueError(f"Module 3.5 {field} is inconsistent.")

    def _recommendation_identity(
        self,
        recommendation: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            key: value
            for key, value in recommendation.items()
            if key != "recommendation_fingerprint"
        }

    def _candidate_sort_key(
        self,
        candidate: Mapping[str, Any],
    ) -> tuple[str, int, str]:
        return (
            str(candidate.get("candidate_id") or ""),
            int(candidate.get("candidate_version") or 0),
            str(candidate.get("candidate_fingerprint") or ""),
        )
