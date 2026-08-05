from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from app.models.audience_feature_contracts import (
    SAFE_PRIVACY_STATUSES,
    normalize_taxonomy_value,
)
from app.models.production_module3_cohort_contracts import (
    GovernedCohortCandidate,
    GovernedCohortCandidateBatch,
    Module3CohortCandidatePolicy,
    Module3CohortGenerationRequest,
    stable_fingerprint,
)
from app.services.sensitive_poi_privacy_risk_service import (
    SensitivePOIPrivacyRiskService,
)


class ProductionModule3CohortCandidateService:
    """Create deterministic, privacy-gated candidates from aggregate features.

    One source feature creates one candidate. The service never sums cohort sizes
    or claims unique reach because Module 2 rows do not contain privacy-safe
    membership-intersection evidence.
    """

    def __init__(
        self,
        *,
        policy: Module3CohortCandidatePolicy | None = None,
        sensitive_poi_service: SensitivePOIPrivacyRiskService | None = None,
    ) -> None:
        self._policy = policy or Module3CohortCandidatePolicy()
        self._sensitive_poi_service = (
            sensitive_poi_service or SensitivePOIPrivacyRiskService()
        )

    def generate(
        self,
        *,
        request: Module3CohortGenerationRequest,
        feature_set: Mapping[str, Any],
        feature_rows: Sequence[Mapping[str, Any]],
    ) -> GovernedCohortCandidateBatch:
        self._validate_feature_set(request=request, feature_set=feature_set)

        source_rows = [dict(row) for row in feature_rows if isinstance(row, Mapping)]
        source_rows.sort(key=lambda row: str(row.get("feature_id") or ""))

        eligible_rows: list[dict[str, Any]] = []
        excluded_below_k = 0
        excluded_unsafe_privacy = 0
        excluded_ineligible = 0
        excluded_invalid = 0

        for row in source_rows:
            if not bool(row.get("eligible_for_retrieval")):
                excluded_ineligible += 1
                continue

            privacy_status = normalize_taxonomy_value(row.get("privacy_status"))
            if privacy_status not in SAFE_PRIVACY_STATUSES:
                excluded_unsafe_privacy += 1
                continue

            try:
                cohort_size = int(row.get("cohort_size") or 0)
            except (TypeError, ValueError):
                excluded_invalid += 1
                continue

            if cohort_size < self._policy.min_cohort_size:
                excluded_below_k += 1
                continue

            required_values = (
                row.get("feature_id"),
                row.get("location_name"),
                row.get("primary_poi_type"),
                row.get("created_day_part"),
            )
            if any(not normalize_taxonomy_value(value) for value in required_values):
                excluded_invalid += 1
                continue

            eligible_rows.append(row)

        batch_identity = {
            "tenant_id": request.tenant_id,
            "feature_set_id": request.feature_set_id,
            "feature_set_version": request.feature_set_version,
            "execution_mode": request.execution_mode,
            "purpose": request.purpose,
            "policy": self._policy.to_record(),
            "source_fingerprint": str(feature_set.get("source_fingerprint") or ""),
            "eligible_source_feature_ids": [
                str(row.get("feature_id") or "") for row in eligible_rows
            ],
        }
        batch_fingerprint = stable_fingerprint(batch_identity)

        all_candidates = [
            self._candidate_from_row(
                request=request,
                feature_set=feature_set,
                row=row,
                batch_fingerprint=batch_fingerprint,
            )
            for row in eligible_rows
        ]
        all_candidates.sort(
            key=lambda candidate: (
                -candidate.quality_score,
                candidate.location_name,
                candidate.primary_poi_type,
                candidate.created_day_part,
                candidate.source_feature_id,
            )
        )
        candidates = all_candidates[: self._policy.max_candidates]
        truncated_candidates = max(0, len(all_candidates) - len(candidates))

        blocked_sensitive = sum(
            candidate.lifecycle_status == "blocked_sensitive_poi"
            for candidate in candidates
        )
        review_sensitive = sum(
            candidate.lifecycle_status == "review_required_sensitive_poi"
            for candidate in candidates
        )

        return GovernedCohortCandidateBatch(
            status="engineering_preview_ready",
            request=request,
            policy=self._policy,
            batch_fingerprint=batch_fingerprint,
            source_feature_count=len(source_rows),
            generated_candidate_count=len(candidates),
            truncated_candidate_count=truncated_candidates,
            excluded_below_k_count=excluded_below_k,
            excluded_unsafe_privacy_count=excluded_unsafe_privacy,
            excluded_ineligible_retrieval_count=excluded_ineligible,
            excluded_invalid_count=excluded_invalid,
            blocked_sensitive_count=blocked_sensitive,
            review_required_sensitive_count=review_sensitive,
            candidates=candidates,
            safety={
                "aggregation_strategy": (
                    "one_source_feature_per_candidate_no_overlap_or_unique_reach_claim"
                ),
                "raw_identifiers_read": False,
                "raw_identifiers_stored": False,
                "raw_identifiers_returned": False,
                "overlap_or_unique_reach_computed": False,
                "lookalike_generation_performed": False,
                "database_candidate_write_performed": False,
                "production_routing_enabled": False,
                "activation_or_export_performed": False,
                "downstream_export_enabled": False,
            },
        )

    def _candidate_from_row(
        self,
        *,
        request: Module3CohortGenerationRequest,
        feature_set: Mapping[str, Any],
        row: Mapping[str, Any],
        batch_fingerprint: str,
    ) -> GovernedCohortCandidate:
        feature_id = " ".join(str(row.get("feature_id") or "").split())
        location = normalize_taxonomy_value(row.get("location_name"))
        poi = normalize_taxonomy_value(row.get("primary_poi_type"))
        daypart = normalize_taxonomy_value(row.get("created_day_part"))
        lookback = normalize_taxonomy_value(row.get("lookback_bucket")) or None
        cohort_size = int(row.get("cohort_size") or 0)
        source_quality = self._unit_score(row.get("quality_score"))
        privacy_status = normalize_taxonomy_value(row.get("privacy_status"))
        rights_status = normalize_taxonomy_value(row.get("rights_status")) or "unknown"
        freshness_status = (
            normalize_taxonomy_value(row.get("freshness_status")) or "unknown"
        )
        data_use_mode = (
            normalize_taxonomy_value(row.get("data_use_mode"))
            or request.execution_mode
        )

        risk = self._sensitive_poi_service.assess(
            prompt="",
            selected_cohorts=[
                {
                    "audience_name": f"{poi} - {daypart} - {location}",
                    "location_name": location,
                    "primary_poi_type": poi,
                    "created_day_part": daypart,
                }
            ],
        )
        sensitive_decision = str(risk.get("overall_decision") or "review_required")

        quality_components = self._quality_components(
            row=row,
            cohort_size=cohort_size,
            source_quality=source_quality,
            freshness_status=freshness_status,
        )
        quality_score = round(
            (quality_components["source_quality"] * self._policy.source_quality_weight)
            + (
                quality_components["size_adequacy"]
                * self._policy.size_adequacy_weight
            )
            + (quality_components["freshness"] * self._policy.freshness_weight)
            + (
                quality_components["data_completeness"]
                * self._policy.completeness_weight
            )
            + (quality_components["privacy"] * self._policy.privacy_weight),
            8,
        )

        if sensitive_decision == "block_export":
            lifecycle_status = "blocked_sensitive_poi"
        elif sensitive_decision == "review_required":
            lifecycle_status = "review_required_sensitive_poi"
        elif (
            request.execution_mode != "production"
            or data_use_mode != "production"
            or freshness_status != "fresh"
            or not bool(feature_set.get("eligible_for_activation"))
            or not bool(row.get("eligible_for_activation"))
        ):
            lifecycle_status = "historical_preview_only"
        else:
            lifecycle_status = "quality_review_pending"

        semantic_identity = {
            "tenant_id": request.tenant_id,
            "source_feature_id": feature_id,
            "location_name": location,
            "primary_poi_type": poi,
            "created_day_part": daypart,
            "lookback_bucket": lookback,
            "purpose": request.purpose,
        }
        candidate_id = "cohort_candidate_" + stable_fingerprint(semantic_identity)[:24]
        candidate_version = int(request.feature_set_version)
        candidate_fingerprint = stable_fingerprint(
            {
                **semantic_identity,
                "candidate_id": candidate_id,
                "candidate_version": candidate_version,
                "source_feature_set_id": request.feature_set_id,
                "source_feature_set_version": request.feature_set_version,
                "cohort_size": cohort_size,
                "source_quality_score": source_quality,
                "privacy_status": privacy_status,
                "rights_status": rights_status,
                "freshness_status": freshness_status,
                "data_use_mode": data_use_mode,
                "sensitive_poi_decision": sensitive_decision,
                "policy_version": self._policy.policy_version,
            }
        )

        return GovernedCohortCandidate(
            tenant_id=request.tenant_id,
            candidate_id=candidate_id,
            candidate_version=candidate_version,
            candidate_fingerprint=candidate_fingerprint,
            batch_fingerprint=batch_fingerprint,
            source_feature_set_id=request.feature_set_id,
            source_feature_set_version=request.feature_set_version,
            source_feature_id=feature_id,
            location_name=location,
            primary_poi_type=poi,
            created_day_part=daypart,
            lookback_bucket=lookback,
            cohort_size=cohort_size,
            source_quality_score=source_quality,
            quality_score=quality_score,
            quality_components=quality_components,
            metric_disclosure={
                "measured": [
                    "cohort_size",
                    "source_quality_score",
                    "location_name",
                    "primary_poi_type",
                    "created_day_part",
                    "lookback_bucket",
                    "privacy_status",
                    "rights_status",
                    "freshness_status",
                    "data_use_mode",
                ],
                "derived": [
                    "size_adequacy_score",
                    "freshness_score",
                    "data_completeness_score",
                    "module3_quality_score",
                    "sensitive_poi_decision",
                ],
                "unavailable_not_inferred": [
                    "unique_reach",
                    "cohort_overlap",
                    "deduplicated_reach",
                    "dwell_distribution",
                    "repeat_visit_distribution",
                    "campaign_lift",
                    "demographic_or_sensitive_traits",
                ],
            },
            privacy_status=privacy_status,
            privacy_decision="k_anonymous_aggregate_passed",
            sensitive_poi_decision=sensitive_decision,
            rights_status=rights_status,
            freshness_status=freshness_status,
            data_use_mode=data_use_mode,
            lifecycle_status=lifecycle_status,
            approval_required=True,
            eligible_for_activation=False,
            eligible_for_export=False,
            lineage={
                "source_feature_set_id": request.feature_set_id,
                "source_feature_set_version": request.feature_set_version,
                "source_feature_id": feature_id,
                "source_fingerprint": str(feature_set.get("source_fingerprint") or ""),
                "privacy_policy_version": str(
                    feature_set.get("privacy_policy_version") or "unknown"
                ),
                "rights_policy_id": str(
                    feature_set.get("rights_policy_id") or "unknown"
                ),
                "module3_policy_version": self._policy.policy_version,
            },
        )

    def _quality_components(
        self,
        *,
        row: Mapping[str, Any],
        cohort_size: int,
        source_quality: float,
        freshness_status: str,
    ) -> dict[str, float]:
        size_ratio = max(cohort_size / self._policy.min_cohort_size, 1.0)
        size_adequacy = min(
            1.0,
            math.log10(size_ratio + 1.0) / math.log10(101.0),
        )
        freshness = {
            "fresh": 1.0,
            "unknown": 0.5,
            "stale": 0.0,
        }.get(freshness_status, 0.0)
        completeness_fields = (
            "location_name",
            "primary_poi_type",
            "created_day_part",
            "lookback_bucket",
            "source_latest_at",
            "rights_status",
            "purpose",
        )
        present = sum(
            bool(str(row.get(field) or "").strip()) for field in completeness_fields
        )
        completeness = present / len(completeness_fields)
        return {
            "source_quality": round(source_quality, 8),
            "size_adequacy": round(size_adequacy, 8),
            "freshness": round(freshness, 8),
            "data_completeness": round(completeness, 8),
            "privacy": 1.0,
        }

    def _validate_feature_set(
        self,
        *,
        request: Module3CohortGenerationRequest,
        feature_set: Mapping[str, Any],
    ) -> None:
        feature_tenant = normalize_taxonomy_value(
            feature_set.get("tenant_id") or request.tenant_id
        )
        if feature_tenant != request.tenant_id:
            raise ValueError("Feature set tenant does not match Module 3 request.")
        if str(feature_set.get("feature_set_id") or "") != request.feature_set_id:
            raise ValueError("Feature set ID does not match Module 3 request.")
        if int(feature_set.get("version") or 0) != request.feature_set_version:
            raise ValueError("Feature set version does not match Module 3 request.")
        if not bool(feature_set.get("eligible_for_retrieval")):
            raise ValueError("Feature set is not eligible for governed retrieval.")
        if (
            normalize_taxonomy_value(feature_set.get("data_use_mode"))
            != request.execution_mode
        ):
            raise ValueError("Feature set data_use_mode does not match execution_mode.")

    def _unit_score(self, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(parsed):
            return 0.0
        return round(min(max(parsed, 0.0), 1.0), 8)
