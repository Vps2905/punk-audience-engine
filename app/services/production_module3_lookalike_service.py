from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.models.audience_feature_contracts import normalize_taxonomy_value
from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    finite_unit_interval,
    stable_fingerprint,
)
from app.models.production_module3_lookalike_contracts import (
    GovernedLookalikePair,
    GovernedLookalikeReport,
    Module3LookalikePolicy,
    Module3LookalikeRequest,
)
from app.services.production_module3_overlap_deduplication_service import (
    ProductionModule3OverlapDeduplicationService,
)


class ProductionModule3LookalikeService:
    """Generate aggregate-only, review-gated candidate-to-candidate suggestions."""

    def __init__(
        self,
        *,
        policy: Module3LookalikePolicy | None = None,
        overlap_service: ProductionModule3OverlapDeduplicationService | None = None,
    ) -> None:
        self._policy = policy or Module3LookalikePolicy()
        self._overlap_service = (
            overlap_service or ProductionModule3OverlapDeduplicationService()
        )

    def generate(
        self,
        *,
        request: Module3LookalikeRequest,
        overlap_report: Mapping[str, Any],
    ) -> GovernedLookalikeReport:
        source_report = self._validated_source_report(
            request=request,
            overlap_report=overlap_report,
        )

        retained_candidates = [
            dict(candidate)
            for candidate in source_report.get("retained_candidates") or []
            if isinstance(candidate, Mapping)
        ]

        overlap_review_identities = self._overlap_review_identities(source_report)

        eligible_candidates: list[dict[str, Any]] = []
        excluded_overlap = 0
        excluded_policy = 0

        for candidate in retained_candidates:
            identity = self._candidate_identity(candidate)

            if (
                self._policy.exclude_overlap_review_candidates
                and identity in overlap_review_identities
            ):
                excluded_overlap += 1
                continue

            if not self._candidate_is_policy_eligible(candidate):
                excluded_policy += 1
                continue

            eligible_candidates.append(candidate)

        eligible_candidates.sort(key=self._candidate_sort_key)

        pairs: list[GovernedLookalikePair] = []

        for seed in eligible_candidates:
            ranked_targets: list[GovernedLookalikePair] = []

            for target in eligible_candidates:
                if self._candidate_identity(seed) == self._candidate_identity(target):
                    continue

                pair = self._build_pair(
                    request=request,
                    seed=seed,
                    target=target,
                )
                if pair is not None:
                    ranked_targets.append(pair)

            ranked_targets.sort(
                key=lambda pair: (
                    -pair.similarity_score,
                    pair.target_candidate_id,
                    pair.target_candidate_version,
                    pair.lookalike_fingerprint,
                )
            )

            pairs.extend(
                ranked_targets[: self._policy.max_targets_per_seed]
            )

        pairs.sort(
            key=lambda pair: (
                pair.seed_candidate_id,
                pair.seed_candidate_version,
                -pair.similarity_score,
                pair.target_candidate_id,
                pair.target_candidate_version,
            )
        )

        safety = {
            "comparison_strategy": (
                "aggregate_candidate_metadata_similarity_review_only"
            ),
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
            "raw_identifiers_returned": False,
            "audience_membership_read": False,
            "membership_similarity_computed": False,
            "audience_membership_generated": False,
            "overlap_rate_computed": False,
            "overlap_estimate_available": False,
            "unique_reach_computed": False,
            "unique_reach_claimed": False,
            "cohort_sizes_summed": False,
            "candidate_lifecycle_mutated": False,
            "lookalike_pair_suggestions_generated": bool(pairs),
            "database_write_performed": False,
            "production_routing_enabled": False,
            "activation_or_export_performed": False,
            "downstream_export_enabled": False,
        }

        identity = self._report_identity(
            request=request,
            policy=self._policy,
            source_retained_candidate_count=len(retained_candidates),
            excluded_overlap_review_candidate_count=excluded_overlap,
            excluded_policy_candidate_count=excluded_policy,
            eligible_seed_count=len(eligible_candidates),
            lookalike_candidates=pairs,
            safety=safety,
        )
        report_fingerprint = stable_fingerprint(identity)

        return GovernedLookalikeReport(
            status="engineering_preview_ready",
            request=request,
            policy=self._policy,
            report_fingerprint=report_fingerprint,
            source_retained_candidate_count=len(retained_candidates),
            excluded_overlap_review_candidate_count=excluded_overlap,
            excluded_policy_candidate_count=excluded_policy,
            eligible_seed_count=len(eligible_candidates),
            generated_lookalike_candidate_count=len(pairs),
            lookalike_candidates=pairs,
            safety=safety,
        )

    def validate_report(
        self,
        report: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)

        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported Module 3.4 report status.")

        request_value = payload.get("request")
        policy_value = payload.get("policy")
        pair_values = payload.get("lookalike_candidates")
        safety_value = payload.get("safety")

        if not isinstance(request_value, Mapping):
            raise ValueError("Module 3.4 request metadata is required.")
        if not isinstance(policy_value, Mapping):
            raise ValueError("Module 3.4 policy metadata is required.")
        if not isinstance(pair_values, list):
            raise ValueError("lookalike_candidates must be a list.")
        if not isinstance(safety_value, Mapping):
            raise ValueError("Module 3.4 safety evidence is required.")

        request = Module3LookalikeRequest(**dict(request_value))
        policy = Module3LookalikePolicy(**dict(policy_value))

        pairs = [
            GovernedLookalikePair(
                lookalike_id=str(value.get("lookalike_id") or ""),
                lookalike_version=int(value.get("lookalike_version") or 0),
                lookalike_fingerprint=str(
                    value.get("lookalike_fingerprint") or ""
                ),
                seed_candidate_id=str(value.get("seed_candidate_id") or ""),
                seed_candidate_version=int(
                    value.get("seed_candidate_version") or 0
                ),
                seed_candidate_fingerprint=str(
                    value.get("seed_candidate_fingerprint") or ""
                ),
                target_candidate_id=str(value.get("target_candidate_id") or ""),
                target_candidate_version=int(
                    value.get("target_candidate_version") or 0
                ),
                target_candidate_fingerprint=str(
                    value.get("target_candidate_fingerprint") or ""
                ),
                similarity_score=float(value.get("similarity_score") or 0.0),
                similarity_components=dict(
                    value.get("similarity_components") or {}
                ),
                reason_codes=list(value.get("reason_codes") or []),
                review_required=value.get("review_required") is True,
                membership_generated=value.get("membership_generated") is True,
                eligible_for_activation=(
                    value.get("eligible_for_activation") is True
                ),
                eligible_for_export=value.get("eligible_for_export") is True,
            )
            for value in pair_values
            if isinstance(value, Mapping)
        ]

        if len(pairs) != len(pair_values):
            raise ValueError("lookalike_candidates contains a non-object value.")

        required_false_safety = (
            "raw_identifiers_read",
            "raw_identifiers_stored",
            "raw_identifiers_returned",
            "audience_membership_read",
            "membership_similarity_computed",
            "audience_membership_generated",
            "overlap_rate_computed",
            "overlap_estimate_available",
            "unique_reach_computed",
            "unique_reach_claimed",
            "cohort_sizes_summed",
            "candidate_lifecycle_mutated",
            "database_write_performed",
            "production_routing_enabled",
            "activation_or_export_performed",
            "downstream_export_enabled",
        )

        for field in required_false_safety:
            if safety_value.get(field) is not False:
                raise ValueError(
                    f"Unsafe Module 3.4 report safety field: {field}."
                )

        generated_count = int(
            payload.get("generated_lookalike_candidate_count") or 0
        )
        if generated_count != len(pairs):
            raise ValueError(
                "generated_lookalike_candidate_count is inconsistent."
            )

        identity = self._report_identity(
            request=request,
            policy=policy,
            source_retained_candidate_count=int(
                payload.get("source_retained_candidate_count") or 0
            ),
            excluded_overlap_review_candidate_count=int(
                payload.get("excluded_overlap_review_candidate_count") or 0
            ),
            excluded_policy_candidate_count=int(
                payload.get("excluded_policy_candidate_count") or 0
            ),
            eligible_seed_count=int(payload.get("eligible_seed_count") or 0),
            lookalike_candidates=pairs,
            safety=dict(safety_value),
        )

        expected_fingerprint = stable_fingerprint(identity)
        if payload.get("report_fingerprint") != expected_fingerprint:
            raise ValueError("Module 3.4 report fingerprint mismatch.")

        return payload

    def _validated_source_report(
        self,
        *,
        request: Module3LookalikeRequest,
        overlap_report: Mapping[str, Any],
    ) -> dict[str, Any]:
        source_report = self._overlap_service.validate_report(overlap_report)

        if source_report.get("report_fingerprint") != (
            request.overlap_report_fingerprint
        ):
            raise ValueError(
                "Module 3.3 report fingerprint does not match Module 3.4 request."
            )

        source_request = source_report.get("request")
        if not isinstance(source_request, Mapping):
            raise ValueError("Module 3.3 request metadata is required.")

        source_tenant = normalize_taxonomy_value(
            source_request.get("tenant_id")
        )
        if source_tenant != request.tenant_id:
            raise ValueError("Module 3.3 tenant does not match Module 3.4 request.")

        source_mode = normalize_taxonomy_value(
            source_request.get("execution_mode")
        )
        if source_mode != request.execution_mode:
            raise ValueError(
                "Module 3.3 execution_mode does not match Module 3.4 request."
            )

        safety = source_report.get("safety")
        if not isinstance(safety, Mapping):
            raise ValueError("Module 3.3 safety evidence is required.")

        for field in (
            "raw_identifiers_read",
            "raw_identifiers_stored",
            "raw_identifiers_returned",
            "membership_intersection_read",
            "overlap_rate_computed",
            "unique_reach_computed",
            "unique_reach_claimed",
            "cohort_sizes_summed",
            "candidate_lifecycle_mutated",
            "database_write_performed",
            "lookalike_generation_performed",
            "production_routing_enabled",
            "activation_or_export_performed",
            "downstream_export_enabled",
        ):
            if safety.get(field) is not False:
                raise ValueError(
                    f"Unsafe Module 3.3 source safety field: {field}."
                )

        return source_report

    def _candidate_is_policy_eligible(
        self,
        candidate: Mapping[str, Any],
    ) -> bool:
        assert_no_raw_identifier_fields(candidate)

        if int(candidate.get("cohort_size") or 0) < 1000:
            return False

        quality_score = finite_unit_interval(
            candidate.get("quality_score"),
            label="candidate.quality_score",
        )
        if quality_score < self._policy.min_seed_quality_score:
            return False

        if quality_score < self._policy.min_target_quality_score:
            return False

        if candidate.get("approval_required") is not True:
            return False

        if candidate.get("eligible_for_activation") is not False:
            return False

        if candidate.get("eligible_for_export") is not False:
            return False

        lifecycle_status = normalize_taxonomy_value(
            candidate.get("lifecycle_status")
        )
        if lifecycle_status in {
            "blocked_sensitive_poi",
            "review_required_sensitive_poi",
        }:
            return False

        return True

    def _build_pair(
        self,
        *,
        request: Module3LookalikeRequest,
        seed: Mapping[str, Any],
        target: Mapping[str, Any],
    ) -> GovernedLookalikePair | None:
        seed_poi = normalize_taxonomy_value(seed.get("primary_poi_type"))
        target_poi = normalize_taxonomy_value(target.get("primary_poi_type"))
        seed_daypart = normalize_taxonomy_value(seed.get("created_day_part"))
        target_daypart = normalize_taxonomy_value(target.get("created_day_part"))
        seed_location = normalize_taxonomy_value(seed.get("location_name"))
        target_location = normalize_taxonomy_value(target.get("location_name"))

        if self._policy.require_same_poi_type and seed_poi != target_poi:
            return None

        if self._policy.require_same_daypart and seed_daypart != target_daypart:
            return None

        if (
            self._policy.require_distinct_location
            and seed_location == target_location
        ):
            return None

        components = self._similarity_components(
            seed=seed,
            target=target,
        )

        score = round(
            (components["poi_type"] * 0.30)
            + (components["daypart"] * 0.20)
            + (components["lookback"] * 0.15)
            + (components["quality_alignment"] * 0.20)
            + (components["source_quality_alignment"] * 0.15),
            8,
        )

        if score < self._policy.min_similarity_score:
            return None

        reason_codes = []

        if components["poi_type"] == 1.0:
            reason_codes.append("same_poi_type")
        if components["daypart"] == 1.0:
            reason_codes.append("same_daypart")
        if components["lookback"] == 1.0:
            reason_codes.append("same_lookback_bucket")
        if components["quality_alignment"] >= 0.90:
            reason_codes.append("quality_aligned")
        if seed_location != target_location:
            reason_codes.append("cross_location_review")

        semantic_identity = {
            "tenant_id": request.tenant_id,
            "seed_candidate_id": seed["candidate_id"],
            "seed_candidate_version": int(seed["candidate_version"]),
            "seed_candidate_fingerprint": seed["candidate_fingerprint"],
            "target_candidate_id": target["candidate_id"],
            "target_candidate_version": int(target["candidate_version"]),
            "target_candidate_fingerprint": target["candidate_fingerprint"],
            "policy_version": self._policy.policy_version,
        }

        lookalike_fingerprint = stable_fingerprint(
            {
                **semantic_identity,
                "similarity_score": score,
                "similarity_components": components,
                "reason_codes": reason_codes,
            }
        )

        return GovernedLookalikePair(
            lookalike_id=(
                "lookalike_pair_"
                + stable_fingerprint(semantic_identity)[:24]
            ),
            lookalike_version=1,
            lookalike_fingerprint=lookalike_fingerprint,
            seed_candidate_id=str(seed["candidate_id"]),
            seed_candidate_version=int(seed["candidate_version"]),
            seed_candidate_fingerprint=str(seed["candidate_fingerprint"]),
            target_candidate_id=str(target["candidate_id"]),
            target_candidate_version=int(target["candidate_version"]),
            target_candidate_fingerprint=str(target["candidate_fingerprint"]),
            similarity_score=score,
            similarity_components=components,
            reason_codes=reason_codes or ["aggregate_similarity_review"],
        )

    @staticmethod
    def _similarity_components(
        *,
        seed: Mapping[str, Any],
        target: Mapping[str, Any],
    ) -> dict[str, float]:
        seed_poi = normalize_taxonomy_value(seed.get("primary_poi_type"))
        target_poi = normalize_taxonomy_value(target.get("primary_poi_type"))
        seed_daypart = normalize_taxonomy_value(seed.get("created_day_part"))
        target_daypart = normalize_taxonomy_value(target.get("created_day_part"))
        seed_lookback = normalize_taxonomy_value(seed.get("lookback_bucket"))
        target_lookback = normalize_taxonomy_value(target.get("lookback_bucket"))

        seed_quality = finite_unit_interval(
            seed.get("quality_score"),
            label="seed.quality_score",
        )
        target_quality = finite_unit_interval(
            target.get("quality_score"),
            label="target.quality_score",
        )

        seed_source_quality = finite_unit_interval(
            seed.get("source_quality_score"),
            label="seed.source_quality_score",
        )
        target_source_quality = finite_unit_interval(
            target.get("source_quality_score"),
            label="target.source_quality_score",
        )

        if seed_lookback and seed_lookback == target_lookback:
            lookback_score = 1.0
        elif not seed_lookback and not target_lookback:
            lookback_score = 0.75
        elif not seed_lookback or not target_lookback:
            lookback_score = 0.60
        else:
            lookback_score = 0.50

        return {
            "poi_type": 1.0 if seed_poi == target_poi else 0.0,
            "daypart": 1.0 if seed_daypart == target_daypart else 0.0,
            "lookback": lookback_score,
            "quality_alignment": round(
                max(0.0, 1.0 - abs(seed_quality - target_quality)),
                8,
            ),
            "source_quality_alignment": round(
                max(
                    0.0,
                    1.0 - abs(seed_source_quality - target_source_quality),
                ),
                8,
            ),
        }

    @staticmethod
    def _candidate_identity(
        candidate: Mapping[str, Any],
    ) -> tuple[str, int]:
        return (
            str(candidate.get("candidate_id") or ""),
            int(candidate.get("candidate_version") or 0),
        )

    @staticmethod
    def _candidate_sort_key(
        candidate: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        return (
            str(candidate.get("candidate_id") or ""),
            int(candidate.get("candidate_version") or 0),
            str(candidate.get("candidate_fingerprint") or ""),
        )

    @classmethod
    def _overlap_review_identities(
        cls,
        report: Mapping[str, Any],
    ) -> set[tuple[str, int]]:
        identities: set[tuple[str, int]] = set()

        for group in report.get("overlap_groups") or []:
            if not isinstance(group, Mapping):
                continue
            for candidate_ref in group.get("candidate_refs") or []:
                if not isinstance(candidate_ref, Mapping):
                    continue
                identities.add(
                    (
                        str(candidate_ref.get("candidate_id") or ""),
                        int(candidate_ref.get("candidate_version") or 0),
                    )
                )

        return identities

    @staticmethod
    def _report_identity(
        *,
        request: Module3LookalikeRequest,
        policy: Module3LookalikePolicy,
        source_retained_candidate_count: int,
        excluded_overlap_review_candidate_count: int,
        excluded_policy_candidate_count: int,
        eligible_seed_count: int,
        lookalike_candidates: Sequence[GovernedLookalikePair],
        safety: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "request": request.to_record(),
            "policy": policy.to_record(),
            "source_retained_candidate_count": int(
                source_retained_candidate_count
            ),
            "excluded_overlap_review_candidate_count": int(
                excluded_overlap_review_candidate_count
            ),
            "excluded_policy_candidate_count": int(
                excluded_policy_candidate_count
            ),
            "eligible_seed_count": int(eligible_seed_count),
            "lookalike_candidates": [
                pair.to_record() for pair in lookalike_candidates
            ],
            "safety": dict(safety),
        }
