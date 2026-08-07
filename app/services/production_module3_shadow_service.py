from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    stable_fingerprint,
)
from app.models.production_module3_shadow_contracts import (
    MODULE3_SHADOW_POLICY_VERSION,
    Module3ShadowObservationRequest,
    PunkAIShadowCandidateObservation,
    PunkAIShadowObservationReport,
)
from app.services.production_module3_lifecycle_service import (
    ProductionModule3LifecycleService,
)
from app.services.production_module3_overlap_deduplication_service import (
    ProductionModule3OverlapDeduplicationService,
)


class ProductionModule3ShadowService:
    """Compare Punk AI proposal selections with Module 3 review evidence.

    This service observes an already-produced privacy-safe proposal. It never
    creates or changes a proposal, mutates lifecycle state, routes traffic,
    activates an audience, or exports data.
    """

    _SAFE_FALSE_FIELDS = (
        "raw_identifiers_read",
        "raw_identifiers_stored",
        "raw_identifiers_returned",
        "audience_membership_read",
        "proposal_created",
        "proposal_modified",
        "candidate_lifecycle_mutated",
        "automatic_approval_performed",
        "shadow_routing_enabled",
        "production_routing_enabled",
        "activation_or_export_performed",
        "downstream_export_enabled",
    )

    def observe(
        self,
        *,
        request: Module3ShadowObservationRequest,
        overlap_report: Mapping[str, Any],
        lifecycle_report: Mapping[str, Any],
        proposal_response: Mapping[str, Any],
    ) -> PunkAIShadowObservationReport:
        overlap = ProductionModule3OverlapDeduplicationService().validate_report(
            overlap_report
        )
        lifecycle = ProductionModule3LifecycleService().validate_report(
            lifecycle_report
        )
        proposal = self._validate_proposal(request, proposal_response)
        self._validate_sources(request, overlap, lifecycle)

        source_candidates = [
            dict(value)
            for value in overlap.get("retained_candidates") or []
            if isinstance(value, Mapping)
        ]
        self._validate_feature_set(proposal, source_candidates)
        candidates_by_feature: dict[str, list[dict[str, Any]]] = {}
        for candidate in source_candidates:
            feature_id = str(candidate.get("source_feature_id") or "").strip()
            if feature_id:
                candidates_by_feature.setdefault(feature_id, []).append(candidate)
        lifecycle_by_candidate = {
            str(value.get("candidate_id") or ""): dict(value)
            for value in lifecycle.get("recommendations") or []
            if isinstance(value, Mapping)
        }
        proposal_candidates = [
            dict(value)
            for value in proposal.get("candidate_cohorts") or []
            if isinstance(value, Mapping)
        ]

        observation_fingerprint = stable_fingerprint(
            {
                "policy_version": MODULE3_SHADOW_POLICY_VERSION,
                "request": request.to_record(),
                "proposal_contract_version": proposal.get("contract_version"),
                "proposal_status": proposal.get("status"),
                "proposal_candidates": proposal_candidates,
            }
        )
        observations = [
            self._candidate_observation(
                request=request,
                observation_fingerprint=observation_fingerprint,
                proposal_candidate=value,
                candidates_by_feature=candidates_by_feature,
                lifecycle_by_candidate=lifecycle_by_candidate,
            )
            for value in proposal_candidates
        ]
        statuses = [value.alignment_status for value in observations]
        aligned_count = sum(status.startswith("aligned_") for status in statuses)
        blocked_count = statuses.count("blocked_lifecycle")
        unmatched_count = statuses.count("unmatched_candidate")
        ambiguous_count = statuses.count("ambiguous_candidate")
        alignment_passed = bool(
            observations
            and aligned_count == len(observations)
            and blocked_count == 0
            and unmatched_count == 0
            and ambiguous_count == 0
        )
        safety = {
            "observation_strategy": (
                "existing_proposal_to_governed_candidate_alignment_only"
            ),
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
            "raw_identifiers_returned": False,
            "audience_membership_read": False,
            "proposal_created": False,
            "proposal_modified": False,
            "candidate_lifecycle_mutated": False,
            "automatic_approval_performed": False,
            "manual_approval_required": True,
            "monitoring_required": True,
            "shadow_routing_enabled": False,
            "production_routing_enabled": False,
            "activation_or_export_performed": False,
            "downstream_export_enabled": False,
        }
        return PunkAIShadowObservationReport(
            status="engineering_preview_ready",
            policy_version=MODULE3_SHADOW_POLICY_VERSION,
            request=request,
            observation_fingerprint=observation_fingerprint,
            proposal_status=str(proposal.get("status") or "unknown"),
            proposal_candidate_count=len(proposal_candidates),
            observation_count=len(observations),
            aligned_candidate_count=aligned_count,
            blocked_candidate_count=blocked_count,
            unmatched_candidate_count=unmatched_count,
            ambiguous_candidate_count=ambiguous_count,
            shadow_alignment_passed=alignment_passed,
            observations=observations,
            safety=safety,
        )

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") != "engineering_preview_ready":
            raise ValueError("Unsupported Module 3.6 report status.")
        if payload.get("policy_version") != MODULE3_SHADOW_POLICY_VERSION:
            raise ValueError("Unsupported Module 3.6 policy version.")
        request_value = payload.get("request")
        if not isinstance(request_value, Mapping):
            raise ValueError("Module 3.6 request is required.")
        request = Module3ShadowObservationRequest(**dict(request_value))
        observation_fingerprint = str(
            payload.get("observation_fingerprint") or ""
        )
        values = payload.get("observations")
        if not isinstance(values, list):
            raise ValueError("Module 3.6 observations must be a list.")
        observations = [
            PunkAIShadowCandidateObservation(**dict(value))
            for value in values
            if isinstance(value, Mapping)
        ]
        if len(observations) != len(values):
            raise ValueError("Module 3.6 observations contain invalid values.")
        for observation in observations:
            if observation.tenant_id != request.tenant_id:
                raise ValueError("Shadow observation tenant mismatch.")
            if observation.observation_fingerprint != observation_fingerprint:
                raise ValueError("Shadow observation fingerprint mismatch.")
            expected = stable_fingerprint(
                self._observation_identity(observation.to_record())
            )
            if expected != observation.candidate_observation_fingerprint:
                raise ValueError("Candidate shadow fingerprint mismatch.")
        safety = payload.get("safety")
        if not isinstance(safety, Mapping):
            raise ValueError("Module 3.6 safety evidence is required.")
        for field in self._SAFE_FALSE_FIELDS:
            if safety.get(field) is not False:
                raise ValueError(f"Unsafe Module 3.6 safety field: {field}.")
        if safety.get("manual_approval_required") is not True:
            raise ValueError("Module 3.6 manual approval must remain required.")
        if safety.get("monitoring_required") is not True:
            raise ValueError("Module 3.6 monitoring must remain required.")
        rebuilt = PunkAIShadowObservationReport(
            status="engineering_preview_ready",
            policy_version=MODULE3_SHADOW_POLICY_VERSION,
            request=request,
            observation_fingerprint=observation_fingerprint,
            proposal_status=str(payload.get("proposal_status") or ""),
            proposal_candidate_count=int(
                payload.get("proposal_candidate_count") or 0
            ),
            observation_count=int(payload.get("observation_count") or 0),
            aligned_candidate_count=int(
                payload.get("aligned_candidate_count") or 0
            ),
            blocked_candidate_count=int(
                payload.get("blocked_candidate_count") or 0
            ),
            unmatched_candidate_count=int(
                payload.get("unmatched_candidate_count") or 0
            ),
            ambiguous_candidate_count=int(
                payload.get("ambiguous_candidate_count") or 0
            ),
            shadow_alignment_passed=bool(
                payload.get("shadow_alignment_passed")
            ),
            observations=observations,
            safety=dict(safety),
        ).to_record()
        self._validate_counts(rebuilt)
        if rebuilt != payload:
            raise ValueError("Module 3.6 report contains non-canonical values.")
        return rebuilt

    def _validate_proposal(
        self,
        request: Module3ShadowObservationRequest,
        proposal_response: Mapping[str, Any],
    ) -> dict[str, Any]:
        proposal = dict(proposal_response)
        assert_no_raw_identifier_fields(proposal)
        if str(proposal.get("tenant_id") or "").strip().lower().replace("-", "_") != request.tenant_id:
            raise ValueError("Punk AI proposal tenant does not match request.")
        if proposal.get("proposal_id") != request.proposal_id:
            raise ValueError("Punk AI proposal ID does not match request.")
        if proposal.get("execution_mode") != request.execution_mode:
            raise ValueError("Punk AI proposal execution mode mismatch.")
        if proposal.get("downstream_export_enabled") is not False:
            raise ValueError("Unsafe Punk AI proposal export state.")
        if proposal.get("approval_required") is not True:
            raise ValueError("Punk AI proposal must require manual approval.")
        candidates = proposal.get("candidate_cohorts")
        if not isinstance(candidates, list):
            raise ValueError("Punk AI proposal candidates must be a list.")
        if any(not isinstance(value, Mapping) for value in candidates):
            raise ValueError("Punk AI proposal contains invalid candidates.")
        ranks = [int(value.get("rank") or 0) for value in candidates]
        feature_ids = [
            str(value.get("feature_id") or "").strip() for value in candidates
        ]
        if any(rank < 1 for rank in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError("Punk AI proposal candidate ranks are invalid.")
        if any(not feature_id for feature_id in feature_ids):
            raise ValueError("Punk AI proposal candidate feature ID is invalid.")
        return proposal

    def _validate_feature_set(
        self,
        proposal: Mapping[str, Any],
        source_candidates: list[dict[str, Any]],
    ) -> None:
        proposal_feature_set = proposal.get("feature_set")
        if not isinstance(proposal_feature_set, Mapping):
            raise ValueError("Punk AI proposal feature-set identity is required.")
        proposal_identity = (
            str(proposal_feature_set.get("feature_set_id") or ""),
            int(proposal_feature_set.get("version") or 0),
        )
        source_identities = {
            (
                str(value.get("source_feature_set_id") or ""),
                int(value.get("source_feature_set_version") or 0),
            )
            for value in source_candidates
        }
        if source_identities and source_identities != {proposal_identity}:
            raise ValueError(
                "Punk AI proposal feature set does not match Module 3 evidence."
            )

    def _validate_sources(
        self,
        request: Module3ShadowObservationRequest,
        overlap: Mapping[str, Any],
        lifecycle: Mapping[str, Any],
    ) -> None:
        if overlap.get("report_fingerprint") != request.overlap_report_fingerprint:
            raise ValueError("Module 3.6 overlap fingerprint mismatch.")
        if lifecycle.get("evaluation_fingerprint") != request.lifecycle_evaluation_fingerprint:
            raise ValueError("Module 3.6 lifecycle fingerprint mismatch.")
        if lifecycle.get("request", {}).get("tenant_id") != request.tenant_id:
            raise ValueError("Module 3.6 lifecycle tenant mismatch.")
        if overlap.get("request", {}).get("tenant_id") != request.tenant_id:
            raise ValueError("Module 3.6 overlap tenant mismatch.")
        if lifecycle.get("request", {}).get("overlap_report_fingerprint") != overlap.get("report_fingerprint"):
            raise ValueError("Lifecycle evidence is not derived from overlap evidence.")

    def _candidate_observation(
        self,
        *,
        request: Module3ShadowObservationRequest,
        observation_fingerprint: str,
        proposal_candidate: Mapping[str, Any],
        candidates_by_feature: Mapping[str, list[dict[str, Any]]],
        lifecycle_by_candidate: Mapping[str, dict[str, Any]],
    ) -> PunkAIShadowCandidateObservation:
        rank = int(proposal_candidate.get("rank") or 0)
        feature_id = str(proposal_candidate.get("feature_id") or "").strip()
        if rank < 1 or not feature_id:
            raise ValueError("Punk AI proposal candidate identity is invalid.")
        matches = candidates_by_feature.get(feature_id, [])
        candidate = matches[0] if len(matches) == 1 else None
        lifecycle = (
            lifecycle_by_candidate.get(str(candidate.get("candidate_id") or ""))
            if candidate
            else None
        )
        if not matches:
            status = "unmatched_candidate"
            reasons = ["proposal_feature_missing_from_module3_candidates"]
        elif len(matches) > 1:
            status = "ambiguous_candidate"
            reasons = ["proposal_feature_maps_to_multiple_module3_candidates"]
        elif not lifecycle:
            status = "unmatched_candidate"
            reasons = ["module3_lifecycle_recommendation_missing"]
        else:
            recommendation = lifecycle.get("recommended_lifecycle_status")
            if (
                request.execution_mode != "production"
                and recommendation == "historical_preview_only"
            ):
                status = "aligned_historical_preview"
                reasons = ["historical_proposal_matches_preview_only_lifecycle"]
            elif (
                request.execution_mode == "production"
                and recommendation == "shadow_review_pending"
            ):
                status = "aligned_manual_shadow_review"
                reasons = ["production_proposal_matches_manual_shadow_review"]
            else:
                status = "blocked_lifecycle"
                reasons = ["proposal_candidate_not_allowed_by_lifecycle_review"]
        identity = {
            "tenant_id": request.tenant_id,
            "observation_fingerprint": observation_fingerprint,
            "proposal_id": request.proposal_id,
            "proposal_rank": rank,
            "feature_id": feature_id,
            "candidate_id": candidate.get("candidate_id") if candidate else None,
            "candidate_version": (
                int(candidate.get("candidate_version") or 0) if candidate else None
            ),
            "candidate_fingerprint": (
                candidate.get("candidate_fingerprint") if candidate else None
            ),
            "recommended_lifecycle_status": (
                lifecycle.get("recommended_lifecycle_status")
                if lifecycle
                else None
            ),
            "alignment_status": status,
            "reason_codes": reasons,
            "proposal_modified": False,
            "candidate_lifecycle_mutated": False,
            "shadow_routing_enabled": False,
            "eligible_for_activation": False,
            "eligible_for_export": False,
        }
        return PunkAIShadowCandidateObservation(
            candidate_observation_fingerprint=stable_fingerprint(identity),
            **identity,
        )

    def _observation_identity(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.get(key)
            for key in (
                "tenant_id",
                "observation_fingerprint",
                "proposal_id",
                "proposal_rank",
                "feature_id",
                "candidate_id",
                "candidate_version",
                "candidate_fingerprint",
                "recommended_lifecycle_status",
                "alignment_status",
                "reason_codes",
                "proposal_modified",
                "candidate_lifecycle_mutated",
                "shadow_routing_enabled",
                "eligible_for_activation",
                "eligible_for_export",
            )
        }

    def _validate_counts(self, report: Mapping[str, Any]) -> None:
        statuses = [
            str(value.get("alignment_status") or "")
            for value in report.get("observations") or []
        ]
        expected = {
            "observation_count": len(statuses),
            "aligned_candidate_count": sum(
                status.startswith("aligned_") for status in statuses
            ),
            "blocked_candidate_count": statuses.count("blocked_lifecycle"),
            "unmatched_candidate_count": statuses.count("unmatched_candidate"),
            "ambiguous_candidate_count": statuses.count("ambiguous_candidate"),
        }
        if int(report.get("proposal_candidate_count") or 0) != len(statuses):
            raise ValueError("Module 3.6 proposal candidate count is inconsistent.")
        for field, value in expected.items():
            if int(report.get(field) or 0) != value:
                raise ValueError(f"Module 3.6 {field} is inconsistent.")
        passed = bool(statuses and expected["aligned_candidate_count"] == len(statuses))
        if bool(report.get("shadow_alignment_passed")) != passed:
            raise ValueError("Module 3.6 alignment result is inconsistent.")
