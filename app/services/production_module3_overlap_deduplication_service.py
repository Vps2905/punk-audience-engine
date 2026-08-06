from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from app.models.audience_feature_contracts import normalize_taxonomy_value
from app.models.production_module3_cohort_contracts import (
    assert_no_raw_identifier_fields,
    canonical_json,
    finite_unit_interval,
    required_text,
    stable_fingerprint,
)
from app.models.production_module3_overlap_contracts import (
    ExactDuplicateSuppression,
    GovernedOverlapDeduplicationReport,
    Module3OverlapDeduplicationPolicy,
    Module3OverlapDeduplicationRequest,
    PotentialOverlapGroup,
)


class ProductionModule3OverlapDeduplicationService:
    """Suppress exact repeated records and disclose potential aggregate overlap.

    The service does not read membership data and therefore never estimates overlap,
    merges cohort sizes, or claims unique reach. Potential overlap groups are review
    evidence only and do not mutate candidate lifecycle state.
    """

    def __init__(
        self,
        *,
        policy: Module3OverlapDeduplicationPolicy | None = None,
    ) -> None:
        self._policy = policy or Module3OverlapDeduplicationPolicy()

    def analyze(
        self,
        *,
        request: Module3OverlapDeduplicationRequest,
        candidate_batch: Mapping[str, Any],
    ) -> GovernedOverlapDeduplicationReport:
        batch = dict(candidate_batch)
        assert_no_raw_identifier_fields(batch)
        self._validate_batch(request=request, batch=batch)

        source_candidates = [
            self._validated_candidate(
                request=request,
                candidate=value,
            )
            for value in batch.get("candidates") or []
            if isinstance(value, Mapping)
        ]
        if len(source_candidates) > self._policy.max_input_candidates:
            raise ValueError(
                "Candidate batch exceeds the governed max_input_candidates bound."
            )

        source_candidates.sort(key=self._candidate_sort_key)
        self._reject_identity_conflicts(source_candidates)

        retained_candidates, suppressions = self._suppress_exact_duplicates(
            source_candidates
        )
        overlap_groups = self._potential_overlap_groups(retained_candidates)
        review_candidate_ids = {
            str(candidate_ref["candidate_id"])
            for group in overlap_groups
            for candidate_ref in group.candidate_refs
        }

        safety = {
            "analysis_strategy": (
                "exact_record_suppression_and_metadata_only_overlap_review"
            ),
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
            "raw_identifiers_returned": False,
            "membership_intersection_read": False,
            "overlap_rate_computed": False,
            "overlap_estimate_available": False,
            "unique_reach_computed": False,
            "unique_reach_claimed": False,
            "cohort_sizes_summed": False,
            "candidate_lifecycle_mutated": False,
            "potential_overlap_candidates_suppressed": False,
            "exact_duplicate_suppression_performed": bool(suppressions),
            "database_write_performed": False,
            "lookalike_generation_performed": False,
            "production_routing_enabled": False,
            "activation_or_export_performed": False,
            "downstream_export_enabled": False,
        }
        report_identity = self._report_identity(
            request=request,
            policy=self._policy,
            source_candidate_count=len(source_candidates),
            retained_candidates=retained_candidates,
            suppressions=suppressions,
            overlap_groups=overlap_groups,
            safety=safety,
        )
        report_fingerprint = stable_fingerprint(report_identity)

        return GovernedOverlapDeduplicationReport(
            status="engineering_preview_ready",
            request=request,
            policy=self._policy,
            report_fingerprint=report_fingerprint,
            source_candidate_count=len(source_candidates),
            retained_candidate_count=len(retained_candidates),
            exact_duplicate_group_count=len(suppressions),
            suppressed_exact_duplicate_occurrence_count=sum(
                suppression.suppressed_occurrence_count
                for suppression in suppressions
            ),
            potential_overlap_group_count=len(overlap_groups),
            candidates_requiring_overlap_review_count=len(review_candidate_ids),
            retained_candidates=retained_candidates,
            exact_duplicate_suppressions=suppressions,
            overlap_groups=overlap_groups,
            safety=safety,
        )


    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        """Validate persisted Module 3.3 evidence and its content fingerprint."""

        payload = dict(report)
        assert_no_raw_identifier_fields(payload)
        if str(payload.get("status") or "") != "engineering_preview_ready":
            raise ValueError("Unsupported Module 3.3 report status.")

        request_value = payload.get("request")
        policy_value = payload.get("policy")
        if not isinstance(request_value, Mapping) or not isinstance(
            policy_value, Mapping
        ):
            raise ValueError("Module 3.3 report request and policy are required.")
        request = Module3OverlapDeduplicationRequest(**dict(request_value))
        policy = Module3OverlapDeduplicationPolicy(**dict(policy_value))

        retained_values = payload.get("retained_candidates")
        suppression_values = payload.get("exact_duplicate_suppressions")
        group_values = payload.get("overlap_groups")
        if not isinstance(retained_values, list):
            raise ValueError("retained_candidates must be a list.")
        if not isinstance(suppression_values, list):
            raise ValueError("exact_duplicate_suppressions must be a list.")
        if not isinstance(group_values, list):
            raise ValueError("overlap_groups must be a list.")

        retained_candidates = [
            self._validated_candidate(request=request, candidate=value)
            for value in retained_values
            if isinstance(value, Mapping)
        ]
        if len(retained_candidates) != len(retained_values):
            raise ValueError("retained_candidates contains a non-object value.")
        suppressions = [
            ExactDuplicateSuppression(**dict(value))
            for value in suppression_values
            if isinstance(value, Mapping)
        ]
        if len(suppressions) != len(suppression_values):
            raise ValueError("exact_duplicate_suppressions contains invalid values.")
        overlap_groups = [
            PotentialOverlapGroup(
                group_id=str(value.get("group_id") or ""),
                group_version=int(value.get("group_version") or 0),
                group_fingerprint=str(value.get("group_fingerprint") or ""),
                group_type=str(value.get("group_type") or ""),
                normalized_signature=dict(
                    value.get("normalized_signature") or {}
                ),
                candidate_refs=[
                    dict(candidate_ref)
                    for candidate_ref in value.get("candidate_refs") or []
                    if isinstance(candidate_ref, Mapping)
                ],
                review_required=value.get("review_required") is True,
                overlap_estimate_available=(
                    value.get("overlap_estimate_available") is True
                ),
                unique_reach_claimed=(
                    value.get("unique_reach_claimed") is True
                ),
            )
            for value in group_values
            if isinstance(value, Mapping)
        ]
        if len(overlap_groups) != len(group_values):
            raise ValueError("overlap_groups contains invalid values.")

        safety = payload.get("safety")
        if not isinstance(safety, Mapping):
            raise ValueError("Module 3.3 report safety evidence is required.")
        for field in (
            "raw_identifiers_read",
            "raw_identifiers_stored",
            "raw_identifiers_returned",
            "membership_intersection_read",
            "overlap_rate_computed",
            "overlap_estimate_available",
            "unique_reach_computed",
            "unique_reach_claimed",
            "cohort_sizes_summed",
            "candidate_lifecycle_mutated",
            "potential_overlap_candidates_suppressed",
            "database_write_performed",
            "lookalike_generation_performed",
            "production_routing_enabled",
            "activation_or_export_performed",
            "downstream_export_enabled",
        ):
            if safety.get(field) is not False:
                raise ValueError(f"Unsafe Module 3.3 report safety field: {field}.")

        source_candidate_count = int(payload.get("source_candidate_count") or 0)
        retained_candidate_count = int(
            payload.get("retained_candidate_count") or 0
        )
        suppressed_count = int(
            payload.get("suppressed_exact_duplicate_occurrence_count") or 0
        )
        if source_candidate_count != retained_candidate_count + suppressed_count:
            raise ValueError("Module 3.3 candidate accounting is inconsistent.")
        if retained_candidate_count != len(retained_candidates):
            raise ValueError("retained_candidate_count is inconsistent.")
        if int(payload.get("exact_duplicate_group_count") or 0) != len(
            suppressions
        ):
            raise ValueError("exact_duplicate_group_count is inconsistent.")
        if int(payload.get("potential_overlap_group_count") or 0) != len(
            overlap_groups
        ):
            raise ValueError("potential_overlap_group_count is inconsistent.")

        expected_identity = self._report_identity(
            request=request,
            policy=policy,
            source_candidate_count=source_candidate_count,
            retained_candidates=retained_candidates,
            suppressions=suppressions,
            overlap_groups=overlap_groups,
            safety=dict(safety),
        )
        expected_fingerprint = stable_fingerprint(expected_identity)
        if str(payload.get("report_fingerprint") or "") != expected_fingerprint:
            raise ValueError("Module 3.3 report fingerprint mismatch.")

        return payload

    @staticmethod
    def _report_identity(
        *,
        request: Module3OverlapDeduplicationRequest,
        policy: Module3OverlapDeduplicationPolicy,
        source_candidate_count: int,
        retained_candidates: Sequence[Mapping[str, Any]],
        suppressions: Sequence[ExactDuplicateSuppression],
        overlap_groups: Sequence[PotentialOverlapGroup],
        safety: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "request": request.to_record(),
            "policy": policy.to_record(),
            "source_candidate_count": int(source_candidate_count),
            "retained_candidates": [
                dict(candidate) for candidate in retained_candidates
            ],
            "exact_duplicate_suppressions": [
                suppression.to_record() for suppression in suppressions
            ],
            "overlap_groups": [group.to_record() for group in overlap_groups],
            "safety": dict(safety),
        }

    def _validate_batch(
        self,
        *,
        request: Module3OverlapDeduplicationRequest,
        batch: Mapping[str, Any],
    ) -> None:
        if str(batch.get("status") or "") != "engineering_preview_ready":
            raise ValueError(
                "Module 3.3 requires an engineering_preview_ready candidate batch."
            )
        batch_fingerprint = str(batch.get("batch_fingerprint") or "").lower()
        if batch_fingerprint != request.batch_fingerprint:
            raise ValueError("Candidate batch fingerprint does not match request.")

        batch_request = batch.get("request")
        if not isinstance(batch_request, Mapping):
            raise ValueError("Candidate batch request metadata is required.")
        tenant_id = normalize_taxonomy_value(batch_request.get("tenant_id"))
        if tenant_id != request.tenant_id:
            raise ValueError("Candidate batch tenant does not match request.")
        execution_mode = normalize_taxonomy_value(
            batch_request.get("execution_mode")
        )
        if execution_mode != request.execution_mode:
            raise ValueError("Candidate batch execution_mode does not match request.")

        candidates = batch.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(
            candidates, (str, bytes)
        ):
            raise ValueError("Candidate batch candidates must be a sequence.")
        declared_count = int(batch.get("generated_candidate_count") or 0)
        if declared_count != len(candidates):
            raise ValueError(
                "generated_candidate_count does not match candidate batch rows."
            )

        safety = batch.get("safety")
        if not isinstance(safety, Mapping):
            raise ValueError("Candidate batch safety evidence is required.")
        required_false = (
            "raw_identifiers_read",
            "raw_identifiers_stored",
            "raw_identifiers_returned",
            "overlap_or_unique_reach_computed",
            "lookalike_generation_performed",
            "activation_or_export_performed",
            "downstream_export_enabled",
        )
        for field in required_false:
            if safety.get(field) is not False:
                raise ValueError(
                    f"Candidate batch safety field must be false: {field}."
                )

    def _validated_candidate(
        self,
        *,
        request: Module3OverlapDeduplicationRequest,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = dict(candidate)
        assert_no_raw_identifier_fields(value)

        tenant_id = normalize_taxonomy_value(value.get("tenant_id"))
        if tenant_id != request.tenant_id:
            raise ValueError("Candidate tenant does not match overlap request.")
        if str(value.get("batch_fingerprint") or "").lower() != (
            request.batch_fingerprint
        ):
            raise ValueError("Candidate batch fingerprint does not match request.")

        for field in (
            "candidate_id",
            "candidate_fingerprint",
            "source_feature_set_id",
            "source_feature_id",
            "location_name",
            "primary_poi_type",
            "created_day_part",
            "privacy_status",
            "rights_status",
            "freshness_status",
            "data_use_mode",
            "lifecycle_status",
        ):
            required_text(value.get(field), label=f"candidate.{field}")
        if int(value.get("candidate_version") or 0) < 1:
            raise ValueError("candidate_version must be >= 1.")
        if int(value.get("source_feature_set_version") or 0) < 1:
            raise ValueError("source_feature_set_version must be >= 1.")
        if int(value.get("cohort_size") or 0) < 1000:
            raise ValueError("Module 3.3 candidates must remain k >= 1000.")
        finite_unit_interval(
            value.get("source_quality_score"),
            label="candidate.source_quality_score",
        )
        finite_unit_interval(
            value.get("quality_score"),
            label="candidate.quality_score",
        )
        fingerprint = str(value.get("candidate_fingerprint") or "").lower()
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError(
                "candidate_fingerprint must be a lowercase SHA-256 digest."
            )
        value["candidate_fingerprint"] = fingerprint

        if value.get("approval_required") is not True:
            raise ValueError("Module 3.3 candidates must remain approval gated.")
        if value.get("eligible_for_activation") is not False:
            raise ValueError("Module 3.3 cannot accept activation-eligible candidates.")
        if value.get("eligible_for_export") is not False:
            raise ValueError("Module 3.3 cannot accept export-eligible candidates.")

        metric_disclosure = value.get("metric_disclosure")
        if not isinstance(metric_disclosure, Mapping):
            raise ValueError("Candidate metric_disclosure is required.")
        unavailable = {
            normalize_taxonomy_value(item)
            for item in metric_disclosure.get("unavailable_not_inferred") or []
        }
        for field in ("unique_reach", "cohort_overlap", "deduplicated_reach"):
            if field not in unavailable:
                raise ValueError(
                    "Candidate must disclose unavailable overlap and reach metrics."
                )

        normalized = dict(value)
        normalized["tenant_id"] = tenant_id
        normalized["location_name"] = normalize_taxonomy_value(
            value.get("location_name")
        )
        normalized["primary_poi_type"] = normalize_taxonomy_value(
            value.get("primary_poi_type")
        )
        normalized["created_day_part"] = normalize_taxonomy_value(
            value.get("created_day_part")
        )
        lookback = normalize_taxonomy_value(value.get("lookback_bucket"))
        normalized["lookback_bucket"] = lookback or None
        return normalized

    @staticmethod
    def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            str(candidate.get("candidate_id") or ""),
            int(candidate.get("candidate_version") or 0),
            str(candidate.get("candidate_fingerprint") or ""),
            str(candidate.get("source_feature_id") or ""),
        )

    @staticmethod
    def _reject_identity_conflicts(
        candidates: Sequence[Mapping[str, Any]],
    ) -> None:
        identity_fingerprints: dict[tuple[str, int], str] = {}
        fingerprint_payloads: dict[str, str] = {}
        for candidate in candidates:
            identity = (
                str(candidate["candidate_id"]),
                int(candidate["candidate_version"]),
            )
            fingerprint = str(candidate["candidate_fingerprint"])
            previous_fingerprint = identity_fingerprints.setdefault(
                identity,
                fingerprint,
            )
            if previous_fingerprint != fingerprint:
                raise ValueError(
                    "Conflicting fingerprints share a candidate identity."
                )
            payload = canonical_json(candidate)
            previous_payload = fingerprint_payloads.setdefault(
                fingerprint,
                payload,
            )
            if previous_payload != payload:
                raise ValueError(
                    "Conflicting candidate payloads share a fingerprint."
                )

    def _suppress_exact_duplicates(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[ExactDuplicateSuppression]]:
        by_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            by_fingerprint[str(candidate["candidate_fingerprint"])].append(
                dict(candidate)
            )

        retained: list[dict[str, Any]] = []
        suppressions: list[ExactDuplicateSuppression] = []
        for fingerprint in sorted(by_fingerprint):
            group = sorted(by_fingerprint[fingerprint], key=self._candidate_sort_key)
            representative = group[0]
            retained.append(representative)
            if len(group) > 1:
                suppressions.append(
                    ExactDuplicateSuppression(
                        retained_candidate_id=str(
                            representative["candidate_id"]
                        ),
                        retained_candidate_version=int(
                            representative["candidate_version"]
                        ),
                        candidate_fingerprint=fingerprint,
                        suppressed_occurrence_count=len(group) - 1,
                    )
                )

        retained.sort(
            key=lambda candidate: (
                -float(candidate.get("quality_score") or 0.0),
                str(candidate.get("location_name") or ""),
                str(candidate.get("primary_poi_type") or ""),
                str(candidate.get("created_day_part") or ""),
                str(candidate.get("lookback_bucket") or ""),
                str(candidate.get("candidate_id") or ""),
            )
        )
        suppressions.sort(
            key=lambda suppression: (
                suppression.retained_candidate_id,
                suppression.retained_candidate_version,
            )
        )
        return retained, suppressions

    def _potential_overlap_groups(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> list[PotentialOverlapGroup]:
        groups: list[PotentialOverlapGroup] = []

        by_constraint_signature: dict[
            tuple[str, str, str, str | None],
            list[Mapping[str, Any]],
        ] = defaultdict(list)
        by_temporal_core: dict[
            tuple[str, str, str],
            list[Mapping[str, Any]],
        ] = defaultdict(list)

        for candidate in candidates:
            constraint_signature = (
                str(candidate["location_name"]),
                str(candidate["primary_poi_type"]),
                str(candidate["created_day_part"]),
                candidate.get("lookback_bucket"),
            )
            by_constraint_signature[constraint_signature].append(candidate)
            temporal_core = constraint_signature[:3]
            by_temporal_core[temporal_core].append(candidate)

        for signature, members in sorted(
            by_constraint_signature.items(),
            key=lambda item: canonical_json(item[0]),
        ):
            if len(members) < self._policy.min_potential_overlap_group_size:
                continue
            groups.append(
                self._group(
                    group_type="potential_constraint_overlap",
                    signature={
                        "location_name": signature[0],
                        "primary_poi_type": signature[1],
                        "created_day_part": signature[2],
                        "lookback_bucket": signature[3],
                    },
                    members=members,
                )
            )

        for signature, members in sorted(by_temporal_core.items()):
            lookbacks = {
                candidate.get("lookback_bucket") for candidate in members
            }
            if (
                len(members) < self._policy.min_potential_overlap_group_size
                or len(lookbacks) < 2
            ):
                continue
            groups.append(
                self._group(
                    group_type="potential_temporal_overlap",
                    signature={
                        "location_name": signature[0],
                        "primary_poi_type": signature[1],
                        "created_day_part": signature[2],
                        "lookback_bucket": None,
                    },
                    members=members,
                )
            )

        groups.sort(
            key=lambda group: (
                group.group_type,
                canonical_json(group.normalized_signature),
                group.group_id,
            )
        )
        return groups

    def _group(
        self,
        *,
        group_type: str,
        signature: Mapping[str, str | None],
        members: Sequence[Mapping[str, Any]],
    ) -> PotentialOverlapGroup:
        if len(members) > self._policy.max_group_members:
            raise ValueError(
                "Potential overlap group exceeds the governed max_group_members bound."
            )
        candidate_refs = [
            {
                "candidate_id": str(candidate["candidate_id"]),
                "candidate_version": int(candidate["candidate_version"]),
                "candidate_fingerprint": str(
                    candidate["candidate_fingerprint"]
                ),
            }
            for candidate in sorted(members, key=self._candidate_sort_key)
        ]
        group_identity = {
            "group_type": group_type,
            "normalized_signature": dict(signature),
            "candidate_refs": candidate_refs,
            "policy_version": self._policy.policy_version,
        }
        fingerprint = stable_fingerprint(group_identity)
        return PotentialOverlapGroup(
            group_id="cohort_overlap_group_" + fingerprint[:24],
            group_version=1,
            group_fingerprint=fingerprint,
            group_type=group_type,
            normalized_signature=dict(signature),
            candidate_refs=candidate_refs,
        )
