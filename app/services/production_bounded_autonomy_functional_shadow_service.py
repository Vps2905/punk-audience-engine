from __future__ import annotations

import json
import os
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from app.models.audience_feature_contracts import (
    SAFE_PRIVACY_STATUSES,
    normalize_taxonomy_value,
)
from app.models.production_agent_security_contracts import (
    AGENT_AUTHORIZATION_POLICY_VERSION,
    AgentAuthorizationContext,
)
from app.models.production_audience_retrieval_contracts import (
    GovernedAudienceRetrievalRequest,
)
from app.models.production_bounded_autonomy_contracts import (
    AutonomyGoal,
    CapabilityExecutionResult,
    required_metadata_token,
)
from app.models.production_bounded_autonomy_functional_shadow_contracts import (
    BOUNDED_AUTONOMY_FUNCTIONAL_SHADOW_VERSION,
    FunctionalShadowPolicy,
    FunctionalShadowRunRequest,
    FunctionalStageRecord,
)
from app.models.production_module3_cohort_contracts import (
    Module3CohortGenerationRequest,
    assert_no_raw_identifier_fields,
    stable_fingerprint,
)
from app.models.production_module3_lookalike_contracts import (
    Module3LookalikeRequest,
)
from app.models.production_module3_overlap_contracts import (
    Module3OverlapDeduplicationRequest,
    required_sha256_digest,
)
from app.models.production_module4_evolution_contracts import (
    Module4EvolutionSnapshotRequest,
)
from app.models.production_module4_evolution_control_contracts import (
    Module4DriftAnalysisRequest,
)
from app.services.audience_proposal_request_safety_service import (
    AudienceProposalRequestSafetyService,
)
from app.services.audience_supervisor_decision_service import (
    AudienceSupervisorDecisionService,
)
from app.services.production_agent_authorization_service import (
    AgentCapabilityAuthorizationEnforcer,
)
from app.services.production_bounded_autonomy_service import (
    CapabilityHandler,
    ProductionBoundedAutonomyService,
)
from app.services.production_bounded_autonomy_shadow_service import (
    LegacyOrchestratorObservationAdapter,
)
from app.services.production_feature_snapshot_reader_service import (
    ProductionFeatureSnapshotReaderService,
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
from app.services.production_module4_evolution_control_service import (
    ProductionModule4DriftDetectionService,
    ProductionModule4EvolutionRecommendationService,
)
from app.services.production_module4_evolution_snapshot_service import (
    ProductionModule4EvolutionSnapshotService,
)

_EFFECT_FIELDS = {
    "activation_or_export_performed",
    "automatic_approval_performed",
    "automatic_evolution_performed",
    "automatic_mutation_performed",
    "automatic_proposal_creation_enabled",
    "candidate_lifecycle_mutated",
    "cohort_lifecycle_mutated",
    "database_candidate_write_performed",
    "database_write_performed",
    "downstream_export_enabled",
    "existing_proposal_flow_modified",
    "production_effect_performed",
    "production_routing_enabled",
    "raw_identifiers_read",
    "raw_identifiers_returned",
    "raw_identifiers_stored",
}

_PROHIBITED_SOURCE_PAYLOAD_KEYS = {
    "document_vector",
    "embedding",
    "embedding_vector",
    "query_vector",
    "raw_vector",
}

_PROHIBITED_IDENTIFIER_KEY_TOKENS = (
    "advertising_id",
    "device_id",
    "email",
    "individual_id",
    "latitude",
    "longitude",
    "maid",
    "phone",
    "raw_identifier",
)


class FunctionalFeatureSnapshotReader(Protocol):
    read_only: bool
    writes_enabled: bool

    def read(
        self,
        *,
        tenant_id: str,
        feature_set_id: str,
        feature_set_version: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]: ...


class FunctionalRetrievalService(Protocol):
    def retrieve(self, request: GovernedAudienceRetrievalRequest) -> dict: ...


class ProductionFeatureSnapshotReaderAdapter:
    """Declare and expose the existing rollback-only production reader."""

    read_only = True
    writes_enabled = False

    def __init__(
        self,
        service: ProductionFeatureSnapshotReaderService | None = None,
    ) -> None:
        self._service = service or ProductionFeatureSnapshotReaderService()

    def read(
        self,
        *,
        tenant_id: str,
        feature_set_id: str,
        feature_set_version: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return self._service.read(
            tenant_id=tenant_id,
            feature_set_id=feature_set_id,
            feature_set_version=feature_set_version,
        )


def _recordable(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_record"):
        value = value.to_record()
    if not isinstance(value, Mapping):
        raise TypeError("Functional service output must be a mapping contract.")
    payload = json.loads(json.dumps(dict(value), default=str))
    assert_no_raw_identifier_fields(payload)
    return payload


def _truthy_effect(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = normalize_taxonomy_value(key)
            if normalized in _EFFECT_FIELDS and value is not False:
                return True
            if _truthy_effect(value):
                return True
    elif isinstance(payload, (list, tuple)):
        return any(_truthy_effect(value) for value in payload)
    return False


def _contains_source_payload_key(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        return any(
            normalize_taxonomy_value(key) in _PROHIBITED_SOURCE_PAYLOAD_KEYS
            or any(
                token in normalize_taxonomy_value(key)
                for token in _PROHIBITED_IDENTIFIER_KEY_TOKENS
            )
            or _contains_source_payload_key(value)
            for key, value in payload.items()
        )
    if isinstance(payload, (list, tuple)):
        return any(_contains_source_payload_key(value) for value in payload)
    return False


def _count(value: Any) -> int:
    try:
        return max(0, min(int(value or 0), 100_000_000))
    except (TypeError, ValueError):
        return 0


class _FunctionalStageRecorder:
    def __init__(self) -> None:
        self.records: list[FunctionalStageRecord] = []
        self.outputs: dict[str, dict[str, Any]] = {}

    def execute(
        self,
        *,
        stage_id: str,
        module_id: int,
        operation: Callable[[], Any],
        metrics: Callable[[dict[str, Any]], Mapping[str, int | float | bool]],
        reason_code: str,
    ) -> dict[str, Any]:
        if any(value.stage_id == stage_id for value in self.records):
            raise ValueError("Functional stage identifiers must be unique.")
        started_tracing = not tracemalloc.is_tracing()
        if started_tracing:
            tracemalloc.start()
        before_current, _ = tracemalloc.get_traced_memory()
        started = time.perf_counter_ns()
        try:
            payload = _recordable(operation())
            if _truthy_effect(payload):
                raise ValueError("Functional service attempted an unsafe effect.")
            elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            _, peak = tracemalloc.get_traced_memory()
            record = FunctionalStageRecord(
                stage_id=stage_id,
                module_id=module_id,
                status="completed",
                latency_ms=elapsed,
                peak_python_bytes=max(0, peak - before_current),
                result_fingerprint=stable_fingerprint(payload),
                reason_codes=(reason_code,),
                metrics=metrics(payload),
            )
            self.records.append(record)
            self.outputs[stage_id] = payload
            return payload
        except Exception:
            elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            _, peak = tracemalloc.get_traced_memory()
            self.records.append(
                FunctionalStageRecord(
                    stage_id=stage_id,
                    module_id=module_id,
                    status="failed",
                    latency_ms=elapsed,
                    peak_python_bytes=max(0, peak - before_current),
                    reason_codes=("functional_service_contract_failure",),
                    metrics={},
                )
            )
            raise
        finally:
            if started_tracing:
                tracemalloc.stop()

    def skip(
        self,
        *,
        stage_id: str,
        module_id: int,
        reason_code: str,
        metrics: Mapping[str, int | float | bool] | None = None,
    ) -> None:
        if any(value.stage_id == stage_id for value in self.records):
            raise ValueError("Functional stage identifiers must be unique.")
        self.records.append(
            FunctionalStageRecord(
                stage_id=stage_id,
                module_id=module_id,
                status="skipped",
                latency_ms=0.0,
                peak_python_bytes=0,
                reason_codes=(reason_code,),
                metrics=metrics or {},
            )
        )


class FunctionalServiceCapabilityAdapterFactory:
    """Invoke real safe services while retaining only bounded stage evidence."""

    def __init__(
        self,
        *,
        request: FunctionalShadowRunRequest,
        goal: AutonomyGoal,
        retrieval_request: GovernedAudienceRetrievalRequest,
        source_reader: FunctionalFeatureSnapshotReader,
        retrieval_service: FunctionalRetrievalService,
        policy: FunctionalShadowPolicy,
        fallback_source_reader: FunctionalFeatureSnapshotReader | None = None,
        cohort_service: ProductionModule3CohortCandidateService | None = None,
        overlap_service: ProductionModule3OverlapDeduplicationService | None = None,
        lookalike_service: ProductionModule3LookalikeService | None = None,
        evolution_snapshot_service: (
            ProductionModule4EvolutionSnapshotService | None
        ) = None,
        drift_service: ProductionModule4DriftDetectionService | None = None,
        evolution_recommendation_service: (
            ProductionModule4EvolutionRecommendationService | None
        ) = None,
        decision_service: AudienceSupervisorDecisionService | None = None,
        baseline_snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        self.request = request
        self.goal = goal
        self.retrieval_request = retrieval_request
        self.source_reader = source_reader
        self.fallback_source_reader = fallback_source_reader
        self.retrieval_service = retrieval_service
        self.policy = policy
        self.cohort_service = (
            cohort_service or ProductionModule3CohortCandidateService()
        )
        self.overlap_service = (
            overlap_service or ProductionModule3OverlapDeduplicationService()
        )
        self.lookalike_service = (
            lookalike_service or ProductionModule3LookalikeService()
        )
        self.evolution_snapshot_service = (
            evolution_snapshot_service or ProductionModule4EvolutionSnapshotService()
        )
        self.drift_service = drift_service or ProductionModule4DriftDetectionService()
        self.evolution_recommendation_service = (
            evolution_recommendation_service
            or ProductionModule4EvolutionRecommendationService()
        )
        self.decision_service = decision_service or AudienceSupervisorDecisionService()
        self.baseline_snapshot = (
            dict(baseline_snapshot) if isinstance(baseline_snapshot, Mapping) else None
        )
        self.recorder = _FunctionalStageRecorder()
        self.functional_decision: dict[str, Any] | None = None
        self.functional_summary: dict[str, Any] | None = None

    def handlers(self) -> Mapping[str, CapabilityHandler]:
        values: dict[str, CapabilityHandler] = {
            "module1_provider_source_discovery": self._source_discovery,
            "module1_privacy_safe_aggregation": self._privacy_assessment,
            "module2_semantic_retrieval": self._retrieval,
            "module3_cohort_strategy": self._cohort_strategy,
            "module4_evolution_review": self._evolution_review,
            "module5_governed_recommendation": self._recommendation,
            "module5_delivery_review": self._delivery_review,
        }
        if self.fallback_source_reader is not None:
            values["module1_public_source_discovery"] = self._fallback_discovery
        return values

    def _source_discovery(self, _goal, _capability, _invocation):
        payload = self._read_source(self.source_reader, "module1_source_snapshot")
        return CapabilityExecutionResult(
            status="completed",
            provided_outcomes=("source_inventory", "coverage_assessment"),
            reason_codes=("read_only_feature_snapshot_loaded",),
            metrics={
                "safe_feature_row_count": len(payload["feature_rows"]),
                "source_is_fresh": payload["freshness_status"] == "fresh",
            },
            output_values={
                "source_inventory": payload,
                "coverage_assessment": {
                    "safe_feature_row_count": len(payload["feature_rows"]),
                    "freshness_status": payload["freshness_status"],
                },
            },
        )

    def _fallback_discovery(self, _goal, _capability, _invocation):
        if self.fallback_source_reader is None:
            raise RuntimeError("Functional fallback source reader is unavailable.")
        payload = self._read_source(
            self.fallback_source_reader,
            "module1_fallback_source_snapshot",
        )
        return CapabilityExecutionResult(
            status="completed",
            provided_outcomes=("source_inventory", "coverage_assessment"),
            reason_codes=("read_only_fallback_snapshot_loaded",),
            metrics={"safe_feature_row_count": len(payload["feature_rows"])},
            output_values={
                "source_inventory": payload,
                "coverage_assessment": {
                    "safe_feature_row_count": len(payload["feature_rows"]),
                    "freshness_status": payload["freshness_status"],
                },
            },
        )

    def _read_source(
        self,
        reader: FunctionalFeatureSnapshotReader,
        stage_id: str,
    ) -> dict[str, Any]:
        if reader.read_only is not True or reader.writes_enabled is not False:
            raise ValueError("Functional source adapter must be explicitly read-only.")

        def operation() -> dict[str, Any]:
            feature_set, rows = reader.read(
                tenant_id=self.request.tenant_id,
                feature_set_id=self.request.feature_set_id,
                feature_set_version=self.request.feature_set_version,
            )
            payload = {
                "feature_set": dict(feature_set),
                "feature_rows": [dict(value) for value in rows],
            }
            assert_no_raw_identifier_fields(payload)
            if _contains_source_payload_key(payload):
                raise ValueError("Functional source snapshot contains vector payloads.")
            if len(payload["feature_rows"]) > self.policy.max_safe_feature_rows:
                raise ValueError("Functional source snapshot exceeds the row bound.")
            if normalize_taxonomy_value(feature_set.get("tenant_id")) != (
                normalize_taxonomy_value(self.request.tenant_id)
            ):
                raise ValueError("Functional source snapshot tenant mismatch.")
            if str(feature_set.get("feature_set_id") or "") != (
                self.request.feature_set_id
            ) or int(feature_set.get("version") or 0) != (
                self.request.feature_set_version
            ):
                raise ValueError("Functional source snapshot lineage mismatch.")
            if not bool(feature_set.get("eligible_for_retrieval")):
                raise ValueError(
                    "Functional source snapshot is not retrieval eligible."
                )
            if int(feature_set.get("feature_count") or 0) != len(
                payload["feature_rows"]
            ):
                raise ValueError("Functional source feature count mismatch.")
            payload["freshness_status"] = normalize_taxonomy_value(
                feature_set.get("freshness_status") or "unknown"
            )
            return payload

        return self.recorder.execute(
            stage_id=stage_id,
            module_id=1,
            operation=operation,
            metrics=lambda value: {
                "safe_feature_row_count": len(value["feature_rows"]),
                "source_is_fresh": value["freshness_status"] == "fresh",
            },
            reason_code="read_only_snapshot_evaluated",
        )

    def _privacy_assessment(self, _goal, _capability, invocation):
        inventory = dict(invocation["input_values"]["source_inventory"] or {})

        def operation() -> dict[str, Any]:
            rows = [dict(value) for value in inventory.get("feature_rows") or []]
            assert_no_raw_identifier_fields(rows)
            safe_rows = sum(
                normalize_taxonomy_value(value.get("privacy_status"))
                in SAFE_PRIVACY_STATUSES
                for value in rows
            )
            eligible_rows = sum(
                bool(value.get("eligible_for_retrieval")) for value in rows
            )
            return {
                **inventory,
                "privacy_safe_metadata_row_count": safe_rows,
                "retrieval_eligible_row_count": eligible_rows,
                "raw_identifiers_present": False,
                "privacy_transformation_performed": False,
                "source_snapshot_modified": False,
            }

        payload = self.recorder.execute(
            stage_id="module1_privacy_validation",
            module_id=1,
            operation=operation,
            metrics=lambda value: {
                "safe_metadata_row_count": value["privacy_safe_metadata_row_count"],
                "retrieval_eligible_row_count": value["retrieval_eligible_row_count"],
            },
            reason_code="aggregate_privacy_metadata_validated",
        )
        return CapabilityExecutionResult(
            status="completed",
            provided_outcomes=("privacy_safe_dataset", "privacy_assessment"),
            reason_codes=("aggregate_privacy_contract_validated",),
            metrics={
                "safe_metadata_row_count": payload["privacy_safe_metadata_row_count"],
                "raw_identifiers_present": False,
            },
            output_values={
                "privacy_safe_dataset": payload,
                "privacy_assessment": {
                    "raw_identifiers_present": False,
                    "privacy_safe_metadata_row_count": payload[
                        "privacy_safe_metadata_row_count"
                    ],
                },
            },
        )

    def _retrieval(self, _goal, _capability, _invocation):
        def operation() -> dict[str, Any]:
            value = _recordable(self.retrieval_service.retrieve(self.retrieval_request))
            if normalize_taxonomy_value(value.get("tenant_id")) != (
                normalize_taxonomy_value(self.request.tenant_id)
            ):
                raise ValueError("Functional retrieval tenant mismatch.")
            if value.get("status") not in {
                "retrieval_ready_for_human_review",
                "blocked",
            }:
                raise ValueError("Unsupported functional retrieval status.")
            return value

        payload = self.recorder.execute(
            stage_id="module2_governed_retrieval",
            module_id=2,
            operation=operation,
            metrics=lambda value: {
                "candidate_pool_count": _count(
                    value.get("retrieval", {}).get("candidate_pool_size")
                ),
                "selected_candidate_count": len(value.get("selected_candidates") or []),
                "retrieval_blocked": value.get("status") == "blocked",
            },
            reason_code="governed_retrieval_executed",
        )
        return CapabilityExecutionResult(
            status="completed",
            provided_outcomes=("retrieval_evidence",),
            reason_codes=(
                "retrieval_completed"
                if payload.get("status") != "blocked"
                else "retrieval_safely_blocked",
            ),
            metrics={
                "candidate_pool_count": _count(
                    payload.get("retrieval", {}).get("candidate_pool_size")
                ),
                "selected_candidate_count": len(
                    payload.get("selected_candidates") or []
                ),
            },
            output_values={"retrieval_evidence": payload},
        )

    def _cohort_strategy(self, _goal, _capability, invocation):
        dataset = dict(self.recorder.outputs.get("module1_privacy_validation") or {})
        retrieval = dict(invocation["input_values"]["retrieval_evidence"] or {})

        def operation() -> dict[str, Any]:
            selected_ids = {
                str(
                    value.get("feature_id")
                    or (value.get("feature_ids") or {}).get("primary")
                    or ""
                )
                for value in retrieval.get("selected_candidates") or []
                if isinstance(value, Mapping)
            }
            rows = [
                dict(value)
                for value in dataset.get("feature_rows") or []
                if str(value.get("feature_id") or "") in selected_ids
            ]
            candidate_batch = self.cohort_service.generate(
                request=Module3CohortGenerationRequest(
                    tenant_id=self.request.tenant_id,
                    feature_set_id=self.request.feature_set_id,
                    feature_set_version=self.request.feature_set_version,
                    execution_mode="historical_preview",
                    purpose=self.request.purpose,
                ),
                feature_set=dict(dataset.get("feature_set") or {}),
                feature_rows=rows,
            ).to_record()
            overlap = self.overlap_service.analyze(
                request=Module3OverlapDeduplicationRequest(
                    tenant_id=self.request.tenant_id,
                    batch_fingerprint=candidate_batch["batch_fingerprint"],
                    execution_mode="historical_preview",
                    purpose=self.request.purpose,
                ),
                candidate_batch=candidate_batch,
            ).to_record()
            lookalike = self.lookalike_service.generate(
                request=Module3LookalikeRequest(
                    tenant_id=self.request.tenant_id,
                    overlap_report_fingerprint=overlap["report_fingerprint"],
                    execution_mode="historical_preview",
                    purpose=self.request.purpose,
                ),
                overlap_report=overlap,
            ).to_record()
            return {
                "candidate_batch": candidate_batch,
                "overlap_report": overlap,
                "lookalike_report": lookalike,
            }

        payload = self.recorder.execute(
            stage_id="module3_cohort_intelligence",
            module_id=3,
            operation=operation,
            metrics=lambda value: {
                "generated_candidate_count": _count(
                    value["candidate_batch"].get("generated_candidate_count")
                ),
                "retained_candidate_count": _count(
                    value["overlap_report"].get("retained_candidate_count")
                ),
                "overlap_review_group_count": _count(
                    value["overlap_report"].get("potential_overlap_group_count")
                ),
                "lookalike_pair_count": _count(
                    value["lookalike_report"].get("generated_lookalike_candidate_count")
                ),
            },
            reason_code="aggregate_cohort_services_executed",
        )
        candidate_batch = payload["candidate_batch"]
        return CapabilityExecutionResult(
            status="completed",
            provided_outcomes=("audience_candidates",),
            reason_codes=("aggregate_candidates_review_only",),
            metrics={
                "generated_candidate_count": _count(
                    candidate_batch.get("generated_candidate_count")
                ),
                "lookalike_pair_count": _count(
                    payload["lookalike_report"].get(
                        "generated_lookalike_candidate_count"
                    )
                ),
            },
            output_values={"audience_candidates": payload},
        )

    def _evolution_review(self, _goal, _capability, invocation):
        audience = dict(invocation["input_values"]["audience_candidates"] or {})
        retained = [
            dict(value)
            for value in audience.get("overlap_report", {}).get("retained_candidates")
            or []
            if isinstance(value, Mapping)
        ]
        if not retained:
            payload = {
                "status": "not_evaluated_no_candidates",
                "current_snapshot": None,
                "drift_report": None,
                "recommendation_report": None,
                "candidate_count": 0,
                "lifecycle_mutated": False,
                "production_routing_enabled": False,
                "activation_or_export_performed": False,
                "downstream_export_enabled": False,
            }
            self.recorder.skip(
                stage_id="module4_evolution_review",
                module_id=4,
                reason_code="no_candidates_for_evolution",
                metrics={"candidate_count": 0},
            )
        else:

            def operation() -> dict[str, Any]:
                cohort_rows = [
                    {
                        "export_cohort_id": value["candidate_id"],
                        "management_quality_score": value["quality_score"],
                        "freshness_status": value["freshness_status"],
                        "approval_status": "pending_approval",
                        "data_safety_status": "safe_aggregate",
                        "risk_decision": value["sensitive_poi_decision"],
                    }
                    for value in retained
                ]
                current = self.evolution_snapshot_service.build(
                    request=Module4EvolutionSnapshotRequest(
                        tenant_id=self.request.tenant_id,
                        source_run_id=self.request.functional_run_id,
                        execution_mode="historical_preview",
                    ),
                    cohort_rows=cohort_rows,
                ).to_record()
                drift = None
                recommendations = None
                if self.baseline_snapshot is not None:
                    baseline = self.evolution_snapshot_service.validate_report(
                        self.baseline_snapshot
                    )
                    drift = self.drift_service.analyze(
                        request=Module4DriftAnalysisRequest(
                            tenant_id=self.request.tenant_id,
                            baseline_snapshot_fingerprint=baseline[
                                "snapshot_fingerprint"
                            ],
                            current_snapshot_fingerprint=current[
                                "snapshot_fingerprint"
                            ],
                            execution_mode="historical_preview",
                        ),
                        baseline_snapshot=baseline,
                        current_snapshot=current,
                    )
                    recommendations = self.evolution_recommendation_service.recommend(
                        drift_report=drift
                    )
                return {
                    "status": "engineering_preview_ready",
                    "current_snapshot": current,
                    "drift_report": drift,
                    "recommendation_report": recommendations,
                    "candidate_count": len(retained),
                    "lifecycle_mutated": False,
                    "production_routing_enabled": False,
                    "activation_or_export_performed": False,
                    "downstream_export_enabled": False,
                }

            payload = self.recorder.execute(
                stage_id="module4_evolution_review",
                module_id=4,
                operation=operation,
                metrics=lambda value: {
                    "snapshot_cohort_count": _count(
                        (value.get("current_snapshot") or {}).get("source_cohort_count")
                    ),
                    "drift_changed_count": _count(
                        (value.get("drift_report") or {}).get("changed_count")
                    ),
                    "recommendation_count": _count(
                        (value.get("recommendation_report") or {}).get(
                            "recommendation_count"
                        )
                    ),
                },
                reason_code="evolution_services_executed",
            )
        return CapabilityExecutionResult(
            status="completed",
            provided_outcomes=("evolution_assessment",),
            reason_codes=(
                "evolution_review_only"
                if retained
                else "evolution_not_evaluated_no_candidates",
            ),
            metrics={
                "candidate_count": len(retained),
                "drift_evaluated": payload.get("drift_report") is not None,
            },
            output_values={"evolution_assessment": payload},
        )

    def _recommendation(self, _goal, _capability, invocation):
        audience = dict(invocation["input_values"]["audience_candidates"] or {})
        evolution = dict(invocation["input_values"]["evolution_assessment"] or {})
        source = self.recorder.outputs.get("module1_source_snapshot") or (
            self.recorder.outputs.get("module1_fallback_source_snapshot") or {}
        )
        retrieval = self.recorder.outputs.get("module2_governed_retrieval") or {}
        candidate_batch = audience.get("candidate_batch") or {}

        def operation() -> dict[str, Any]:
            freshness = normalize_taxonomy_value(
                source.get("freshness_status") or "unknown"
            )
            candidate_count = _count(candidate_batch.get("generated_candidate_count"))
            if freshness != "fresh":
                approval_status = "blocked_stale_source"
            elif candidate_count == 0:
                approval_status = "blocked_no_safe_exact_match"
            else:
                approval_status = "pending_approval"
            pipeline_result = {
                "status": "completed",
                "freshness_status": freshness,
                "approval_status": approval_status,
                "approval_required": True,
                "downstream_export_enabled": False,
                "prompt_selected_cohorts": candidate_count,
                "prepared_audience_candidates": 0,
                "safe_export": {
                    "approval_status": approval_status,
                    "approval_required": True,
                    "exported_cohorts": 0,
                    "downstream_export_enabled": False,
                },
            }
            decision = self.decision_service.decide(pipeline_result)
            summary = {
                "route": decision["route"],
                "stage": decision["stage"],
                "freshness_status": freshness,
                "source_evaluated": True,
                "safe_feature_row_count": len(source.get("feature_rows") or []),
                "vector_count": _count(
                    source.get("feature_set", {}).get("feature_count")
                ),
                "ranked_match_count": _count(
                    retrieval.get("retrieval", {}).get("candidate_pool_size")
                ),
                "selected_cohort_count": candidate_count,
                "prepared_candidate_count": 0,
                "terminal": bool(decision.get("terminal", True)),
                "approval_required": bool(decision.get("approval_required", True)),
                "downstream_export_enabled": False,
                "raw_identifiers_returned": False,
                "activation_or_export_performed": False,
            }
            return {
                "status": "review_only",
                "decision": decision,
                "functional_summary": summary,
                "evolution_evaluated": evolution.get("current_snapshot") is not None,
                "manual_approval_required": True,
                "automatic_approval_performed": False,
                "automatic_mutation_performed": False,
                "production_routing_enabled": False,
                "activation_or_export_performed": False,
                "downstream_export_enabled": False,
            }

        payload = self.recorder.execute(
            stage_id="module5_governed_recommendation",
            module_id=5,
            operation=operation,
            metrics=lambda value: {
                "selected_cohort_count": value["functional_summary"][
                    "selected_cohort_count"
                ],
                "approval_required": value["functional_summary"]["approval_required"],
                "terminal": value["functional_summary"]["terminal"],
            },
            reason_code="deterministic_governed_recommendation",
        )
        self.functional_decision = dict(payload["decision"])
        self.functional_summary = dict(payload["functional_summary"])
        return CapabilityExecutionResult(
            status="completed",
            provided_outcomes=("governed_recommendation",),
            reason_codes=("manual_review_recommendation_created",),
            metrics={
                "selected_cohort_count": self.functional_summary[
                    "selected_cohort_count"
                ],
                "downstream_export_enabled": False,
            },
            output_values={"governed_recommendation": payload},
        )

    def _delivery_review(self, _goal, _capability, invocation):
        recommendation = dict(
            invocation["input_values"]["governed_recommendation"] or {}
        )

        def operation() -> dict[str, Any]:
            return {
                "status": "manual_review_required",
                "decision_route": recommendation.get("decision", {}).get("route"),
                "approval_required": True,
                "delivery_authorized": False,
                "production_effect_performed": False,
                "activation_or_export_performed": False,
                "downstream_export_enabled": False,
            }

        payload = self.recorder.execute(
            stage_id="module5_delivery_review",
            module_id=5,
            operation=operation,
            metrics=lambda value: {
                "approval_required": value["approval_required"],
                "delivery_authorized": value["delivery_authorized"],
            },
            reason_code="delivery_kept_manual_and_disabled",
        )
        return CapabilityExecutionResult(
            status="completed",
            provided_outcomes=("delivery_review",),
            reason_codes=("delivery_not_authorized",),
            metrics={"delivery_authorized": False},
            output_values={"delivery_review": payload},
        )


class ProductionBoundedAutonomyFunctionalShadowService:
    """Run real safe module services in an isolated, non-authorizing shadow."""

    def __init__(
        self,
        *,
        source_reader: FunctionalFeatureSnapshotReader | None = None,
        retrieval_service: FunctionalRetrievalService | None = None,
        policy: FunctionalShadowPolicy | None = None,
        bounded_service: ProductionBoundedAutonomyService | None = None,
        decision_service: AudienceSupervisorDecisionService | None = None,
        safety_service: AudienceProposalRequestSafetyService | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.source_reader = source_reader
        self.retrieval_service = retrieval_service
        self._environment = environment if environment is not None else os.environ
        self.policy = policy or self._policy_from_environment()
        self.bounded_service = bounded_service or ProductionBoundedAutonomyService()
        self.decision_service = decision_service or AudienceSupervisorDecisionService()
        self.safety_service = safety_service or AudienceProposalRequestSafetyService()

    def run(
        self,
        *,
        request: FunctionalShadowRunRequest,
        goal: AutonomyGoal,
        retrieval_request: GovernedAudienceRetrievalRequest,
        legacy_result: Mapping[str, Any],
        authorization_context: AgentAuthorizationContext | None = None,
        baseline_snapshot: Mapping[str, Any] | None = None,
        fallback_source_reader: FunctionalFeatureSnapshotReader | None = None,
    ) -> dict[str, Any]:
        self._validate_inputs(request, goal, retrieval_request)
        legacy = LegacyOrchestratorObservationAdapter(self.decision_service).adapt(
            goal=goal, legacy_result=legacy_result
        )
        safety_started = time.perf_counter_ns()
        terminal = self.safety_service.evaluate({"audience_intent": goal.objective})
        safety_latency = (time.perf_counter_ns() - safety_started) / 1_000_000.0
        if terminal.terminal:
            return self._terminal_report(
                request=request,
                goal=goal,
                legacy=legacy.to_record(),
                reason_code=str(terminal.reason_code or "terminal_safety_block"),
                latency_ms=safety_latency,
            )
        if self.source_reader is None or self.retrieval_service is None:
            raise RuntimeError(
                "Functional shadow requires configured read-only source and "
                "retrieval service adapters."
            )
        if authorization_context is None:
            raise ValueError(
                "Functional capability execution requires an authorization context."
            )
        self._validate_authorization_context(
            request=request,
            goal=goal,
            authorization_context=authorization_context,
        )
        factory = FunctionalServiceCapabilityAdapterFactory(
            request=request,
            goal=goal,
            retrieval_request=retrieval_request,
            source_reader=self.source_reader,
            fallback_source_reader=fallback_source_reader,
            retrieval_service=self.retrieval_service,
            policy=self.policy,
            decision_service=self.decision_service,
            baseline_snapshot=baseline_snapshot,
        )
        enforcer = AgentCapabilityAuthorizationEnforcer(
            goal=goal,
            authorization_context=authorization_context,
            registry=self.bounded_service.registry,
        )
        started = time.perf_counter_ns()
        bounded = self.bounded_service.run(
            goal=goal,
            handlers=enforcer.wrap(factory.handlers()),
        )
        total_latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if factory.functional_summary is None:
            failed_decision = self.decision_service.decide(
                {"status": "failed", "error": "functional_shadow_failed"}
            )
            summary = {
                "route": failed_decision["route"],
                "stage": failed_decision["stage"],
                "freshness_status": "not_evaluated",
                "source_evaluated": any(
                    value.module_id == 1 and value.status == "completed"
                    for value in factory.recorder.records
                ),
                "safe_feature_row_count": 0,
                "vector_count": 0,
                "ranked_match_count": 0,
                "selected_cohort_count": 0,
                "prepared_candidate_count": 0,
                "terminal": True,
                "approval_required": True,
                "downstream_export_enabled": False,
                "raw_identifiers_returned": False,
                "activation_or_export_performed": False,
            }
        else:
            summary = factory.functional_summary
        return self._build_report(
            request=request,
            goal=goal,
            legacy=legacy.to_record(),
            bounded_report=bounded,
            stage_records=factory.recorder.records,
            functional_summary=summary,
            total_latency_ms=total_latency_ms,
            terminal_preflight=False,
            authorization_evidence=enforcer.minimized_evidence(),
        )

    def _validate_authorization_context(
        self,
        *,
        request: FunctionalShadowRunRequest,
        goal: AutonomyGoal,
        authorization_context: AgentAuthorizationContext,
    ) -> None:
        principal = authorization_context.principal
        if authorization_context.request_id != request.request_id:
            raise ValueError("Functional authorization request lineage mismatch.")
        if principal.tenant_id != request.tenant_id:
            raise ValueError("Functional authorization tenant lineage mismatch.")
        if principal.principal_type != "bounded_agent":
            raise ValueError("Functional shadow requires a bounded agent principal.")
        if "audience:read" not in principal.scopes or (
            "audience:propose" not in principal.scopes
        ):
            raise ValueError("Functional agent requires read and propose scopes.")
        if authorization_context.request_id != goal.request_id:
            raise ValueError("Functional authorization goal lineage mismatch.")

    def _validate_inputs(self, request, goal, retrieval_request) -> None:
        if goal.execution_mode == "production":
            raise ValueError("Functional shadow cannot execute a production goal.")
        if goal.execution_mode not in {"shadow", "offline_evaluation"}:
            raise ValueError("Functional shadow goal mode must be shadow or offline.")
        if retrieval_request.execution_mode != "historical_preview":
            raise ValueError("Functional retrieval must use historical_preview mode.")
        if not (request.tenant_id == goal.tenant_id == retrieval_request.tenant_id):
            raise ValueError("Functional shadow tenant lineage mismatch.")
        if request.request_id != goal.request_id:
            raise ValueError("Functional shadow request lineage mismatch.")
        if request.feature_set_id != retrieval_request.primary_model.feature_set_id or (
            request.feature_set_version
            != retrieval_request.primary_model.feature_set_version
        ):
            raise ValueError("Functional shadow primary feature-set lineage mismatch.")
        if " ".join(goal.objective.split()) != " ".join(
            retrieval_request.query_text.split()
        ):
            raise ValueError("Functional retrieval query must match the autonomy goal.")

    def _terminal_report(
        self,
        *,
        request: FunctionalShadowRunRequest,
        goal: AutonomyGoal,
        legacy: Mapping[str, Any],
        reason_code: str,
        latency_ms: float,
    ) -> dict[str, Any]:
        if reason_code == "blocked_privacy_identifier_request":
            filter_mode = "privacy_identifier_request_blocked"
        elif reason_code == "blocked_approval_bypass_attempt":
            filter_mode = "approval_bypass_attempt_blocked"
        else:
            filter_mode = "export_action_requires_existing_audience"
        decision = self.decision_service.decide(
            {
                "status": "skipped",
                "approval_status": reason_code,
                "prompt_filter_report": {"filter_mode": filter_mode},
                "approval_required": True,
                "downstream_export_enabled": False,
            }
        )
        summary = {
            "route": decision["route"],
            "stage": decision["stage"],
            "freshness_status": "not_evaluated",
            "source_evaluated": False,
            "safe_feature_row_count": 0,
            "vector_count": 0,
            "ranked_match_count": 0,
            "selected_cohort_count": 0,
            "prepared_candidate_count": 0,
            "terminal": True,
            "approval_required": True,
            "downstream_export_enabled": False,
            "raw_identifiers_returned": False,
            "activation_or_export_performed": False,
        }
        record = FunctionalStageRecord(
            stage_id="module5_terminal_safety_preflight",
            module_id=5,
            status="completed",
            latency_ms=latency_ms,
            peak_python_bytes=0,
            result_fingerprint=stable_fingerprint(
                {
                    "reason_code": reason_code,
                    "route": decision["route"],
                    "stage": decision["stage"],
                }
            ),
            reason_codes=(required_metadata_token(reason_code, label="reason_code"),),
            metrics={"source_evaluated": False, "terminal": True},
        )
        return self._build_report(
            request=request,
            goal=goal,
            legacy=legacy,
            bounded_report=None,
            stage_records=[record],
            functional_summary=summary,
            total_latency_ms=latency_ms,
            terminal_preflight=True,
            authorization_evidence={
                "policy_version": AGENT_AUTHORIZATION_POLICY_VERSION,
                "principal_fingerprint": None,
                "authorization_context_fingerprint": None,
                "decision_count": 0,
                "allowed_decision_count": 0,
                "denied_decision_count": 0,
                "decision_fingerprints": [],
                "terminal_safety_preflight": True,
                "credential_material_stored": False,
                "authentication_token_stored": False,
            },
        )

    def _build_report(
        self,
        *,
        request: FunctionalShadowRunRequest,
        goal: AutonomyGoal,
        legacy: Mapping[str, Any],
        bounded_report: Mapping[str, Any] | None,
        stage_records: Sequence[FunctionalStageRecord],
        functional_summary: Mapping[str, Any],
        total_latency_ms: float,
        terminal_preflight: bool,
        authorization_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        comparison = self._compare(legacy, functional_summary)
        peak_python_bytes = max(
            (value.peak_python_bytes for value in stage_records),
            default=0,
        )
        stage_latency_passed = all(
            value.latency_ms <= self.policy.max_stage_latency_ms
            for value in stage_records
        )
        resource_gates = {
            "total_latency": {
                "passed": total_latency_ms <= self.policy.max_total_latency_ms,
                "observed_ms": round(total_latency_ms, 6),
                "maximum_ms": self.policy.max_total_latency_ms,
            },
            "stage_latency": {
                "passed": stage_latency_passed,
                "maximum_ms": self.policy.max_stage_latency_ms,
            },
            "python_allocation": {
                "passed": peak_python_bytes <= self.policy.max_peak_python_bytes,
                "observed_peak_bytes": peak_python_bytes,
                "maximum_peak_bytes": self.policy.max_peak_python_bytes,
            },
        }
        stages_safe = all(
            value.status in {"completed", "skipped"} for value in stage_records
        )
        bounded_ready = bool(
            terminal_preflight
            or (
                bounded_report
                and bounded_report.get("status") == "engineering_preview_ready"
            )
        )
        eligible = bool(
            bounded_ready
            and stages_safe
            and int(authorization_evidence.get("denied_decision_count") or 0)
            == 0
            and comparison["critical_divergence_count"] == 0
            and comparison["overall_divergence_count"] == 0
            and all(value["passed"] for value in resource_gates.values())
        )
        report = {
            "status": (
                "engineering_preview_ready"
                if eligible
                else "engineering_preview_blocked"
            ),
            "policy_version": BOUNDED_AUTONOMY_FUNCTIONAL_SHADOW_VERSION,
            "request": request.to_record(),
            "goal": goal.to_record(),
            "policy": self.policy.to_record(),
            "lineage": {
                "source_certification_report_fingerprint": request.source_certification_report_fingerprint,
                "bounded_autonomy_report_fingerprint": (
                    bounded_report.get("bounded_autonomy_report_fingerprint")
                    if bounded_report
                    else None
                ),
            },
            "legacy_observation": dict(legacy),
            "functional_summary": dict(functional_summary),
            "functional_execution": {
                "terminal_preflight": terminal_preflight,
                "stage_count": len(stage_records),
                "completed_stage_count": sum(
                    value.status == "completed" for value in stage_records
                ),
                "skipped_stage_count": sum(
                    value.status == "skipped" for value in stage_records
                ),
                "failed_stage_count": sum(
                    value.status in {"failed", "blocked"} for value in stage_records
                ),
                "stage_records": [value.to_record() for value in stage_records],
                "total_latency_ms": round(float(total_latency_ms), 6),
                "peak_python_bytes": peak_python_bytes,
                "full_service_outputs_stored": False,
            },
            "authorization": dict(authorization_evidence),
            "comparison": comparison,
            "resource_gates": resource_gates,
            "review": {
                "eligible_for_staging_review": eligible,
                "live_cutover_authorized": False,
                "fresh_data_certified": False,
                "automatic_cutover_performed": False,
            },
            "safety": {
                "shadow_only": True,
                "isolated_historical_snapshot_only": True,
                "read_only_feature_access": True,
                "prompt_content_stored": False,
                "query_text_stored": False,
                "tool_arguments_stored": False,
                "tool_results_stored": False,
                "full_service_outputs_stored": False,
                "raw_identifiers_read": False,
                "raw_identifiers_stored": False,
                "raw_identifiers_returned": False,
                "database_write_performed": False,
                "capability_authorization_enforced": True,
                "agent_approval_authority": False,
                "agent_delivery_authority": False,
                "automatic_approval_performed": False,
                "automatic_mutation_performed": False,
                "production_routing_changed": False,
                "production_effect_performed": False,
                "activation_or_export_performed": False,
                "downstream_export_enabled": False,
                "manual_approval_required": True,
            },
        }
        report["functional_shadow_report_fingerprint"] = stable_fingerprint(report)
        return self.validate_report(report)

    def _compare(
        self,
        legacy: Mapping[str, Any],
        functional: Mapping[str, Any],
    ) -> dict[str, Any]:
        checks = {
            "route": (
                not self.policy.require_route_match
                or legacy.get("decision_route") == functional.get("route")
            ),
            "stage": (
                not self.policy.require_stage_match
                or legacy.get("decision_stage") == functional.get("stage")
            ),
            "freshness": (
                not self.policy.require_freshness_match
                or legacy.get("freshness_status") == functional.get("freshness_status")
            ),
            "source_evaluated": (
                bool(legacy.get("source_evaluated"))
                == bool(functional.get("source_evaluated"))
            ),
            "vector_count": abs(
                _count(legacy.get("vector_count"))
                - _count(functional.get("vector_count"))
            )
            <= self.policy.max_vector_count_delta,
            "ranked_match_count": abs(
                _count(legacy.get("ranked_match_count"))
                - _count(functional.get("ranked_match_count"))
            )
            <= self.policy.max_ranked_match_delta,
            "selected_cohort_count": abs(
                _count(legacy.get("selected_cohort_count"))
                - _count(functional.get("selected_cohort_count"))
            )
            <= self.policy.max_selected_cohort_delta,
            "terminal": (
                not self.policy.require_terminal_match
                or bool(legacy.get("terminal")) == bool(functional.get("terminal"))
            ),
            "approval_required": (
                not self.policy.require_approval_match
                or bool(legacy.get("approval_required"))
                == bool(functional.get("approval_required"))
            ),
            "downstream_export_disabled": (
                legacy.get("downstream_export_enabled") is False
                and functional.get("downstream_export_enabled") is False
            ),
            "raw_identifiers_absent": (
                legacy.get("raw_identifiers_returned") is False
                and functional.get("raw_identifiers_returned") is False
            ),
            "production_effects_absent": (
                legacy.get("activation_or_export_performed") is False
                and functional.get("activation_or_export_performed") is False
            ),
        }
        critical_names = {
            "route",
            "stage",
            "terminal",
            "approval_required",
            "downstream_export_disabled",
            "raw_identifiers_absent",
            "production_effects_absent",
        }
        divergences = sorted(name for name, passed in checks.items() if not passed)
        return {
            "checks": checks,
            "divergence_codes": [f"{name}_divergence" for name in divergences],
            "overall_divergence_count": len(divergences),
            "critical_divergence_count": sum(
                name in critical_names for name in divergences
            ),
            "source_row_count_comparison_performed": False,
            "source_row_count_comparison_reason": (
                "legacy_source_rows_and_safe_feature_rows_have_different_semantics"
            ),
            "prepared_candidate_count_comparison_performed": False,
            "prepared_candidate_count_comparison_reason": (
                "functional_shadow_does_not_persist_audience_candidates"
            ),
        }

    def validate_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        payload = json.loads(json.dumps(dict(report)))
        assert_no_raw_identifier_fields(payload)
        if payload.get("status") not in {
            "engineering_preview_ready",
            "engineering_preview_blocked",
        }:
            raise ValueError("Unsupported functional shadow report status.")
        if payload.get("policy_version") != (
            BOUNDED_AUTONOMY_FUNCTIONAL_SHADOW_VERSION
        ):
            raise ValueError("Unsupported functional shadow policy version.")
        request = FunctionalShadowRunRequest(**dict(payload.get("request") or {}))
        policy = FunctionalShadowPolicy(**dict(payload.get("policy") or {}))
        goal = payload.get("goal")
        execution = payload.get("functional_execution")
        authorization = payload.get("authorization")
        comparison = payload.get("comparison")
        review = payload.get("review")
        safety = payload.get("safety")
        if not all(
            isinstance(value, Mapping)
            for value in (
                goal,
                execution,
                authorization,
                comparison,
                review,
                safety,
            )
        ):
            raise ValueError("Functional shadow evidence sections are required.")
        serialized = json.dumps(payload, sort_keys=True).lower()
        if '"objective"' in serialized or '"query_text"' in serialized:
            raise ValueError(
                "Prompt or query content cannot enter functional evidence."
            )
        stages = [
            FunctionalStageRecord(**dict(value))
            for value in execution.get("stage_records") or []
            if isinstance(value, Mapping)
        ]
        if len(stages) != int(execution.get("stage_count") or 0):
            raise ValueError("Functional stage evidence count is inconsistent.")
        if len({value.stage_id for value in stages}) != len(stages):
            raise ValueError("Functional stage evidence must be unique.")
        if (
            any(value.latency_ms > policy.max_stage_latency_ms for value in stages)
            and payload.get("status") != "engineering_preview_blocked"
        ):
            raise ValueError("Stage latency budget violation must block evidence.")
        if payload.get("legacy_observation", {}).get("tenant_id") != request.tenant_id:
            raise ValueError("Functional legacy observation tenant mismatch.")
        if authorization.get("policy_version") != AGENT_AUTHORIZATION_POLICY_VERSION:
            raise ValueError("Functional authorization policy is unsupported.")
        decision_count = int(authorization.get("decision_count") or 0)
        allowed_count = int(authorization.get("allowed_decision_count") or 0)
        denied_count = int(authorization.get("denied_decision_count") or 0)
        decision_fingerprints = authorization.get("decision_fingerprints")
        if not isinstance(decision_fingerprints, list):
            raise TypeError("Functional authorization decisions are required.")
        if decision_count != allowed_count + denied_count or (
            decision_count != len(decision_fingerprints)
        ):
            raise ValueError("Functional authorization counts are inconsistent.")
        for value in decision_fingerprints:
            required_sha256_digest(
                value,
                label="authorization_decision_fingerprint",
            )
        terminal_preflight = execution.get("terminal_preflight") is True
        if terminal_preflight:
            if decision_count != 0 or authorization.get(
                "terminal_safety_preflight"
            ) is not True:
                raise ValueError("Terminal safety authorization evidence is invalid.")
        else:
            required_sha256_digest(
                authorization.get("principal_fingerprint"),
                label="principal_fingerprint",
            )
            required_sha256_digest(
                authorization.get("authorization_context_fingerprint"),
                label="authorization_context_fingerprint",
            )
            if decision_count < 1:
                raise ValueError("Functional capabilities require authorization.")
        for field in (
            "credential_material_stored",
            "authentication_token_stored",
        ):
            if authorization.get(field) is not False:
                raise ValueError(f"Unsafe functional authorization field: {field}.")
        lineage = payload.get("lineage")
        if not isinstance(lineage, Mapping):
            raise TypeError("Functional shadow lineage is required.")
        if (
            required_sha256_digest(
                lineage.get("source_certification_report_fingerprint"),
                label="source_certification_report_fingerprint",
            )
            != request.source_certification_report_fingerprint
        ):
            raise ValueError("Functional certification lineage mismatch.")
        if (
            review.get("live_cutover_authorized") is not False
            or review.get("fresh_data_certified") is not False
            or review.get("automatic_cutover_performed") is not False
        ):
            raise ValueError("Functional shadow cannot authorize production cutover.")
        required_false = (
            "prompt_content_stored",
            "query_text_stored",
            "tool_arguments_stored",
            "tool_results_stored",
            "full_service_outputs_stored",
            "raw_identifiers_read",
            "raw_identifiers_stored",
            "raw_identifiers_returned",
            "database_write_performed",
            "agent_approval_authority",
            "agent_delivery_authority",
            "automatic_approval_performed",
            "automatic_mutation_performed",
            "production_routing_changed",
            "production_effect_performed",
            "activation_or_export_performed",
            "downstream_export_enabled",
        )
        for field in required_false:
            if safety.get(field) is not False:
                raise ValueError(f"Unsafe functional shadow field: {field}.")
        for field in (
            "shadow_only",
            "isolated_historical_snapshot_only",
            "read_only_feature_access",
            "capability_authorization_enforced",
            "manual_approval_required",
        ):
            if safety.get(field) is not True:
                raise ValueError(f"Functional shadow requires {field}.")
        eligible = review.get("eligible_for_staging_review") is True
        gates_pass = bool(
            comparison.get("overall_divergence_count") == 0
            and denied_count == 0
            and all(
                value.get("passed") is True
                for value in payload.get("resource_gates", {}).values()
            )
            and int(execution.get("failed_stage_count") or 0) == 0
        )
        if eligible != gates_pass:
            raise ValueError("Functional staging-review eligibility is inconsistent.")
        supplied = payload.pop("functional_shadow_report_fingerprint", None)
        if stable_fingerprint(payload) != supplied:
            raise ValueError("Functional shadow report fingerprint mismatch.")
        payload["functional_shadow_report_fingerprint"] = required_sha256_digest(
            supplied,
            label="functional_shadow_report_fingerprint",
        )
        return payload

    def _policy_from_environment(self) -> FunctionalShadowPolicy:
        def integer(name: str, default: int) -> int:
            raw = str(self._environment.get(name) or "").strip()
            return int(raw) if raw else default

        def number(name: str, default: float) -> float:
            raw = str(self._environment.get(name) or "").strip()
            return float(raw) if raw else default

        return FunctionalShadowPolicy(
            max_safe_feature_rows=integer(
                "MODULE5_FUNCTIONAL_SHADOW_MAX_SAFE_FEATURE_ROWS",
                5000,
            ),
            max_vector_count_delta=integer(
                "MODULE5_FUNCTIONAL_SHADOW_MAX_VECTOR_COUNT_DELTA",
                0,
            ),
            max_ranked_match_delta=integer(
                "MODULE5_FUNCTIONAL_SHADOW_MAX_RANKED_MATCH_DELTA",
                0,
            ),
            max_selected_cohort_delta=integer(
                "MODULE5_FUNCTIONAL_SHADOW_MAX_SELECTED_COHORT_DELTA",
                0,
            ),
            max_total_latency_ms=number(
                "MODULE5_FUNCTIONAL_SHADOW_MAX_TOTAL_LATENCY_MS",
                15000.0,
            ),
            max_stage_latency_ms=number(
                "MODULE5_FUNCTIONAL_SHADOW_MAX_STAGE_LATENCY_MS",
                10000.0,
            ),
            max_peak_python_bytes=integer(
                "MODULE5_FUNCTIONAL_SHADOW_MAX_PEAK_PYTHON_BYTES",
                1_073_741_824,
            ),
        )
