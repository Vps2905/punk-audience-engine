from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from app.models.audience_feature_contracts import parse_utc_datetime
from app.models.production_fresh_data_workflow_contracts import (
    ProductionFreshDataWorkflowRequest,
)
from app.models.production_module3_cohort_contracts import (
    Module3CohortGenerationRequest,
)
from app.models.production_module3_overlap_contracts import (
    Module3OverlapDeduplicationRequest,
)
from app.services.production_module3_cohort_candidate_service import (
    ProductionModule3CohortCandidateService,
)
from app.services.production_module3_overlap_deduplication_service import (
    ProductionModule3OverlapDeduplicationService,
)


class WorkflowState(Protocol):
    def claim(self, request: Any, **kwargs: Any) -> dict[str, Any]: ...
    def transition(self, **kwargs: Any) -> dict[str, Any]: ...


class IngestionReceiptReader(Protocol):
    def get(self, ingestion_id: str) -> dict[str, Any]: ...


class CanonicalSourceReader(Protocol):
    def read(self, manifest: Any) -> Sequence[dict[str, Any]]: ...


class FeatureBuildExecutor(Protocol):
    def execute(
        self,
        request: Any,
        rows: Sequence[dict[str, Any]],
    ) -> dict[str, Any]: ...


class FeatureSnapshotReader(Protocol):
    def read(
        self,
        *,
        tenant_id: str,
        feature_set_id: str,
        feature_set_version: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]: ...


class ProductionFreshDataWorkflowService:
    """Durably orchestrate completed Module 1 output through Modules 2 and 3.

    The workflow never reads raw identifiers, never persists Module 3
    candidates, never generates lookalikes, and never activates or exports.
    Its terminal success state is awaiting_review.
    """

    def __init__(
        self,
        *,
        state_service: WorkflowState,
        ingestion_reader: IngestionReceiptReader,
        canonical_source_reader: CanonicalSourceReader,
        feature_build_executor: FeatureBuildExecutor,
        feature_snapshot_reader: FeatureSnapshotReader,
    ) -> None:
        self._state = state_service
        self._ingestion_reader = ingestion_reader
        self._canonical_source_reader = canonical_source_reader
        self._feature_build_executor = feature_build_executor
        self._feature_snapshot_reader = feature_snapshot_reader

    def execute(
        self,
        request: ProductionFreshDataWorkflowRequest,
        *,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> dict[str, Any]:
        claim = self._state.claim(
            request,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )
        record = dict(claim["record"])
        if not claim["acquired"]:
            return self._replay(record, busy=bool(claim.get("busy")))

        current_status = str(record.get("status") or "claimed")
        try:
            current_status = self._advance(
                request,
                worker_id,
                current_status,
                "validating_ingestion",
                lease_seconds,
            )
            ingestion = self._ingestion_reader.get(request.ingestion_id)
            self._validate_ingestion(request, ingestion)

            current_status = self._advance(
                request,
                worker_id,
                current_status,
                "reading_canonical",
                lease_seconds,
            )
            rows = list(self._canonical_source_reader.read(request.source))
            if len(rows) != request.source.expected_row_count:
                raise ValueError(
                    "Canonical source row count changed after verification."
                )

            current_status = self._advance(
                request,
                worker_id,
                current_status,
                "building_features",
                lease_seconds,
            )
            feature_receipt = self._feature_build_executor.execute(
                request.feature_build_request,
                rows,
            )
            self._validate_feature_receipt(request, feature_receipt)

            feature_set, feature_rows = self._feature_snapshot_reader.read(
                tenant_id=request.source.tenant_id,
                feature_set_id=str(feature_receipt["feature_set_id"]),
                feature_set_version=int(
                    feature_receipt["feature_set_version"]
                ),
            )

            current_status = self._advance(
                request,
                worker_id,
                current_status,
                "generating_candidates",
                lease_seconds,
                feature_receipt=feature_receipt,
            )
            candidate_report = ProductionModule3CohortCandidateService(
                policy=request.cohort_policy
            ).generate(
                request=Module3CohortGenerationRequest(
                    tenant_id=request.source.tenant_id,
                    feature_set_id=str(feature_receipt["feature_set_id"]),
                    feature_set_version=int(
                        feature_receipt["feature_set_version"]
                    ),
                    execution_mode=request.source.data_use_mode,
                    purpose=request.source.purpose,
                ),
                feature_set=feature_set,
                feature_rows=feature_rows,
            ).to_record()

            current_status = self._advance(
                request,
                worker_id,
                current_status,
                "analyzing_overlap",
                lease_seconds,
                candidate_report=candidate_report,
            )
            overlap_report = ProductionModule3OverlapDeduplicationService(
                policy=request.overlap_policy
            ).analyze(
                request=Module3OverlapDeduplicationRequest(
                    tenant_id=request.source.tenant_id,
                    batch_fingerprint=str(
                        candidate_report["batch_fingerprint"]
                    ),
                    execution_mode=request.source.data_use_mode,
                    purpose=request.source.purpose,
                ),
                candidate_batch=candidate_report,
            ).to_record()

            result = self._result_receipt(
                request=request,
                ingestion=ingestion,
                feature_receipt=feature_receipt,
                candidate_report=candidate_report,
                overlap_report=overlap_report,
            )
            final = self._state.transition(
                tenant_id=request.source.tenant_id,
                workflow_id=request.workflow_id,
                worker_id=worker_id,
                status="awaiting_review",
                feature_receipt=feature_receipt,
                candidate_report=candidate_report,
                overlap_report=overlap_report,
                result_receipt=result,
                event_details={
                    "generated_candidate_count": int(
                        candidate_report["generated_candidate_count"]
                    ),
                    "potential_overlap_group_count": int(
                        overlap_report["potential_overlap_group_count"]
                    ),
                },
                lease_seconds=lease_seconds,
            )
            return {
                **result,
                "durable_state": final["status"],
                "durable_claim_replayed": bool(claim["duplicate"]),
            }
        except ValueError as exc:
            self._terminal_failure(
                request=request,
                worker_id=worker_id,
                current_status=current_status,
                status="quarantined",
                reason_code="fresh_data_contract_validation_failed",
                lease_seconds=lease_seconds,
                error=exc,
            )
            raise
        except RuntimeError as exc:
            reason = (
                "fresh_data_policy_or_model_blocked"
                if any(
                    token in str(exc).lower()
                    for token in ("approved", "eligible", "policy")
                )
                else "fresh_data_workflow_runtime_failed"
            )
            terminal = "blocked" if reason.endswith("blocked") else "failed"
            self._terminal_failure(
                request=request,
                worker_id=worker_id,
                current_status=current_status,
                status=terminal,
                reason_code=reason,
                lease_seconds=lease_seconds,
                error=exc,
            )
            raise
        except Exception as exc:
            self._terminal_failure(
                request=request,
                worker_id=worker_id,
                current_status=current_status,
                status="failed",
                reason_code="fresh_data_workflow_unexpected_failure",
                lease_seconds=lease_seconds,
                error=exc,
            )
            raise

    def _advance(
        self,
        request: ProductionFreshDataWorkflowRequest,
        worker_id: str,
        current_status: str,
        target_status: str,
        lease_seconds: int,
        **kwargs: Any,
    ) -> str:
        stage_order = {
            "claimed": 0,
            "validating_ingestion": 1,
            "reading_canonical": 2,
            "building_features": 3,
            "generating_candidates": 4,
            "analyzing_overlap": 5,
            "awaiting_review": 6,
        }
        current_rank = stage_order.get(current_status)
        target_rank = stage_order.get(target_status)
        if current_rank is None or target_rank is None:
            raise RuntimeError("Fresh-data workflow is in an invalid resumable state.")
        if current_rank >= target_rank:
            return current_status
        self._state.transition(
            tenant_id=request.source.tenant_id,
            workflow_id=request.workflow_id,
            worker_id=worker_id,
            status=target_status,
            lease_seconds=lease_seconds,
            **kwargs,
        )
        return target_status

    def _validate_ingestion(
        self,
        request: ProductionFreshDataWorkflowRequest,
        ingestion: Mapping[str, Any],
    ) -> None:
        metadata = dict(ingestion.get("metadata") or {})
        expected = {
            "ingestion_id": request.ingestion_id,
            "tenant_id": request.source.tenant_id,
            "provider_id": request.source.provider_id,
            "dataset_id": request.source.dataset_id,
            "canonical_ref": request.source.source_ref,
            "output_rows": request.source.expected_row_count,
        }
        for key, value in expected.items():
            if ingestion.get(key) != value:
                raise ValueError(
                    f"Completed ingestion {key} does not match workflow request."
                )
        if ingestion.get("status") != "completed":
            raise ValueError("Module 1 ingestion is not completed.")
        metadata_checks = {
            "canonical_checksum_sha256": request.source.source_fingerprint,
            "canonical_object_version": request.source.source_version,
            "canonical_object_version_kind": (
                request.source.source_version_kind
            ),
            "canonical_size_bytes": request.source.source_size_bytes,
        }
        for key, value in metadata_checks.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"Completed ingestion {key} does not match canonical manifest."
                )
        metadata_identity = {
            "schema_version": request.source.schema_version,
            "purpose": request.source.purpose,
            "rights_policy_id": request.source.rights_policy_id,
            "privacy_policy_version": request.source.privacy_policy_version,
            "rights_status": request.source.rights_status,
        }
        for key, value in metadata_identity.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"Completed ingestion {key} does not match canonical manifest."
                )
        source_latest = parse_utc_datetime(
            metadata.get("canonical_source_latest_at")
        )
        if source_latest != request.source.source_latest_at:
            raise ValueError(
                "Completed ingestion canonical_source_latest_at does not match "
                "canonical manifest."
            )

        controls = set(metadata.get("canonical_privacy_controls") or ())
        required_controls = {
            "daily_contribution_bounding",
            "k_anonymity",
            "no_identifier_output",
        }
        if not required_controls.issubset(controls):
            raise ValueError(
                "Completed ingestion is missing canonical privacy controls."
            )

    def _validate_feature_receipt(
        self,
        request: ProductionFreshDataWorkflowRequest,
        receipt: Mapping[str, Any],
    ) -> None:
        if receipt.get("status") != "completed":
            raise RuntimeError("Module 2 feature build did not complete.")
        if receipt.get("tenant_id") != request.source.tenant_id:
            raise RuntimeError("Module 2 feature receipt tenant mismatch.")
        if int(receipt.get("feature_count") or -1) != int(
            request.source.expected_row_count
        ):
            raise RuntimeError("Module 2 feature receipt count mismatch.")
        if bool(receipt.get("eligible_for_activation")):
            raise RuntimeError(
                "Module 2 feature receipt unexpectedly enables activation."
            )

    def _result_receipt(
        self,
        *,
        request: ProductionFreshDataWorkflowRequest,
        ingestion: Mapping[str, Any],
        feature_receipt: Mapping[str, Any],
        candidate_report: Mapping[str, Any],
        overlap_report: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": "awaiting_human_review",
            "workflow_id": request.workflow_id,
            "request_fingerprint": request.request_fingerprint,
            "tenant_id": request.source.tenant_id,
            "ingestion_id": request.ingestion_id,
            "privacy_job_id": ingestion.get("privacy_job_id"),
            "feature_build_id": request.feature_build_request.build_id,
            "feature_set_id": feature_receipt["feature_set_id"],
            "feature_set_version": int(
                feature_receipt["feature_set_version"]
            ),
            "feature_count": int(feature_receipt["feature_count"]),
            "candidate_batch_fingerprint": candidate_report[
                "batch_fingerprint"
            ],
            "generated_candidate_count": int(
                candidate_report["generated_candidate_count"]
            ),
            "blocked_sensitive_count": int(
                candidate_report["blocked_sensitive_count"]
            ),
            "review_required_sensitive_count": int(
                candidate_report["review_required_sensitive_count"]
            ),
            "overlap_report_fingerprint": overlap_report[
                "report_fingerprint"
            ],
            "potential_overlap_group_count": int(
                overlap_report["potential_overlap_group_count"]
            ),
            "candidates_requiring_overlap_review_count": int(
                overlap_report[
                    "candidates_requiring_overlap_review_count"
                ]
            ),
            "approval_required": True,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
            "membership_intersection_read": False,
            "overlap_rate_computed": False,
            "unique_reach_claimed": False,
            "cohort_sizes_summed": False,
            "candidate_database_write_performed": False,
            "lookalike_generation_performed": False,
            "production_routing_enabled": False,
            "activation_or_export_performed": False,
            "downstream_export_enabled": False,
        }

    def _terminal_failure(
        self,
        *,
        request: ProductionFreshDataWorkflowRequest,
        worker_id: str,
        current_status: str,
        status: str,
        reason_code: str,
        lease_seconds: int,
        error: Exception,
    ) -> None:
        if current_status in {"awaiting_review", "blocked", "quarantined", "failed"}:
            return
        try:
            self._state.transition(
                tenant_id=request.source.tenant_id,
                workflow_id=request.workflow_id,
                worker_id=worker_id,
                status=status,
                reason_code=reason_code,
                event_details={"error_type": type(error).__name__},
                lease_seconds=lease_seconds,
            )
        except Exception:
            # Preserve the original exception. The durable lease allows an
            # operator to inspect/recover the workflow if terminal recording
            # itself fails.
            return

    def _replay(self, record: Mapping[str, Any], *, busy: bool) -> dict[str, Any]:
        if record.get("status") == "awaiting_review":
            result = dict(record.get("result_receipt") or {})
            if not result:
                raise RuntimeError(
                    "Completed fresh-data workflow is missing its result receipt."
                )
            return {
                **result,
                "durable_state": "awaiting_review",
                "durable_claim_replayed": True,
                "workflow_busy": False,
            }
        return {
            "status": record.get("status"),
            "reason_code": record.get("reason_code"),
            "workflow_id": record.get("workflow_id"),
            "tenant_id": record.get("tenant_id"),
            "terminal": bool(record.get("terminal")),
            "workflow_busy": busy,
            "durable_claim_replayed": True,
            "approval_required": True,
            "activation_or_export_performed": False,
            "downstream_export_enabled": False,
        }
