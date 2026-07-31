from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any, Dict, Optional

import pandas as pd

from app.core.schema_validator import SafeSchemaValidator
from app.models.provider_ingestion_contracts import (
    CanonicalObjectTarget,
    ProviderContractValidator,
    ProviderDatasetContract,
    ProviderObjectDescriptor,
    ProviderObjectManifest,
    ProviderObjectValidationError,
)
from app.models.provider_scale_contracts import (
    ProviderDistributedDispatchError,
    ProviderDistributedJobRequest,
)
from app.services.privacy_ingestion_pipeline_service import (
    PrivacyIngestionConfig,
    PrivacyIngestionPipelineService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_object_store_service import ProviderObjectStore
from app.services.provider_payload_decoder_service import (
    ProviderPayloadDecodeError,
    ProviderPayloadDecoderService,
)
from app.services.provider_scale_execution_service import (
    ProviderDistributedJobLauncher,
    ProviderScaleExecutionPlanner,
)
from app.services.provider_data_rights_service import (
    ProviderDataRightsService,
)
from app.utils.serialization import make_serializable


class ProviderObjectIngestionGatewayService:
    """
    Fail-closed boundary from immutable provider objects to safe feature rows.

    Object identity is claimed before provider reads, privacy processing, or
    canonical writes. Duplicate notifications therefore return the existing
    durable state without repeating downstream side effects.
    """

    def __init__(
        self,
        *,
        state_service: ProviderIngestionStateService,
        object_store: ProviderObjectStore,
        privacy_pipeline: PrivacyIngestionPipelineService,
        decoder: Optional[ProviderPayloadDecoderService] = None,
        contract_validator: Optional[ProviderContractValidator] = None,
        safe_schema_validator: Optional[SafeSchemaValidator] = None,
        execution_planner: Optional[ProviderScaleExecutionPlanner] = None,
        distributed_launcher: Optional[ProviderDistributedJobLauncher] = None,
        data_rights_service: Optional[ProviderDataRightsService] = None,
    ) -> None:
        self._state = state_service
        self._object_store = object_store
        self._privacy_pipeline = privacy_pipeline
        self._decoder = decoder or ProviderPayloadDecoderService()
        self._contract_validator = contract_validator or ProviderContractValidator()
        self._safe_schema_validator = safe_schema_validator or SafeSchemaValidator()
        self._execution_planner = (
            execution_planner or ProviderScaleExecutionPlanner()
        )
        self._distributed_launcher = distributed_launcher
        self._data_rights = data_rights_service

    def ingest_object(
        self,
        *,
        contract: ProviderDatasetContract,
        descriptor: ProviderObjectDescriptor,
        manifest: ProviderObjectManifest,
        canonical_target: CanonicalObjectTarget,
        actor: str = "provider_ingestion_worker",
        retry_failed: bool = False,
    ) -> Dict[str, Any]:
        claim = self._state.claim_object(
            descriptor,
            metadata={
                "contract_key": contract.contract_key,
                "schema_version": contract.schema_version,
                "purpose": manifest.purpose,
                "rights_policy_id": manifest.rights_policy_id,
                "event_time_start": manifest.event_time_start,
                "event_time_end": manifest.event_time_end,
                "content_type": descriptor.content_type,
                "size_bytes": descriptor.size_bytes,
                "server_side_encryption": descriptor.server_side_encryption,
                "last_modified": descriptor.last_modified,
                "event_id": descriptor.event_id,
            },
        )
        record = claim["record"]
        resume_dispatch = (
            claim["duplicate"] and record.get("status") == "dispatching"
        )
        if claim["duplicate"]:
            if retry_failed and record.get("status") == "failed":
                record = self._state.transition(
                    record["ingestion_id"],
                    status="validating",
                    reason_code=None,
                    metadata_update={"retry_requested": True},
                )
            elif not resume_dispatch:
                return self._result(
                    record,
                    duplicate=True,
                    privacy_pipeline_started=False,
                )

        ingestion_id = record["ingestion_id"]
        if record.get("status") == "received":
            self._state.transition(ingestion_id, status="validating")

        try:
            self._contract_validator.validate(
                contract=contract,
                descriptor=descriptor,
                manifest=manifest,
            )
            execution_decision = self._execution_planner.choose(
                contract=contract,
                descriptor=descriptor,
                manifest=manifest,
            )
            if execution_decision.execution_mode == "distributed":
                return self._dispatch_distributed(
                    contract=contract,
                    descriptor=descriptor,
                    manifest=manifest,
                    canonical_target=canonical_target,
                    actor=actor,
                    ingestion_id=ingestion_id,
                    decision_reason=execution_decision.reason_code,
                    duplicate=claim["duplicate"],
                )
            payload = self._object_store.read_object(
                descriptor,
                max_bytes=contract.max_object_bytes,
            )
            self._validate_payload_identity(
                payload=payload,
                contract=contract,
                descriptor=descriptor,
                manifest=manifest,
            )
            rows = self._decoder.decode(
                payload,
                data_format=contract.data_format,
                max_rows=contract.max_rows_per_object,
            )
            if manifest.row_count is not None and len(rows) != manifest.row_count:
                raise ProviderObjectValidationError(
                    "row_count_mismatch",
                    "The provider payload row count does not match its manifest.",
                )
        except (ProviderObjectValidationError, ProviderPayloadDecodeError) as exc:
            record = self._state.transition(
                ingestion_id,
                status="quarantined",
                reason_code=exc.reason_code,
            )
            return self._result(
                record,
                duplicate=False,
                privacy_pipeline_started=False,
            )
        except Exception:
            record = self._state.transition(
                ingestion_id,
                status="failed",
                reason_code="provider_object_read_failed",
            )
            return self._result(
                record,
                duplicate=False,
                privacy_pipeline_started=False,
            )

        self._state.transition(
            ingestion_id,
            status="processing",
            input_rows=len(rows),
            increment_attempt=True,
        )

        try:
            suppressed_token_digests = (
                self._data_rights.active_token_digests(
                    tenant_id=contract.tenant_id,
                    provider_id=contract.provider_id,
                    dataset_id=contract.dataset_id,
                )
                if self._data_rights is not None
                else ()
            )
            privacy_result = self._privacy_pipeline.process_events(
                rows,
                config=PrivacyIngestionConfig(
                    source_type="provider_s3_object",
                    source_ref=descriptor.source_ref,
                    entity_id_column=contract.entity_id_column,
                    timestamp_column=contract.timestamp_column,
                    cohort_columns=tuple(contract.cohort_columns),
                    min_cohort_size=contract.min_cohort_size,
                    epsilon=contract.epsilon,
                    delta=contract.delta,
                    sensitivity=contract.sensitivity,
                    mechanism=contract.mechanism,
                    random_seed=self._deterministic_dp_seed(
                        descriptor.fingerprint
                    ),
                    suppressed_token_digests=suppressed_token_digests,
                ),
                run_id=ingestion_id,
                actor=actor,
            )
        except Exception:
            record = self._state.transition(
                ingestion_id,
                status="failed",
                reason_code="privacy_pipeline_failed",
                input_rows=len(rows),
            )
            return self._result(
                record,
                duplicate=False,
                privacy_pipeline_started=True,
            )

        privacy_job_id = privacy_result.get("job_id")
        if privacy_result.get("status") != "completed":
            record = self._state.transition(
                ingestion_id,
                status="blocked",
                reason_code=self._privacy_block_reason(privacy_result),
                privacy_job_id=privacy_job_id,
                input_rows=len(rows),
                output_rows=0,
            )
            return self._result(
                record,
                duplicate=False,
                privacy_pipeline_started=True,
            )

        safe_rows = make_serializable(privacy_result.get("safe_feature_rows") or [])
        try:
            safe_frame = pd.DataFrame(safe_rows)
            self._safe_schema_validator.validate_safe_cohort_dataframe(
                safe_frame,
                context="provider canonical output",
            )
            canonical_payload = self._encode_jsonl(safe_rows)
            canonical_key = self._canonical_key(
                target=canonical_target,
                contract=contract,
                fingerprint=descriptor.fingerprint,
            )
            canonical_receipt = self._object_store.write_canonical(
                target=canonical_target,
                key=canonical_key,
                payload=canonical_payload,
                content_type="application/x-ndjson",
                metadata={
                    "tenant-id": contract.tenant_id,
                    "provider-id": contract.provider_id,
                    "dataset-id": contract.dataset_id,
                    "schema-version": contract.schema_version,
                    "source-fingerprint": descriptor.fingerprint,
                    "privacy-job-id": str(privacy_job_id or ""),
                },
            )
        except (TypeError, ValueError):
            record = self._state.transition(
                ingestion_id,
                status="quarantined",
                reason_code="unsafe_canonical_output",
                privacy_job_id=privacy_job_id,
                input_rows=len(rows),
                output_rows=0,
            )
            return self._result(
                record,
                duplicate=False,
                privacy_pipeline_started=True,
            )
        except Exception:
            record = self._state.transition(
                ingestion_id,
                status="failed",
                reason_code="canonical_write_failed",
                privacy_job_id=privacy_job_id,
                input_rows=len(rows),
                output_rows=0,
            )
            return self._result(
                record,
                duplicate=False,
                privacy_pipeline_started=True,
            )

        canonical_ref = str(canonical_receipt.get("source_ref") or "")
        if not canonical_ref:
            record = self._state.transition(
                ingestion_id,
                status="failed",
                reason_code="canonical_receipt_invalid",
                privacy_job_id=privacy_job_id,
                input_rows=len(rows),
                output_rows=0,
            )
            return self._result(
                record,
                duplicate=False,
                privacy_pipeline_started=True,
            )

        record = self._state.transition(
            ingestion_id,
            status="completed",
            reason_code=None,
            privacy_job_id=privacy_job_id,
            canonical_ref=canonical_ref,
            input_rows=len(rows),
            output_rows=len(safe_rows),
            metadata_update={
                "canonical_checksum_sha256": hashlib.sha256(
                    canonical_payload
                ).hexdigest(),
                "privacy_controls": list(privacy_result.get("privacy_controls") or []),
            },
        )
        return self._result(
            record,
            duplicate=False,
            privacy_pipeline_started=True,
        )

    def _dispatch_distributed(
        self,
        *,
        contract: ProviderDatasetContract,
        descriptor: ProviderObjectDescriptor,
        manifest: ProviderObjectManifest,
        canonical_target: CanonicalObjectTarget,
        actor: str,
        ingestion_id: str,
        decision_reason: str,
        duplicate: bool,
    ) -> Dict[str, Any]:
        current = self._state.get(ingestion_id)
        if self._distributed_launcher is None:
            target_status = (
                "failed"
                if current.get("status") == "dispatching"
                else "blocked"
            )
            record = self._state.transition(
                ingestion_id,
                status=target_status,
                reason_code="distributed_processing_not_configured",
                metadata_update={
                    "execution_mode": "distributed",
                    "execution_reason": decision_reason,
                },
            )
            return self._result(
                record,
                duplicate=duplicate,
                privacy_pipeline_started=False,
            )

        if current.get("status") == "validating":
            current = self._state.transition(
                ingestion_id,
                status="dispatching",
                reason_code=None,
                increment_attempt=True,
                metadata_update={
                    "execution_mode": "distributed",
                    "execution_reason": decision_reason,
                },
            )
        if current.get("status") != "dispatching":
            raise RuntimeError(
                "Distributed ingestion is not in a dispatchable state."
            )

        request = ProviderDistributedJobRequest(
            ingestion_id=ingestion_id,
            fingerprint=descriptor.fingerprint,
            contract=contract,
            descriptor=descriptor,
            manifest=manifest,
            canonical_target=canonical_target,
            actor=actor,
            dispatch_attempt=int(current.get("attempt_count") or 1),
        )
        try:
            receipt = self._distributed_launcher.submit_or_get(request)
        except ProviderDistributedDispatchError as exc:
            failure_reason = (
                exc.reason_code
                if exc.retryable
                else "distributed_dispatch_permanent_failure"
            )
            record = self._state.transition(
                ingestion_id,
                status="failed",
                reason_code=failure_reason,
                metadata_update={
                    "execution_mode": "distributed",
                    "distributed_dispatch_retryable": exc.retryable,
                },
            )
            return self._result(
                record,
                duplicate=duplicate,
                privacy_pipeline_started=False,
            )
        except Exception:
            record = self._state.transition(
                ingestion_id,
                status="failed",
                reason_code="distributed_dispatch_failed",
                metadata_update={
                    "execution_mode": "distributed",
                    "distributed_dispatch_retryable": True,
                },
            )
            return self._result(
                record,
                duplicate=duplicate,
                privacy_pipeline_started=False,
            )

        record = self._state.transition(
            ingestion_id,
            status="dispatched",
            reason_code=None,
            metadata_update={
                "execution_mode": "distributed",
                "distributed_job_id": receipt.job_id,
                "distributed_backend": receipt.backend,
                "distributed_dispatch_replayed": receipt.replayed,
                "distributed_submitted_at": receipt.submitted_at,
            },
        )
        return self._result(
            record,
            duplicate=duplicate,
            privacy_pipeline_started=False,
        )

    def _validate_payload_identity(
        self,
        *,
        payload: bytes,
        contract: ProviderDatasetContract,
        descriptor: ProviderObjectDescriptor,
        manifest: ProviderObjectManifest,
    ) -> None:
        if len(payload) != descriptor.size_bytes:
            raise ProviderObjectValidationError(
                "object_size_mismatch",
                "The provider payload size does not match its object descriptor.",
            )
        if len(payload) > contract.max_object_bytes:
            raise ProviderObjectValidationError(
                "object_too_large",
                "The provider object exceeds the configured maximum size.",
            )

        actual_checksum = hashlib.sha256(payload).hexdigest()
        expected_checksum = manifest.checksum_sha256 or descriptor.checksum_sha256
        if expected_checksum and actual_checksum != expected_checksum:
            raise ProviderObjectValidationError(
                "checksum_verification_failed",
                "The provider payload failed SHA-256 checksum verification.",
            )

    def _privacy_block_reason(self, result: Dict[str, Any]) -> str:
        reason = str(result.get("reason") or "").lower()
        if "k-anonymity" in reason:
            return "k_anonymity_threshold_not_met"
        if "required column" in reason:
            return "privacy_required_columns_missing"
        if "contribution" in reason:
            return "contribution_bounding_blocked"
        return "privacy_policy_blocked"

    def _deterministic_dp_seed(self, fingerprint: str) -> int:
        secret = (
            os.getenv("AUDIENCE_DP_SEED_SECRET")
            or os.getenv("AUDIENCE_HASH_SALT")
            or "dev_only_change_me"
        )
        digest = hmac.new(
            secret.encode("utf-8"),
            fingerprint.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def _canonical_key(
        self,
        *,
        target: CanonicalObjectTarget,
        contract: ProviderDatasetContract,
        fingerprint: str,
    ) -> str:
        prefix = target.prefix
        return (
            f"{prefix}{contract.tenant_id}/{contract.provider_id}/"
            f"{contract.dataset_id}/{contract.schema_version}/"
            f"{fingerprint}.jsonl"
        )

    def _encode_jsonl(self, rows: list[Dict[str, Any]]) -> bytes:
        return (
            "\n".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
            )
            + "\n"
        ).encode("utf-8")

    def _result(
        self,
        record: Dict[str, Any],
        *,
        duplicate: bool,
        privacy_pipeline_started: bool,
    ) -> Dict[str, Any]:
        metadata = record.get("metadata") or {}
        return {
            "ingestion_id": record.get("ingestion_id"),
            "status": record.get("status"),
            "reason_code": record.get("reason_code"),
            "duplicate": duplicate,
            "privacy_pipeline_started": privacy_pipeline_started,
            "execution_mode": metadata.get("execution_mode") or "in_process",
            "distributed_job_id": metadata.get("distributed_job_id"),
            "distributed_backend": metadata.get("distributed_backend"),
            "distributed_dispatch_replayed": bool(
                metadata.get("distributed_dispatch_replayed")
            ),
            "tenant_id": record.get("tenant_id"),
            "provider_id": record.get("provider_id"),
            "dataset_id": record.get("dataset_id"),
            "source_ref": record.get("source_ref"),
            "object_version": record.get("object_version"),
            "input_rows": record.get("input_rows"),
            "output_rows": record.get("output_rows"),
            "privacy_job_id": record.get("privacy_job_id"),
            "canonical_ref": record.get("canonical_ref"),
            "attempt_count": record.get("attempt_count"),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "completed_at": record.get("completed_at"),
        }
