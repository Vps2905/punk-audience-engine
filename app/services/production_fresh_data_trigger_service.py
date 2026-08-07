from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Protocol

from app.models.production_feature_build_contracts import (
    CanonicalFeatureSourceManifest,
    EmbeddingModelSpec,
)
from app.models.production_fresh_data_workflow_contracts import (
    ProductionFreshDataWorkflowRequest,
)
from app.models.provider_ingestion_contracts import ProviderDatasetContract


class ProviderIngestionLister(Protocol):
    def list_recent(self, **kwargs: Any) -> list[dict[str, Any]]: ...


class ProviderContractResolver(Protocol):
    def get_active(self, **kwargs: Any) -> ProviderDatasetContract: ...


class WorkflowExecutor(Protocol):
    def execute(self, request: ProductionFreshDataWorkflowRequest, **kwargs: Any) -> dict[str, Any]: ...


class ProductionFreshDataWorkflowRequestFactory:
    """Build a fail-closed Module 1 -> 2 -> 3 request from a completed receipt.

    The factory never guesses provider schema semantics. It derives feature
    columns from the immutable provider contract and requires canonical privacy
    evidence written by Module 1.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DEFAULT_MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        self._env = dict(environment or os.environ)

    def build(
        self,
        *,
        ingestion: Mapping[str, Any],
        contract: ProviderDatasetContract,
    ) -> ProductionFreshDataWorkflowRequest:
        if str(ingestion.get("status") or "") != "completed":
            raise ValueError("Only completed provider ingestion can trigger a workflow.")
        for field in ("tenant_id", "provider_id", "dataset_id"):
            if str(ingestion.get(field) or "") != str(getattr(contract, field)):
                raise ValueError(f"Provider ingestion {field} does not match its contract.")

        metadata = dict(ingestion.get("metadata") or {})
        schema_version = str(metadata.get("schema_version") or "")
        if schema_version != contract.schema_version:
            raise ValueError("Provider ingestion schema_version does not match its contract.")

        canonical_ref = str(ingestion.get("canonical_ref") or "").strip()
        checksum = str(metadata.get("canonical_checksum_sha256") or "").strip().lower()
        source_version = str(metadata.get("canonical_object_version") or "").strip()
        version_kind = str(metadata.get("canonical_object_version_kind") or "").strip()
        latest_at = metadata.get("canonical_source_latest_at")
        source_size = metadata.get("canonical_size_bytes")
        output_rows = int(ingestion.get("output_rows") or 0)
        privacy_controls = tuple(metadata.get("canonical_privacy_controls") or ())
        purpose = str(metadata.get("purpose") or "").strip()
        rights_policy_id = str(metadata.get("rights_policy_id") or contract.rights_policy_id).strip()
        rights_status = str(metadata.get("rights_status") or "").strip()
        privacy_policy_version = str(metadata.get("privacy_policy_version") or "").strip()

        required = {
            "canonical_ref": canonical_ref,
            "canonical_checksum_sha256": checksum,
            "canonical_object_version": source_version,
            "canonical_object_version_kind": version_kind,
            "canonical_source_latest_at": latest_at,
            "canonical_size_bytes": source_size,
            "output_rows": output_rows,
            "purpose": purpose,
            "rights_policy_id": rights_policy_id,
            "rights_status": rights_status,
            "privacy_policy_version": privacy_policy_version,
        }
        missing = sorted(k for k, v in required.items() if v in (None, "", 0))
        if missing:
            raise ValueError(
                "Completed provider ingestion is missing canonical workflow evidence: "
                + ", ".join(missing)
            )

        cohort_columns = tuple(contract.cohort_columns)
        if len(cohort_columns) < 2:
            raise ValueError(
                "Provider contract requires at least location and category cohort columns."
            )

        mode = str(self._env.get("FRESH_DATA_WORKFLOW_DATA_USE_MODE") or "offline_evaluation").strip().lower()
        if mode not in {"offline_evaluation", "production"}:
            raise ValueError("FRESH_DATA_WORKFLOW_DATA_USE_MODE must be offline_evaluation or production.")

        source = CanonicalFeatureSourceManifest(
            tenant_id=contract.tenant_id,
            provider_id=contract.provider_id,
            dataset_id=contract.dataset_id,
            schema_version=contract.schema_version,
            source_ref=canonical_ref,
            source_version=source_version,
            source_fingerprint=checksum,
            source_latest_at=latest_at,
            expected_row_count=output_rows,
            data_use_mode=mode,
            privacy_status="privacy_safe",
            privacy_policy_version=privacy_policy_version,
            privacy_controls=privacy_controls,
            rights_status=rights_status,
            rights_policy_id=rights_policy_id,
            purpose=purpose,
            location_column=cohort_columns[0],
            category_column=cohort_columns[1],
            daypart_column=(cohort_columns[2] if len(cohort_columns) >= 3 else None),
            source_version_kind=version_kind,
            data_format="jsonl",
            source_size_bytes=int(source_size),
            max_object_bytes=int(contract.max_object_bytes),
        )
        model = EmbeddingModelSpec(
            backend="sentence_transformers",
            model_name=str(self._env.get("FRESH_DATA_EMBEDDING_MODEL_NAME") or self.DEFAULT_MODEL_NAME),
            model_revision=str(self._env.get("FRESH_DATA_EMBEDDING_MODEL_REVISION") or self.DEFAULT_MODEL_REVISION),
            dimension=384,
            normalize_embeddings=True,
        )
        return ProductionFreshDataWorkflowRequest(
            ingestion_id=str(ingestion["ingestion_id"]),
            source=source,
            model=model,
            batch_size=int(self._env.get("FRESH_DATA_WORKFLOW_BATCH_SIZE") or 256),
            max_features=int(self._env.get("FRESH_DATA_WORKFLOW_MAX_FEATURES") or 1_000_000),
            requested_by="fresh_data_trigger_worker",
        )


class ProductionFreshDataTriggerService:
    """Poll completed Module 1 receipts and idempotently trigger workflows."""

    def __init__(
        self,
        *,
        ingestion_state: ProviderIngestionLister,
        contract_registry: ProviderContractResolver,
        workflow_executor: WorkflowExecutor,
        request_factory: ProductionFreshDataWorkflowRequestFactory | None = None,
    ) -> None:
        self._ingestion_state = ingestion_state
        self._contract_registry = contract_registry
        self._workflow_executor = workflow_executor
        self._request_factory = request_factory or ProductionFreshDataWorkflowRequestFactory()

    def run_once(
        self,
        *,
        worker_id: str,
        tenant_id: str | None = None,
        limit: int = 50,
        lease_seconds: int = 300,
    ) -> dict[str, Any]:
        if not 1 <= int(limit) <= 200:
            raise ValueError("limit must be between 1 and 200.")
        rows = self._ingestion_state.list_recent(
            tenant_id=tenant_id,
            status="completed",
            limit=int(limit),
        )
        results: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        # Oldest-first prevents a high-frequency provider from starving earlier deliveries.
        for ingestion in reversed(rows):
            try:
                metadata = dict(ingestion.get("metadata") or {})
                contract = self._contract_registry.get_active(
                    tenant_id=str(ingestion["tenant_id"]),
                    provider_id=str(ingestion["provider_id"]),
                    dataset_id=str(ingestion["dataset_id"]),
                    schema_version=str(metadata.get("schema_version") or ""),
                )
                request = self._request_factory.build(
                    ingestion=ingestion,
                    contract=contract,
                )
                outcome = self._workflow_executor.execute(
                    request,
                    worker_id=worker_id,
                    lease_seconds=int(lease_seconds),
                )
                results.append(
                    {
                        "ingestion_id": str(ingestion["ingestion_id"]),
                        "workflow_id": request.workflow_id,
                        "status": str(outcome.get("durable_state") or outcome.get("status") or "unknown"),
                        "replayed": bool(outcome.get("durable_claim_replayed")),
                    }
                )
            except (KeyError, ValueError, RuntimeError) as exc:
                failures.append(
                    {
                        "ingestion_id": str(ingestion.get("ingestion_id") or "unknown"),
                        "error_code": type(exc).__name__,
                    }
                )
        return {
            "status": "completed_with_failures" if failures else "completed",
            "examined_completed_ingestions": len(rows),
            "triggered_or_replayed": len(results),
            "failed": len(failures),
            "results": results,
            "failures": failures,
            "raw_identifiers_read": False,
            "activation_or_export_performed": False,
        }
