from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from app.models.audience_feature_contracts import CanonicalFeatureSet
from app.models.production_feature_build_contracts import (
    ProductionFeatureBuildRequest,
)
from app.services.production_feature_embedding_service import (
    ProductionCanonicalFeatureEmbeddingService,
)


class VersionedFeatureStore(Protocol):
    def save_feature_set(
        self,
        feature_set: CanonicalFeatureSet,
    ) -> dict[str, Any]: ...

    def get_feature_set(
        self,
        *,
        tenant_id: str,
        feature_set_id: str | None = None,
        version: int | None = None,
        data_use_mode: str | None = None,
    ) -> dict[str, Any]: ...


class ApprovedEmbeddingModelRegistry(Protocol):
    def require_approved(
        self,
        *,
        tenant_id: str,
        model: Any,
    ) -> dict[str, Any]: ...


class ProductionFeatureBuildPipelineService:
    """
    Idempotent publication boundary for production canonical features.

    The content-addressed feature set is built before publication. A replay
    returns the existing immutable identity; a conflicting record fails closed.
    This service does not change approval or activation state.
    """

    def __init__(
        self,
        *,
        embedding_service: ProductionCanonicalFeatureEmbeddingService,
        feature_store: VersionedFeatureStore,
        model_registry: ApprovedEmbeddingModelRegistry,
    ) -> None:
        self._embedding_service = embedding_service
        self._feature_store = feature_store
        self._model_registry = model_registry

    def execute(
        self,
        request: ProductionFeatureBuildRequest,
        rows: Sequence[dict[str, Any]],
        *,
        phase_observer: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        model_approval = self._model_registry.require_approved(
            tenant_id=request.source.tenant_id,
            model=request.model,
        )
        if (
            not model_approval.get("approved")
            or model_approval.get("model_fingerprint")
            != request.model.fingerprint
        ):
            raise RuntimeError(
                "Embedding model approval receipt failed verification."
            )
        if phase_observer is not None:
            phase_observer("embedding")
        feature_set = self._embedding_service.build(request, rows)
        existing = self._existing(feature_set)
        if existing is not None:
            self._validate_replay(
                existing=existing,
                expected=feature_set,
                request=request,
            )
            if phase_observer is not None:
                phase_observer("publishing")
            return self._receipt(
                feature_set,
                idempotency_replayed=True,
                storage_status="already_published",
            )

        if phase_observer is not None:
            phase_observer("publishing")
        storage_receipt = self._feature_store.save_feature_set(feature_set)
        if (
            str(storage_receipt.get("feature_set_id") or "")
            != feature_set.feature_set_id
            or int(storage_receipt.get("feature_set_version") or 0)
            != feature_set.version
            or int(storage_receipt.get("feature_count") or -1)
            != feature_set.feature_count
        ):
            raise RuntimeError(
                "Feature-store publication receipt failed verification."
            )
        return self._receipt(
            feature_set,
            idempotency_replayed=False,
            storage_status=str(
                storage_receipt.get("status") or "completed"
            ),
        )

    def _existing(
        self,
        feature_set: CanonicalFeatureSet,
    ) -> dict[str, Any] | None:
        try:
            return self._feature_store.get_feature_set(
                tenant_id=feature_set.tenant_id,
                feature_set_id=feature_set.feature_set_id,
                version=feature_set.version,
                data_use_mode=feature_set.data_use_mode,
            )
        except FileNotFoundError:
            return None

    def _validate_replay(
        self,
        *,
        existing: dict[str, Any],
        expected: CanonicalFeatureSet,
        request: ProductionFeatureBuildRequest,
    ) -> None:
        lineage = dict(existing.get("lineage") or {})
        expected_values = {
            "tenant_id": expected.tenant_id,
            "feature_set_id": expected.feature_set_id,
            "version": expected.version,
            "source_fingerprint": expected.source_fingerprint,
            "model_name": expected.model_name,
            "model_version": expected.model_version,
            "embedding_dimension": expected.embedding_dimension,
            "feature_count": expected.feature_count,
            "data_use_mode": expected.data_use_mode,
        }
        for key, expected_value in expected_values.items():
            if existing.get(key) != expected_value:
                raise RuntimeError(
                    "Existing feature-set identity conflicts with this build."
                )
        if lineage.get("request_fingerprint") != request.request_fingerprint:
            raise RuntimeError(
                "Existing feature-set lineage conflicts with this build."
            )

    def _receipt(
        self,
        feature_set: CanonicalFeatureSet,
        *,
        idempotency_replayed: bool,
        storage_status: str,
    ) -> dict[str, Any]:
        return {
            "status": "completed",
            "storage_status": storage_status,
            "tenant_id": feature_set.tenant_id,
            "feature_build_id": feature_set.lineage["feature_build_id"],
            "feature_set_id": feature_set.feature_set_id,
            "feature_set_version": feature_set.version,
            "feature_count": feature_set.feature_count,
            "model_backend": feature_set.model_backend,
            "model_name": feature_set.model_name,
            "model_version": feature_set.model_version,
            "model_fingerprint": feature_set.lineage[
                "embedding_model_fingerprint"
            ],
            "freshness_status": feature_set.freshness_status,
            "quality_status": "not_scored",
            "eligible_for_retrieval": (
                feature_set.eligible_for_retrieval
            ),
            "eligible_for_activation": False,
            "downstream_export_enabled": False,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
            "idempotency_replayed": idempotency_replayed,
        }
