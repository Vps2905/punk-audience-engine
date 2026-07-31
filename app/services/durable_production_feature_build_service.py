from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from app.models.production_feature_build_contracts import (
    ProductionFeatureBuildRequest,
)
from app.services.production_feature_build_pipeline_service import (
    ProductionFeatureBuildPipelineService,
)


class FeatureBuildState(Protocol):
    def claim(
        self,
        request: ProductionFeatureBuildRequest,
    ) -> dict[str, Any]: ...

    def transition(self, **kwargs: Any) -> dict[str, Any]: ...


class DurableProductionFeatureBuildService:
    """Durable fail-closed execution around one bounded feature partition."""

    def __init__(
        self,
        *,
        pipeline: ProductionFeatureBuildPipelineService,
        state_service: FeatureBuildState,
    ) -> None:
        self._pipeline = pipeline
        self._state = state_service

    def execute(
        self,
        request: ProductionFeatureBuildRequest,
        rows: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        claim = self._state.claim(request)
        record = dict(claim["record"])
        if claim["duplicate"]:
            return self._duplicate_receipt(record)
        self._transition(request, "validating")

        def observe(status: str) -> None:
            if status in {"embedding", "publishing"}:
                self._transition(request, status)

        try:
            receipt = self._pipeline.execute(
                request,
                rows,
                phase_observer=observe,
            )
        except ValueError:
            self._transition(
                request,
                "quarantined",
                reason_code="invalid_canonical_feature_input",
            )
            raise
        except RuntimeError as exc:
            reason = (
                "embedding_model_not_approved"
                if "model" in str(exc).lower()
                and "approv" in str(exc).lower()
                else "feature_build_runtime_failed"
            )
            terminal = "blocked" if reason.endswith("not_approved") else "failed"
            self._transition(
                request,
                terminal,
                reason_code=reason,
            )
            raise
        except Exception:
            self._transition(
                request,
                "failed",
                reason_code="feature_build_unexpected_failure",
            )
            raise

        self._transition(
            request,
            "completed",
            processed_feature_count=receipt["feature_count"],
            feature_set_id=receipt["feature_set_id"],
            feature_set_version=receipt["feature_set_version"],
            result_receipt=receipt,
        )
        return {
            **receipt,
            "durable_state": "completed",
            "durable_claim_replayed": False,
        }

    def _transition(
        self,
        request: ProductionFeatureBuildRequest,
        status: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._state.transition(
            tenant_id=request.source.tenant_id,
            feature_build_id=request.build_id,
            status=status,
            **kwargs,
        )

    def _duplicate_receipt(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        if record.get("status") == "completed":
            receipt = dict(record.get("result_receipt") or {})
            if not receipt:
                raise RuntimeError(
                    "Completed feature build is missing its durable receipt."
                )
            return {
                **receipt,
                "idempotency_replayed": True,
                "durable_state": "completed",
                "durable_claim_replayed": True,
            }
        return {
            "status": record.get("status"),
            "reason_code": record.get("reason_code"),
            "tenant_id": record.get("tenant_id"),
            "feature_build_id": record.get("feature_build_id"),
            "terminal": bool(record.get("terminal")),
            "durable_claim_replayed": True,
            "eligible_for_activation": False,
            "downstream_export_enabled": False,
            "activation_or_export_performed": False,
        }
