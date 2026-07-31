from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.core.api_key_auth import require_audience_api_key
from app.core.tenant_request_auth import require_verified_audience_tenant
from app.models.provider_privacy_window_contracts import (
    ProviderDataRightsRequest,
)
from app.services.provider_contract_registry_service import (
    ProviderContractRegistryService,
)
from app.services.provider_delivery_monitor_service import (
    ProviderDeliveryMonitorService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_queue_service import SQSProviderQueue
from app.services.provider_data_rights_service import (
    ProviderDataRightsService,
)
from app.services.provider_privacy_window_service import (
    ProviderPrivacyWindowService,
)
from app.services.provider_scale_acceptance_service import (
    ProviderScaleAcceptanceService,
)


class ProviderDataRightsRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    provider_id: str = Field(min_length=1, max_length=128)
    dataset_id: str = Field(min_length=1, max_length=128)
    request_type: str = Field(pattern="^(delete|opt_out)$")
    subject_token_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern="^[a-fA-F0-9]{64}$",
    )
    requested_at: str = Field(min_length=1, max_length=64)
    event_time_start: str | None = Field(default=None, max_length=64)
    event_time_end: str | None = Field(default=None, max_length=64)
    source_request_ref: str | None = Field(default=None, max_length=512)


router = APIRouter(
    prefix="/api/audience-intelligence/provider-ingestion",
    tags=["Provider Ingestion"],
    dependencies=[Depends(require_audience_api_key)],
)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _database_url() -> str | None:
    # Provider control-plane writes must never fall back to the historical
    # Echo/source database. The target must be explicitly Punk-owned.
    return os.getenv("PROVIDER_INGESTION_DATABASE_URL")


@router.get("/status")
def provider_ingestion_status(
    verified_tenant_id: str = Depends(require_verified_audience_tenant),
) -> Dict[str, Any]:
    enabled = _truthy(os.getenv("PROVIDER_GATEWAY_ENABLED"))
    if not enabled:
        return {
            "status": "disabled",
            "enabled": False,
            "live_provider_data_required": False,
            "message": (
                "The provider gateway is built but disabled by configuration."
            ),
        }

    db_url = _database_url()
    queue_url = os.getenv("PROVIDER_SQS_QUEUE_URL")
    dlq_url = os.getenv("PROVIDER_SQS_DLQ_URL")
    if not db_url or not queue_url or not dlq_url:
        return {
            "status": "blocked_configuration_incomplete",
            "enabled": True,
            "database_configured": bool(db_url),
            "queue_configured": bool(queue_url),
            "dlq_configured": bool(dlq_url),
        }

    queue = SQSProviderQueue(
        queue_url=queue_url,
        dlq_url=dlq_url,
        region_name=os.getenv("PROVIDER_S3_REGION"),
    )
    monitor = ProviderDeliveryMonitorService(
        registry=ProviderContractRegistryService(database_url=db_url),
        state_service=ProviderIngestionStateService(database_url=db_url),
        queue=queue,
        privacy_window_service=ProviderPrivacyWindowService(
            database_url=db_url
        ),
    )
    try:
        report = monitor.report(tenant_id=verified_tenant_id)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "provider_ingestion_status_unavailable",
                "error_type": type(exc).__name__,
            },
        ) from None
    return {
        "enabled": True,
        "tenant_id": verified_tenant_id,
        "live_provider_data_required": False,
        "distributed_processing_enabled": _truthy(
            os.getenv("PROVIDER_DISTRIBUTED_PROCESSING_ENABLED")
        ),
        "distributed_state_machine_configured": bool(
            os.getenv("PROVIDER_DISTRIBUTED_STATE_MACHINE_ARN")
        ),
        **report,
    }


@router.get("/runs/{ingestion_id}")
def provider_ingestion_run(
    ingestion_id: str,
    verified_tenant_id: str = Depends(require_verified_audience_tenant),
) -> Dict[str, Any]:
    db_url = _database_url()
    if not db_url:
        raise HTTPException(
            status_code=503,
            detail="Provider ingestion database is not configured.",
        )
    try:
        record = ProviderIngestionStateService(
            database_url=db_url
        ).get(ingestion_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Provider ingestion run was not found.",
        ) from None
    if str(record.get("tenant_id") or "").lower() != verified_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Provider ingestion run was not found.",
        )
    return {
        "ingestion_id": record.get("ingestion_id"),
        "tenant_id": record.get("tenant_id"),
        "provider_id": record.get("provider_id"),
        "dataset_id": record.get("dataset_id"),
        "status": record.get("status"),
        "reason_code": record.get("reason_code"),
        "source_ref": record.get("source_ref"),
        "object_version": record.get("object_version"),
        "attempt_count": record.get("attempt_count"),
        "input_rows": record.get("input_rows"),
        "output_rows": record.get("output_rows"),
        "privacy_job_id": record.get("privacy_job_id"),
        "canonical_ref": record.get("canonical_ref"),
        "execution_mode": (record.get("metadata") or {}).get(
            "execution_mode"
        ),
        "distributed_job_id": (record.get("metadata") or {}).get(
            "distributed_job_id"
        ),
        "distributed_backend": (record.get("metadata") or {}).get(
            "distributed_backend"
        ),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "completed_at": record.get("completed_at"),
    }


@router.get("/privacy-windows/{window_key}")
def provider_privacy_window(
    window_key: str,
    verified_tenant_id: str = Depends(require_verified_audience_tenant),
) -> Dict[str, Any]:
    db_url = _database_url()
    if not db_url:
        raise HTTPException(
            status_code=503,
            detail="Provider ingestion database is not configured.",
        )
    try:
        result = ProviderPrivacyWindowService(
            database_url=db_url
        ).get_window(window_key, tenant_id=verified_tenant_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Provider privacy window was not found.",
        ) from None
    if str(result["tenant_id"]).lower() != verified_tenant_id:
        raise HTTPException(
            status_code=404,
            detail="Provider privacy window was not found.",
        )
    return result


@router.post("/data-rights")
def submit_provider_data_rights_request(
    body: ProviderDataRightsRequestBody,
    verified_tenant_id: str = Depends(require_verified_audience_tenant),
) -> Dict[str, Any]:
    if body.tenant_id.strip().lower() != verified_tenant_id:
        raise HTTPException(
            status_code=403,
            detail="Tenant header does not match request tenant.",
        )
    db_url = _database_url()
    if not db_url:
        raise HTTPException(
            status_code=503,
            detail="Provider ingestion database is not configured.",
        )
    try:
        request = ProviderDataRightsRequest(**body.model_dump())
        return ProviderDataRightsService(database_url=db_url).submit(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/data-rights/{request_id}/apply")
def apply_provider_data_rights_request(
    request_id: str,
    verified_tenant_id: str = Depends(require_verified_audience_tenant),
) -> Dict[str, Any]:
    db_url = _database_url()
    if not db_url:
        raise HTTPException(
            status_code=503,
            detail="Provider ingestion database is not configured.",
        )
    service = ProviderDataRightsService(database_url=db_url)
    try:
        result = service.apply(
            request_id,
            tenant_id=verified_tenant_id,
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Data-rights request was not found.",
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return result


@router.get("/scale-readiness")
def provider_scale_readiness(
    verified_tenant_id: str = Depends(require_verified_audience_tenant),
) -> Dict[str, Any]:
    db_url = _database_url()
    if not db_url:
        raise HTTPException(
            status_code=503,
            detail="Provider ingestion database is not configured.",
        )
    return ProviderScaleAcceptanceService(
        database_url=db_url
    ).latest(tenant_id=verified_tenant_id)
