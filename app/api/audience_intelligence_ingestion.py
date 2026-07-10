from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
import pandas as pd

from app.core.api_key_auth import require_audience_api_key
from app.services.ingestion_job_service import IngestionJobService
from app.services.ingestion_lineage_service import IngestionLineageService
from app.services.privacy_ingestion_pipeline_service import (
    PrivacyIngestionConfig,
    PrivacyIngestionPipelineService,
)


router = APIRouter(
    prefix="/api/audience-intelligence/ingest",
    tags=["Audience Intelligence Ingestion"],
    dependencies=[Depends(require_audience_api_key)],
)


class PrivacyIngestionConfigRequest(BaseModel):
    source_type: str = "api"
    source_ref: Optional[str] = None
    entity_id_column: str = "entity_id"
    timestamp_column: str = "created_at"
    cohort_columns: List[str] = Field(
        default_factory=lambda: [
            "location_name",
            "primary_poi_type",
            "created_day_part",
        ]
    )
    min_cohort_size: int = 1000
    epsilon: float = 1.0
    delta: float = 1e-5
    sensitivity: float = 1.0
    mechanism: str = "gaussian"
    hash_salt: Optional[str] = None
    random_seed: Optional[int] = None


class PrivacyIngestionRequest(BaseModel):
    events: List[Dict[str, Any]]
    config: Optional[PrivacyIngestionConfigRequest] = None
    run_id: Optional[str] = None
    actor: str = "api"


@router.post("")
def ingest_privacy_safe_features(request: PrivacyIngestionRequest) -> Dict[str, Any]:
    """
    Module 1 API:
    raw/API/CSV-like event rows -> privacy-safe feature table.

    Pipeline:
    hashing -> contribution bounding -> aggregation -> k-anonymity -> DP noise
    -> lineage logging -> job status tracking.
    """
    config_payload = request.config or PrivacyIngestionConfigRequest()

    try:
        config = PrivacyIngestionConfig(
            source_type=config_payload.source_type,
            source_ref=config_payload.source_ref,
            entity_id_column=config_payload.entity_id_column,
            timestamp_column=config_payload.timestamp_column,
            cohort_columns=tuple(config_payload.cohort_columns),
            min_cohort_size=config_payload.min_cohort_size,
            epsilon=config_payload.epsilon,
            delta=config_payload.delta,
            sensitivity=config_payload.sensitivity,
            mechanism=config_payload.mechanism,
            hash_salt=config_payload.hash_salt,
            random_seed=config_payload.random_seed,
        )

        return PrivacyIngestionPipelineService().process_events(
            request.events,
            config=config,
            run_id=request.run_id,
            actor=request.actor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{job_id}/status")
def get_ingestion_status(job_id: str) -> Dict[str, Any]:
    """
    Returns ingestion job status + lineage chain summary.
    """
    job_result = IngestionJobService().get_job(job_id)

    if job_result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Ingestion job not found.")

    lineage_summary = IngestionLineageService().build_chain_summary(job_id=job_id)

    return {
        "job": job_result,
        "lineage": lineage_summary,
    }


@router.post("/csv")
async def ingest_privacy_safe_csv(
    file: UploadFile = File(...),
    source_type: str = Query(default="csv"),
    min_cohort_size: int = Query(default=1000, ge=1),
    epsilon: float = Query(default=1.0, gt=0),
    delta: float = Query(default=1e-5, gt=0, lt=1),
    sensitivity: float = Query(default=1.0, gt=0),
    mechanism: str = Query(default="gaussian"),
    hash_salt: Optional[str] = Query(default=None),
    random_seed: Optional[int] = Query(default=None),
    run_id: Optional[str] = Query(default=None),
    actor: str = Query(default="api"),
) -> Dict[str, Any]:
    """
    Module 1 CSV API:
    CSV upload -> privacy-safe feature table.

    Required CSV columns by default:
        entity_id, created_at, location_name, primary_poi_type, created_day_part
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

        df = pd.read_csv(BytesIO(content))
        df = df.where(pd.notna(df), None)

        events = df.to_dict(orient="records")

        config = PrivacyIngestionConfig(
            source_type=source_type,
            source_ref=file.filename,
            min_cohort_size=min_cohort_size,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            mechanism=mechanism,
            hash_salt=hash_salt,
            random_seed=random_seed,
        )

        return PrivacyIngestionPipelineService().process_events(
            events,
            config=config,
            run_id=run_id,
            actor=actor,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV ingestion failed: {exc}") from exc
