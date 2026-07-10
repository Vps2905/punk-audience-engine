from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.api_key_auth import require_audience_api_key
from app.services.ingestion_job_service import IngestionJobCreate, IngestionJobService
from app.services.ingestion_lineage_service import IngestionLineageService, LineageEvent
from app.services.synthetic_service import generate_synthetic_for_job


router = APIRouter(
    prefix="/api/audience-intelligence/synthetic",
    tags=["Audience Intelligence Synthetic"],
    dependencies=[Depends(require_audience_api_key)],
)


@router.post("/generate/{job_id}")
def generate_synthetic_for_ingestion_job(
    job_id: str,
    num_rows: int = Query(default=1000, ge=1, le=100000),
    use_sdv: bool = Query(default=False),
    actor: str = Query(default="api"),
) -> Dict[str, Any]:
    """
    Module 1 synthetic API.

    Uses existing synthetic generator, but wraps it with:
    - API key protection
    - synthetic job tracking
    - lineage audit
    """
    job_service = IngestionJobService()
    lineage_service = IngestionLineageService()

    synthetic_job = job_service.create_job(
        IngestionJobCreate(
            source_type="synthetic_generation",
            source_ref=job_id,
            run_id=job_id,
            actor=actor,
            metadata={
                "parent_job_id": job_id,
                "num_rows_requested": num_rows,
                "use_sdv": use_sdv,
                "module": "module_1_synthetic_generation",
            },
        )
    )

    synthetic_job_id = synthetic_job.get("job_id") or f"synthetic_{job_id}"

    try:
        job_service.mark_running(synthetic_job_id)

        lineage_service.record_event(
            LineageEvent(
                job_id=synthetic_job_id,
                run_id=job_id,
                stage="synthetic_generation_started",
                transformation="safe_feature_table_to_synthetic_profiles",
                input_ref=job_id,
                output_ref=None,
                input_rows=None,
                output_rows=None,
                dropped_rows=0,
                actor=actor,
                details={
                    "parent_job_id": job_id,
                    "num_rows_requested": num_rows,
                    "use_sdv": use_sdv,
                    "privacy_note": (
                        "Synthetic generation must use processed safe feature tables, "
                        "not raw identifiers or raw observations."
                    ),
                },
            )
        )

        result = generate_synthetic_for_job(
            job_id=job_id,
            num_rows=num_rows,
            use_sdv=use_sdv,
        )

        generated_rows = int(result.get("num_rows_generated") or 0)

        lineage_service.record_event(
            LineageEvent(
                job_id=synthetic_job_id,
                run_id=job_id,
                stage="synthetic_export",
                transformation=str(result.get("backend") or "synthetic_generation"),
                input_ref=job_id,
                output_ref=str(result.get("synthetic_path") or ""),
                input_rows=None,
                output_rows=generated_rows,
                dropped_rows=0,
                actor=actor,
                details={
                    "parent_job_id": job_id,
                    "backend": result.get("backend"),
                    "synthetic_path": result.get("synthetic_path"),
                    "manifest_path": result.get("manifest_path"),
                    "privacy_mode": result.get("privacy_mode"),
                    "safe_for_export_seed": result.get("safe_for_export_seed"),
                },
            )
        )

        job_service.mark_completed(
            job_id=synthetic_job_id,
            input_rows=0,
            output_rows=generated_rows,
            dropped_rows=0,
            metadata_update={
                "parent_job_id": job_id,
                "synthetic_generation_completed": True,
                "backend": result.get("backend"),
                "synthetic_path": result.get("synthetic_path"),
                "manifest_path": result.get("manifest_path"),
            },
        )

        return {
            "status": "completed",
            "parent_job_id": job_id,
            "synthetic_job_id": synthetic_job_id,
            "num_rows_requested": num_rows,
            "num_rows_generated": generated_rows,
            "result": result,
            "privacy_note": (
                "Synthetic output is generated from processed safe features. "
                "Lineage has been recorded for audit."
            ),
        }

    except ValueError as exc:
        job_service.mark_failed(job_id=synthetic_job_id, error_message=str(exc))
        lineage_service.record_event(
            LineageEvent(
                job_id=synthetic_job_id,
                run_id=job_id,
                stage="synthetic_generation_failed",
                transformation="safe_feature_table_to_synthetic_profiles",
                input_ref=job_id,
                output_ref=None,
                input_rows=None,
                output_rows=0,
                dropped_rows=0,
                actor=actor,
                details={"error": str(exc), "parent_job_id": job_id},
            )
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{synthetic_job_id}/lineage")
def get_synthetic_lineage(synthetic_job_id: str) -> Dict[str, Any]:
    """
    Returns synthetic generation lineage chain.
    """
    lineage = IngestionLineageService().build_chain_summary(job_id=synthetic_job_id)
    job = IngestionJobService().get_job(synthetic_job_id)

    if job.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Synthetic job not found.")

    return {
        "job": job,
        "lineage": lineage,
    }
