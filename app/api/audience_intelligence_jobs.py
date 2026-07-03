from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent
from app.core.audience_job_store import AudienceJobStore

from app.core.api_key_auth import require_audience_api_key


router = APIRouter(
    prefix="/api/audience-intelligence/jobs",
    tags=["Audience Intelligence Jobs"],
    dependencies=[Depends(require_audience_api_key)],
)

job_store = AudienceJobStore()


class AudienceJobRequest(BaseModel):
    prompt: str = Field(..., min_length=3)
    source: str = Field(default="postgres", pattern="^(postgres|safe_artifact)$")
    safe_cohort_path: Optional[str] = None
    output_root: str = "data/prompt_runs"
    postgres_limit: int = 10000
    k_min: int = 1000
    epsilon: float = 1.0
    synthetic_rows: int = 1000
    max_export_cohorts: int = 25
    min_export_quality: float = 0.25
    approval_required: bool = True


class ApprovalRequest(BaseModel):
    approver: str = "internal_reviewer"
    note: Optional[str] = None


def _run_job_background(job_id: str) -> None:
    record = job_store.get(job_id)
    payload = record["payload"]

    try:
        job_store.update_status(
            job_id,
            status="running",
            stage="orchestration_started",
            message="Running Audience Intelligence pipeline.",
        )

        agent = AudienceIntelligenceOrchestratorAgent()

        result = agent.run(
            prompt=payload["prompt"],
            output_root=payload.get("output_root", "data/prompt_runs"),
            source=payload.get("source", "postgres"),
            safe_cohort_path=payload.get("safe_cohort_path"),
            postgres_limit=int(payload.get("postgres_limit", 10000)),
            k_min=int(payload.get("k_min", 1000)),
            epsilon=float(payload.get("epsilon", 1.0)),
            synthetic_rows=int(payload.get("synthetic_rows", 1000)),
            max_export_cohorts=int(payload.get("max_export_cohorts", 25)),
            min_export_quality=float(payload.get("min_export_quality", 0.25)),
            approval_required=bool(payload.get("approval_required", True)),
        )

        business_summary = _build_business_summary(result)
        business_summary_path = Path(result["run_dir"]) / "business_prompt_summary.md"
        business_summary_path.write_text(business_summary)

        result["business_summary_path"] = str(business_summary_path)
        result["business_summary"] = business_summary

        job_store.update_status(
            job_id,
            status="completed",
            stage="completed",
            message="Audience Intelligence job completed.",
            result=_safe_result_for_job(result),
        )

    except Exception as exc:
        job_store.update_status(
            job_id,
            status="failed",
            stage="failed",
            message="Audience Intelligence job failed.",
            error=str(exc),
        )


@router.post("/run")
def run_job(request: AudienceJobRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    if request.source == "safe_artifact" and not request.safe_cohort_path:
        raise HTTPException(
            status_code=400,
            detail="safe_cohort_path is required when source is safe_artifact.",
        )

    record = job_store.create_job(request.model_dump())
    background_tasks.add_task(_run_job_background, record["job_id"])

    return {
        "status": "queued",
        "job_id": record["job_id"],
        "status_url": f"/api/audience-intelligence/jobs/status/{record['job_id']}",
        "result_url": f"/api/audience-intelligence/jobs/result/{record['job_id']}",
    }


@router.get("/status/{job_id}")
def get_status(job_id: str) -> Dict[str, Any]:
    try:
        record = job_store.get(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "job_id": record["job_id"],
        "status": record["status"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "progress": record["progress"],
        "error": record.get("error"),
    }


@router.get("/result/{job_id}")
def get_result(job_id: str) -> Dict[str, Any]:
    try:
        record = job_store.get(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if record["status"] != "completed":
        return {
            "job_id": job_id,
            "status": record["status"],
            "progress": record["progress"],
            "error": record.get("error"),
            "result": None,
        }

    return {
        "job_id": job_id,
        "status": record["status"],
        "result": record["result"],
    }


@router.post("/approve/{job_id}")
def approve_job_export(job_id: str, request: ApprovalRequest) -> Dict[str, Any]:
    try:
        record = job_store.get(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if record["status"] != "completed":
        raise HTTPException(status_code=400, detail="Only completed jobs can be approved.")

    result = record.get("result") or {}
    run_dir = result.get("run_dir")

    if not run_dir:
        raise HTTPException(status_code=400, detail="Completed job does not have run_dir.")

    run_dir_path = Path(run_dir)
    safe_export_dir = run_dir_path / "05_safe_export"

    manifest_path = safe_export_dir / "safe_export_manifest.json"
    payload_path = safe_export_dir / "safe_export_payload.json"
    approval_path = safe_export_dir / "export_approval_request.json"
    cohorts_path = safe_export_dir / "safe_export_cohorts.csv"
    approval_decision_path = safe_export_dir / "approval_decision.json"

    for path in [manifest_path, payload_path, approval_path, cohorts_path]:
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"Missing export artifact: {path}")

    manifest = json.loads(manifest_path.read_text())
    payload = json.loads(payload_path.read_text())
    approval = json.loads(approval_path.read_text())
    cohorts = pd.read_csv(cohorts_path)

    manifest["approval_status"] = "approved"
    manifest["downstream_export_enabled"] = True
    manifest["export_blocked_until_approved"] = False
    manifest["approved_at"] = datetime.now(timezone.utc).isoformat()
    manifest["approved_by"] = request.approver

    payload["approval_status"] = "approved"
    payload["downstream_export_enabled"] = True

    approval["approval_status"] = "approved"
    approval["export_blocked_until_approved"] = False
    approval["review_required_before_downstream_delivery"] = False
    approval["approved_at"] = manifest["approved_at"]
    approval["approved_by"] = request.approver
    approval["approval_note"] = request.note

    if "export_status" in cohorts.columns:
        cohorts["export_status"] = "approved"

    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    payload_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    approval_path.write_text(json.dumps(approval, indent=2, allow_nan=False))
    cohorts.to_csv(cohorts_path, index=False)

    approval_decision = {
        "job_id": job_id,
        "run_dir": str(run_dir_path),
        "approval_status": "approved",
        "approved_at": manifest["approved_at"],
        "approved_by": request.approver,
        "note": request.note,
        "downstream_export_enabled": True,
        "meta_upload_performed": False,
        "message": "Safe export package approved. Meta upload is still a separate explicit step.",
    }

    approval_decision_path.write_text(json.dumps(approval_decision, indent=2, allow_nan=False))

    result["safe_export"]["approval_status"] = "approved"
    result["safe_export"]["downstream_export_enabled"] = True
    result["safe_export"]["export_blocked_until_approved"] = False
    result["approval_decision_path"] = str(approval_decision_path)

    record["result"] = result
    job_store.save(record)

    return approval_decision


def _safe_result_for_job(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": result.get("status"),
        "run_id": result.get("run_id"),
        "prompt": result.get("prompt"),
        "source_mode": result.get("source_mode"),
        "source_rows": result.get("source_rows"),
        "prompt_selected_cohorts": result.get("prompt_selected_cohorts"),
        "coverage_warnings": result.get("coverage_warnings", []),
        "business_summary": result.get("business_summary"),
        "business_summary_path": result.get("business_summary_path"),
        "final_summary_path": result.get("final_summary_path"),
        "run_dir": result.get("run_dir"),
        "safe_export": result.get("safe_export"),
        "privacy_guarantees": result.get("privacy_guarantees"),
    }


def _build_business_summary(result: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Audience Intelligence Result")
    lines.append("")
    lines.append(f"Prompt: {result.get('prompt')}")
    lines.append(f"Run ID: {result.get('run_id')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"Source mode: {result.get('source_mode')}")
    lines.append(f"Source rows checked: {result.get('source_rows')}")
    lines.append(f"Prompt-selected cohorts: {result.get('prompt_selected_cohorts')}")
    lines.append(f"Exported audiences: {result.get('safe_export', {}).get('exported_cohorts')}")
    lines.append(f"Lookalike pairs: {result.get('safe_export', {}).get('exported_lookalike_pairs')}")
    lines.append(f"Approval status: {result.get('safe_export', {}).get('approval_status')}")
    lines.append(f"Downstream export enabled: {result.get('safe_export', {}).get('downstream_export_enabled')}")
    lines.append("")

    filter_report = result.get("prompt_filter_report", {}) or {}
    filter_mode = str(filter_report.get("filter_mode") or "")

    if filter_mode and filter_mode != "location+poi+daypart":
        lines.append("## Match note")
        lines.append("")
        lines.append(
            "Exact location + business/category + time match was not strong enough, so the system used the safest available fallback match. Review these audiences before approval."
        )
        lines.append("")

    warnings = result.get("coverage_warnings", []) or []
    if warnings:
        lines.append("## Coverage warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    export_outputs = result.get("safe_export", {}).get("outputs", {})
    cohorts_path = Path(export_outputs.get("safe_export_cohorts", ""))

    if cohorts_path.exists():
        cohorts = pd.read_csv(cohorts_path)
        lines.append("## Created approval-gated audiences")
        lines.append("")

        for idx, row in cohorts.head(10).iterrows():
            lines.append(f"{idx + 1}. {row.get('audience_name', 'Unnamed audience')}")
            lines.append(f"   - Location: {row.get('location_name')}")
            lines.append(f"   - POI type: {row.get('primary_poi_type')}")
            lines.append(f"   - Daypart: {row.get('created_day_part')}")
            lines.append(f"   - Quality: {float(row.get('management_quality_score', 0)):.3f}")
            lines.append(f"   - Status: {row.get('export_status')}")
            lines.append("")

    privacy = result.get("privacy_guarantees", {})

    lines.append("## Privacy and export safety")
    lines.append("")
    lines.append(f"Raw MAIDs exported: {privacy.get('raw_maids_exported')}")
    lines.append(f"Hashed identifiers exported: {privacy.get('hashed_identifiers_exported')}")
    lines.append(f"Raw observations exported: {privacy.get('raw_observations_exported')}")
    lines.append(f"Raw lat/lng exported: {privacy.get('raw_lat_lng_exported')}")
    lines.append(f"Email/phone exported: {privacy.get('raw_email_exported')} / {privacy.get('raw_phone_exported')}")
    lines.append(f"Individual user data exported: {privacy.get('individual_user_data_exported')}")
    lines.append("")
    lines.append("Export package is ready for review, but downstream delivery is blocked until approval.")
    lines.append("")
    lines.append(f"Run folder: {result.get('run_dir')}")

    return "\n".join(lines)
