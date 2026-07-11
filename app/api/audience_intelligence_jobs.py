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
from app.core.production_guardrails import local_file_storage_allowed
from app.services.audience_run_history_service import AudienceRunHistoryService


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


def _job_nested_get(data: Any, *path: str) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _job_norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _job_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _job_norm(value) in {"true", "1", "yes", "y"}


def _job_collect_warning_text(*sources: Any) -> list[str]:
    warnings: list[str] = []
    for source in sources:
        if not source:
            continue
        if isinstance(source, str):
            warnings.append(source)
        elif isinstance(source, list):
            warnings.extend(str(item) for item in source if item)
        elif isinstance(source, dict):
            for key in ["coverage_warnings", "warnings", "approval_blockers", "blockers"]:
                value = source.get(key)
                if isinstance(value, str):
                    warnings.append(value)
                elif isinstance(value, list):
                    warnings.extend(str(item) for item in value if item)
    return warnings


def _collect_job_approval_blockers(
    *,
    result: Dict[str, Any],
    manifest: Dict[str, Any],
    payload: Dict[str, Any],
    approval: Dict[str, Any],
    cohorts: pd.DataFrame,
) -> list[str]:
    """
    Final approval safety gate for async job approval.

    Manual approval should only unlock downstream export when the package is
    otherwise safe: fresh enough, non-empty, privacy-safe, and not already
    blocked by prompt/category/freshness guardrails.
    """
    blockers: list[str] = []
    result = result or {}
    manifest = manifest or {}
    payload = payload or {}
    approval = approval or {}

    safe_export = result.get("safe_export") or {}

    if cohorts is None or cohorts.empty:
        blockers.append("No exportable cohorts exist for this job.")

    blocked_statuses = {
        "blocked",
        "blocked_approval",
        "blocked_no_safe_exact_match",
        "blocked_privacy_budget",
        "blocked_privacy_leak",
        "blocked_source_freshness",
        "failed",
        "rejected",
    }

    status_sources = {
        "result": result.get("approval_status"),
        "result.safe_export": safe_export.get("approval_status"),
        "manifest": manifest.get("approval_status"),
        "payload": payload.get("approval_status"),
        "approval_request": approval.get("approval_status"),
    }

    for source, status in status_sources.items():
        status_norm = _job_norm(status)
        if status_norm in blocked_statuses or status_norm.startswith("blocked_"):
            blockers.append(f"{source} approval_status is {status_norm}.")

    freshness_values = [
        _job_nested_get(result, "v2_autonomous", "data_freshness", "freshness_status"),
        _job_nested_get(result, "data_freshness", "freshness_status"),
        _job_nested_get(safe_export, "data_freshness", "freshness_status"),
        _job_nested_get(manifest, "v2_autonomous", "data_freshness", "freshness_status"),
        _job_nested_get(manifest, "data_freshness", "freshness_status"),
        manifest.get("freshness"),
        manifest.get("freshness_status"),
        _job_nested_get(payload, "data_freshness", "freshness_status"),
        payload.get("freshness"),
        payload.get("freshness_status"),
        _job_nested_get(approval, "data_freshness", "freshness_status"),
        approval.get("freshness"),
        approval.get("freshness_status"),
    ]

    if any(_job_norm(value) == "stale" for value in freshness_values):
        blockers.append("Source data is stale; refresh or verify source data before approval.")

    swarm_values = [
        _job_nested_get(result, "v2_swarm_review", "overall_review_status"),
        result.get("swarm_review_status"),
        _job_nested_get(manifest, "v2_swarm_review", "overall_review_status"),
        manifest.get("swarm_review_status"),
        payload.get("swarm_review_status"),
        approval.get("swarm_review_status"),
    ]

    if any(_job_norm(value) == "blocked" for value in swarm_values):
        blockers.append("Swarm review is blocked.")

    prompt_filter_report = (
        result.get("prompt_filter_report")
        or manifest.get("prompt_filter_report")
        or payload.get("prompt_filter_report")
        or approval.get("prompt_filter_report")
        or {}
    )

    blocked_filter_modes = {
        "location_category_gap_no_export",
        "broad_location_no_export",
        "privacy_identifier_request_blocked",
        "export_action_requires_existing_audience",
    }

    filter_mode = _job_norm(prompt_filter_report.get("filter_mode"))
    if filter_mode in blocked_filter_modes:
        blockers.append(f"Prompt filter mode blocks approval: {filter_mode}.")

    privacy_sources = [
        result.get("privacy_guarantees") or {},
        safe_export.get("privacy_guarantees") or {},
        manifest.get("privacy_guarantees") or {},
        payload.get("privacy_guarantees") or {},
        approval.get("privacy_guarantees") or {},
    ]

    privacy_flags = [
        "raw_maids_exported",
        "hashed_identifiers_exported",
        "raw_observations_exported",
        "raw_lat_lng_exported",
        "raw_email_exported",
        "raw_phone_exported",
        "individual_user_data_exported",
    ]

    for source in privacy_sources:
        for flag in privacy_flags:
            if _job_bool(source.get(flag)):
                blockers.append(f"Privacy flag blocks approval: {flag}=true.")

    warning_text = " ".join(
        _job_collect_warning_text(result, safe_export, manifest, payload, approval, prompt_filter_report)
    ).lower()

    warning_block_terms = [
        "export blocked",
        "blocked instead of falling back",
        "no exact safe cohort",
        "cannot be exported",
        "raw maids",
        "device ids",
        "individual-level user data",
    ]

    if any(term in warning_text for term in warning_block_terms):
        blockers.append("Coverage/privacy warnings require review before approval.")

    deduped: list[str] = []
    for blocker in blockers:
        if blocker not in deduped:
            deduped.append(blocker)


    top_level_freshness = str(
        result.get("freshness_status")
        or _job_nested_get(
            result,
            "source_freshness",
            "freshness_status",
        )
        or ""
    ).strip().lower()

    if top_level_freshness in {"stale", "expired", "outdated"}:
        blockers.append(
            "Source data is stale; refresh or verify source data before approval."
        )

    explicit_export_block = bool(
        result.get("block_export")
        or safe_export.get("block_export")
        or safe_export.get("export_blocked")
        or safe_export.get("export_blocked_until_source_refresh")
        or manifest.get("block_export")
        or manifest.get("export_blocked")
        or manifest.get("export_blocked_until_source_refresh")
        or payload.get("block_export")
        or approval.get("block_export")
    )

    if explicit_export_block:
        blockers.append(
            "Export is explicitly blocked by production safety guardrails."
        )

    return list(dict.fromkeys(blockers))


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

        if local_file_storage_allowed():
            business_summary_path = (
                Path(result["run_dir"])
                / "business_prompt_summary.md"
            )
            business_summary_path.write_text(
                business_summary,
                encoding="utf-8",
            )
            result["business_summary_path"] = str(
                business_summary_path
            )
        else:
            result["business_summary_path"] = (
                "postgres://audience_jobs.result"
                f"?job_id={job_id}&field=business_summary"
            )

        result["business_summary"] = business_summary

        # Async jobs must persist their completed run before they can be
        # approved through the DB-backed run-history workflow.
        run_history = AudienceRunHistoryService().persist_run(
            final_summary=result,
        )
        result["run_history"] = run_history

        uses_postgres_job_store = (
            getattr(job_store, "backend", "local")
            in AudienceJobStore.POSTGRES_BACKENDS
        )

        if (
            uses_postgres_job_store
            and run_history.get("status") != "persisted"
        ):
            raise RuntimeError(
                "Audience run-history persistence failed for async job "
                f"{job_id}: {run_history}"
            )

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
def approve_job_export(
    job_id: str,
    request: ApprovalRequest,
) -> Dict[str, Any]:
    try:
        record = job_store.get(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if record["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Only completed jobs can be approved.",
        )

    result = record.get("result") or {}
    run_id = result.get("run_id")

    if not run_id:
        raise HTTPException(
            status_code=400,
            detail="Completed job does not contain a run_id.",
        )

    service = AudienceRunHistoryService()
    decision = service.approve_run(
        run_id=run_id,
        actor=request.approver,
        note=request.note,
        downstream_export_enabled=True,
    )

    decision_status = decision.get("status")

    if decision_status == "skipped":
        raise HTTPException(status_code=503, detail=decision)

    if decision_status == "not_found":
        raise HTTPException(status_code=404, detail=decision)

    if decision_status == "failed":
        raise HTTPException(status_code=500, detail=decision)

    refreshed = service.get_run(run_id)

    if refreshed.get("status") == "ok":
        run = refreshed.get("run") or {}
        final_summary = run.get("final_summary") or {}

        if isinstance(final_summary, str):
            try:
                final_summary = json.loads(final_summary)
            except json.JSONDecodeError:
                final_summary = {}

        refreshed_export = (
            final_summary.get("safe_export")
            or run.get("safe_export")
            or {}
        )

        if refreshed_export:
            result["safe_export"] = refreshed_export

        result["approval_status"] = run.get(
            "approval_status",
            decision.get("approval_status"),
        )
        result["downstream_export_enabled"] = bool(
            run.get(
                "downstream_export_enabled",
                decision.get(
                    "downstream_export_enabled",
                    False,
                ),
            )
        )

    if decision_status == "blocked":
        blocked_status = (
            decision.get("approval_status")
            or "blocked_approval"
        )

        safe_export = result.get("safe_export") or {}
        safe_export["approval_status"] = blocked_status
        safe_export["downstream_export_enabled"] = False
        safe_export["approval_blockers"] = (
            decision.get("blockers")
            or decision.get("approval_blockers")
            or []
        )

        result["safe_export"] = safe_export
        result["approval_status"] = blocked_status
        result["downstream_export_enabled"] = False

    result["approval_decision"] = decision
    record["result"] = result
    job_store.save(record)

    response = {
        "job_id": job_id,
        **decision,
    }

    if decision_status == "blocked":
        raise HTTPException(status_code=400, detail=response)

    return response


def _safe_result_for_job(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": result.get("status"),
        "run_id": result.get("run_id"),
        "freshness_status": result.get("freshness_status"),
        "source_freshness": result.get("source_freshness"),
        "approval_status": (
            result.get("approval_status")
            or (result.get("safe_export") or {}).get(
                "approval_status"
            )
        ),
        "downstream_export_enabled": bool(
            result.get(
                "downstream_export_enabled",
                (result.get("safe_export") or {}).get(
                    "downstream_export_enabled",
                    False,
                ),
            )
        ),
        "block_export": bool(
            result.get(
                "block_export",
                (result.get("safe_export") or {}).get(
                    "block_export",
                    False,
                ),
            )
        ),
        "prompt": result.get("prompt"),
        "source_mode": result.get("source_mode"),
        "source_rows": result.get("source_rows"),
        "prompt_selected_cohorts": result.get("prompt_selected_cohorts"),
        "coverage_warnings": result.get("coverage_warnings", []),
        "prompt_filter_report": result.get("prompt_filter_report"),
        "v2_autonomous": result.get("v2_autonomous"),
        "v2_swarm_review": result.get("v2_swarm_review"),
        "business_summary": result.get("business_summary"),
        "business_summary_path": result.get("business_summary_path"),
        "final_summary_path": result.get("final_summary_path"),
        "run_dir": result.get("run_dir"),
        "safe_export": result.get("safe_export"),
        "privacy_guarantees": result.get("privacy_guarantees"),
        "run_history": result.get("run_history"),
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
