from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent
from app.agents.audience_supervisor_agent import build_audience_execution_agent
from app.services.audience_run_history_service import AudienceRunHistoryService

from app.core.audience_request_context import (
    AudienceRequestContext,
    require_authenticated_audience_request,
)
from app.core.local_ui_session import require_local_ui_session
from app.core.production_guardrails import local_file_storage_allowed

router = APIRouter(
    prefix="/api/audience-intelligence/prompt",
    tags=["Audience Intelligence Prompt"],
)



def _audience_execution_agent():
    return build_audience_execution_agent(
        orchestrator_factory=(
            AudienceIntelligenceOrchestratorAgent
        ),
    )


class AudiencePromptRequest(BaseModel):
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


class LocalAudiencePromptRequest(BaseModel):
    """Browser-safe request contract for the local workspace."""

    prompt: str = Field(..., min_length=3, max_length=4000)


def _business_label(value: Any) -> str:
    normalized = str(value or "not available").replace("_", " ")
    return normalized[:1].upper() + normalized[1:]


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
    terminal_policy_decision = bool(
        result.get("terminal_policy_decision")
    )
    source_rows_checked = result.get("source_rows")
    if source_rows_checked is None and not terminal_policy_decision:
        source_rows_checked = (
            (
                (result.get("v2_autonomous") or {}).get(
                    "data_freshness"
                )
                or {}
            ).get("source_rows_checked")
        )
    if source_rows_checked is None:
        source_rows_checked = (
            "not evaluated"
            if terminal_policy_decision
            else "unknown"
        )
    lines.append(f"Source rows checked: {source_rows_checked}")
    if terminal_policy_decision:
        safe_export = result.get("safe_export", {}) or {}
        approval_status = str(
            result.get("approval_status")
            or safe_export.get("approval_status")
            or "blocked"
        )
        lines.append(
            "Decision: " + _business_label(approval_status)
        )
        lines.append("Downstream export enabled: False")
        lines.append("")
        lines.append("## Safety decision")
        lines.append("")
        if approval_status == "blocked_privacy_identifier_request":
            lines.append(
                "Raw MAIDs, device IDs, and individual-level user data "
                "cannot be provided or exported. Only privacy-safe "
                "aggregated cohorts are allowed."
            )
            lines.append(
                "No audience ranking, preparation, or export was "
                "attempted. No activation was performed."
            )
        elif approval_status == "blocked_approval_bypass_attempt":
            lines.append(
                "Safety, freshness, governance, and manual approval "
                "controls cannot be bypassed."
            )
            lines.append(
                "The request was stopped before source access, ranking, "
                "audience preparation, activation, or export."
            )
        else:
            lines.append(
                "An export action requires an existing selected "
                "run/audience and manual approval through the "
                "authenticated approval workflow."
            )
            lines.append(
                "No new audience ranking, preparation, or export was "
                "attempted. No activation was performed."
            )
        lines.append("")
        lines.append("## Execution")
        lines.append("")
        lines.append("- Source data access: not evaluated")
        lines.append("- Semantic retrieval: skipped")
        lines.append("- Cohort intelligence: skipped")
        lines.append("- Evolution review: skipped")
        lines.append("- Downstream activation/export: blocked")
        warnings = result.get("coverage_warnings", []) or []
        if warnings:
            lines.append("")
            lines.append("## Review note")
            lines.append("")
            for warning in warnings:
                lines.append(f"- {warning}")
        lines.append("")
        lines.append("## Privacy and delivery")
        lines.append("")
        lines.append("Raw or hashed identifiers returned: False")
        lines.append("Individual-level data returned: False")
        lines.append("Audience activated or exported: False")
        return "\n".join(lines)

    lines.append(f"Prompt-selected cohorts: {result.get('prompt_selected_cohorts')}")
    safe_export = result.get("safe_export", {}) or {}
    audience_count = int(
        safe_export.get("exported_cohorts") or 0
    )
    downstream_enabled = bool(
        safe_export.get(
            "downstream_export_enabled"
        )
    )

    audience_count_label = (
        "Delivered audiences"
        if downstream_enabled
        else "Prepared audience candidates"
    )

    lines.append(
        f"{audience_count_label}: "
        f"{audience_count}"
    )
    lines.append(
        "Lookalike pairs: "
        f"{safe_export.get('exported_lookalike_pairs')}"
    )
    lines.append(
        "Approval status: "
        f"{safe_export.get('approval_status')}"
    )
    lines.append(
        "Downstream export enabled: "
        f"{safe_export.get('downstream_export_enabled')}"
    )
    lines.append("")

    filter_report = result.get("prompt_filter_report", {})
    lines.append("## Prompt understanding")
    lines.append("")
    lines.append(f"Filter mode used: {filter_report.get('filter_mode')}")

    fulfillment = filter_report.get('fulfillment_status')
    if fulfillment:
        lines.append(f"Fulfillment status: {fulfillment}")

    lines.append(f"Locations detected: {', '.join(filter_report.get('locations_detected', []) or ['none'])}")

    if filter_report.get('matched_requested_locations'):
        lines.append(f"Matched locations: {', '.join(filter_report.get('matched_requested_locations', []))}")
    if filter_report.get('missing_requested_locations'):
        lines.append(f"Missing locations: {', '.join(filter_report.get('missing_requested_locations', []))}")

    quality = filter_report.get('quality_intent')
    if quality:
        lines.append(f"Requested quality: {quality}")

    q_report = filter_report.get('quality_policy_report')
    if q_report:
        lines.append(f"Quality policy: {q_report.get('quality_policy_status', 'not_requested')}")
        if q_report.get('quality_policy_status') != 'not_requested':
            lines.append(f"Quality-qualified candidates: {q_report.get('quality_candidates_after', 0)}")
            lines.append(f"Candidates excluded below requested quality: {q_report.get('quality_excluded_count', 0)}")

            loc_meeting = q_report.get('locations_meeting_quality', [])
            if loc_meeting:
                lines.append(f"Locations meeting requested quality: {', '.join(loc_meeting)}")

            loc_missing = q_report.get('locations_missing_quality', [])
            if loc_missing:
                lines.append(f"Locations unable to meet requested quality: {', '.join(loc_missing)}")

    lines.append(f"POI terms detected: {', '.join(filter_report.get('poi_terms_detected', []) or ['none'])}")
    lines.append(f"Dayparts detected: {', '.join(filter_report.get('dayparts_detected', []) or ['none'])}")
    lines.append("")

    filter_mode_for_safety = str(filter_report.get("filter_mode") or "")
    terminal_safety_modes = {
        "privacy_identifier_request_blocked",
        "export_action_requires_existing_audience",
        "approval_bypass_attempt_blocked",
    }
    if filter_mode_for_safety == "privacy_identifier_request_blocked":
        lines.append("## Safety decision")
        lines.append("")
        lines.append(
            "This request asked for raw MAIDs, device IDs, or individual-level user data. "
            "Those cannot be provided or exported. Only privacy-safe aggregated cohorts are allowed."
        )
        lines.append("")
    elif filter_mode_for_safety == "export_action_requires_existing_audience":
        lines.append("## Safety decision")
        lines.append("")
        lines.append(
            "This was an export-action-only request. The system will not create or export a new audience "
            "without an existing selected run/audience and manual approval."
        )
        lines.append("")
    elif filter_mode_for_safety == "approval_bypass_attempt_blocked":
        lines.append("## Safety decision")
        lines.append("")
        lines.append(
            "Safety, freshness, governance, and manual approval controls "
            "cannot be bypassed. No source data was read and no audience "
            "was ranked, prepared, activated, or exported."
        )
        lines.append("")

    filter_report = result.get("prompt_filter_report", {}) or {}
    filter_mode = str(filter_report.get("filter_mode") or "")

    if (
        filter_mode
        and filter_mode != "location+poi+daypart"
        and filter_mode not in terminal_safety_modes
    ):
        lines.append("## Match note")
        lines.append("")
        lines.append(
            "Exact match and fallback status are shown in Coverage warnings and Swarm review. Review before approval."
        )
        lines.append("")

    warnings = result.get("coverage_warnings", []) or []
    if warnings:
        lines.append("## Coverage warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    v2 = result.get("v2_autonomous", {}) or {}
    v2_review = result.get("v2_swarm_review", {}) or {}
    if v2 and terminal_policy_decision:
        lines.append("## Autonomous Audience Intelligence v2")
        lines.append("")
        lines.append("Status: skipped")
        lines.append(
            "Retrieval, ranking, mutation, and evolution were not evaluated "
            "because the request was stopped by the terminal safety policy."
        )
        lines.append("")
    elif v2:
        lines.append("## Autonomous Audience Intelligence v2")
        lines.append("")
        lines.append(f"Status: {v2.get('status')}")
        lines.append(f"Pipeline: {v2.get('pipeline_version')}")
        lines.append(f"Prompt confidence: {(v2.get('prompt_intent') or {}).get('confidence_score')}")
        lines.append(f"Freshness: {(v2.get('data_freshness') or {}).get('freshness_status')}")
        lines.append(f"Latest source timestamp: {(v2.get('data_freshness') or {}).get('latest_source_timestamp')}")
        lines.append(f"Source rows checked: {(v2.get('data_freshness') or {}).get('source_rows_checked')}")
        lines.append(f"Vector count: {(v2.get('embedding_manifest') or {}).get('vector_count')}")
        lines.append(f"Vector dimension: {(v2.get('embedding_manifest') or {}).get('vector_dimension')}")
        lines.append(f"Ranked matches: {v2.get('ranked_match_count')}")
        lines.append(f"Mutation suggestions: {(v2.get('mutation') or {}).get('suggestion_count')}")
        lines.append(f"Approval required: {v2.get('approval_required')}")
        lines.append(f"Downstream export enabled: {v2.get('downstream_export_enabled')}")

        if v2_review:
            lines.append(f"Swarm review status: {v2_review.get('overall_review_status')}")
            lines.append(f"Coverage warnings: {v2_review.get('coverage_warning_count')}")
            lines.append(f"Data gaps: {v2_review.get('data_gap_count')}")
            recommendations = v2_review.get("recommendations") or []
            if recommendations:
                lines.append("Swarm recommendations:")
                for recommendation in recommendations:
                    lines.append(f"- {recommendation}")

        lines.append("")

    export_outputs = result.get("safe_export", {}).get("outputs", {}) or {}
    cohorts_path_value = export_outputs.get("safe_export_cohorts")

    if cohorts_path_value:
        cohorts_path = Path(str(cohorts_path_value))
    else:
        cohorts_path = None

    if cohorts_path and cohorts_path.is_file():
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
    else:
        lines.append("## Created approval-gated audiences")
        lines.append("")

        if filter_mode in terminal_safety_modes:
            lines.append(
                "No audience selection or preparation was attempted because "
                "this request was resolved by the terminal safety decision."
            )
        elif audience_count > 0:
            lines.append(
                "No local audience file was created. "
                "Approval-gated candidates are stored in "
                "Postgres run history."
            )
        else:
            lines.append(
                "No audience candidate was created for this run."
            )

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

    safe_export = result.get("safe_export", {}) or {}
    freshness_status = (
        (
            (result.get("v2_autonomous") or {}).get(
                "data_freshness"
            )
            or {}
        ).get("freshness_status")
    )
    downstream_enabled = bool(
        safe_export.get("downstream_export_enabled")
    )
    approval_status = str(
        safe_export.get("approval_status") or ""
    )

    if approval_status == "blocked_privacy_identifier_request":
        lines.append(
            "The raw-identifier request was blocked by the privacy guardrail. "
            "No audience ranking, preparation, or export was attempted."
        )
    elif approval_status == "blocked_export_action_requires_existing_audience":
        lines.append(
            "The export action was blocked because no existing approved "
            "audience was supplied. No new audience ranking, preparation, "
            "or export was attempted."
        )
    elif approval_status == "blocked_approval_bypass_attempt":
        lines.append(
            "The request attempted to bypass authoritative safety or "
            "approval controls. It was stopped before source access, "
            "ranking, audience preparation, activation, or export."
        )
    elif approval_status == "blocked_no_safe_exact_match":
        lines.append(
            "No audience candidate was created because no exact "
            "privacy-safe cohort matched the requested location, "
            "category, and daypart."
        )
    elif approval_status == "blocked_requested_quality_unmet":
        lines.append(
            "Exact privacy-safe cohorts were available, but none met the requested "
            "quality requirement."
        )
    elif approval_status == "blocked_v2_failure":
        lines.append(
            "No audience candidate was approved because the "
            "Autonomous Audience Intelligence v2 stage failed. "
            "Downstream delivery remains blocked."
        )
    elif audience_count == 0:
        lines.append(
            "No audience candidate was created for this run."
        )
    elif (
        not downstream_enabled
        and str(freshness_status).lower() == "stale"
    ):
        lines.append(
            "The prepared audience candidate is reviewable, but "
            "downstream delivery is blocked because source data is "
            "stale and manual approval is required."
        )
    elif not downstream_enabled:
        lines.append(
            "The prepared audience candidate is reviewable, but "
            "downstream delivery is blocked until safety checks and "
            "manual approval pass."
        )
    else:
        lines.append(
            "The audience is approved for downstream delivery."
        )

    lines.append("")
    lines.append(f"Run folder: {result.get('run_dir')}")

    return "\n".join(lines)


def _build_prompt_api_response(
    *,
    result: Dict[str, Any],
    business_summary: str,
    business_summary_path: Path,
) -> Dict[str, Any]:
    safe_export = result.get("safe_export", {}) or {}
    v2 = result.get("v2_autonomous", {}) or {}
    v2_freshness = v2.get("data_freshness", {}) or {}
    source_freshness = result.get("source_freshness", {}) or {}

    freshness_status = (
        result.get("freshness_status")
        or source_freshness.get("freshness_status")
        or v2_freshness.get("freshness_status")
        or v2_freshness.get("status")
    )

    approval_status = (
        result.get("approval_status")
        or safe_export.get("approval_status")
    )

    downstream_export_enabled = bool(
        result.get(
            "downstream_export_enabled",
            safe_export.get("downstream_export_enabled", False),
        )
    )

    block_export = bool(
        result.get("block_export")
        or safe_export.get("block_export")
        or approval_status == "blocked_stale_source"
    )

    block_export_reason = (
        result.get("block_export_reason")
        or safe_export.get("block_export_reason")
        or source_freshness.get("block_export_reason")
        or source_freshness.get("reason")
        or v2_freshness.get("reason")
        or v2_freshness.get("diagnosis")
    )

    safe_export_response = {
        "approval_status": approval_status,
        "downstream_export_enabled": downstream_export_enabled,
        "block_export": block_export,
        "block_export_reason": block_export_reason,
        "export_blocked_until_source_refresh": bool(
            safe_export.get("export_blocked_until_source_refresh")
            or approval_status == "blocked_stale_source"
        ),
        "exported_cohorts": safe_export.get("exported_cohorts", 0),
        "exported_lookalike_pairs": safe_export.get("exported_lookalike_pairs", 0),
        "outputs": safe_export.get("outputs", {}),
    }

    response = {
        "status": result.get("status"),
        "run_id": result.get("run_id"),
        "source_mode": result.get("source_mode"),
        "source_rows": (
            result.get("source_rows")
            if result.get("source_rows") is not None
            else (
                None
                if result.get("terminal_policy_decision")
                else v2_freshness.get("source_rows_checked")
            )
        ),
        "freshness_status": freshness_status,
        "source_freshness": source_freshness or v2_freshness,
        "approval_status": approval_status,
        "downstream_export_enabled": downstream_export_enabled,
        "block_export": block_export,
        "block_export_reason": block_export_reason,
        "privacy_cohorts": result.get("privacy_cohorts"),
        "business_summary": business_summary,
        "business_summary_path": str(business_summary_path),
        "final_summary_path": result.get("final_summary_path"),
        "run_dir": result.get("run_dir"),
        "prompt_selected_cohorts": result.get("prompt_selected_cohorts", 0),
        "prompt_filter_report": result.get("prompt_filter_report", {}),
        "coverage_warnings": result.get("coverage_warnings", []),
        "v2_autonomous": v2,
        "v2_swarm_review": result.get("v2_swarm_review", {}),
        "safe_export": safe_export_response,
        "privacy_guarantees": result.get("privacy_guarantees", {}),
        "run_history": result.get("run_history", {}),
        "terminal_policy_decision": bool(
            result.get("terminal_policy_decision")
        ),
        "policy_reason_code": result.get("policy_reason_code"),
        "pipeline_stages": list(
            result.get("pipeline_stages") or []
        ),
    }

    if result.get("supervisor_decision"):
        response.update(
            {
                "supervisor_decision": result.get(
                    "supervisor_decision"
                ),
                "supervisor_route": result.get(
                    "supervisor_route"
                ),
                "supervisor_stage": result.get(
                    "supervisor_stage"
                ),
                "supervisor_reason_codes": list(
                    result.get(
                        "supervisor_reason_codes"
                    )
                    or []
                ),
                "supervisor_trace": list(
                    result.get("supervisor_trace")
                    or []
                ),
            }
        )

        if result.get("graph_terminal_status"):
            response["graph_terminal_status"] = result.get(
                "graph_terminal_status"
            )
            response["supervisor_graph_trace"] = list(
                result.get("supervisor_graph_trace")
                or []
            )

        if result.get("supervisor_graph_error_type"):
            response["supervisor_graph_error_type"] = result.get(
                "supervisor_graph_error_type"
            )

    return response


def _execute_audience_prompt(
    request: AudiencePromptRequest,
    context: AudienceRequestContext,
) -> Dict[str, Any]:
    try:
        agent = _audience_execution_agent()

        result = agent.run(
            prompt=request.prompt,
            output_root=request.output_root,
            source=request.source,
            safe_cohort_path=request.safe_cohort_path,
            postgres_limit=request.postgres_limit,
            k_min=request.k_min,
            epsilon=request.epsilon,
            synthetic_rows=request.synthetic_rows,
            max_export_cohorts=request.max_export_cohorts,
            min_export_quality=request.min_export_quality,
            approval_required=request.approval_required,
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
            business_summary_path = (
                "postgres://audience_run_history.final_summary"
                f"?run_id={result['run_id']}"
                "&field=business_summary"
            )
            result["business_summary_path"] = business_summary_path
        result["business_summary"] = business_summary
        result["tenant_id"] = context.tenant_id

        run_history = AudienceRunHistoryService().persist_run(
            tenant_id=context.tenant_id,
            final_summary=result,
        )
        result["run_history"] = run_history

        return _build_prompt_api_response(
            result=result,
            business_summary=business_summary,
            business_summary_path=business_summary_path,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/run")
def run_audience_prompt(
    request: AudiencePromptRequest,
    context: AudienceRequestContext = Depends(
        require_authenticated_audience_request
    ),
) -> Dict[str, Any]:
    return _execute_audience_prompt(request, context)


@router.post("/ui/run")
def run_local_audience_prompt(
    request: LocalAudiencePromptRequest,
    context: AudienceRequestContext = Depends(
        require_local_ui_session
    ),
) -> Dict[str, Any]:
    operational_request = AudiencePromptRequest(
        prompt=request.prompt,
        source="postgres",
        output_root="data/prompt_runs",
        safe_cohort_path=None,
        postgres_limit=10000,
        k_min=1000,
        epsilon=1.0,
        synthetic_rows=1000,
        max_export_cohorts=25,
        min_export_quality=0.25,
        approval_required=True,
    )
    return _execute_audience_prompt(
        operational_request,
        context,
    )


@router.get("/ui", response_class=RedirectResponse)
def audience_prompt_ui() -> RedirectResponse:
    return RedirectResponse(
        url="/audience-workspace",
        status_code=307,
    )
