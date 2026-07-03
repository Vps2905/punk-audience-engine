from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent

from app.core.api_key_auth import require_audience_api_key


router = APIRouter(
    prefix="/api/audience-intelligence/prompt",
    tags=["Audience Intelligence Prompt"],
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

    filter_report = result.get("prompt_filter_report", {})
    lines.append("## Prompt understanding")
    lines.append("")
    lines.append(f"Filter mode used: {filter_report.get('filter_mode')}")
    lines.append(f"Locations detected: {', '.join(filter_report.get('locations_detected', []) or ['none'])}")
    lines.append(f"POI terms detected: {', '.join(filter_report.get('poi_terms_detected', []) or ['none'])}")
    lines.append(f"Dayparts detected: {', '.join(filter_report.get('dayparts_detected', []) or ['none'])}")
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


@router.post("/run", dependencies=[Depends(require_audience_api_key)])
def run_audience_prompt(request: AudiencePromptRequest) -> Dict[str, Any]:
    try:
        agent = AudienceIntelligenceOrchestratorAgent()

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

        business_summary_path = Path(result["run_dir"]) / "business_prompt_summary.md"
        business_summary_path.write_text(business_summary)

        return {
            "status": result["status"],
            "run_id": result["run_id"],
            "business_summary": business_summary,
            "business_summary_path": str(business_summary_path),
            "final_summary_path": result["final_summary_path"],
            "run_dir": result["run_dir"],
            "prompt_selected_cohorts": result["prompt_selected_cohorts"],
            "coverage_warnings": result.get("coverage_warnings", []),
            "safe_export": {
                "approval_status": result["safe_export"]["approval_status"],
                "downstream_export_enabled": result["safe_export"]["downstream_export_enabled"],
                "exported_cohorts": result["safe_export"]["exported_cohorts"],
                "exported_lookalike_pairs": result["safe_export"]["exported_lookalike_pairs"],
                "outputs": result["safe_export"]["outputs"],
            },
            "privacy_guarantees": result["privacy_guarantees"],
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/ui", response_class=HTMLResponse)
def audience_prompt_ui() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Audience Intelligence Prompt Runner</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; max-width: 1100px; }
    textarea { width: 100%; height: 110px; font-size: 15px; }
    button { padding: 12px 18px; font-size: 15px; cursor: pointer; }
    pre { background: #111; color: #eee; padding: 16px; border-radius: 8px; white-space: pre-wrap; }
    .row { margin: 14px 0; }
    input, select { padding: 8px; font-size: 14px; width: 100%; }
  </style>
</head>
<body>
  <h1>Audience Intelligence Prompt Runner</h1>

  <div class="row">
    <label>Prompt</label>
    <textarea id="prompt">Build me a high-quality restaurant evening audience for Montreal and San Francisco</textarea>
  </div>

  <div class="row">
    <label>Source</label>
    <select id="source">
      <option value="postgres">postgres</option>
      <option value="safe_artifact">safe_artifact</option>
    </select>
  </div>

  <div class="row">
    <label>Safe cohort path, only for safe_artifact mode</label>
    <input id="safePath" value="data/modular_runs/postgres_privacy_to_synthetic/01_privacy/clean_feature_table.csv" />
  </div>

  <div class="row">
    <label>Audience API Key</label>
    <input id="apiKey" type="password" placeholder="Paste local API key from .env" />
  </div>

  <button onclick="runPrompt()">Run Audience Intelligence</button>

  <h2>Result</h2>
  <pre id="output">Waiting...</pre>

  <script>
    async function runPrompt() {
      const output = document.getElementById("output");
      output.textContent = "Running pipeline...";

      const source = document.getElementById("source").value;
      const payload = {
        prompt: document.getElementById("prompt").value,
        source: source,
        safe_cohort_path: source === "safe_artifact" ? document.getElementById("safePath").value : null,
        approval_required: true,
        postgres_limit: 10000,
        k_min: 1000,
        epsilon: 1.0,
        synthetic_rows: 1000,
        max_export_cohorts: 25,
        min_export_quality: 0.25
      };

      try {
        const res = await fetch("/api/audience-intelligence/prompt/run", {
          method: "POST",
          headers: {"Content-Type": "application/json", "X-Audience-API-Key": document.getElementById("apiKey").value},
          body: JSON.stringify(payload)
        });

        const data = await res.json();

        if (!res.ok) {
          output.textContent = "Error:\\n" + JSON.stringify(data, null, 2);
          return;
        }

        const summary = String(data.business_summary || "");
        output.textContent = summary.split("\\n").join(String.fromCharCode(10));
      } catch (err) {
        output.textContent = "Request failed: " + err;
      }
    }
  </script>
</body>
</html>
"""


