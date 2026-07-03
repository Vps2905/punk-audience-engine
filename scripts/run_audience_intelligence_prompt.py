from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent


def build_business_summary(result: Dict[str, Any]) -> str:
    run_dir = Path(result["run_dir"])

    export_outputs = result.get("safe_export", {}).get("outputs", {})
    cohorts_path = Path(export_outputs.get("safe_export_cohorts", ""))
    lookalikes_path = Path(export_outputs.get("safe_export_lookalikes", ""))

    lines = []
    lines.append("# Audience Intelligence Result")
    lines.append("")
    lines.append(f"Prompt: {result.get('prompt', 'not_available')}")
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
    filter_mode = filter_report.get("filter_mode", "unknown")

    lines.append("## Prompt understanding")
    lines.append("")
    lines.append(f"Filter mode used: {filter_mode}")
    lines.append(f"Locations detected: {', '.join(filter_report.get('locations_detected', []) or ['none'])}")
    lines.append(f"POI terms detected: {', '.join(filter_report.get('poi_terms_detected', []) or ['none'])}")
    lines.append(f"Dayparts detected: {', '.join(filter_report.get('dayparts_detected', []) or ['none'])}")
    lines.append("")

    if filter_mode != "location+poi+daypart":
        lines.append("Note: Exact location + POI + daypart match was not strong enough, so the system used the safest available fallback match.")
        lines.append("")

    coverage_warnings = result.get("coverage_warnings", []) or []
    if coverage_warnings:
        lines.append("## Coverage warnings")
        lines.append("")
        for warning in coverage_warnings:
            lines.append(f"- {warning}")
        lines.append("")

    if cohorts_path.exists():
        cohorts = pd.read_csv(cohorts_path)

        lines.append("## Created approval-gated audiences")
        lines.append("")

        for idx, row in cohorts.head(10).iterrows():
            audience_name = row.get("audience_name", "Unnamed audience")
            quality = float(row.get("management_quality_score", 0))
            status = row.get("export_status", "unknown")
            location = row.get("location_name", "unknown")
            poi = row.get("primary_poi_type", "unknown")
            daypart = row.get("created_day_part", "unknown")
            lookback = row.get("lookback_bucket", "unknown")

            lines.append(f"{idx + 1}. {audience_name}")
            lines.append(f"   - Location: {location}")
            lines.append(f"   - POI type: {poi}")
            lines.append(f"   - Daypart: {daypart}")
            lines.append(f"   - Lookback: {lookback}")
            lines.append(f"   - Management quality: {quality:.3f}")
            lines.append(f"   - Export status: {status}")
            lines.append("")

    if lookalikes_path.exists():
        lookalikes = pd.read_csv(lookalikes_path)
        lines.append("## Lookalike package")
        lines.append("")
        lines.append(f"Safe lookalike pairs created: {len(lookalikes)}")
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

    lines.append("## Final decision")
    lines.append("")

    approval_status = result.get("safe_export", {}).get("approval_status")
    downstream_enabled = result.get("safe_export", {}).get("downstream_export_enabled")

    if approval_status == "pending_approval" and downstream_enabled is False:
        lines.append("Export package is ready for review, but downstream delivery is blocked until approval.")
    else:
        lines.append("Export package is ready based on the current approval configuration.")

    lines.append("")
    lines.append(f"Run folder: {run_dir}")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full Audience Intelligence pipeline from a prompt.")

    parser.add_argument("--prompt", required=True, help="Business/audience prompt.")
    parser.add_argument("--source", choices=["postgres", "safe_artifact"], default="postgres")
    parser.add_argument("--safe-cohort-path", default=None)
    parser.add_argument("--output-root", default="data/prompt_runs")
    parser.add_argument("--postgres-limit", type=int, default=10000)
    parser.add_argument("--k-min", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--synthetic-rows", type=int, default=1000)
    parser.add_argument("--max-export-cohorts", type=int, default=25)
    parser.add_argument("--min-export-quality", type=float, default=0.25)
    parser.add_argument("--no-approval-required", action="store_true")

    args = parser.parse_args()

    agent = AudienceIntelligenceOrchestratorAgent()

    result = agent.run(
        prompt=args.prompt,
        output_root=args.output_root,
        source=args.source,
        safe_cohort_path=args.safe_cohort_path,
        postgres_limit=args.postgres_limit,
        k_min=args.k_min,
        epsilon=args.epsilon,
        synthetic_rows=args.synthetic_rows,
        max_export_cohorts=args.max_export_cohorts,
        min_export_quality=args.min_export_quality,
        approval_required=not args.no_approval_required,
    )

    business_summary = build_business_summary(result)
    business_summary_path = Path(result["run_dir"]) / "business_prompt_summary.md"
    business_summary_path.write_text(business_summary)

    print()
    print("=== BUSINESS SUMMARY ===")
    print(business_summary)
    print()
    print("BUSINESS SUMMARY PATH:", business_summary_path)

    print()
    print("=== FINAL RESULT ===")
    print(json.dumps(
        {
            "status": result["status"],
            "run_id": result["run_id"],
            "prompt_selected_cohorts": result["prompt_selected_cohorts"],
            "approval_status": result["safe_export"]["approval_status"],
            "exported_cohorts": result["safe_export"]["exported_cohorts"],
            "exported_lookalike_pairs": result["safe_export"]["exported_lookalike_pairs"],
            "run_dir": result["run_dir"],
            "final_summary_path": result["final_summary_path"],
            "business_summary_path": str(business_summary_path),
        },
        indent=2,
    ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
