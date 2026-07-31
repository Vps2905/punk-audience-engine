from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent


def persist_business_summary(
    result: dict,
    business_summary: str,
) -> str:
    """
    Persist the CLI business summary only when the run uses local storage.

    Production runs may return durable postgres:// artifact references.
    Those references must never be interpreted as local filesystem paths.
    """
    run_dir = str(result.get("run_dir") or "").strip()

    if run_dir.startswith(("postgres://", "postgresql://")):
        existing_reference = str(
            result.get("business_summary_path") or ""
        ).strip()

        if existing_reference:
            return existing_reference

        final_summary_reference = str(
            result.get("final_summary_path") or ""
        ).strip()

        if final_summary_reference:
            separator = (
                "&"
                if "?" in final_summary_reference
                else "?"
            )
            return (
                f"{final_summary_reference}"
                f"{separator}section=business_summary"
            )

        run_id = str(result.get("run_id") or "").strip()

        if not run_id:
            raise ValueError(
                "Postgres-backed run is missing run_id"
            )

        return (
            "postgres://audience_run_history.final_summary"
            f"?run_id={run_id}"
            "&section=business_summary"
        )

    if not run_dir:
        raise ValueError("Audience run is missing run_dir")

    business_summary_path = (
        Path(run_dir) / "business_prompt_summary.md"
    )
    business_summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    business_summary_path.write_text(
        business_summary,
        encoding="utf-8",
    )

    return str(business_summary_path)


def build_business_summary(result: Dict[str, Any]) -> str:
    run_dir = str(
        result.get("run_dir") or ""
    ).strip()

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
    lines.append(
        "Prepared safe export candidates: "
        f"{result.get('safe_export', {}).get('exported_cohorts')}"
    )
    lines.append(
        "Prepared lookalike pairs: "
        f"{result.get('safe_export', {}).get('exported_lookalike_pairs')}"
    )
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

    if cohorts_path.is_file():
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

    if lookalikes_path.is_file():
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

    safe_export = result.get("safe_export") or {}
    approval_status = safe_export.get("approval_status")
    downstream_enabled = safe_export.get(
        "downstream_export_enabled"
    )
    safe_export_status = safe_export.get("status")
    prepared_cohorts = (
        safe_export.get("exported_cohorts") or 0
    )

    if downstream_enabled is True:
        lines.append(
            "A privacy-safe export package is ready and "
            "downstream delivery is enabled under the "
            "current approval configuration."
        )
    elif approval_status == "pending_approval":
        lines.append(
            "Privacy-safe export candidates were prepared, "
            "but downstream delivery is blocked until "
            "explicit approval."
        )
    elif (
        safe_export_status == "completed"
        and prepared_cohorts
    ):
        lines.append(
            "Privacy-safe export candidates were prepared, "
            "but downstream delivery is blocked. "
            f"Approval status: {approval_status or 'unknown'}."
        )
    else:
        lines.append(
            "No downstream export was performed. "
            f"Approval status: {approval_status or 'unknown'}."
        )

    lines.append("")

    if run_dir.startswith(
        ("postgres://", "postgresql://")
    ):
        lines.append(f"Run reference: {run_dir}")
    else:
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
    business_summary_path = persist_business_summary(
        result,
        business_summary,
    )

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
