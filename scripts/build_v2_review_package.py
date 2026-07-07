from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


SAFE_RANKED_COLUMNS = [
    "rank",
    "location_name",
    "primary_poi_type",
    "created_day_part",
    "lookback_bucket",
    "privacy_status",
    "quality_score",
    "total_maid_volume",
    "noisy_maid_volume",
    "location_match_score",
    "category_match_score",
    "daypart_match_score",
    "cohort_quality_component",
    "volume_score",
    "freshness_score",
    "privacy_score",
    "fallback_penalty",
    "final_match_score",
    "confidence_score",
    "match_type",
    "match_reason",
]

BANNED_COLUMN_KEYWORDS = [
    "maid",
    "hash",
    "identifier",
    "session_id",
    "email",
    "phone",
    "lat",
    "lng",
    "longitude",
    "latitude",
    "raw",
    "center",
    "pois",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="data/review_packages")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    final_summary = read_json(run_dir / "final_prompt_summary.json")
    v2 = final_summary.get("v2_autonomous", {})
    review = final_summary.get("v2_swarm_review", {})

    if not v2:
        raise RuntimeError("No v2_autonomous section found in final_prompt_summary.json")

    package_name = f"v2_review_package_{final_summary.get('run_id', run_dir.name)}"
    package_dir = output_root / package_name

    if package_dir.exists():
        shutil.rmtree(package_dir)

    package_dir.mkdir(parents=True)

    write_report(package_dir / "V2_REVIEW_REPORT.md", final_summary, v2, review)

    copy_if_exists(run_dir / "final_prompt_summary.json", package_dir / "final_prompt_summary.json")
    copy_if_exists(run_dir / "business_prompt_summary.md", package_dir / "business_prompt_summary.md")

    v2_dir = run_dir / "06_v2_autonomous_preview"

    copy_if_exists(v2_dir / "data_freshness_report.json", package_dir / "data_freshness_report.json")
    copy_if_exists(v2_dir / "prompt_intent.json", package_dir / "prompt_intent.json")
    copy_if_exists(v2_dir / "v2_preview_summary.json", package_dir / "v2_preview_summary.json")
    copy_if_exists(v2_dir / "mutation_suggestions.json", package_dir / "mutation_suggestions.json")
    copy_if_exists(v2_dir / "v2_swarm_review.json", package_dir / "v2_swarm_review.json")
    copy_if_exists(
        v2_dir / "embeddings" / "all_safe_cohort_embedding_manifest.json",
        package_dir / "all_safe_cohort_embedding_manifest.json",
    )

    create_safe_ranked_preview(
        source_path=v2_dir / "ranked_audience_matches.csv",
        output_path=package_dir / "safe_ranked_matches_top50.csv",
    )

    create_package_manifest(package_dir / "PACKAGE_MANIFEST.json", final_summary, v2, review)

    zip_path = output_root / f"{package_name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(package_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(package_dir))

    print("V2 review package created ✅")
    print(f"Package folder: {package_dir}")
    print(f"Package zip: {zip_path}")


def write_report(path: Path, final_summary: dict[str, Any], v2: dict[str, Any], review: dict[str, Any]) -> None:
    freshness = v2.get("data_freshness", {})
    prompt_intent = v2.get("prompt_intent", {})
    embed = v2.get("embedding_manifest", {})
    mutation = v2.get("mutation", {})

    lines = [
        "# Audience Intelligence v2 Review Report",
        "",
        "## Executive summary",
        "",
        f"- Run ID: `{final_summary.get('run_id')}`",
        f"- Source mode: `{final_summary.get('source_mode')}`",
        f"- Source rows: `{final_summary.get('source_rows')}`",
        f"- Privacy-safe cohorts: `{final_summary.get('privacy_cohorts')}`",
        f"- v2 status: `{v2.get('status')}`",
        f"- Swarm review status: `{review.get('overall_review_status')}`",
        "",
        "## Real data freshness",
        "",
        f"- Freshness status: `{freshness.get('freshness_status')}`",
        f"- Latest source timestamp: `{freshness.get('latest_source_timestamp')}`",
        f"- Source rows checked: `{freshness.get('source_rows_checked')}`",
        "",
        "## Prompt understanding",
        "",
        f"- Business intent: `{prompt_intent.get('business_intent')}`",
        f"- Prompt confidence: `{prompt_intent.get('confidence_score')}`",
        f"- Locations: `{prompt_intent.get('locations')}`",
        f"- Categories: `{prompt_intent.get('canonical_categories')}`",
        f"- Dayparts: `{prompt_intent.get('dayparts')}`",
        "",
        "## Embedding and ranking proof",
        "",
        f"- Embedding scope: `{embed.get('embedding_scope')}`",
        f"- Embedding backend: `{embed.get('embedding_backend')}`",
        f"- Vector count: `{embed.get('vector_count')}`",
        f"- Vector dimension: `{embed.get('vector_dimension')}`",
        f"- Ranked matches: `{v2.get('ranked_match_count')}`",
        "",
        "## Mutation and coverage review",
        "",
        f"- Coverage warning count: `{review.get('coverage_warning_count')}`",
        f"- Data gap count: `{review.get('data_gap_count')}`",
        f"- Mutation suggestions: `{mutation.get('suggestion_count')}`",
        "",
        "### Coverage warnings",
        "",
    ]

    for warning in v2.get("coverage_warnings", []) or []:
        lines.append(f"- {warning}")

    lines.extend([
        "",
        "## Safety",
        "",
        f"- Approval required: `{v2.get('approval_required')}`",
        f"- Downstream export enabled: `{v2.get('downstream_export_enabled')}`",
        "",
        "No raw MAIDs, hashed identifiers, raw lat/lng, email, phone, or individual user rows are included in this package.",
        "",
        "## Swarm recommendations",
        "",
    ])

    for recommendation in review.get("recommendations", []) or []:
        lines.append(f"- {recommendation}")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def create_safe_ranked_preview(source_path: Path, output_path: Path) -> None:
    if not source_path.exists():
        return

    df = pd.read_csv(source_path)

    allowed = [
        column for column in SAFE_RANKED_COLUMNS
        if column in df.columns and not is_banned_column(column)
    ]

    safe = df[allowed].head(50).copy()
    safe.to_csv(output_path, index=False)


def create_package_manifest(
    path: Path,
    final_summary: dict[str, Any],
    v2: dict[str, Any],
    review: dict[str, Any],
) -> None:
    manifest = {
        "status": "completed",
        "package_type": "audience_intelligence_v2_review",
        "run_id": final_summary.get("run_id"),
        "source_mode": final_summary.get("source_mode"),
        "source_rows": final_summary.get("source_rows"),
        "privacy_cohorts": final_summary.get("privacy_cohorts"),
        "v2_status": v2.get("status"),
        "swarm_review_status": review.get("overall_review_status"),
        "vector_count": (v2.get("embedding_manifest") or {}).get("vector_count"),
        "vector_dimension": (v2.get("embedding_manifest") or {}).get("vector_dimension"),
        "freshness_status": (v2.get("data_freshness") or {}).get("freshness_status"),
        "latest_source_timestamp": (v2.get("data_freshness") or {}).get("latest_source_timestamp"),
        "approval_required": v2.get("approval_required"),
        "downstream_export_enabled": v2.get("downstream_export_enabled"),
        "privacy_safety": {
            "raw_maids_included": False,
            "hashed_identifiers_included": False,
            "raw_lat_lng_included": False,
            "email_phone_included": False,
            "individual_user_rows_included": False,
        },
    }

    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def is_banned_column(column: str) -> bool:
    lower = column.lower()
    return any(keyword in lower for keyword in BANNED_COLUMN_KEYWORDS)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(errors="ignore"))


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
