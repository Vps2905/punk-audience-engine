from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.services.autonomous_audience_intelligence_v2_service import (
    AutonomousAudienceIntelligenceV2Service,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--clean-feature-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--previous-freshness-report-path", required=False, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_cohorts = pd.read_csv(args.clean_feature_table)

    summary = AutonomousAudienceIntelligenceV2Service().run(
        prompt=args.prompt,
        safe_cohorts=safe_cohorts,
        output_dir=output_dir,
        previous_freshness_report_path=args.previous_freshness_report_path,
    )

    print("\nAudience Intelligence v2 Preview Completed ✅")
    print(f"Output dir: {output_dir}")
    print(f"Freshness: {summary['data_freshness'].get('freshness_status')}")
    print(f"Prompt confidence: {summary['prompt_intent'].get('confidence_score')}")
    print(f"Vector count: {summary['embedding_manifest'].get('vector_count')}")
    print(f"Vector dimension: {summary['embedding_manifest'].get('vector_dimension')}")
    print(f"Ranked matches: {summary.get('ranked_match_count')}")
    print(f"Mutation suggestions: {summary['mutation'].get('suggestion_count')}")
    print(f"Coverage warnings: {len(summary.get('coverage_warnings', []))}")
    print(f"Approval required: {summary.get('approval_required')}")
    print(f"Downstream export enabled: {summary.get('downstream_export_enabled')}")


if __name__ == "__main__":
    main()
