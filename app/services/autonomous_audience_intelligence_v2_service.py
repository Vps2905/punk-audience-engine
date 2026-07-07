from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.agents.autonomous_mutation_agent import AutonomousMutationAgent
from app.agents.data_freshness_agent import DataFreshnessAgent
from app.agents.semantic_prompt_intelligence_agent import SemanticPromptIntelligenceAgent
from app.agents.hybrid_semantic_intent_agent import HybridSemanticIntentAgent
from app.services.all_safe_cohort_embedding_service import AllSafeCohortEmbeddingService
from app.services.audience_match_ranking_service import DynamicAudienceRankingService


class AutonomousAudienceIntelligenceV2Service:
    """
    Reusable v2 autonomous audience intelligence pipeline.

    Flow:
    - freshness check
    - semantic prompt understanding
    - all-safe-cohort 384-dim embeddings
    - dynamic ranking
    - coverage warnings
    - autonomous mutation suggestions
    - approval-gated v2 summary
    """

    def run(
        self,
        prompt: str,
        safe_cohorts: pd.DataFrame,
        output_dir: str | Path,
        previous_freshness_report_path: str | Path | None = None,
        freshness_source_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if safe_cohorts is None or safe_cohorts.empty:
            raise ValueError("safe_cohorts is empty. Cannot run v2 intelligence.")

        freshness_df = freshness_source_df if freshness_source_df is not None else safe_cohorts

        freshness = DataFreshnessAgent().analyze_dataframe(
            df=freshness_df,
            run_dir=output_path,
            previous_report_path=previous_freshness_report_path,
        )

        prompt_intent = HybridSemanticIntentAgent().resolve(prompt=prompt, safe_cohorts=safe_cohorts, output_dir=output_dir)

        embedding_manifest = AllSafeCohortEmbeddingService().build_index(
            safe_cohorts=safe_cohorts,
            output_dir=output_path / "embeddings",
        )

        ranked = DynamicAudienceRankingService().rank(
            safe_cohorts=safe_cohorts,
            prompt_intent=prompt_intent,
            freshness_report=freshness,
        )

        ranked_path = output_path / "ranked_audience_matches.csv"
        ranked.to_csv(ranked_path, index=False)

        coverage_warnings = self._build_coverage_warnings(prompt_intent, ranked)

        mutation = AutonomousMutationAgent().generate_suggestions(
            prompt_intent=prompt_intent,
            ranked_cohorts=ranked,
            coverage_warnings=coverage_warnings,
            run_dir=output_path,
        )

        summary = {
            "status": "completed",
            "pipeline_version": "v2_autonomous_preview",
            "prompt": prompt,
            "data_freshness": freshness,
            "prompt_intent": prompt_intent,
            "embedding_manifest": embedding_manifest,
            "ranked_match_count": int(len(ranked)),
            "ranked_matches_path": str(ranked_path),
            "top_ranked_matches": ranked.head(10).to_dict(orient="records"),
            "coverage_warnings": coverage_warnings,
            "mutation": mutation,
            "approval_required": True,
            "downstream_export_enabled": False,
            "privacy_note": (
                "V2 uses privacy-safe cohort metadata only. "
                "No raw MAIDs, hashed identifiers, raw lat/lng, email, phone, "
                "or individual rows are exported."
            ),
        }

        summary_path = output_path / "v2_preview_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )

        return summary

    def _build_coverage_warnings(
        self,
        prompt_intent: dict[str, Any],
        ranked: pd.DataFrame,
    ) -> list[str]:
        warnings = []

        if ranked is None or ranked.empty:
            return ["No privacy-safe cohorts were available for ranking."]

        requested_locations = prompt_intent.get("locations", [])
        requested_categories = prompt_intent.get("canonical_categories", [])
        requested_dayparts = prompt_intent.get("dayparts", [])

        for loc in requested_locations:
            loc_rows = ranked[
                ranked["location_name"]
                .astype(str)
                .str.lower()
                .str.contains(str(loc).lower(), na=False)
            ] if "location_name" in ranked.columns else pd.DataFrame()

            if loc_rows.empty:
                warnings.append(
                    f"{loc} was requested, but no privacy-safe cohort exists for that location."
                )
                continue

            exact = loc_rows.copy()

            if requested_categories and "category_match_score" in exact.columns:
                exact = exact[exact["category_match_score"] >= 0.9]

            if requested_dayparts and "daypart_match_score" in exact.columns:
                exact = exact[exact["daypart_match_score"] >= 0.9]

            if exact.empty:
                warnings.append(
                    f"{loc} was requested, but no exact safe cohort matched the requested category/daypart."
                )

        return warnings
