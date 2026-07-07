from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class AutonomousMutationAgent:
    """
    Creates safe audience variants when exact coverage is missing or weak.

    Important:
    - Does not invent fake audiences.
    - Uses existing privacy-safe cohorts only.
    - Data gaps are clearly marked.
    - Everything remains approval-gated.
    """

    def generate_suggestions(
        self,
        prompt_intent: dict[str, Any],
        ranked_cohorts: pd.DataFrame,
        coverage_warnings: list[str] | None = None,
        run_dir: str | Path | None = None,
        max_suggestions: int = 10,
    ) -> dict[str, Any]:
        coverage_warnings = coverage_warnings or []
        data_gap_suggestions = self._data_gap_suggestions(prompt_intent, ranked_cohorts)
        fallback_suggestions: list[dict[str, Any]] = []

        remaining_slots = max(max_suggestions - len(data_gap_suggestions), 0)

        if ranked_cohorts is not None and not ranked_cohorts.empty and remaining_slots > 0:
            fallback_rows = ranked_cohorts[
                ranked_cohorts["match_type"].astype(str).ne("exact_match")
            ].head(remaining_slots)

            for _, row in fallback_rows.iterrows():
                fallback_suggestions.append(self._row_to_suggestion(row))

        suggestions = self._dedupe(data_gap_suggestions + fallback_suggestions)[:max_suggestions]

        result = {
            "status": "completed",
            "suggestion_count": len(suggestions),
            "coverage_warnings": coverage_warnings,
            "mutation_suggestions": suggestions,
            "approval_required": True,
            "downstream_export_enabled": False,
            "safety_note": (
                "Mutation suggestions are generated only from privacy-safe cohorts "
                "or explicit data-gap recommendations. No fake audiences are created."
            ),
        }

        if run_dir:
            output_dir = Path(run_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "mutation_suggestions.json").write_text(
                json.dumps(result, indent=2),
                encoding="utf-8",
            )

        return result

    def _row_to_suggestion(self, row: pd.Series) -> dict[str, Any]:
        match_type = str(row.get("match_type", "safe_fallback_candidate"))

        if match_type == "exact_match":
            mutation_type = "fresh_exact_opportunity"
        elif match_type == "adjacent_category":
            mutation_type = "adjacent_category"
        elif match_type == "broader_daypart":
            mutation_type = "broader_daypart"
        elif match_type == "nearby_or_other_location":
            mutation_type = "nearby_or_other_location"
        else:
            mutation_type = "safe_fallback_candidate"

        return {
            "mutation_type": mutation_type,
            "audience_name": (
                f"Review Candidate - {row.get('primary_poi_type')} - "
                f"{row.get('created_day_part')} - {row.get('location_name')}"
            ),
            "location_name": str(row.get("location_name", "")),
            "primary_poi_type": str(row.get("primary_poi_type", "")),
            "created_day_part": str(row.get("created_day_part", "")),
            "final_match_score": self._safe_float(row.get("final_match_score", 0)),
            "confidence_score": self._safe_float(row.get("confidence_score", 0)),
            "reason": str(row.get("match_reason", "Safe fallback candidate.")),
            "source_cohort_reference": {
                "cohort_index": self._safe_int(row.get("cohort_index", row.name if hasattr(row, "name") else -1)),
                "vector_index": self._safe_int(row.get("vector_index", -1)),
            },
            "approval_required": True,
            "downstream_export_enabled": False,
        }

    def _data_gap_suggestions(
        self,
        prompt_intent: dict[str, Any],
        ranked_cohorts: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        suggestions = []

        requested_locations = prompt_intent.get("locations", [])
        requested_categories = prompt_intent.get("canonical_categories", [])
        requested_dayparts = prompt_intent.get("dayparts", [])

        if ranked_cohorts is None or ranked_cohorts.empty:
            for loc in requested_locations or ["requested_location"]:
                for cat in requested_categories or ["requested_category"]:
                    suggestions.append(self._data_gap(loc, cat, requested_dayparts))
            return suggestions

        for loc in requested_locations:
            loc_rows = ranked_cohorts[
                ranked_cohorts["location_name"].astype(str).str.lower().str.contains(str(loc).lower(), na=False)
            ] if "location_name" in ranked_cohorts.columns else pd.DataFrame()

            if loc_rows.empty:
                for cat in requested_categories or ["requested_category"]:
                    suggestions.append(self._data_gap(loc, cat, requested_dayparts))
                continue

            for cat in requested_categories:
                cat_rows = loc_rows[
                    loc_rows["primary_poi_type"].astype(str).str.lower().str.contains(str(cat).lower(), na=False)
                ] if "primary_poi_type" in loc_rows.columns else pd.DataFrame()

                if cat_rows.empty:
                    suggestions.append(self._data_gap(loc, cat, requested_dayparts))

        return suggestions

    def _data_gap(self, location: str, category: str, dayparts: list[str]) -> dict[str, Any]:
        return {
            "mutation_type": "data_gap",
            "audience_name": f"Data Gap - {location} - {category}",
            "location_name": location,
            "primary_poi_type": category,
            "created_day_part": ",".join(dayparts) if dayparts else "unknown",
            "final_match_score": 0.0,
            "confidence_score": 0.0,
            "reason": (
                "No exact privacy-safe cohort exists for this requested location/category/daypart. "
                "Use fresh Echo/Postgres data or review adjacent safe cohorts."
            ),
            "source_cohort_reference": None,
            "approval_required": True,
            "downstream_export_enabled": False,
        }

    def _dedupe(self, suggestions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        output = []

        for item in suggestions:
            key = (
                item.get("mutation_type"),
                item.get("location_name"),
                item.get("primary_poi_type"),
                item.get("created_day_part"),
                item.get("audience_name"),
            )
            if key not in seen:
                seen.add(key)
                output.append(item)

        return output

    def _safe_float(self, value: Any) -> float:
        try:
            return round(float(value), 3)
        except Exception:
            return 0.0

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return -1
