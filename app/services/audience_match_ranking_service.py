from __future__ import annotations

import math
from typing import Any

import pandas as pd


class DynamicAudienceRankingService:
    """
    Precise audience matching/ranking engine.

    Scores cohorts using:
    - location match
    - category match
    - daypart match
    - cohort quality
    - volume strength
    - freshness
    - privacy status
    """

    CATEGORY_NEIGHBORS = {
        "restaurant": {
            "restaurant",
            "shawarma_restaurant",
            "middle_eastern_restaurant",
            "food",
            "fast_food",
            "takeaway",
            "cafe",
        },
        "cafe": {"cafe", "coffee", "restaurant", "bakery"},
        "gym": {"gym", "fitness", "health_club", "yoga", "pilates"},
        "office": {
            "office",
            "corporate_office",
            "coworking_space",
            "consultant",
            "point_of_interest",
        },
        "tattoo": {"tattoo", "body_art_service", "beauty"},
        "retail": {"retail", "store", "shopping_mall", "point_of_interest"},
        "healthcare": {"clinic", "hospital", "healthcare", "medical", "pharmacy"},
        "education": {"school", "college", "university", "campus"},
        "nightlife": {"bar", "pub", "club", "lounge"},
        "beauty": {"salon", "spa", "beauty"},
        "auto": {"auto", "car", "automotive", "garage"},
    }

    def rank(
        self,
        safe_cohorts: pd.DataFrame,
        prompt_intent: dict[str, Any],
        freshness_report: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if safe_cohorts is None or safe_cohorts.empty:
            return pd.DataFrame()

        ranked = safe_cohorts.copy().reset_index(drop=True)
        scored_rows = []

        for _, row in ranked.iterrows():
            scored_rows.append(self._score_row(row, prompt_intent, freshness_report or {}))

        scored_df = pd.DataFrame(scored_rows)
        ranked = pd.concat([ranked, scored_df], axis=1)
        ranked = ranked.sort_values("final_match_score", ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1

        return ranked

    def _score_row(
        self,
        row: pd.Series,
        intent: dict[str, Any],
        freshness: dict[str, Any],
    ) -> dict[str, Any]:
        location_score = self._location_score(row, intent)
        category_score = self._category_score(row, intent)
        daypart_score = self._daypart_score(row, intent)
        quality_score = self._quality_score(row)
        volume_score = self._volume_score(row)
        freshness_score = self._freshness_score(freshness)
        privacy_score = 1.0 if str(row.get("privacy_status", "passed")).lower() == "passed" else 0.0

        exact = location_score >= 0.95 and category_score >= 0.9 and daypart_score >= 0.9

        if exact:
            match_type = "exact_match"
            fallback_penalty = 0.0
        elif location_score >= 0.7 and category_score >= 0.9 and daypart_score < 0.9:
            match_type = "broader_daypart"
            fallback_penalty = 0.12
        elif location_score >= 0.7 and category_score >= 0.5:
            match_type = "adjacent_category"
            fallback_penalty = 0.12
        elif location_score < 0.7 and category_score >= 0.5:
            match_type = "nearby_or_other_location"
            fallback_penalty = 0.18
        else:
            match_type = "data_gap_candidate"
            fallback_penalty = 0.25

        final_score = (
            0.25 * location_score
            + 0.25 * category_score
            + 0.15 * daypart_score
            + 0.15 * quality_score
            + 0.10 * volume_score
            + 0.05 * freshness_score
            + 0.05 * privacy_score
            - fallback_penalty
        )
        final_score = max(0.0, min(1.0, final_score))
        confidence = max(0.0, min(1.0, (location_score + category_score + daypart_score) / 3))

        return {
            "location_match_score": round(location_score, 3),
            "category_match_score": round(category_score, 3),
            "daypart_match_score": round(daypart_score, 3),
            "cohort_quality_component": round(quality_score, 3),
            "volume_score": round(volume_score, 3),
            "freshness_score": round(freshness_score, 3),
            "privacy_score": round(privacy_score, 3),
            "fallback_penalty": round(fallback_penalty, 3),
            "final_match_score": round(final_score, 3),
            "confidence_score": round(confidence, 3),
            "match_type": match_type,
            "match_reason": (
                f"{match_type}: {row.get('location_name')} / "
                f"{row.get('primary_poi_type')} / {row.get('created_day_part')}"
            ),
        }

    def _location_score(self, row: pd.Series, intent: dict[str, Any]) -> float:
        requested = [str(x).lower() for x in intent.get("locations", [])]
        loc = str(row.get("location_name", "")).lower()

        if not requested:
            return 0.7
        if loc in requested:
            return 1.0
        if any(req in loc or loc in req for req in requested):
            return 0.8
        return 0.0

    def _category_score(self, row: pd.Series, intent: dict[str, Any]) -> float:
        requested = [str(x).lower() for x in intent.get("canonical_categories", [])]
        poi = str(row.get("primary_poi_type", "")).lower().replace(" ", "_")

        if not requested:
            return 0.7

        for req in requested:
            neighbors = self.CATEGORY_NEIGHBORS.get(req, {req})

            if poi == req:
                return 1.0
            if poi in neighbors:
                return 0.9
            if any(n in poi or poi in n for n in neighbors):
                return 0.75

        return 0.0

    def _daypart_score(self, row: pd.Series, intent: dict[str, Any]) -> float:
        requested = [str(x).lower() for x in intent.get("dayparts", [])]
        daypart = str(row.get("created_day_part", "")).lower()

        if not requested:
            return 0.7
        if daypart in requested:
            return 1.0
        if "evening" in requested and daypart in {"afternoon", "night"}:
            return 0.45
        if "morning" in requested and daypart == "afternoon":
            return 0.35

        return 0.0

    def _quality_score(self, row: pd.Series) -> float:
        for col in ["management_quality_score", "quality_score"]:
            if col in row and pd.notna(row[col]):
                try:
                    return max(0.0, min(1.0, float(row[col])))
                except Exception:
                    continue
        return 0.3

    def _volume_score(self, row: pd.Series) -> float:
        for col in ["total_maid_volume", "noisy_maid_volume"]:
            if col in row and pd.notna(row[col]):
                try:
                    volume = max(float(row[col]), 1.0)
                    return max(0.0, min(1.0, math.log10(volume) / 6.0))
                except Exception:
                    continue
        return 0.3

    def _freshness_score(self, freshness: dict[str, Any]) -> float:
        status = str(freshness.get("freshness_status", "unknown")).lower()

        if status == "fresh":
            return 1.0
        if status == "unchanged":
            return 0.7
        if status == "stale":
            return 0.2

        return 0.5
