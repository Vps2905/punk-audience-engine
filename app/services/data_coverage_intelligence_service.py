from __future__ import annotations

import re
from typing import Any, Dict, List

from app.services.hybrid_audience_retrieval_service import CATEGORY_SYNONYMS, _norm


UNKNOWN_VALUES = {
    "unknown",
    "unknown_business_intent",
    "none",
    "null",
    "n/a",
    "na",
}


EXTRA_CATEGORY_GROUPS = {
    "restaurant": {
        "restaurant",
        "restaurants",
        "shawarma_restaurant",
        "middle_eastern_restaurant",
        "lebanese_restaurant",
        "bar_and_grill",
        "food",
        "food_spots",
        "dining",
    },
    "fast_food_restaurant": {
        "fast_food_restaurant",
        "fast_food",
        "fast-food",
        "burger",
        "burger_place",
        "burger_places",
        "chicken_restaurant",
        "food_spots",
        "restaurant",
    },
    "healthcare": {
        "healthcare",
        "health",
        "hospital",
        "clinic",
        "clinics",
        "pharmacy",
        "pharmacies",
        "medical_center",
        "diagnostic_center",
        "diagnostic_centers",
        "dentist",
    },
    "retail": {
        "retail",
        "store",
        "stores",
        "shopping",
        "shopping_mall",
        "clothing_store",
        "womens_clothing_store",
        "shoe_store",
        "book_store",
        "toy_store",
        "boutique",
        "boutiques",
        "fashion",
        "fashion_store",
        "fashion_stores",
        "home_decor",
        "home_decor_shop",
        "home_decor_shops",
    },
    "casino": {
        "casino",
        "casinos",
        "gaming",
        "gaming_venue",
        "gaming_venues",
        "tourist_entertainment",
        "entertainment",
        "event_venue",
    },
}


class DataCoverageIntelligenceService:
    def build_report(
        self,
        intent: Dict[str, Any],
        safe_cohorts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        requested_locations = self._clean_list(intent.get("locations"))
        requested_categories = self._clean_list(
            intent.get("requested_categories")
            or intent.get("canonical_categories")
            or intent.get("poi_terms")
            or [intent.get("business_intent")]
        )
        requested_dayparts = self._clean_dayparts(intent.get("dayparts"))

        rows = [row for row in safe_cohorts if isinstance(row, dict)]

        available_locations = sorted(
            {
                _norm(row.get("location_name"))
                for row in rows
                if _norm(row.get("location_name"))
            }
        )
        available_categories = sorted(
            {
                _norm(row.get("primary_poi_type"))
                for row in rows
                if _norm(row.get("primary_poi_type"))
            }
        )
        available_dayparts = sorted(
            {
                _norm(row.get("created_day_part"))
                for row in rows
                if _norm(row.get("created_day_part"))
            }
        )

        available_categories_by_location: Dict[str, List[str]] = {}
        available_dayparts_by_location: Dict[str, List[str]] = {}

        for row in rows:
            location = _norm(row.get("location_name"))
            category = _norm(row.get("primary_poi_type"))
            daypart = _norm(row.get("created_day_part"))
            if not location:
                continue
            if category:
                available_categories_by_location.setdefault(location, [])
                if category not in available_categories_by_location[location]:
                    available_categories_by_location[location].append(category)
            if daypart:
                available_dayparts_by_location.setdefault(location, [])
                if daypart not in available_dayparts_by_location[location]:
                    available_dayparts_by_location[location].append(daypart)

        for value in available_categories_by_location.values():
            value.sort()
        for value in available_dayparts_by_location.values():
            value.sort()

        per_location = []
        missing_requested_locations = []
        missing_requested_categories_by_location: Dict[str, List[str]] = {}
        missing_requested_dayparts_by_location: Dict[str, List[str]] = {}

        if requested_locations:
            for requested_location in requested_locations:
                location_rows = [
                    row for row in rows
                    if self._location_match(requested_location, row.get("location_name"))
                ]

                category_rows = [
                    row for row in location_rows
                    if self._category_allowed(requested_categories, row.get("primary_poi_type"))
                ]

                daypart_rows = [
                    row for row in category_rows
                    if self._daypart_allowed(requested_dayparts, row.get("created_day_part"))
                ]

                location_available = bool(location_rows)
                category_available = bool(category_rows)
                daypart_available = bool(daypart_rows)
                exact_combo_available = location_available and category_available and daypart_available

                if not location_available:
                    missing_requested_locations.append(requested_location)
                elif requested_categories and not category_available:
                    missing_requested_categories_by_location[requested_location] = requested_categories
                elif requested_dayparts and not daypart_available:
                    missing_requested_dayparts_by_location[requested_location] = requested_dayparts

                per_location.append(
                    {
                        "requested_location": requested_location,
                        "location_available": location_available,
                        "category_available": category_available,
                        "daypart_available": daypart_available,
                        "exact_combo_available": exact_combo_available,
                        "matching_safe_rows": int(len(daypart_rows)),
                    }
                )
        else:
            category_rows = [
                row for row in rows
                if self._category_allowed(requested_categories, row.get("primary_poi_type"))
            ]
            daypart_rows = [
                row for row in category_rows
                if self._daypart_allowed(requested_dayparts, row.get("created_day_part"))
            ]
            per_location.append(
                {
                    "requested_location": None,
                    "location_available": True,
                    "category_available": bool(category_rows),
                    "daypart_available": bool(daypart_rows),
                    "exact_combo_available": bool(daypart_rows),
                    "matching_safe_rows": int(len(daypart_rows)),
                }
            )

        exact_combo_available = bool(per_location) and all(
            item["exact_combo_available"] for item in per_location
        )
        any_exact_combo_available = any(
            item["exact_combo_available"] for item in per_location
        )

        location_available_any = any(item["location_available"] for item in per_location)
        category_available_any = any(item["category_available"] for item in per_location)
        daypart_available_any = any(item["daypart_available"] for item in per_location)

        if exact_combo_available:
            coverage_decision = "exact_safe_combo_available"
            recommended_next_action = "Proceed to ranking, guardrail validation, and approval-gated export."
        elif any_exact_combo_available:
            coverage_decision = "partial_safe_data_available"
            recommended_next_action = (
                "Use only available exact safe cohorts and clearly disclose missing requested locations/categories."
            )
        elif requested_locations and not location_available_any:
            coverage_decision = "requested_location_not_available"
            recommended_next_action = (
                f"Run new safe data extraction for {', '.join(requested_locations)}."
            )
        elif requested_categories and not category_available_any:
            coverage_decision = "requested_category_not_available"
            recommended_next_action = (
                "Run new safe data extraction for the requested category in the requested location."
            )
        elif requested_dayparts and not daypart_available_any:
            coverage_decision = "requested_daypart_not_available"
            recommended_next_action = (
                "Run new safe data extraction for the requested time/daypart."
            )
        else:
            coverage_decision = "partial_safe_data_available"
            recommended_next_action = (
                "Use available partial data only if guardrails allow it; otherwise block export and request new extraction."
            )

        return {
            "pipeline": "data_coverage_intelligence",
            "requested_locations": requested_locations,
            "requested_categories": requested_categories,
            "requested_dayparts": requested_dayparts,
            "available_locations": available_locations,
            "available_categories": available_categories,
            "available_dayparts": available_dayparts,
            "available_categories_by_location": available_categories_by_location,
            "available_dayparts_by_location": available_dayparts_by_location,
            "per_requested_location": per_location,
            "exact_combo_available": exact_combo_available,
            "any_exact_combo_available": any_exact_combo_available,
            "partial_location_coverage": bool(any_exact_combo_available and not exact_combo_available),
            "missing_requested_locations": missing_requested_locations,
            "missing_requested_categories_by_location": missing_requested_categories_by_location,
            "missing_requested_dayparts_by_location": missing_requested_dayparts_by_location,
            "coverage_decision": coverage_decision,
            "recommended_next_action": recommended_next_action,
        }

    def _clean_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            values = [value]

        cleaned = []
        for item in values:
            normed = _norm(item)
            if normed and normed not in UNKNOWN_VALUES and normed not in cleaned:
                cleaned.append(normed)
        return cleaned

    def _clean_dayparts(self, value: Any) -> List[str]:
        raw = self._clean_list(value)
        cleaned = []

        for item in raw:
            if item in {"weekend", "weekends", "weekday", "weekdays"}:
                continue
            if "morning" in item:
                cleaned.append("morning")
            elif "afternoon" in item:
                cleaned.append("afternoon")
            elif "evening" in item:
                cleaned.append("evening")
            elif "night" in item:
                cleaned.append("night")
            elif item:
                cleaned.append(item)

        return list(dict.fromkeys(cleaned))

    def _location_match(self, requested: Any, actual: Any) -> bool:
        requested_norm = _norm(requested)
        actual_norm = _norm(actual)
        if not requested_norm or not actual_norm:
            return False
        return (
            requested_norm == actual_norm
            or requested_norm in actual_norm
            or actual_norm in requested_norm
        )

    def _daypart_allowed(self, requested_dayparts: List[str], actual: Any) -> bool:
        if not requested_dayparts:
            return True
        actual_norm = _norm(actual)
        return actual_norm in requested_dayparts

    def _category_allowed(self, requested_categories: List[str], actual: Any) -> bool:
        if not requested_categories:
            return True

        actual_norm = _norm(actual)
        if not actual_norm:
            return False

        return any(self._category_match(requested, actual_norm) for requested in requested_categories)

    def _category_match(self, requested: str, actual: str) -> bool:
        requested = _norm(requested)
        actual = _norm(actual)

        if not requested or not actual:
            return False

        if requested == actual or requested in actual or actual in requested:
            return True

        groups = []

        for group_name, values in CATEGORY_SYNONYMS.items():
            group_values = {_norm(group_name), *{_norm(v) for v in values}}
            groups.append(group_values)

        for group_name, values in EXTRA_CATEGORY_GROUPS.items():
            group_values = {_norm(group_name), *{_norm(v) for v in values}}
            groups.append(group_values)

        for group in groups:
            if requested in group and actual in group:
                return True

        requested_tokens = set(re.split(r"[_\s]+", requested))
        actual_tokens = set(re.split(r"[_\s]+", actual))

        if requested_tokens & actual_tokens and (
            "restaurant" in requested_tokens
            or "restaurant" in actual_tokens
            or "store" in requested_tokens
            or "store" in actual_tokens
        ):
            return True

        return False
