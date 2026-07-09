from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    return {t for t in re.findall(r"[a-z0-9]+", text) if len(t) >= 3}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


CATEGORY_SYNONYMS: Dict[str, set[str]] = {
    "cafe": {"cafe", "coffee", "coffee_shop", "espresso", "caffeine", "tea", "bakery"},
    "restaurant": {
        "restaurant", "food", "food_spot", "dining", "fast_food_restaurant",
        "burger", "chicken_restaurant", "shawarma_restaurant",
        "middle_eastern_restaurant", "pizza_restaurant", "lebanese_restaurant",
        "bar_and_grill"
    },
    "fast_food_restaurant": {
        "fast_food", "fast_food_restaurant", "burger", "burger_place",
        "chicken_restaurant", "food_spot", "quick_service"
    },
    "casino": {"casino", "gaming", "gaming_venue", "entertainment", "tourist_entertainment"},
    "retail": {
        "retail", "store", "shopping_mall", "clothing_store", "womens_clothing_store",
        "boutique", "fashion", "lifestyle", "home_decor", "shoe_store", "book_store"
    },
    "pet_care": {
        "pet", "pet_care", "pet_store", "pet_supply", "grooming", "veterinary_care",
        "animal", "vet"
    },
    "healthcare": {
        "healthcare", "hospital", "clinic", "pharmacy", "diagnostic", "health_checkup",
        "medical", "doctor"
    },
    "fitness": {"fitness", "gym", "yoga_studio", "sports", "recovery", "physio"},
    "office": {"office", "corporate_office", "coworking_space", "business_center"},
}

GENERIC_POI = {
    "store", "point_of_interest", "establishment", "business", "selected_location",
    "locality", "premise"
}


@dataclass
class RetrievalCandidate:
    cohort: Dict[str, Any]
    final_score: float
    score_breakdown: Dict[str, float]
    decision: str
    reason_codes: List[str]


class HybridAudienceRetrievalService:
    """
    Hybrid retrieval over privacy-safe audience cohorts.

    Combines:
    - structured location/category/daypart match
    - keyword/POI match
    - semantic token overlap
    - quality score
    - privacy status

    It expects already privacy-safe cohort metadata.
    """

    def __init__(
        self,
        min_location_score: float = 0.62,
        min_category_score: float = 0.62,
        min_daypart_score: float = 0.80,
        min_final_score: float = 0.60,
    ) -> None:
        self.min_location_score = min_location_score
        self.min_category_score = min_category_score
        self.min_daypart_score = min_daypart_score
        self.min_final_score = min_final_score

    def retrieve(
        self,
        prompt: str,
        intent: Dict[str, Any],
        safe_cohorts: Sequence[Dict[str, Any]],
        top_k: int = 25,
    ) -> Dict[str, Any]:
        requested_locations = self._list_from_intent(intent, "locations")
        requested_categories = self._requested_categories(intent)
        requested_dayparts = self._list_from_intent(intent, "dayparts")
        requested_terms = self._requested_terms(prompt, intent)

        candidates: List[RetrievalCandidate] = []
        for cohort in safe_cohorts:
            candidate = self._score_candidate(
                prompt=prompt,
                cohort=cohort,
                requested_locations=requested_locations,
                requested_categories=requested_categories,
                requested_dayparts=requested_dayparts,
                requested_terms=requested_terms,
            )
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.final_score, reverse=True)

        selected = [
            c for c in candidates
            if c.decision in {"selected_for_review", "strong_candidate"}
        ][:top_k]

        return {
            "pipeline": "hybrid_audience_retrieval",
            "requested": {
                "locations": requested_locations,
                "categories": requested_categories,
                "dayparts": requested_dayparts,
                "terms": sorted(requested_terms),
            },
            "total_safe_cohorts_checked": len(safe_cohorts),
            "ranked_candidates": [self._candidate_to_dict(c) for c in candidates[:top_k]],
            "selected_candidates": [self._candidate_to_dict(c) for c in selected],
            "selection_count": len(selected),
        }

    def _score_candidate(
        self,
        prompt: str,
        cohort: Dict[str, Any],
        requested_locations: List[str],
        requested_categories: List[str],
        requested_dayparts: List[str],
        requested_terms: set[str],
    ) -> RetrievalCandidate:
        location_score = self._location_score(requested_locations, cohort)
        category_score = self._category_score(requested_categories, cohort)
        daypart_score = self._daypart_score(requested_dayparts, cohort)
        keyword_score = self._keyword_score(requested_terms, cohort)
        semantic_score = self._semantic_overlap_score(prompt, cohort)
        quality_score = min(max(_safe_float(cohort.get("quality_score"), 0.5), 0.0), 1.0)
        privacy_score = 1.0 if _norm(cohort.get("privacy_status", "safe")) in {"safe", "passed", "privacy_safe"} else 0.0

        final_score = (
            location_score * 0.30
            + category_score * 0.30
            + daypart_score * 0.15
            + semantic_score * 0.10
            + keyword_score * 0.10
            + quality_score * 0.03
            + privacy_score * 0.02
        )

        breakdown = {
            "location_score": round(location_score, 4),
            "category_score": round(category_score, 4),
            "daypart_score": round(daypart_score, 4),
            "semantic_score": round(semantic_score, 4),
            "keyword_score": round(keyword_score, 4),
            "quality_score": round(quality_score, 4),
            "privacy_score": round(privacy_score, 4),
            "final_score": round(final_score, 4),
        }

        decision, reason_codes = self._decision(
            requested_locations=requested_locations,
            requested_categories=requested_categories,
            requested_dayparts=requested_dayparts,
            breakdown=breakdown,
            cohort=cohort,
        )

        return RetrievalCandidate(
            cohort=cohort,
            final_score=round(final_score, 4),
            score_breakdown=breakdown,
            decision=decision,
            reason_codes=reason_codes,
        )

    def _decision(
        self,
        requested_locations: List[str],
        requested_categories: List[str],
        requested_dayparts: List[str],
        breakdown: Dict[str, float],
        cohort: Dict[str, Any],
    ) -> tuple[str, List[str]]:
        if breakdown["privacy_score"] < 1.0:
            return "blocked_privacy_status", ["privacy_status_not_safe"]

        if requested_locations and breakdown["location_score"] < self.min_location_score:
            return "blocked_wrong_location", ["requested_location_not_matched"]

        if requested_categories and breakdown["category_score"] < self.min_category_score:
            return "blocked_wrong_category", ["requested_category_not_matched"]

        if requested_dayparts and breakdown["daypart_score"] < self.min_daypart_score:
            return "blocked_wrong_daypart", ["requested_daypart_not_matched"]

        poi = _norm(cohort.get("primary_poi_type"))
        if requested_categories and poi in GENERIC_POI:
            return "blocked_generic_poi", ["generic_poi_blocked_for_specific_request"]

        if breakdown["final_score"] >= 0.78:
            return "selected_for_review", ["strong_structured_and_semantic_match"]

        if breakdown["final_score"] >= self.min_final_score:
            return "strong_candidate", ["candidate_above_minimum_score"]

        return "not_selected_low_score", ["candidate_below_minimum_score"]

    def _location_score(self, requested_locations: List[str], cohort: Dict[str, Any]) -> float:
        if not requested_locations:
            return 0.75

        cohort_loc = _norm(cohort.get("location_name") or cohort.get("location") or cohort.get("city"))
        if not cohort_loc:
            return 0.0

        best = 0.0
        cohort_tokens = set(cohort_loc.split("_"))

        for loc in requested_locations:
            req = _norm(loc)
            if not req:
                continue

            if req == cohort_loc:
                best = max(best, 1.0)
            elif req in cohort_loc or cohort_loc in req:
                best = max(best, 0.88)
            else:
                req_tokens = set(req.split("_"))
                if req_tokens:
                    overlap = len(req_tokens & cohort_tokens) / max(len(req_tokens), 1)
                    best = max(best, min(overlap, 0.75))

        return best

    def _category_score(self, requested_categories: List[str], cohort: Dict[str, Any]) -> float:
        if not requested_categories:
            return 0.70

        poi = _norm(cohort.get("primary_poi_type") or cohort.get("poi_type") or cohort.get("category"))
        if not poi:
            return 0.0

        best = 0.0
        for category in requested_categories:
            req = _norm(category)
            if not req:
                continue

            if req == poi:
                best = max(best, 1.0)
                continue

            if req in poi or poi in req:
                best = max(best, 0.82)

            req_group = self._category_group(req)
            poi_group = self._category_group(poi)

            if req_group and poi_group and req_group == poi_group:
                best = max(best, 0.92)

            if req_group and poi in CATEGORY_SYNONYMS.get(req_group, set()):
                best = max(best, 0.92)

        return best

    def _daypart_score(self, requested_dayparts: List[str], cohort: Dict[str, Any]) -> float:
        if not requested_dayparts:
            return 0.70

        cohort_daypart = _norm(cohort.get("created_day_part") or cohort.get("daypart"))
        if not cohort_daypart:
            return 0.0

        requested = {_norm(d) for d in requested_dayparts}
        if cohort_daypart in requested:
            return 1.0

        schedule_qualifiers = {"weekend", "weekday", "weekdays", "weekends"}
        if requested and requested.issubset(schedule_qualifiers):
            return 0.70

        return 0.0

    def _keyword_score(self, requested_terms: set[str], cohort: Dict[str, Any]) -> float:
        if not requested_terms:
            return 0.50

        cohort_tokens = _tokens(self._cohort_text(cohort))
        if not cohort_tokens:
            return 0.0

        overlap = len(requested_terms & cohort_tokens)
        return min(overlap / max(len(requested_terms), 1), 1.0)

    def _semantic_overlap_score(self, prompt: str, cohort: Dict[str, Any]) -> float:
        prompt_tokens = _tokens(prompt)
        cohort_tokens = _tokens(self._cohort_text(cohort))

        if not prompt_tokens or not cohort_tokens:
            return 0.0

        overlap = len(prompt_tokens & cohort_tokens)
        union = len(prompt_tokens | cohort_tokens)
        return min(overlap / max(union, 1), 1.0)

    def _cohort_text(self, cohort: Dict[str, Any]) -> str:
        fields = [
            cohort.get("location_name"),
            cohort.get("primary_poi_type"),
            cohort.get("created_day_part"),
            cohort.get("lookback_bucket"),
            cohort.get("trait_text"),
            cohort.get("embedding_text"),
            cohort.get("audience_name"),
        ]
        return " ".join(str(f or "") for f in fields)

    def _category_group(self, value: str) -> Optional[str]:
        value = _norm(value)
        for group, terms in CATEGORY_SYNONYMS.items():
            normalized_terms = {_norm(t) for t in terms}
            if value == group or value in normalized_terms:
                return group
        return None

    def _requested_terms(self, prompt: str, intent: Dict[str, Any]) -> set[str]:
        terms = set()
        for value in self._requested_categories(intent):
            terms |= _tokens(value)
        for value in self._list_from_intent(intent, "poi_terms"):
            terms |= _tokens(value)
        terms |= _tokens(prompt)
        return {
            t for t in terms
            if t not in {"find", "people", "during", "near", "offer", "build", "audience"}
        }

    def _requested_categories(self, intent: Dict[str, Any]) -> List[str]:
        values: List[str] = []
        for key in ("requested_categories", "canonical_categories", "matched_available_poi_types", "poi_terms"):
            values.extend(self._list_from_intent(intent, key))

        business_intent = intent.get("business_intent")
        if business_intent:
            values.append(str(business_intent))

        deduped: List[str] = []
        seen = set()
        for value in values:
            n = _norm(value)
            if n and n not in seen:
                deduped.append(value)
                seen.add(n)
        return deduped

    def _list_from_intent(self, intent: Dict[str, Any], key: str) -> List[str]:
        value = intent.get(key)
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Iterable):
            return [str(v) for v in value if v is not None and str(v).strip()]
        return [str(value)]

    def _candidate_to_dict(self, candidate: RetrievalCandidate) -> Dict[str, Any]:
        cohort = candidate.cohort
        return {
            "audience_name": cohort.get("audience_name") or self._default_audience_name(cohort),
            "location_name": cohort.get("location_name"),
            "primary_poi_type": cohort.get("primary_poi_type"),
            "created_day_part": cohort.get("created_day_part"),
            "quality_score": cohort.get("quality_score"),
            "privacy_status": cohort.get("privacy_status"),
            "final_score": candidate.final_score,
            "score_breakdown": candidate.score_breakdown,
            "decision": candidate.decision,
            "reason_codes": candidate.reason_codes,
            "cohort": cohort,
        }

    def _default_audience_name(self, cohort: Dict[str, Any]) -> str:
        poi = str(cohort.get("primary_poi_type") or "Audience").replace("_", " ").title()
        daypart = str(cohort.get("created_day_part") or "Anytime").title()
        loc = str(cohort.get("location_name") or "Unknown").title()
        return f"{poi} - {daypart} - {loc}"
