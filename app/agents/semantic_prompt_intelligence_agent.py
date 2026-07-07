from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class SemanticPromptIntent:
    status: str
    original_prompt: str
    business_intent: str
    audience_goal: str
    locations: list[str]
    poi_terms: list[str]
    canonical_categories: list[str]
    dayparts: list[str]
    quality_intent: str
    fallback_tolerance: str
    confidence_score: float
    extraction_method: str
    reasons: list[str]


class SemanticPromptIntelligenceAgent:
    """
    Dynamic-ish semantic prompt intelligence.

    This is not prompt-specific hardcoding.
    It uses a broad business taxonomy + alias expansion + confidence scoring.

    Current purpose:
    - map natural language business prompts into safe structured intent
    - support synonyms like:
      after-office -> evening
      coworking hubs/business centers -> office
      coffee lovers/espresso -> cafe
      quick bites/dinner time -> restaurant/evening
      body ink/piercing -> tattoo
    - never invent real audiences; matching still happens against real safe cohorts
    """

    CATEGORY_TAXONOMY = {
        "restaurant": [
            "restaurant",
            "restaurants",
            "food",
            "food street",
            "food streets",
            "dining",
            "casual dining",
            "casual diners",
            "eatery",
            "eateries",
            "shawarma",
            "burger",
            "pizza",
            "fast food",
            "quick service food",
            "quick bites",
            "fast meals",
            "middle eastern",
            "takeaway",
            "lunch",
            "dinner",
            "dinner time",
        ],
        "cafe": [
            "cafe",
            "cafes",
            "café",
            "cafés",
            "coffee",
            "coffee shop",
            "coffee lovers",
            "espresso",
            "snacks",
            "bakery cafe",
        ],
        "gym": [
            "gym",
            "fitness",
            "workout",
            "workouts",
            "workout people",
            "fitness routines",
            "health club",
            "training center",
            "training sessions",
            "yoga",
            "pilates",
        ],
        "tattoo": [
            "tattoo",
            "tattoo studio",
            "body art",
            "body_art_service",
            "body ink",
            "body ink studio",
            "body ink studios",
            "piercing",
            "piercing studio",
            "piercing studios",
        ],
        "office": [
            "office",
            "corporate",
            "corporate office",
            "coworking",
            "coworking space",
            "coworking spaces",
            "coworking hub",
            "coworking hubs",
            "workspace",
            "workspaces",
            "business district",
            "business districts",
            "business center",
            "business centers",
            "business place",
            "business places",
            "business space",
            "business spaces",
            "business hubs",
            "working professional",
            "working professionals",
            "professionals",
            "after office crowd",
            "after-office crowd",
            "office crowd",
            "consultant",
        ],
        "retail": [
            "retail",
            "store",
            "stores",
            "shopping",
            "shopping area",
            "shopping areas",
            "mall",
            "shop",
            "fashion",
            "apparel",
        ],
        "healthcare": [
            "clinic",
            "clinics",
            "hospital",
            "healthcare",
            "medical",
            "doctor",
            "pharmacy",
            "pharmacies",
            "dentist",
        ],
        "education": [
            "school",
            "college",
            "university",
            "education",
            "campus",
            "students",
            "student",
            "campus visitors",
        ],
        "nightlife": [
            "bar",
            "pub",
            "club",
            "nightlife",
            "lounge",
            "late evening",
            "go out late",
        ],
        "beauty": [
            "salon",
            "spa",
            "beauty",
            "hair",
            "makeup",
            "skincare",
            "wellness",
            "grooming",
            "self care",
            "self-care",
        ],
        "auto": [
            "auto",
            "car",
            "vehicle",
            "automotive",
            "garage",
            "service center",
        ],
    }

    DAYPART_TAXONOMY = {
        "morning": [
            "morning",
            "breakfast",
            "early morning",
            "before work",
            "early in the day",
            "early day",
            "am",
        ],
        "afternoon": [
            "afternoon",
            "lunch",
            "noon",
            "afternoon hours",
        ],
        "evening": [
            "evening",
            "dinner",
            "dinner time",
            "after work",
            "after-work",
            "after office",
            "after-office",
            "after office hours",
            "after work hours",
            "post work",
            "late evening",
            "sunset",
        ],
        "night": [
            "night",
            "late night",
            "midnight",
            "go out late",
        ],
        "weekend": [
            "weekend",
            "weekends",
            "saturday",
            "sunday",
        ],
        "weekday": [
            "weekday",
            "weekdays",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
        ],
    }

    KNOWN_LOCATIONS = [
        "montreal downtown",
        "montreal qc",
        "san francisco",
        "los angeles",
        "new york",
        "westmount",
        "montreal",
        "toronto",
        "vancouver",
        "calgary",
        "chicago",
        "brooklyn",
        "hyderabad",
        "bangalore",
        "mumbai",
        "delhi",
        "quebec",
    ]

    QUALITY_TERMS = {
        "high": [
            "high quality",
            "high-quality",
            "high intent",
            "high-intent",
            "premium",
            "best",
            "top",
            "strong",
        ],
        "broad": [
            "broad",
            "large",
            "scale",
            "reach",
        ],
        "balanced": [
            "balanced",
            "standard",
            "normal",
        ],
    }

    def analyze_prompt(self, prompt: str, run_dir: str | Path | None = None) -> dict[str, Any]:
        text = self._normalize(prompt)

        locations = self._extract_locations(text)
        canonical_categories, poi_terms = self._extract_categories(text)
        dayparts = self._extract_dayparts(text)
        quality_intent = self._extract_quality_intent(text)
        fallback_tolerance = self._extract_fallback_tolerance(text)
        audience_goal = self._infer_audience_goal(text)
        business_intent = self._infer_business_intent(canonical_categories)

        reasons = []
        if locations:
            reasons.append("Detected location intent from prompt.")
        if canonical_categories:
            reasons.append("Mapped business/category terms using semantic taxonomy.")
        if dayparts:
            reasons.append("Detected daypart/time intent.")
        if quality_intent:
            reasons.append("Detected quality preference.")
        if fallback_tolerance != "unknown":
            reasons.append("Detected fallback/review preference.")

        confidence_score = self._confidence_score(
            locations=locations,
            categories=canonical_categories,
            dayparts=dayparts,
            audience_goal=audience_goal,
            quality_intent=quality_intent,
        )

        result = SemanticPromptIntent(
            status="completed",
            original_prompt=prompt,
            business_intent=business_intent,
            audience_goal=audience_goal,
            locations=locations,
            poi_terms=poi_terms,
            canonical_categories=canonical_categories,
            dayparts=dayparts,
            quality_intent=quality_intent,
            fallback_tolerance=fallback_tolerance,
            confidence_score=confidence_score,
            extraction_method="deterministic_semantic_taxonomy_v2",
            reasons=reasons,
        )

        output = asdict(result)

        if run_dir:
            path = Path(run_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "prompt_intent.json").write_text(
                json.dumps(output, indent=2),
                encoding="utf-8",
            )

        return output

    def _normalize(self, prompt: str) -> str:
        text = prompt.lower()
        text = text.replace("café", "cafe").replace("cafés", "cafes")
        text = text.replace("-", " ")
        text = text.replace("/", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_locations(self, text: str) -> list[str]:
        found = []

        for location in sorted(self.KNOWN_LOCATIONS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(location)}\b", text):
                found.append(location)

        return self._dedupe(found)


    def _extract_categories(self, text: str) -> tuple[list[str], list[str]]:
        categories = []
        terms = []

        def has_any(values: list[str]) -> bool:
            return any(self._normalize(value) in text for value in values)

        cafe_signals = [
            "cafe",
            "cafes",
            "coffee",
            "coffee shop",
            "coffee lovers",
            "espresso",
            "caffeine",
            "caffeine break",
            "coffee break",
            "grab coffee",
            "grab espresso",
            "snacks",
        ]

        office_strong_signals = [
            "coworking",
            "coworking space",
            "coworking spaces",
            "coworking hub",
            "coworking hubs",
            "flexible workspace",
            "flexible workspaces",
            "workspace",
            "workspaces",
            "business center",
            "business centers",
            "business place",
            "business places",
            "business hub",
            "business hubs",
            "corporate office",
            "office crowd",
            "working professional",
            "working professionals",
            "young professional",
            "young professionals",
            "professionals around",
        ]

        # Main business object wins.
        # "caffeine break after office" = cafe intent + evening time.
        cafe_requested = has_any(cafe_signals)
        office_requested = has_any(office_strong_signals)

        if cafe_requested:
            categories.append("cafe")
            for signal in cafe_signals:
                if self._normalize(signal) in text:
                    terms.append(self._normalize(signal))

        for canonical, aliases in self.CATEGORY_TAXONOMY.items():
            # Avoid treating "after office" as office category.
            if canonical == "office":
                if not office_requested:
                    continue

            # If cafe is clearly requested, do not let contextual "office" override category.
            if cafe_requested and canonical == "office":
                continue

            for alias in aliases:
                alias_norm = self._normalize(alias)

                # Plain office is too broad; require stronger office context.
                if canonical == "office" and alias_norm == "office" and not office_requested:
                    continue

                if re.search(rf"\b{re.escape(alias_norm)}\b", text):
                    categories.append(canonical)
                    terms.append(alias_norm)

        deduped_categories = self._dedupe(categories)

        # Keep stable canonical business ordering based on taxonomy order.
        # This prevents outputs like cafe+restaurant when the expected
        # canonical intent is restaurant+cafe.
        ordered_categories = [
            category
            for category in self.CATEGORY_TAXONOMY.keys()
            if category in deduped_categories
        ]

        ordered_categories.extend(
            category
            for category in deduped_categories
            if category not in ordered_categories
        )

        return ordered_categories, self._dedupe(terms)


    def _extract_dayparts(self, text: str) -> list[str]:
        found = []

        for canonical, aliases in self.DAYPART_TAXONOMY.items():
            for alias in aliases:
                alias_norm = self._normalize(alias)
                if re.search(rf"\b{re.escape(alias_norm)}\b", text):
                    found.append(canonical)

        return self._dedupe(found)

    def _extract_quality_intent(self, text: str) -> str:
        for quality, aliases in self.QUALITY_TERMS.items():
            if any(self._normalize(alias) in text for alias in aliases):
                return quality
        return "balanced"

    def _extract_fallback_tolerance(self, text: str) -> str:
        if any(term in text for term in ["exact only", "exact matches only", "prioritize exact"]):
            return "strict_exact_first"

        if any(term in text for term in ["fallback", "adjacent", "alternative", "suggest"]):
            return "allow_review_fallbacks"

        return "unknown"

    def _infer_audience_goal(self, text: str) -> str:
        if any(term in text for term in ["high quality", "high intent", "best", "premium"]):
            return "high_quality_audience"

        if any(term in text for term in ["lookalike", "similar"]):
            return "lookalike_expansion"

        if any(term in text for term in ["visitors", "footfall", "traffic", "crowd", "people"]):
            return "visitor_footfall_audience"

        return "general_audience"

    def _infer_business_intent(self, categories: list[str]) -> str:
        if categories:
            return "+".join(categories)
        return "unknown_business_intent"

    def _confidence_score(
        self,
        locations: list[str],
        categories: list[str],
        dayparts: list[str],
        audience_goal: str,
        quality_intent: str,
    ) -> float:
        score = 0.15

        if locations:
            score += 0.25
        if categories:
            score += 0.35
        if dayparts:
            score += 0.25
        if audience_goal != "general_audience":
            score += 0.05
        if quality_intent != "balanced":
            score += 0.05

        return round(min(score, 0.98), 3)

    def _dedupe(self, values: list[str]) -> list[str]:
        seen = set()
        output = []

        for value in values:
            cleaned = value.strip().lower()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                output.append(cleaned)

        return output
