from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from app.agents.hybrid_semantic_intent_agent import HybridSemanticIntentAgent
from app.agents.semantic_prompt_intelligence_agent import SemanticPromptIntelligenceAgent


@dataclass
class AutonomousSelectorConfig:
    top_k: int = 10
    min_final_score: float = 0.38
    min_location_score: float = 0.62
    min_poi_score: float = 0.62
    min_daypart_score: float = 0.90


class AutonomousPromptCohortSelectorAgent:
    """
    Dynamic selector:
    - Uses Hybrid LLM/RAG intent when available.
    - Scores all privacy-safe cohorts with embeddings + generic constraint compatibility.
    - Does not contain city/category-specific business rules.
    - Deterministic gates are only safety gates to prevent fake export.
    """

    def __init__(self, config: AutonomousSelectorConfig | None = None):
        self.config = config or AutonomousSelectorConfig()
        self.vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=2048,
            alternate_sign=False,
            norm="l2",
        )

    def select(
        self,
        *,
        prompt: str,
        cohorts: pd.DataFrame,
        intent: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        df = cohorts.copy()

        if df.empty:
            return df, self._empty_report(prompt, reason="no_safe_cohorts")

        loc_col = self._column(df, "location_name", "location")
        poi_col = self._column(df, "primary_poi_type", "poi_type")
        day_col = self._column(df, "created_day_part", "daypart", "day_part")

        intent = intent or self._resolve_intent(prompt, df)

        requested_locations = self._dedupe(intent.get("locations") or [])

        # Always recover explicit city/location phrases from the prompt.
        # If LLM misses Austin/Dubai/London, this prevents cross-city export.
        explicit_locations = self._extract_explicit_locations_from_prompt(prompt)

        if explicit_locations:
            requested_locations = self._dedupe(explicit_locations + requested_locations)

        # Always recover explicit city/location phrases from the prompt.
        # If LLM misses Austin/Dubai/London, this prevents cross-city export.
        explicit_locations = self._extract_explicit_locations_from_prompt(prompt)

        if explicit_locations:
            requested_locations = self._dedupe(explicit_locations + requested_locations)
        requested_categories = self._dedupe(
            (intent.get("matched_available_poi_types") or [])
            + (intent.get("requested_categories") or [])
            + (intent.get("canonical_categories") or [])
            + (intent.get("poi_terms") or [])
        )

        available_pois = self._available_values(df, poi_col)
        rescued_pois = self._rescue_available_poi_matches(
            prompt=prompt,
            requested_values=requested_categories,
            available_pois=available_pois,
        )
        requested_categories = self._dedupe(rescued_pois + requested_categories)

        # Strong export targets:
        # Priority:
        # 1. Primary LLM/fallback category if that exact POI exists in safe metadata.
        # 2. Dynamic rescued POI only when no primary exact POI exists.
        #
        # This prevents prompt context like "after office" from polluting a cafe
        # intent with coworking/corporate-office export targets.
        available_poi_set = {self._norm(value) for value in available_pois}

        primary_poi_targets = self._dedupe(
            (intent.get("matched_available_poi_types") or [])
            + (intent.get("requested_categories") or [])
            + (intent.get("canonical_categories") or [])
        )

        primary_available_targets = [
            value for value in primary_poi_targets
            if self._norm(value) in available_poi_set
        ]

        if primary_available_targets:
            strong_poi_targets = primary_available_targets
        else:
            strong_poi_targets = [
                value for value in rescued_pois
                if not self._is_generic_poi_value(value)
            ]

        available_locations = self._available_values(df, loc_col)

        # Hard UI/report cleanup:
        # If the user explicitly requested a city/market and the safe cohort
        # metadata has no coverage for it, do not even show cross-city selected
        # cohorts. This keeps Prompt-selected cohorts and Exported audiences both 0.
        if requested_locations and not self._has_requested_location_coverage(
            requested_locations=requested_locations,
            available_locations=available_locations,
        ):
            report = {
                "filter_mode": "location_category_gap_no_export",
                "locations_detected": requested_locations,
                "poi_terms_detected": self._display_poi_terms(intent),
                "dayparts_detected": [],
                "schedule_qualifiers_detected": [],
                "business_intent": intent.get("business_intent"),
                "rescued_available_poi_matches": rescued_pois,
                "coverage_warnings": [
                    "A specific location was requested, but no privacy-safe cohort exists for that location. Cross-location fallback audiences were blocked from export."
                ],
            }
            return df.iloc[0:0].copy(), report

        available_dayparts = self._available_values(df, day_col)
        raw_requested_dayparts = self._dedupe(intent.get("dayparts") or [])
        requested_dayparts, schedule_qualifiers = self._split_dayparts_and_schedule_qualifiers(
            requested_dayparts=raw_requested_dayparts,
            available_dayparts=available_dayparts,
        )

        intent_doc = self._intent_document(prompt, intent)

        cohort_docs = []
        for _, row in df.iterrows():
            cohort_docs.append(self._cohort_document(row, loc_col, poi_col, day_col))

        semantic_scores = self._semantic_scores(intent_doc, cohort_docs)

        scored_rows = []

        for idx, row in df.iterrows():
            location_value = str(row.get(loc_col, "")) if loc_col else ""
            poi_value = str(row.get(poi_col, "")) if poi_col else ""
            daypart_value = str(row.get(day_col, "")) if day_col else ""

            location_score = self._max_location_score(requested_locations, location_value)
            poi_score = self._max_text_score(requested_categories, poi_value)
            daypart_score = self._max_daypart_score(requested_dayparts, daypart_value)

            has_location_request = bool(requested_locations and loc_col)
            has_poi_request = bool(requested_categories and poi_col)
            has_daypart_request = bool(requested_dayparts and day_col)

            # Specific POI export gate:
            # For specific intent, do not export loose semantic matches.
            # Example: requested cafe + available cafe only in Montreal must not
            # export San Francisco coworking/corporate office.
            if has_poi_request and self._has_specific_requested_intent(requested_categories):
                if strong_poi_targets:
                    strong_poi_score = self._max_text_score(strong_poi_targets, poi_value)
                    if strong_poi_score < 0.86:
                        continue
                else:
                    continue

            # Generic specificity gate:
            # If the user requested a specific business/domain intent, do not export
            # broad generic POIs like "store" or "clinic" as if they were exact matches.
            # They can still appear in ranked review/mutation suggestions, but not export.
            if has_poi_request and self._has_specific_requested_intent(requested_categories):
                if self._is_generic_poi_value(poi_value):
                    continue

            # Safety gates, not business rules.
            # They prevent cross-city/category/daypart fallback exports.
            if has_location_request and location_score < self.config.min_location_score:
                continue

            if has_poi_request and poi_score < self.config.min_poi_score:
                continue

            if has_daypart_request and daypart_score < self.config.min_daypart_score:
                continue

            quality_score = self._safe_float(row.get("quality_score"), default=0.0)

            final_score = (
                0.35 * float(semantic_scores[idx])
                + 0.25 * location_score
                + 0.25 * poi_score
                + 0.10 * daypart_score
                + 0.05 * min(max(quality_score, 0.0), 1.0)
            )

            if final_score < self.config.min_final_score:
                continue

            scored_rows.append(
                {
                    "_row_index": idx,
                    "_autonomous_match_score": round(final_score, 6),
                    "_semantic_score": round(float(semantic_scores[idx]), 6),
                    "_location_score": round(location_score, 6),
                    "_poi_score": round(poi_score, 6),
                    "_daypart_score": round(daypart_score, 6),
                }
            )

        if not scored_rows:
            selected = df.iloc[0:0].copy()
            coverage_warnings = []

            if requested_locations and requested_categories:
                coverage_warnings.append(
                    "A specific location and business/category were requested, but no strong privacy-safe match exists. Cross-location or category-only fallback audiences were blocked from export."
                )

            return selected, {
                "status": "completed",
                "filter_mode": "location_category_gap_no_export"
                if requested_locations and requested_categories
                else "semantic_no_match",
                "selector_mode": "autonomous_hybrid_rag_embedding_selector",
                "locations_detected": requested_locations,
                "poi_terms_detected": self._display_poi_terms(intent),
                "dayparts_detected": requested_dayparts,
                "schedule_qualifiers_detected": schedule_qualifiers,
                "coverage_warnings": coverage_warnings,
                "selected_count": 0,
                "llm_used": bool(intent.get("llm_used")),
                "resolver_mode": intent.get("resolver_mode"),
                "business_intent": intent.get("business_intent"),
                "rescued_available_poi_matches": rescued_pois,
            }

        score_df = pd.DataFrame(scored_rows).sort_values(
            "_autonomous_match_score", ascending=False
        )

        top_indices = score_df.head(self.config.top_k)["_row_index"].tolist()
        selected = df.loc[top_indices].copy()

        selected = selected.merge(
            score_df,
            left_index=True,
            right_on="_row_index",
            how="left",
        ).drop(columns=["_row_index"], errors="ignore")

        filter_parts = []
        if requested_locations:
            filter_parts.append("location")
        if requested_categories:
            filter_parts.append("poi")
        if requested_dayparts:
            filter_parts.append("daypart")

        selected = self._filter_generic_export_rows(
            selected=selected,
            poi_col=poi_col,
            requested_values=requested_categories,
            prompt=prompt,
            intent=intent,
        )

        if selected.empty:
            coverage_warnings = []
            if requested_locations and requested_categories:
                coverage_warnings.append(
                    "A specific location and business/category were requested, but only generic POI matches were available. Generic fallback audiences were blocked from export."
                )

            return selected, {
                "status": "completed",
                "filter_mode": "location_category_gap_no_export"
                if requested_locations and requested_categories
                else "semantic_no_specific_match",
                "selector_mode": "autonomous_hybrid_rag_embedding_selector",
                "locations_detected": requested_locations,
                "poi_terms_detected": self._display_poi_terms(intent),
                "dayparts_detected": requested_dayparts,
                "schedule_qualifiers_detected": schedule_qualifiers,
                "coverage_warnings": coverage_warnings,
                "selected_count": 0,
                "llm_used": bool(intent.get("llm_used")),
                "resolver_mode": intent.get("resolver_mode"),
                "business_intent": intent.get("business_intent"),
                "rescued_available_poi_matches": rescued_pois,
            }

        return selected, {
            "status": "completed",
            "filter_mode": "+".join(filter_parts) if filter_parts else "semantic",
            "selector_mode": "autonomous_hybrid_rag_embedding_selector",
            "locations_detected": requested_locations,
            "poi_terms_detected": self._display_poi_terms(intent),
            "dayparts_detected": requested_dayparts,
                "schedule_qualifiers_detected": schedule_qualifiers,
            "coverage_warnings": [],
            "selected_count": int(len(selected)),
            "llm_used": bool(intent.get("llm_used")),
            "resolver_mode": intent.get("resolver_mode"),
            "business_intent": intent.get("business_intent"),
                "rescued_available_poi_matches": rescued_pois,
        }

    def _resolve_intent(self, prompt: str, df: pd.DataFrame) -> dict[str, Any]:
        try:
            return HybridSemanticIntentAgent().resolve(prompt=prompt, safe_cohorts=df)
        except Exception as exc:
            fallback = SemanticPromptIntelligenceAgent().analyze_prompt(prompt)
            fallback["llm_used"] = False
            fallback["resolver_mode"] = "deterministic_fallback_selector_exception"
            fallback["llm_error"] = str(exc)
            return fallback

    def _semantic_scores(self, intent_doc: str, cohort_docs: list[str]) -> np.ndarray:
        if not cohort_docs:
            return np.array([])

        matrix = self.vectorizer.transform([intent_doc] + cohort_docs)
        query = matrix[0]
        docs = matrix[1:]
        scores = docs @ query.T
        return np.asarray(scores.toarray()).reshape(-1)

    def _intent_document(self, prompt: str, intent: dict[str, Any]) -> str:
        parts = [
            prompt,
            str(intent.get("business_intent") or ""),
            " ".join(intent.get("locations") or []),
            " ".join(intent.get("requested_categories") or []),
            " ".join(intent.get("matched_available_poi_types") or []),
            " ".join(intent.get("canonical_categories") or []),
            " ".join(intent.get("poi_terms") or []),
            " ".join(intent.get("dayparts") or []),
            str(intent.get("audience_goal") or ""),
        ]
        return self._norm(" ".join(parts))

    def _cohort_document(self, row: pd.Series, loc_col: str | None, poi_col: str | None, day_col: str | None) -> str:
        parts = []
        for col in [loc_col, poi_col, day_col, "quality_score", "total_maid_volume", "privacy_status"]:
            if col and col in row.index:
                parts.append(str(row.get(col) or ""))
        return self._norm(" ".join(parts))

    def _max_location_score(self, requested_locations: list[str], actual_location: str) -> float:
        if not requested_locations:
            return 1.0

        actual = self._norm(actual_location)
        if not actual:
            return 0.0

        return max(self._location_score(requested, actual) for requested in requested_locations)

    def _location_score(self, requested_location: str, actual_location: str) -> float:
        requested = self._norm(requested_location)
        actual = self._norm(actual_location)

        if not requested or not actual:
            return 0.0

        if requested == actual:
            return 1.0

        # Generic broad-area support:
        # "montreal" can match "montreal downtown".
        if requested in actual:
            return 0.98

        # Specific-to-broad fallback is intentionally weak:
        # "westmount montreal" should not automatically export "montreal".
        if actual in requested:
            return 0.55

        return self._char_similarity(requested, actual)

    def _max_text_score(self, requested_values: list[str], actual_value: str) -> float:
        if not requested_values:
            return 1.0

        actual = self._norm(actual_value)
        if not actual:
            return 0.0

        return max(self._text_score(requested, actual) for requested in requested_values)

    def _text_score(self, requested: str, actual: str) -> float:
        requested_norm = self._norm(requested)
        actual_norm = self._norm(actual)

        if not requested_norm or not actual_norm:
            return 0.0

        if requested_norm == actual_norm:
            return 1.0

        if requested_norm in actual_norm or actual_norm in requested_norm:
            return 0.88

        return self._char_similarity(requested_norm, actual_norm)

    def _max_daypart_score(self, requested_dayparts: list[str], actual_daypart: str) -> float:
        if not requested_dayparts:
            return 1.0

        actual = self._norm(actual_daypart)
        if not actual:
            return 0.0

        return max(1.0 if self._norm(daypart) == actual else 0.0 for daypart in requested_dayparts)

    def _char_similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0

        matrix = self.vectorizer.transform([left, right])
        score = matrix[0] @ matrix[1].T
        return float(score.toarray()[0][0])





    def _is_generic_poi_value(self, value: Any) -> bool:
        text = self._norm(value).replace(" ", "_")
        generic_values = {
            "store",
            "stores",
            "shop",
            "shops",
            "clinic",
            "clinics",
            "service",
            "services",
            "service_center",
            "service_centers",
            "point_of_interest",
            "establishment",
            "business",
            "place",
            "places",
        }
        return text in generic_values


    def _has_specific_requested_intent(self, requested_values: list[str]) -> bool:
        for value in requested_values or []:
            text = self._norm(value).replace(" ", "_")
            if not text:
                continue

            if self._is_generic_poi_value(text):
                continue

            # Any non-generic POI/category is a specific export intent.
            # Examples: cafe, gym, restaurant, pet_store, car_wash,
            # electronics_shop, veterinary_care, yoga_studio.
            if len(text) >= 3:
                return True

        return False


    def _is_generic_poi_value(self, value: Any) -> bool:
        text = self._norm(value).replace(" ", "_")
        generic_values = {
            "store",
            "stores",
            "shop",
            "shops",
            "clinic",
            "clinics",
            "service",
            "services",
            "service_center",
            "service_centers",
            "point_of_interest",
            "establishment",
            "business",
            "place",
            "places",
        }
        return text in generic_values

    def _has_specific_requested_intent(self, requested_values: list[str]) -> bool:
        for value in requested_values or []:
            text = self._norm(value).replace(" ", "_")
            if not text:
                continue
            if self._is_generic_poi_value(text):
                continue

            # Specific domain intent examples:
            # pet_store, veterinary_care, auto_repair_shops, electronics_shop,
            # yoga_studio, computer_store, car_wash, mobile_repair_store, etc.
            if "_" in text or len(text) >= 8:
                return True

        return False




    def _extract_explicit_locations_from_prompt(self, prompt: str) -> list[str]:
        # Generic explicit-location recovery.
        # Purpose: if LLM misses a city like London/Austin/Dubai, do not allow
        # Montreal/San Francisco cohorts to be exported by POI-only matching.
        # This is a safety guard, not city-specific hardcoding.
        if not prompt:
            return []

        prompt = str(prompt)
        candidates = []

        patterns = [
            # in Austin / in Dubai / near New York / around Los Angeles / across London
            r"\b(?:in|near|around|within|across)\s+([A-Z][A-Za-z]+(?:[\s,]+[A-Z][A-Za-z]+){0,3})(?=[\.\!\?\;:]|\s|$)",
            # city of London
            r"\b(?:city of)\s+([A-Z][A-Za-z]+(?:[\s,]+[A-Z][A-Za-z]+){0,3})(?=[\.\!\?\;:]|\s|$)",
        ]

        stop_words = {
            "Find", "Build", "Create", "Reach", "Audience", "People", "Who",
            "Visit", "Visits", "Visited", "After", "Before", "During",
            "Weekend", "Weekday", "Morning", "Afternoon", "Evening", "Night",
            "Premium", "Luxury", "Local", "New", "Late", "Fast", "Food",
            "Pet", "Care", "Health", "Travel", "Business", "Store", "Shop",
            "Service", "Restaurant", "Campaign", "Fashion", "Lifestyle",
            "Sports", "Recovery", "Physiotherapy", "Shopping", "Clothing",
            "Mobile", "Repair", "Gadget", "Accessories",
        }

        for pattern in patterns:
            for match in re.finditer(pattern, prompt):
                raw = match.group(1).strip(" .,!?:;")
                raw = raw.replace(",", " ")

                words = []
                for word in raw.split():
                    if word in stop_words:
                        break
                    words.append(word)

                if not words:
                    continue

                candidate = self._norm(" ".join(words))
                if candidate and len(candidate) >= 3:
                    candidates.append(candidate)

        return self._dedupe(candidates)



    def _has_requested_location_coverage(
        self,
        *,
        requested_locations: list[str],
        available_locations: list[str],
    ) -> bool:
        if not requested_locations:
            return True

        available = [self._norm(value) for value in available_locations if self._norm(value)]

        if not available:
            return False

        for requested in requested_locations:
            req = self._norm(requested)
            if not req:
                continue

            for loc in available:
                # Exact or safe sub-area match:
                # new york can match times square, new york.
                # montreal can match montreal downtown.
                if req == loc or req in loc or loc in req:
                    return True

        return False

    def _available_values(self, df: pd.DataFrame, column: str | None) -> list[str]:
        if not column or column not in df.columns:
            return []

        return (
            df[column]
            .dropna()
            .astype(str)
            .map(self._norm)
            .drop_duplicates()
            .tolist()
        )

    def _split_dayparts_and_schedule_qualifiers(
        self,
        *,
        requested_dayparts: list[str],
        available_dayparts: list[str],
    ) -> tuple[list[str], list[str]]:
        available = {self._norm(value) for value in available_dayparts}

        schedule_words = {
            "weekday",
            "weekdays",
            "weekend",
            "weekends",
            "daily",
            "workday",
            "workdays",
        }

        real_dayparts = []
        schedule_qualifiers = []

        for value in requested_dayparts or []:
            value_norm = self._norm(value)

            if value_norm in schedule_words and value_norm not in available:
                schedule_qualifiers.append(value_norm)
                continue

            if not available or value_norm in available:
                real_dayparts.append(value_norm)
            else:
                schedule_qualifiers.append(value_norm)

        return self._dedupe(real_dayparts), self._dedupe(schedule_qualifiers)


    def _rescue_available_poi_matches(
        self,
        *,
        prompt: str,
        requested_values: list[str],
        available_pois: list[str],
    ) -> list[str]:
        # Dynamic, data-aware rescue:
        # Match available safe POI types using meaningful content tokens only.
        # Generic words like store/shop/place must not rescue unrelated subtypes.
        query = self._norm(
            " ".join([prompt] + [str(value) for value in requested_values or []])
        )

        if not query or not available_pois:
            return []

        query_tokens = set(self._content_tokens(query))
        matches = []

        for poi in available_pois:
            poi_norm = self._norm(poi)
            if not poi_norm:
                continue

            poi_tokens = set(self._content_tokens(poi_norm))
            specific_poi_tokens = {
                token for token in poi_tokens
                if token not in {
                    "store", "shop", "place", "service", "center", "business",
                    "point", "interest", "location", "area", "related"
                }
            }

            if not specific_poi_tokens:
                continue

            overlap = specific_poi_tokens & query_tokens
            token_overlap = len(overlap) / max(len(specific_poi_tokens), 1)

            # Require direct content-token evidence.
            # Example:
            # clothing_store ↔ clothing stores ✅
            # shopping_mall ↔ malls ✅
            # pet_store ↔ lifestyle store ❌
            # grocery_store ↔ lifestyle store ❌
            if token_overlap > 0:
                matches.append((poi_norm, token_overlap, poi_norm))

        matches.sort(key=lambda item: (-item[1], item[2]))
        return self._dedupe([item[0] for item in matches[:8]])


    def _filter_generic_export_rows(
        self,
        *,
        selected: pd.DataFrame,
        poi_col: str | None,
        requested_values: list[str],
        prompt: str = "",
        intent: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if selected.empty or not poi_col or poi_col not in selected.columns:
            return selected

        if not self._has_specific_requested_intent(requested_values):
            return selected

        intent = intent or {}

        query_text = self._norm(
            " ".join(
                [
                    prompt,
                    str(intent.get("business_intent") or ""),
                    " ".join(intent.get("requested_categories") or []),
                    " ".join(intent.get("matched_available_poi_types") or []),
                    " ".join(intent.get("canonical_categories") or []),
                    " ".join(intent.get("poi_terms") or []),
                    " ".join(requested_values or []),
                ]
            )
        )

        def keep_row(poi_value: Any) -> bool:
            # Never export broad generic POI rows for specific requests.
            if self._is_generic_poi_value(poi_value):
                return False

            return self._poi_directly_supported_by_prompt(
                poi_value=poi_value,
                query_text=query_text,
            )

        return selected[selected[poi_col].apply(keep_row)].copy()




    def _poi_directly_supported_by_prompt(self, *, poi_value: Any, query_text: str) -> bool:
        poi_norm = self._norm(poi_value)
        query_norm = self._norm(query_text)

        if not poi_norm or not query_norm:
            return False

        generic_tokens = {
            "store",
            "stores",
            "shop",
            "shops",
            "place",
            "places",
            "center",
            "centers",
            "service",
            "services",
            "business",
            "point",
            "interest",
            "location",
            "area",
            "areas",
            "related",
        }

        poi_tokens = self._content_tokens(poi_norm)
        query_tokens = self._content_tokens(query_norm)

        specific_poi_tokens = [
            token for token in poi_tokens
            if token not in generic_tokens and len(token) >= 3
        ]

        if not specific_poi_tokens:
            return False

        # Approval-gated export must be directly explainable from prompt/intent.
        # Do not allow semantic similarity from generic words like "store" to export
        # unrelated subtypes like pet_store or grocery_store.
        if set(specific_poi_tokens) & set(query_tokens):
            return True

        singular_query_tokens = {self._singularize_token(token) for token in query_tokens}
        singular_poi_tokens = {self._singularize_token(token) for token in specific_poi_tokens}

        if singular_poi_tokens & singular_query_tokens:
            return True

        poi_phrase = " ".join(specific_poi_tokens)
        if poi_phrase and poi_phrase in query_norm:
            return True

        return False


    def _content_tokens(self, value: Any) -> list[str]:
        text = self._norm(value)
        raw_tokens = re.split(r"[^a-z0-9]+", text)

        stop_tokens = {
            "",
            "the",
            "and",
            "or",
            "for",
            "who",
            "with",
            "near",
            "into",
            "from",
            "that",
            "this",
            "they",
            "them",
            "their",
            "people",
            "audience",
            "audiences",
            "build",
            "find",
            "reach",
            "visit",
            "visited",
            "visits",
            "recently",
            "during",
            "after",
            "before",
            "around",
            "related",
            "places",
            "place",
            "locations",
            "location",
        }

        tokens = []
        for token in raw_tokens:
            token = token.strip().lower()
            if token in stop_tokens:
                continue
            if len(token) < 3:
                continue

            tokens.append(token)
            singular = self._singularize_token(token)
            if singular != token:
                tokens.append(singular)

        return self._dedupe(tokens)

    def _singularize_token(self, token: str) -> str:
        token = str(token or "").strip().lower()

        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"

        if len(token) > 3 and token.endswith("s"):
            return token[:-1]

        return token

    def _display_poi_terms(self, intent: dict[str, Any]) -> list[str]:
        # Clean reporting only. Selection stays autonomous LLM/RAG + embedding based.
        values = []

        primary_categories = self._dedupe(
            (intent.get("requested_categories") or [])
            + (intent.get("matched_available_poi_types") or [])
            + (intent.get("canonical_categories") or [])
        )

        raw_terms = self._dedupe(intent.get("poi_terms") or [])

        values.extend(primary_categories)

        noisy_terms = {
            "food",
            "food street",
            "food streets",
            "fast food",
            "fast food restaurant",
            "restaurant",
            "restaurants",
            "dining",
            "casual",
            "casual dining",
            "casual diners",
            "eatery",
            "eateries",
            "takeaway",
            "lunch",
            "dinner",
            "dinner time",
            "shawarma",
            "burger",
            "pizza",
            "middle eastern",
        }

        # Keep support terms only if they are directly useful for human review.
        for term in raw_terms:
            norm_term = self._norm(term)
            if not norm_term or norm_term in noisy_terms:
                continue
            values.append(norm_term)
            if "_" in norm_term:
                values.append(norm_term.replace("_", " "))

        # Compact semantic alias for cafe-like intent without broad taxonomy expansion.
        if "cafe" in primary_categories and "coffee" not in values:
            values.append("coffee")

        return self._dedupe(values)[:8]


    def _column(self, df: pd.DataFrame, *names: str) -> str | None:
        column_map = {str(col).lower(): col for col in df.columns}
        for name in names:
            if name.lower() in column_map:
                return column_map[name.lower()]
        return None

    def _dedupe(self, values: list[Any]) -> list[str]:
        output = []
        seen = set()

        for value in values or []:
            text = self._norm(value)
            if text and text not in seen:
                output.append(text)
                seen.add(text)

        return output

    def _norm(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        text = text.replace("_", " ").replace("-", " ")
        text = re.sub(r"[^a-z0-9\s,]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _empty_report(self, prompt: str, reason: str) -> dict[str, Any]:
        return {
            "status": "completed",
            "filter_mode": reason,
            "selector_mode": "autonomous_hybrid_rag_embedding_selector",
            "locations_detected": [],
            "poi_terms_detected": [],
            "dayparts_detected": [],
            "coverage_warnings": [reason],
            "selected_count": 0,
            "prompt": prompt,
        }
