from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class LocalSemanticIntentUnavailableError(RuntimeError):
    """Raised when the configured local embedding model cannot run."""


class LocalSemanticIntentService:
    """
    Provider-independent semantic intent resolver.

    This layer:
    - Uses a local sentence-transformer model.
    - Receives only privacy-safe cohort metadata.
    - Derives locations from available safe source coverage.
    - Maps unrestricted natural language to canonical audience concepts.
    - Preserves multiple explicitly requested categories.
    - Never decides whether an audience may be exported.

    Privacy, freshness, approval, and export decisions remain deterministic.
    """

    _model = None
    _model_name: str | None = None
    _model_lock = threading.Lock()

    CATEGORY_ONTOLOGY: dict[str, tuple[str, ...]] = {
        "cafe": (
            "people visiting cafés and coffee shops",
            "customers stopping for espresso, coffee, or hot drinks",
            "short refreshment visits for beverages and light snacks",
        ),
        "restaurant": (
            "people visiting restaurants for dining and meals",
            "customers having lunch, dinner, or prepared food",
            "visitors to food venues and meal takeaway locations",
        ),
        "retail": (
            "people shopping at malls and retail stores",
            "customers browsing stores and purchasing products",
            "visitors to fashion stores, department stores, and malls",
        ),
        "fitness": (
            "people visiting gyms and fitness centres",
            "customers attending exercise, yoga, and health clubs",
            "visitors engaged in physical training and workouts",
        ),
        "nightlife": (
            "people visiting bars, clubs, and nightlife venues",
            "evening entertainment and social venue visitors",
        ),
        "office": (
            "people visiting offices, coworking spaces, and business centres",
            "professionals present at corporate and flexible workspaces",
        ),
        "healthcare": (
            "people visiting clinics, pharmacies, and healthcare facilities",
            "patients and visitors to medical service locations",
        ),
        "beauty": (
            "people visiting salons, spas, and personal-care venues",
            "customers seeking beauty and wellness services",
        ),
        "entertainment": (
            "people visiting entertainment and leisure venues",
            "visitors to cinemas, attractions, gaming, and recreation venues",
        ),
        "travel": (
            "people visiting airports, stations, hotels, and travel venues",
            "travellers using transport and accommodation locations",
        ),
    }

    DAYPART_ONTOLOGY: dict[str, tuple[str, ...]] = {
        "morning": (
            "activities early in the day before normal work begins",
            "breakfast-time and morning commute visits",
            "visits shortly after the day starts",
        ),
        "afternoon": (
            "activities occurring after midday and before evening",
            "lunchtime and later daytime visits",
        ),
        "evening": (
            "activities occurring at the end of the working day",
            "visits made after professional duties before returning home",
            "early-night activities following normal office hours",
        ),
        "night": (
            "late-night activities after the evening period",
            "visits during late hours and overnight",
        ),
        "weekend": (
            "activities taking place on Saturday or Sunday",
            "leisure visits during non-working weekend days",
        ),
        "weekday": (
            "activities taking place during normal working days",
            "Monday through Friday behaviour",
        ),
    }


    TEMPORAL_EVIDENCE_ONTOLOGY: dict[
        str,
        tuple[str, ...],
    ] = {
        "specified": (
            "the user explicitly states when the visit happens",
            (
                "the request contains a time constraint, "
                "schedule, daypart, or before or after context"
            ),
            (
                "audience behaviour is limited to a stated "
                "time window or stage of the working day"
            ),
        ),
        "unspecified": (
            (
                "the user asks who or where to target but "
                "does not say when the visit happens"
            ),
            (
                "no time of day, schedule, shift context, "
                "or daypart is requested"
            ),
            (
                "an audience request without a temporal "
                "constraint"
            ),
        ),
    }

    QUALITY_EVIDENCE_ONTOLOGY: dict[
        str,
        tuple[str, ...],
    ] = {
        "audience_high": (
            "the user explicitly asks for a high quality audience",
            "the request is for premium, best, or robust audience cohorts",
            "an explicit focus on targeting high-value, top-tier, or reliable audiences",
        ),
        "audience_broad": (
            "the user explicitly asks for a broad audience",
            "the request is for maximum reach, scale, or high volume audiences",
            "an explicit focus on widespread audience coverage",
        ),
        "business_premium": (
            "the user describes their own business or product as premium",
            "the advertiser states they are a luxury or high-end brand",
            "a self-description identifying the advertiser's service as the best",
        ),
        "neutral": (
            "the request does not specify any audience quality, scale, or performance requirements",
            "no mention of premium, broad, or high-value audience",
        ),
    }

    POI_CATEGORY_ALIASES: dict[str, set[str]] = {
        "cafe": {
            "cafe",
            "coffee",
            "coffee_shop",
            "tea_house",
        },
        "restaurant": {
            "restaurant",
            "food",
            "food_court",
            "meal_takeaway",
            "fast_food",
        },
        "retail": {
            "retail",
            "store",
            "shopping_mall",
            "department_store",
            "fashion",
            "clothing_store",
            "supermarket",
        },
        "fitness": {
            "fitness",
            "gym",
            "health_club",
            "yoga_studio",
            "sports_centre",
        },
        "nightlife": {
            "bar",
            "club",
            "night_club",
            "nightlife",
            "pub",
        },
        "office": {
            "office",
            "coworking",
            "business_centre",
            "corporate_office",
        },
        "healthcare": {
            "clinic",
            "hospital",
            "pharmacy",
            "healthcare",
            "medical_centre",
        },
        "beauty": {
            "salon",
            "spa",
            "beauty",
            "personal_care",
        },
        "entertainment": {
            "entertainment",
            "cinema",
            "theatre",
            "casino",
            "attraction",
            "recreation",
        },
        "travel": {
            "airport",
            "hotel",
            "station",
            "travel",
            "transport",
        },
    }

    def resolve(
        self,
        *,
        prompt: str,
        rag_context: dict[str, Any],
    ) -> dict[str, Any]:
        clean_prompt = str(prompt or "").strip()

        if not clean_prompt:
            raise ValueError("Prompt cannot be empty.")

        available_locations = self._clean_values(
            rag_context.get("available_locations")
        )
        requested_location_candidates = (
            self._clean_values(
                rag_context.get(
                    "requested_location_candidates"
                )
            )
        )
        available_pois = self._clean_values(
            rag_context.get("available_poi_types")
        )
        available_dayparts = self._clean_values(
            rag_context.get("available_dayparts")
        )

        location_candidates = self._dedupe(
            available_locations
            + requested_location_candidates
        )

        locations = self._extract_available_locations(
            prompt=clean_prompt,
            available_locations=location_candidates,
        )

        covered_locations = []

        for location in locations:
            location_norm = self._space_norm(location)

            if any(
                (
                    location_norm
                    == self._space_norm(available)
                    or location_norm
                    in self._space_norm(available)
                    or self._space_norm(available)
                    in location_norm
                )
                for available in available_locations
            ):
                covered_locations.append(location)

        missing_location_coverage = [
            location
            for location in locations
            if location not in covered_locations
        ]

        query_segments = self._build_query_segments(clean_prompt)

        category_descriptions = self._build_category_descriptions(
            available_pois
        )

        category_scores = self._score_labels(
            query_texts=query_segments,
            label_descriptions=category_descriptions,
        )

        direct_categories = self._direct_available_categories(
            prompt=clean_prompt,
            available_pois=available_pois,
        )

        categories = self._select_categories(
            category_scores=category_scores,
            direct_categories=direct_categories,
        )

        direct_dayparts = self._direct_dayparts(
            prompt=clean_prompt,
            available_dayparts=available_dayparts,
        )

        # Build temporal segments once and reuse for both
        # evidence and daypart classification so they share
        # the same semantic representation.
        temporal_query_segments = (
            self._build_temporal_query_segments(
                clean_prompt
            )
        )

        temporal_evidence_scores = (
            self._score_labels(
                query_texts=temporal_query_segments,
                label_descriptions=(
                    self.TEMPORAL_EVIDENCE_ONTOLOGY
                ),
            )
        )

        daypart_scores = self._score_labels(
            query_texts=temporal_query_segments,
            label_descriptions=(
                self._build_daypart_descriptions(
                    available_dayparts
                )
            ),
        )

        dayparts = self._select_dayparts(
            daypart_scores=daypart_scores,
            direct_dayparts=direct_dayparts,
            temporal_evidence_scores=(
                temporal_evidence_scores
            ),
        )

        matched_available_pois = [
            poi
            for poi in available_pois
            if self._canonical_category_for_poi(poi)
            in categories
        ]

        poi_terms = self._dedupe(
            matched_available_pois + categories
        )

        confidence = self._calculate_confidence(
            locations=locations,
            categories=categories,
            dayparts=dayparts,
            category_scores=category_scores,
            daypart_scores=daypart_scores,
            direct_categories=direct_categories,
            direct_dayparts=direct_dayparts,
        )

        quality_evidence_scores = self._score_labels(
            query_texts=query_segments,
            label_descriptions=self.QUALITY_EVIDENCE_ONTOLOGY,
        )

        quality_intent = self._select_quality_intent(
            quality_evidence_scores=quality_evidence_scores
        )

        return {
            "business_intent": (
                "+".join(categories)
                if categories
                else "unknown_business_intent"
            ),
            "audience_goal": "visitor_footfall_audience",
            "locations": locations,
            "requested_categories": categories,
            "matched_available_poi_types": matched_available_pois,
            "poi_terms": poi_terms,
            "dayparts": dayparts,
            "quality_intent": quality_intent,
            "fallback_tolerance": "strict_exact_first",
            "data_gap_likely": bool(
                missing_location_coverage
            ),
            "missing_location_coverage": (
                missing_location_coverage
            ),
            "confidence_score": confidence,
            "reasoning_summary": (
                "Local semantic intent resolved from the user prompt "
                "and privacy-safe available cohort metadata."
            ),
            "_semantic_scores": {
                "categories": {
                    key: round(value, 4)
                    for key, value in category_scores.items()
                },
                "dayparts": {
                    key: round(value, 4)
                    for key, value in daypart_scores.items()
                },
                "temporal_evidence": {
                    key: round(value, 4)
                    for key, value
                    in temporal_evidence_scores.items()
                },
                "quality_evidence": {
                    key: round(value, 4)
                    for key, value
                    in quality_evidence_scores.items()
                },
            },
        }

    def _build_category_descriptions(
        self,
        available_pois: list[str],
    ) -> dict[str, tuple[str, ...]]:
        descriptions = {
            key: list(values)
            for key, values in self.CATEGORY_ONTOLOGY.items()
        }

        for poi in available_pois:
            canonical = self._canonical_category_for_poi(poi)
            descriptions.setdefault(canonical, [])

            readable_poi = poi.replace("_", " ")

            descriptions[canonical].append(
                f"people visiting {readable_poi}"
            )
            descriptions[canonical].append(
                f"audience associated with {readable_poi} venues"
            )

        return {
            key: tuple(self._dedupe(values))
            for key, values in descriptions.items()
        }



    def _build_temporal_query_segments(
        self,
        prompt: str,
    ) -> list[str]:
        """
        Produce generic semantic spans for temporal inference.

        The model compares the whole request, natural clauses,
        and overlapping word windows against the daypart
        ontology. No prompt-specific phrase mapping is used.
        """
        normalized = self._space_norm(prompt)

        if not normalized:
            return []

        min_words = max(
            2,
            self._safe_env_int(
                "LOCAL_SEMANTIC_TEMPORAL_MIN_WORDS",
                default=3,
                min_val=2,
                max_val=20,
            ),
        )
        max_words = max(
            min_words,
            self._safe_env_int(
                "LOCAL_SEMANTIC_TEMPORAL_MAX_WORDS",
                default=8,
                min_val=2,
                max_val=30,
            ),
        )
        max_segments = max(
            16,
            self._safe_env_int(
                "LOCAL_SEMANTIC_TEMPORAL_MAX_SEGMENTS",
                default=96,
                min_val=16,
                max_val=500,
            ),
        )

        output: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            clean_value = self._space_norm(value)

            if (
                not clean_value
                or clean_value in seen
                or len(output) >= max_segments
            ):
                return

            seen.add(clean_value)
            output.append(clean_value)

        # Preserve the full semantic context.
        add(normalized)

        # Add natural clauses without interpreting their meaning.
        clauses = re.split(
            (
                r"[,;:.!?]+|"
                r"\b(?:and|or|but|while|whereas)\b"
            ),
            str(prompt),
            flags=re.IGNORECASE,
        )

        for clause in clauses:
            if len(
                self._space_norm(clause).split()
            ) >= 2:
                add(clause)

        # Add overlapping semantic windows. This lets abstract
        # phrases such as the end of a shift retain their meaning
        # without requiring an exact keyword rule.
        words = normalized.split()
        upper_size = min(
            max_words,
            len(words),
        )

        for size in range(
            upper_size,
            min_words - 1,
            -1,
        ):
            for start in range(
                0,
                len(words) - size + 1,
            ):
                add(
                    " ".join(
                        words[start : start + size]
                    )
                )

                if len(output) >= max_segments:
                    return output

        return output

    def _build_daypart_descriptions(
        self,
        available_dayparts: list[str],
    ) -> dict[str, tuple[str, ...]]:
        """
        Build a semantic label space from privacy-safe source
        coverage.

        Explicit user language may still request any supported
        daypart. For implicit semantic inference, labels that are
        impossible in the available source context are excluded.
        """
        available = self._dedupe(
            available_dayparts
        )

        candidates = (
            available
            if available
            else list(self.DAYPART_ONTOLOGY)
        )

        descriptions: dict[
            str,
            tuple[str, ...],
        ] = {}

        for daypart in candidates:
            if daypart in self.DAYPART_ONTOLOGY:
                descriptions[daypart] = (
                    self.DAYPART_ONTOLOGY[daypart]
                )
                continue

            readable = daypart.replace("_", " ")

            descriptions[daypart] = (
                f"activities during the {readable} period",
                f"visits associated with {readable} time",
            )

        # A semantic abstention class prevents the resolver from
        # inventing a daypart when the user supplied no temporal
        # meaning.
        descriptions["_unspecified"] = (
            "visitors at any time without a requested time period",
            "general audience behaviour with no time-of-day intent",
            "a request that does not mention when visits occur",
        )

        return descriptions

    def _build_query_segments(
        self,
        prompt: str,
    ) -> list[str]:
        segments = [prompt]

        pieces = re.split(
            r"[,;]|\b(?:and|or|with|plus)\b",
            prompt,
            flags=re.IGNORECASE,
        )

        segments.extend(
            piece.strip()
            for piece in pieces
            if len(piece.strip()) >= 3
        )

        return self._dedupe(segments)

    def _extract_available_locations(
        self,
        *,
        prompt: str,
        available_locations: list[str],
    ) -> list[str]:
        prompt_norm = self._space_norm(prompt)
        wrapped_prompt = f" {prompt_norm} "

        matched = []

        for location in sorted(
            available_locations,
            key=len,
            reverse=True,
        ):
            location_norm = self._space_norm(location)

            if not location_norm:
                continue

            if f" {location_norm} " in wrapped_prompt:
                matched.append(location)

        return self._dedupe(matched)

    def _direct_available_categories(
        self,
        *,
        prompt: str,
        available_pois: list[str],
    ) -> list[str]:
        prompt_norm = self._space_norm(prompt)
        wrapped_prompt = f" {prompt_norm} "

        categories = []

        for poi in available_pois:
            poi_norm = self._space_norm(poi)

            if poi_norm and f" {poi_norm} " in wrapped_prompt:
                categories.append(
                    self._canonical_category_for_poi(poi)
                )

        return self._dedupe(categories)

    def _direct_dayparts(
        self,
        *,
        prompt: str,
        available_dayparts: list[str],
    ) -> list[str]:
        prompt_norm = self._space_norm(prompt)
        wrapped_prompt = f" {prompt_norm} "

        candidates = self._dedupe(
            available_dayparts
            + list(self.DAYPART_ONTOLOGY)
        )

        return [
            daypart
            for daypart in candidates
            if f" {self._space_norm(daypart)} "
            in wrapped_prompt
        ]

    def _select_quality_intent(
        self,
        *,
        quality_evidence_scores: dict[str, float],
    ) -> str:
        high_score = quality_evidence_scores.get("audience_high", 0.0)
        broad_score = quality_evidence_scores.get("audience_broad", 0.0)
        business_score = quality_evidence_scores.get("business_premium", 0.0)
        neutral_score = quality_evidence_scores.get("neutral", 0.0)

        min_confidence = self._safe_env_float("AUDIENCE_QUALITY_SEMANTIC_MIN_CONFIDENCE", default=0.20)
        margin = self._safe_env_float("AUDIENCE_QUALITY_SEMANTIC_MARGIN", default=0.02)

        if (
            high_score >= min_confidence
            and high_score > (business_score + margin)
            and high_score > (neutral_score + margin)
            and high_score >= broad_score
        ):
            return "high"

        if (
            broad_score >= min_confidence
            and broad_score > (neutral_score + margin)
            and broad_score > high_score
        ):
            return "broad"

        return "balanced"

    def _select_categories(
        self,
        *,
        category_scores: dict[str, float],
        direct_categories: list[str],
    ) -> list[str]:
        if not category_scores and not direct_categories:
            return []

        threshold = self._safe_env_float(
            "LOCAL_SEMANTIC_CATEGORY_MIN_SCORE",
            default=0.34,
            min_val=0.0,
            max_val=1.0,
        )
        margin = self._safe_env_float(
            "LOCAL_SEMANTIC_MULTI_LABEL_MARGIN",
            default=0.07,
            min_val=0.0,
            max_val=1.0,
        )
        max_labels = max(
            1,
            self._safe_env_int(
                "LOCAL_SEMANTIC_MAX_CATEGORIES",
                default=4,
                min_val=1,
                max_val=20,
            ),
        )

        highest = max(
            category_scores.values(),
            default=0.0,
        )

        semantic_categories = [
            category
            for category, score in sorted(
                category_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if score >= threshold
            and score >= highest - margin
        ]

        selected = self._dedupe(
            direct_categories + semantic_categories
        )

        return selected[:max_labels]

    # ── Clock-time vs calendar temporal dimensions ──────────
    _CLOCK_TIME_LABELS = frozenset(
        {"morning", "afternoon", "evening", "night"}
    )
    _CALENDAR_LABELS = frozenset(
        {"weekday", "weekend"}
    )

    def _select_dayparts(
        self,
        *,
        daypart_scores: dict[str, float],
        direct_dayparts: list[str],
        temporal_evidence_scores: dict[
            str,
            float,
        ],
    ) -> list[str]:
        """Select dayparts using separated temporal dimensions.

        Clock-time labels (morning/afternoon/evening/night) and
        calendar qualifiers (weekday/weekend) are evaluated
        independently so that a high weekday score cannot displace
        a valid evening interpretation.

        When the evidence gate is uncertain but a clock-time
        daypart has a strong, clear score that beats both
        competing clock-time labels and the abstention class, a
        configurable override margin allows the result through.
        This avoids rejecting semantically meaningful temporal
        phrases while preserving venue-category abstention.
        """
        # Directly named time periods remain authoritative.
        if direct_dayparts:
            return self._dedupe(direct_dayparts)

        specified_score = float(
            temporal_evidence_scores.get(
                "specified",
                0.0,
            )
        )
        unspecified_score = float(
            temporal_evidence_scores.get(
                "unspecified",
                0.0,
            )
        )

        evidence_threshold = self._safe_env_float(
            "LOCAL_SEMANTIC_TEMPORAL_EVIDENCE_MIN_SCORE",
            default=0.24,
            min_val=0.0,
            max_val=1.0,
        )
        evidence_margin = self._safe_env_float(
            "LOCAL_SEMANTIC_TEMPORAL_EVIDENCE_MIN_MARGIN",
            default=0.02,
            min_val=0.0,
            max_val=1.0,
        )
        override_margin = self._safe_env_float(
            "LOCAL_SEMANTIC_TEMPORAL_OVERRIDE_MARGIN",
            default=0.05,
            min_val=0.0,
            max_val=1.0,
        )

        # ── Separate semantic dimensions ──────────────────
        clock_time_scores = {
            label: score
            for label, score in daypart_scores.items()
            if label in self._CLOCK_TIME_LABELS
        }
        calendar_scores = {
            label: score
            for label, score in daypart_scores.items()
            if label in self._CALENDAR_LABELS
        }

        unspecified_daypart_score = float(
            daypart_scores.get(
                "_unspecified",
                0.0,
            )
        )

        threshold = self._safe_env_float(
            "LOCAL_SEMANTIC_DAYPART_MIN_SCORE",
            default=0.28,
            min_val=0.0,
            max_val=1.0,
        )
        min_margin = self._safe_env_float(
            "LOCAL_SEMANTIC_DAYPART_MIN_MARGIN",
            default=0.025,
            min_val=0.0,
            max_val=1.0,
        )

        # Venue associations such as coffee → morning must not
        # create a time constraint. First prove that the user
        # actually expressed temporal meaning.
        evidence_clear = (
            specified_score >= evidence_threshold
            and specified_score - unspecified_score
            >= evidence_margin
        )

        selected: list[str] = []

        # ── Clock-time selection ──────────────────────────
        clock_time_selected = self._select_from_dimension(
            dimension_scores=clock_time_scores,
            unspecified_score=unspecified_daypart_score,
            threshold=threshold,
            min_margin=min_margin,
            evidence_clear=evidence_clear,
            override_margin=override_margin,
            evidence_threshold=evidence_threshold,
            specified_score=specified_score,
        )
        if clock_time_selected:
            selected.append(clock_time_selected)

        # ── Calendar selection ────────────────────────────
        calendar_selected = self._select_from_dimension(
            dimension_scores=calendar_scores,
            unspecified_score=unspecified_daypart_score,
            threshold=threshold,
            min_margin=min_margin,
            evidence_clear=evidence_clear,
            override_margin=override_margin,
            evidence_threshold=evidence_threshold,
            specified_score=specified_score,
        )
        if calendar_selected:
            selected.append(calendar_selected)

        logger.debug(
            "temporal_selection: "
            "specified=%.4f unspecified=%.4f "
            "evidence_clear=%s "
            "clock_top=%s calendar_top=%s "
            "selected=%s",
            specified_score,
            unspecified_score,
            evidence_clear,
            clock_time_selected,
            calendar_selected,
            selected,
        )

        return selected

    def _select_from_dimension(
        self,
        *,
        dimension_scores: dict[str, float],
        unspecified_score: float,
        threshold: float,
        min_margin: float,
        evidence_clear: bool,
        override_margin: float,
        evidence_threshold: float,
        specified_score: float,
    ) -> str | None:
        """Select the best label within one temporal dimension.

        Returns the winning label or None if abstention wins.
        When evidence is ambiguous (not clearly specified over
        unspecified) a stronger override margin is required on
        the daypart score itself to proceed.  The override is
        blocked entirely when the evidence classifier clearly
        favours 'unspecified' — i.e. the request has no
        temporal intent at all.
        """
        if not dimension_scores:
            return None

        ranked = sorted(
            dimension_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_label, top_score = ranked[0]
        second_score = (
            ranked[1][1]
            if len(ranked) > 1
            else 0.0
        )

        if top_score < threshold:
            return None

        # The label must clearly beat its within-dimension
        # competitors.
        if (
            len(ranked) > 1
            and top_score - second_score
            < min_margin
        ):
            return None

        # The label must beat the semantic abstention class.
        if (
            unspecified_score > 0
            and top_score - unspecified_score
            < min_margin
        ):
            return None

        if evidence_clear:
            # Evidence gate passed — accept the label.
            return top_label

        # ── Semantic override path ────────────────────────
        # Evidence is ambiguous. Allow the label only when
        # the daypart signal is strong enough to be credible
        # on its own.  This requires:
        # - specified_score meets the evidence threshold;
        # - the daypart clearly beats its within-dimension
        #   competitors by the override margin;
        # - the daypart clearly beats the abstention class
        #   by a doubled margin (to compensate for lacking
        #   clear evidence separation).
        # Venue-category correlations (e.g. coffee → morning)
        # produce small abstention margins and are rejected.
        # Genuine temporal phrases (e.g. after finishing work
        # → evening) produce large abstention margins and pass.
        if specified_score < evidence_threshold:
            return None

        if (
            top_score - second_score
            < override_margin
        ):
            return None

        strong_abstention_margin = override_margin * 2.0

        if (
            unspecified_score > 0
            and top_score - unspecified_score
            < strong_abstention_margin
        ):
            return None

        logger.debug(
            "temporal_override: label=%s score=%.4f "
            "second=%.4f unspecified=%.4f "
            "specified=%.4f",
            top_label,
            top_score,
            second_score,
            unspecified_score,
            specified_score,
        )

        return top_label

    def _score_labels(
        self,
        *,
        query_texts: list[str],
        label_descriptions: dict[str, tuple[str, ...]],
    ) -> dict[str, float]:
        if not query_texts or not label_descriptions:
            return {}

        flat_descriptions: list[str] = []
        label_ranges: dict[str, tuple[int, int]] = {}

        for label, descriptions in label_descriptions.items():
            start = len(flat_descriptions)

            flat_descriptions.extend(
                str(description)
                for description in descriptions
                if str(description).strip()
            )

            label_ranges[label] = (
                start,
                len(flat_descriptions),
            )

        if not flat_descriptions:
            return {}

        all_texts = query_texts + flat_descriptions
        vectors = self._encode(all_texts)

        query_vectors = vectors[: len(query_texts)]
        description_vectors = vectors[len(query_texts) :]

        scores = {}

        for label, (start, end) in label_ranges.items():
            if end <= start:
                continue

            label_vectors = description_vectors[start:end]
            similarities = query_vectors @ label_vectors.T

            scores[label] = float(
                np.max(similarities)
            )

        return scores

    def _encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        model = self._get_model()

        try:
            vectors = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise LocalSemanticIntentUnavailableError(
                f"Local semantic encoding failed: "
                f"{type(exc).__name__}"
            ) from exc

        return np.asarray(
            vectors,
            dtype=np.float32,
        )

    @classmethod
    def _get_model(cls):
        model_name = os.getenv(
            "LOCAL_SEMANTIC_MODEL",
            "all-MiniLM-L6-v2",
        ).strip()

        with cls._model_lock:
            if (
                cls._model is not None
                and cls._model_name == model_name
            ):
                return cls._model

            try:
                from sentence_transformers import (
                    SentenceTransformer,
                )

                cls._model = SentenceTransformer(
                    model_name,
                    local_files_only=True,
                )
                cls._model_name = model_name
            except Exception as exc:
                raise LocalSemanticIntentUnavailableError(
                    "Local semantic model is unavailable: "
                    f"{type(exc).__name__}"
                ) from exc

        return cls._model

    def _canonical_category_for_poi(
        self,
        poi: str,
    ) -> str:
        normalized = self._underscore_norm(poi)

        for category, aliases in (
            self.POI_CATEGORY_ALIASES.items()
        ):
            if normalized in aliases:
                return category

        # Provider-specific POI types are folded into stable
        # business families. They must not become separate
        # requested categories merely because they exist in
        # source metadata.
        if (
            normalized.endswith("_restaurant")
            or normalized
            in {
                "restaurant",
                "food",
                "food_court",
                "meal_takeaway",
                "fast_food",
            }
        ):
            return "restaurant"

        if (
            normalized.endswith("_store")
            or normalized
            in {
                "retail",
                "shopping_mall",
                "department_store",
                "fashion",
            }
        ):
            return "retail"

        if normalized in {
            "cafe",
            "coffee",
            "coffee_shop",
            "tea_house",
        }:
            return "cafe"

        if normalized in {
            "gym",
            "fitness",
            "fitness_center",
            "fitness_centre",
            "health_club",
            "yoga_studio",
        }:
            return "fitness"

        if normalized in {
            "bar",
            "pub",
            "club",
            "night_club",
            "nightlife",
        }:
            return "nightlife"

        return normalized

    def _calculate_confidence(
        self,
        *,
        locations: list[str],
        categories: list[str],
        dayparts: list[str],
        category_scores: dict[str, float],
        daypart_scores: dict[str, float],
        direct_categories: list[str],
        direct_dayparts: list[str],
    ) -> float:
        signals = []

        if locations:
            signals.append(0.95)

        if categories:
            selected_scores = [
                category_scores.get(category, 0.0)
                for category in categories
            ]

            category_confidence = max(
                selected_scores,
                default=0.0,
            )

            if direct_categories:
                category_confidence = max(
                    category_confidence,
                    0.92,
                )

            signals.append(category_confidence)

        if dayparts:
            daypart_confidence = max(
                (
                    daypart_scores.get(daypart, 0.0)
                    for daypart in dayparts
                ),
                default=0.0,
            )

            if direct_dayparts:
                daypart_confidence = max(
                    daypart_confidence,
                    0.94,
                )

            signals.append(daypart_confidence)

        if not signals:
            return 0.0

        confidence = float(sum(signals) / len(signals))

        # This is a routing confidence, not an export decision.
        # A valid location plus a semantic category/daypart should
        # be eligible for deterministic downstream validation.
        if locations and (categories or dayparts):
            confidence = max(confidence, 0.72)

        return round(
            min(max(confidence, 0.0), 0.95),
            3,
        )

    @staticmethod
    def _safe_env_float(
        name: str,
        *,
        default: float,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> float:
        """Parse an environment variable as a bounded float.

        Falls back to *default* when the variable is missing or
        malformed. The result is always clamped to [min_val, max_val].
        """
        raw = os.getenv(name)
        if raw is None:
            return max(min_val, min(default, max_val))
        try:
            value = float(raw)
        except (ValueError, TypeError):
            logger.warning(
                "Malformed env %s=%r, using default %.4f",
                name,
                raw,
                default,
            )
            value = default
        return max(min_val, min(value, max_val))

    @staticmethod
    def _safe_env_int(
        name: str,
        *,
        default: int,
        min_val: int = 0,
        max_val: int = 10000,
    ) -> int:
        """Parse an environment variable as a bounded int.

        Falls back to *default* when the variable is missing or
        malformed. The result is always clamped to [min_val, max_val].
        """
        raw = os.getenv(name)
        if raw is None:
            return max(min_val, min(default, max_val))
        try:
            value = int(raw)
        except (ValueError, TypeError):
            logger.warning(
                "Malformed env %s=%r, using default %d",
                name,
                raw,
                default,
            )
            value = default
        return max(min_val, min(value, max_val))

    def _clean_values(
        self,
        values: Any,
    ) -> list[str]:
        if not isinstance(values, list):
            return []

        return self._dedupe(
            [
                self._underscore_norm(value)
                for value in values
                if str(value or "").strip()
            ]
        )

    def _space_norm(
        self,
        value: Any,
    ) -> str:
        text = str(value or "").lower()
        text = text.replace("_", " ").replace("-", " ")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _underscore_norm(
        self,
        value: Any,
    ) -> str:
        return self._space_norm(value).replace(" ", "_")

    def _dedupe(
        self,
        values: list[Any],
    ) -> list[str]:
        seen = set()
        output = []

        for value in values:
            item = str(value or "").strip().lower()

            if not item or item in seen:
                continue

            seen.add(item)
            output.append(item)

        return output
