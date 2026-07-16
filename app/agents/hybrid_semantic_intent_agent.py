from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from app.agents.semantic_prompt_intelligence_agent import SemanticPromptIntelligenceAgent
from app.services.llm_model_router_service import (
    LLMModelRouterError,
    LLMModelRouterService,
    LLMResponseValidationError,
)


LLMCallable = Callable[[list[dict[str, str]], dict[str, Any]], str]


@dataclass
class HybridIntentConfig:
    enabled: bool
    provider: str
    model: str
    api_key: str
    base_url: str
    timeout_seconds: int
    max_context_items: int
    max_tokens: int
    min_llm_confidence: float


class HybridSemanticIntentAgent:
    """
    Hybrid LLM/RAG semantic intent resolver.

    Responsibilities:
    - Build RAG context from privacy-safe cohort metadata only.
    - Ask LLM to understand natural business language.
    - Validate/sanitize LLM JSON deterministically.
    - Fall back to deterministic SemanticPromptIntelligenceAgent if LLM fails.

    Important:
    - LLM never decides export.
    - LLM never receives raw MAIDs, hashed IDs, raw observations, raw lat/lng, email, or phone.
    - Export safety remains deterministic in orchestrator/ranking/guardrails.
    """

    def __init__(self, llm_client: Optional[LLMCallable] = None) -> None:
        self.llm_client = llm_client

    def resolve(
        self,
        *,
        prompt: str,
        safe_cohorts: pd.DataFrame,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        fallback = SemanticPromptIntelligenceAgent().analyze_prompt(prompt)

        config = self._load_config()
        rag_context = self._build_safe_rag_context(safe_cohorts)
        llm_rag_context = self._build_llm_rag_context(
            prompt=prompt,
            rag_context=rag_context,
            max_context_items=config.max_context_items,
        )

        if not config.enabled:
            result = self._with_hybrid_metadata(
                fallback,
                rag_context=rag_context,
                llm_used=False,
                resolver_mode="deterministic_fallback_llm_disabled",
                llm_error=None,
            )
            result = self._attach_llm_router_metadata(
                result=result,
                telemetry={
                    "llm_attempted": False,
                    "llm_used": False,
                    "attempt_count": 0,
                    "total_latency_ms": 0,
                    "fallback_used": True,
                    "fallback_reason": "llm_disabled",
                    "attempts": [],
                },
            )
            self._write_result(result, output_dir)
            return result

        try:
            routed = self._call_llm(
                prompt=prompt,
                rag_context=llm_rag_context,
                config=config,
            )
            llm_json = routed["validated"]

            result = self._validate_and_merge_llm_intent(
                prompt=prompt,
                fallback=fallback,
                llm_json=llm_json,
                rag_context=rag_context,
                min_confidence=config.min_llm_confidence,
            )
            result = self._attach_llm_router_metadata(
                result=result,
                telemetry=routed.get("telemetry") or {},
            )

            self._write_result(result, output_dir)
            return result

        except LLMModelRouterError as exc:
            telemetry = exc.telemetry or {}
            resolver_mode = self._fallback_resolver_mode(
                telemetry
            )
            result = self._with_hybrid_metadata(
                fallback,
                rag_context=rag_context,
                llm_used=False,
                resolver_mode=resolver_mode,
                llm_error=self._router_error_summary(
                    telemetry
                ),
            )
            result = self._attach_llm_router_metadata(
                result=result,
                telemetry=telemetry,
            )
            self._write_result(result, output_dir)
            return result

        except Exception as exc:
            result = self._with_hybrid_metadata(
                fallback,
                rag_context=rag_context,
                llm_used=False,
                resolver_mode="deterministic_fallback_llm_failed",
                llm_error=str(exc),
            )
            result = self._attach_llm_router_metadata(
                result=result,
                telemetry={
                    "llm_attempted": True,
                    "llm_used": False,
                    "attempt_count": 1,
                    "total_latency_ms": 0,
                    "fallback_used": True,
                    "fallback_reason": "unhandled_llm_error",
                    "attempts": [],
                },
            )
            self._write_result(result, output_dir)
            return result

    def _load_config(self) -> HybridIntentConfig:
        enabled = os.getenv("ENABLE_LLM_INTENT", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        provider = os.getenv("LLM_INTENT_PROVIDER", "openrouter").strip().lower()
        model = os.getenv("LLM_INTENT_MODEL", "").strip()

        api_key = (
            os.getenv("LLM_INTENT_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()

        if provider == "openrouter":
            base_url = os.getenv("LLM_INTENT_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
        elif provider == "openai":
            base_url = os.getenv("LLM_INTENT_BASE_URL", "https://api.openai.com/v1/chat/completions")
        else:
            base_url = os.getenv("LLM_INTENT_BASE_URL", "").strip()

        return HybridIntentConfig(
            enabled=enabled,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=int(os.getenv("LLM_INTENT_TIMEOUT_SECONDS", "20")),
            max_context_items=int(os.getenv("LLM_INTENT_MAX_CONTEXT_ITEMS", "80")),
            max_tokens=int(os.getenv("LLM_INTENT_MAX_TOKENS", "1000")),
            min_llm_confidence=float(os.getenv("LLM_INTENT_MIN_CONFIDENCE", "0.65")),
        )



    def _build_safe_rag_context(self, safe_cohorts: pd.DataFrame) -> dict[str, Any]:
        df = safe_cohorts.copy()

        safe_columns = [
            "location_name",
            "primary_poi_type",
            "created_day_part",
            "quality_score",
            "total_maid_volume",
            "privacy_status",
        ]

        available_columns = [column for column in safe_columns if column in df.columns]
        safe_df = df[available_columns].copy() if available_columns else pd.DataFrame()

        def unique_values(column: str) -> list[str]:
            if column not in safe_df.columns:
                return []
            return (
                safe_df[column]
                .dropna()
                .astype(str)
                .str.lower()
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

        poi_counts = {}
        if "primary_poi_type" in safe_df.columns:
            poi_counts = (
                safe_df["primary_poi_type"]
                .dropna()
                .astype(str)
                .str.lower()
                .str.strip()
                .value_counts()
                .head(80)
                .to_dict()
            )

        location_counts = {}
        if "location_name" in safe_df.columns:
            location_counts = (
                safe_df["location_name"]
                .dropna()
                .astype(str)
                .str.lower()
                .str.strip()
                .value_counts()
                .head(80)
                .to_dict()
            )

        safe_combinations = []
        required_combo_columns = {"location_name", "primary_poi_type", "created_day_part"}

        if required_combo_columns.issubset(set(safe_df.columns)):
            combo_df = (
                safe_df[["location_name", "primary_poi_type", "created_day_part"]]
                .dropna()
                .astype(str)
                .apply(lambda col: col.str.lower().str.strip())
                .drop_duplicates()
            )

            # Important:
            # Use ALL safe combinations for deterministic validation.
            # Do not truncate this list; truncation caused exact combos to be missed.
            for row in combo_df.to_dict(orient="records"):
                safe_combinations.append(
                    {
                        "location_name": row["location_name"],
                        "primary_poi_type": row["primary_poi_type"],
                        "created_day_part": row["created_day_part"],
                    }
                )

        return {
            "privacy_note": "Only privacy-safe cohort metadata is included. No raw MAIDs, hashed IDs, raw observations, raw lat/lng, email, phone, or individual rows.",
            "safe_cohort_count": int(len(safe_df)),
            "available_locations": unique_values("location_name"),
            "available_poi_types": unique_values("primary_poi_type"),
            "available_dayparts": unique_values("created_day_part"),
            "top_poi_counts": poi_counts,
            "top_location_counts": location_counts,
            "safe_available_combinations": safe_combinations,
        }


    def _build_llm_rag_context(
        self,
        *,
        prompt: str,
        rag_context: dict[str, Any],
        max_context_items: int,
    ) -> dict[str, Any]:
        # Compact only the network payload. The full privacy-safe context
        # remains available for deterministic validation after the LLM call.
        limit = max(8, int(max_context_items or 20))
        prompt_norm = self._match_norm(prompt)
        prompt_terms = {
            term
            for term in prompt_norm.split("_")
            if term
        }

        def score_text(value: Any) -> int:
            normalized = self._match_norm(value)
            if not normalized:
                return 0

            score = 0
            if normalized in prompt_norm:
                score += 20

            value_terms = {
                term
                for term in normalized.split("_")
                if term
            }
            score += len(
                prompt_terms.intersection(value_terms)
            ) * 3
            return score

        def compact_values(values: Any) -> list[str]:
            items = [
                str(item)
                for item in (values or [])
                if str(item).strip()
            ]
            ranked = sorted(
                enumerate(items),
                key=lambda pair: (
                    -score_text(pair[1]),
                    pair[0],
                ),
            )
            return [
                item
                for _, item in ranked[:limit]
            ]

        combinations = list(
            rag_context.get(
                "safe_available_combinations"
            )
            or []
        )

        scored_combinations = []
        for index, combo in enumerate(combinations):
            if not isinstance(combo, dict):
                continue

            score = (
                score_text(combo.get("location_name"))
                + score_text(combo.get("primary_poi_type"))
                + score_text(combo.get("created_day_part"))
            )
            scored_combinations.append(
                (score, index, dict(combo))
            )

        scored_combinations.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )
        compact_combinations = [
            combo
            for _, _, combo in scored_combinations[:limit]
        ]

        def compact_counts(value: Any) -> dict[str, Any]:
            if not isinstance(value, dict):
                return {}

            ranked = sorted(
                value.items(),
                key=lambda item: (
                    -score_text(item[0]),
                    -int(item[1] or 0),
                    str(item[0]),
                ),
            )
            return dict(ranked[:limit])

        return {
            "privacy_note": rag_context.get("privacy_note"),
            "safe_cohort_count": int(
                rag_context.get("safe_cohort_count") or 0
            ),
            "available_locations": compact_values(
                rag_context.get("available_locations")
            ),
            "available_poi_types": compact_values(
                rag_context.get("available_poi_types")
            ),
            "available_dayparts": compact_values(
                rag_context.get("available_dayparts")
            ),
            "top_poi_counts": compact_counts(
                rag_context.get("top_poi_counts")
            ),
            "top_location_counts": compact_counts(
                rag_context.get("top_location_counts")
            ),
            "safe_available_combinations": compact_combinations,
            "context_compaction": {
                "max_context_items": limit,
                "full_safe_combination_count": len(combinations),
                "sent_safe_combination_count": len(
                    compact_combinations
                ),
            },
        }

    def _call_llm(
        self,
        *,
        prompt: str,
        rag_context: dict[str, Any],
        config: HybridIntentConfig,
    ) -> dict[str, Any]:
        messages = self._build_messages(
            prompt=prompt,
            rag_context=rag_context,
        )

        def validator(content: str) -> dict[str, Any]:
            parsed = self._parse_json_object(content)
            confidence = self._safe_float(
                parsed.get("confidence_score"),
                default=0.0,
            )
            if confidence < config.min_llm_confidence:
                raise LLMResponseValidationError(
                    "low_confidence",
                    (
                        f"LLM confidence {confidence} below "
                        f"threshold {config.min_llm_confidence}."
                    ),
                )
            return parsed

        if self.llm_client:
            started = time.monotonic()
            content = self.llm_client(
                messages,
                {
                    "provider": config.provider,
                    "model": config.model,
                    "base_url": config.base_url,
                },
            )
            validated = validator(content)
            latency_ms = int(
                (time.monotonic() - started)
                * 1000
            )
            return {
                "content": content,
                "validated": validated,
                "telemetry": {
                    "llm_attempted": True,
                    "llm_used": True,
                    "success": True,
                    "provider_used": config.provider,
                    "model_used": config.model,
                    "attempt_count": 1,
                    "total_latency_ms": latency_ms,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "attempts": [
                        {
                            "provider": config.provider,
                            "model": config.model,
                            "attempt": 1,
                            "status": "succeeded",
                            "latency_ms": latency_ms,
                            "error_category": None,
                            "http_status": None,
                            "error_message": None,
                            "retryable": False,
                        }
                    ],
                },
            }

        return LLMModelRouterService().route(
            messages=messages,
            primary_provider=config.provider,
            primary_model=config.model,
            primary_api_key=config.api_key,
            primary_base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
            max_tokens=config.max_tokens,
            validator=validator,
        )

    def _attach_llm_router_metadata(
        self,
        *,
        result: dict[str, Any],
        telemetry: dict[str, Any],
    ) -> dict[str, Any]:
        output = dict(result)
        output["llm_attempted"] = bool(
            telemetry.get("llm_attempted")
        )
        output["llm_used"] = bool(
            output.get("llm_used")
            and telemetry.get("llm_used")
        )
        output["llm_provider"] = telemetry.get(
            "provider_used"
        )
        output["llm_model"] = telemetry.get(
            "model_used"
        )
        output["llm_attempt_count"] = int(
            telemetry.get("attempt_count") or 0
        )
        output["llm_latency_ms"] = int(
            telemetry.get("total_latency_ms") or 0
        )
        output["llm_fallback_used"] = bool(
            telemetry.get("fallback_used")
        )
        output["llm_fallback_reason"] = telemetry.get(
            "fallback_reason"
        )
        output["llm_router_attempts"] = list(
            telemetry.get("attempts") or []
        )
        return output

    def _fallback_resolver_mode(
        self,
        telemetry: dict[str, Any],
    ) -> str:
        categories = {
            str(item.get("error_category") or "")
            for item in (telemetry.get("attempts") or [])
            if item.get("status") == "failed"
        }
        categories.discard("")

        if categories and categories == {"low_confidence"}:
            return "deterministic_fallback_low_llm_confidence"

        return "deterministic_fallback_llm_failed"

    def _router_error_summary(
        self,
        telemetry: dict[str, Any],
    ) -> str:
        failures = []
        for item in telemetry.get("attempts") or []:
            if item.get("status") != "failed":
                continue
            provider = item.get("provider") or "unknown"
            model = item.get("model") or "unknown"
            category = (
                item.get("error_category")
                or "unknown_error"
            )
            status = item.get("http_status")
            suffix = f" HTTP {status}" if status else ""
            failures.append(
                f"{provider}/{model}: {category}{suffix}"
            )

        if not failures:
            return "All configured LLM models failed."

        return (
            "All configured LLM models failed: "
            + "; ".join(failures[:5])
        )

    def _build_messages(self, *, prompt: str, rag_context: dict[str, Any]) -> list[dict[str, str]]:
        system = """
You are the Semantic Intent Agent for an audience intelligence product.

You only receive privacy-safe cohort metadata. Never ask for or infer raw user/device-level data.

Your job:
- Understand the business meaning of the user prompt.
- Map natural language to available safe cohort metadata.
- Extract location, category/POI intent, daypart/time intent, audience goal, and confidence.
- If the requested category/location/daypart is not available, still return the requested intent, but mark data_gap_likely=true.
- Do not decide export.
- Do not create fake audiences.

Critical intent rules:
- The main business object wins over context words.
- Separate primary destination categories from support terms.
- requested_categories must contain only the primary venue/category the user is trying to reach.
- poi_terms can include support terms like snack, refresh break, coffee, quick stop, before going home.
- Do not expand a support term into unrelated export categories.
- Example: "quick coffee, small snack, refresh break" should keep cafe/coffee as primary unless the user clearly asks for restaurant, dining, meal, lunch, dinner, or food court.
- "caffeine break", "coffee break", "grab espresso", or "grab coffee" means cafe/coffee intent.
- "after office" and "after work" are daypart/time signals for evening; they do not mean office/coworking category by themselves.
- Only map to office/coworking when the prompt explicitly mentions coworking, flexible workspaces, business centers, corporate office, or professionals around workspaces.

Return JSON only with this schema:
{
  "business_intent": "string",
  "audience_goal": "string",
  "locations": ["string"],
  "requested_categories": ["string"],
  "matched_available_poi_types": ["string"],
  "poi_terms": ["string"],
  "dayparts": ["string"],
  "quality_intent": "high|balanced|broad",
  "fallback_tolerance": "strict_exact_first|allow_review_fallbacks|unknown",
  "data_gap_likely": true,
  "confidence_score": 0.0,
  "reasoning_summary": "short explanation"
}
""".strip()

        user = {
            "user_prompt": prompt,
            "safe_rag_context": rag_context,
        }

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, indent=2)},
        ]

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.I).strip()
            cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM response did not contain a JSON object.")

        return json.loads(cleaned[start : end + 1])



    def _validate_and_merge_llm_intent(
        self,
        *,
        prompt: str,
        fallback: dict[str, Any],
        llm_json: dict[str, Any],
        rag_context: dict[str, Any],
        min_confidence: float,
    ) -> dict[str, Any]:
        available_locations = set(rag_context.get("available_locations") or [])
        available_pois = set(rag_context.get("available_poi_types") or [])
        available_dayparts = set(rag_context.get("available_dayparts") or [])

        confidence = self._safe_float(llm_json.get("confidence_score"), default=0.0)
        if confidence < min_confidence:
            return self._with_hybrid_metadata(
                fallback,
                rag_context=rag_context,
                llm_used=False,
                resolver_mode="deterministic_fallback_low_llm_confidence",
                llm_error=f"LLM confidence {confidence} below threshold {min_confidence}",
            )

        llm_locations = self._normalize_list(llm_json.get("locations"))
        requested_categories = self._normalize_list(llm_json.get("requested_categories"))
        matched_available_poi_types = self._normalize_list(llm_json.get("matched_available_poi_types"))
        poi_terms = self._normalize_list(llm_json.get("poi_terms"))
        dayparts = self._normalize_list(llm_json.get("dayparts"))

        fallback_locations = self._normalize_list(fallback.get("locations"))
        fallback_categories = self._normalize_list(fallback.get("canonical_categories"))
        fallback_poi_terms = self._normalize_list(fallback.get("poi_terms"))
        fallback_dayparts = self._normalize_list(fallback.get("dayparts"))

        # Important:
        # If deterministic fallback found an explicit user location like "montreal",
        # do not allow the LLM to replace it with an inferred sub-area like
        # "westmount, montreal" unless the user actually wrote that sub-area.
        locations = self._resolve_final_locations(
            prompt=prompt,
            llm_locations=llm_locations,
            fallback_locations=fallback_locations,
        )

        available_location_matches = [
            location for location in locations
            if self._safe_location_available(location, available_locations)
        ]

        available_poi_matches = [
            poi for poi in matched_available_poi_types + poi_terms + requested_categories
            if self._safe_poi_available(poi, available_pois)
        ]

        available_daypart_matches = [
            daypart for daypart in dayparts
            if daypart in available_dayparts or daypart in {"morning", "afternoon", "evening", "night", "weekend", "weekday"}
        ]

        canonical_categories = requested_categories or fallback_categories or []
        final_poi_terms = self._dedupe(available_poi_matches + poi_terms + fallback_poi_terms)
        final_locations = self._dedupe(locations)
        final_dayparts = self._dedupe(available_daypart_matches or fallback_dayparts)

        normalized_business_intent = self._normalize_business_intent(
            llm_json=llm_json,
            requested_categories=requested_categories,
            canonical_categories=canonical_categories,
            fallback=fallback,
        )

        combo_available = self._exact_safe_combo_available(
            locations=final_locations,
            poi_terms=available_poi_matches or requested_categories or final_poi_terms,
            dayparts=final_dayparts,
            rag_context=rag_context,
        )

        combo_check_required = bool(final_locations and (available_poi_matches or requested_categories or final_poi_terms) and final_dayparts)

        deterministic_data_gap = bool(combo_check_required and not combo_available)
        llm_data_gap = bool(llm_json.get("data_gap_likely", False))
        final_data_gap_likely = bool(llm_data_gap or deterministic_data_gap)

        result = {
            "status": "completed",
            "original_prompt": prompt,
            "business_intent": normalized_business_intent,
            "audience_goal": str(llm_json.get("audience_goal") or fallback.get("audience_goal") or "general_audience"),
            "locations": final_locations,
            "available_location_matches": self._dedupe(available_location_matches),
            "poi_terms": final_poi_terms,
            "canonical_categories": self._dedupe(canonical_categories),
            "requested_categories": self._dedupe(requested_categories),
            "matched_available_poi_types": self._dedupe(available_poi_matches),
            "dayparts": final_dayparts,
            "quality_intent": str(llm_json.get("quality_intent") or fallback.get("quality_intent") or "balanced"),
            "fallback_tolerance": str(llm_json.get("fallback_tolerance") or fallback.get("fallback_tolerance") or "unknown"),
            "data_gap_likely": final_data_gap_likely,
            "exact_safe_combo_available": bool(combo_available),
            "exact_safe_combo_check_required": bool(combo_check_required),
            "confidence_score": round(min(max(confidence, 0.0), 0.99), 3),
            "extraction_method": "hybrid_llm_rag_semantic_intent_v1",
            "reasons": [
                str(llm_json.get("reasoning_summary") or "LLM/RAG semantic intent resolved from safe metadata.")
            ],
            "llm_used": True,
            "resolver_mode": "llm_rag_primary",
            "llm_error": None,
            "rag_context_summary": {
                "safe_cohort_count": rag_context.get("safe_cohort_count", 0),
                "available_location_count": len(rag_context.get("available_locations") or []),
                "available_poi_type_count": len(rag_context.get("available_poi_types") or []),
                "available_daypart_count": len(rag_context.get("available_dayparts") or []),
                "safe_available_combination_count": len(rag_context.get("safe_available_combinations") or []),
            },
        }

        return result



    def _resolve_final_locations(
        self,
        *,
        prompt: str,
        llm_locations: list[str],
        fallback_locations: list[str],
    ) -> list[str]:
        prompt_norm = self._match_norm(prompt)

        # Prefer explicit deterministic location extraction.
        # This prevents "near Montreal" becoming "westmount, montreal".
        if fallback_locations:
            return fallback_locations

        # If no fallback location exists, keep only LLM locations that are
        # directly mentioned in the user prompt.
        direct_llm_locations = []
        for location in llm_locations:
            location_norm = self._match_norm(location)
            if location_norm and location_norm in prompt_norm:
                direct_llm_locations.append(location)

        if direct_llm_locations:
            return self._dedupe(direct_llm_locations)

        return self._dedupe(llm_locations)

    def _normalize_business_intent(
        self,
        *,
        llm_json: dict[str, Any],
        requested_categories: list[str],
        canonical_categories: list[str],
        fallback: dict[str, Any],
    ) -> str:
        if requested_categories:
            return "+".join(self._dedupe(requested_categories))

        if canonical_categories:
            return "+".join(self._dedupe(canonical_categories))

        fallback_intent = str(fallback.get("business_intent") or "").strip()
        if fallback_intent and fallback_intent != "unknown_business_intent":
            return fallback_intent

        raw_intent = str(llm_json.get("business_intent") or "").strip().lower()
        if raw_intent and len(raw_intent) <= 60:
            return raw_intent

        return "unknown_business_intent"

    def _exact_safe_combo_available(
        self,
        *,
        locations,
        poi_terms,
        dayparts,
        rag_context,
    ) -> bool:
        """Return True only when every requested location has the requested POI/daypart combo.

        Production safety rule:
        - Multi-location request must not pass just because one requested city has a match.
        - Each requested location needs its own exact safe location + category + daypart combo.
        """

        def norm(value):
            import re

            if value is None:
                return ""
            value = str(value).lower().replace("_", " ")
            value = re.sub(r"[^a-z0-9]+", " ", value)
            return re.sub(r"\s+", " ", value).strip()

        requested_locations = [norm(x) for x in (locations or []) if norm(x)]
        requested_pois = [norm(x) for x in (poi_terms or []) if norm(x)]
        requested_dayparts = [norm(x) for x in (dayparts or []) if norm(x)]

        combos = (rag_context or {}).get("safe_available_combinations") or []
        if not combos:
            return False

        normalized_combos = []
        for combo in combos:
            normalized_combos.append(
                {
                    "location": norm(combo.get("location_name")),
                    "poi": norm(combo.get("primary_poi_type")),
                    "daypart": norm(combo.get("created_day_part")),
                }
            )

        def poi_matches(combo_poi):
            if not requested_pois:
                return True

            for poi in requested_pois:
                if combo_poi == poi:
                    return True

                # Normalize common aliases.
                if poi in {"coffee", "coffee shop"} and combo_poi in {"cafe", "coffee shop"}:
                    return True

                if poi == "restaurant" and (
                    combo_poi == "restaurant" or combo_poi.endswith(" restaurant")
                ):
                    return True

            return False

        def daypart_matches(combo_daypart):
            if not requested_dayparts:
                return True
            return combo_daypart in requested_dayparts

        # If no location was requested, any exact category/daypart-safe combo is enough.
        if not requested_locations:
            return any(
                poi_matches(combo["poi"]) and daypart_matches(combo["daypart"])
                for combo in normalized_combos
            )

        # Critical rule: every requested location must have the requested category/daypart.
        for requested_location in requested_locations:
            location_has_exact_safe_combo = any(
                combo["location"] == requested_location
                and poi_matches(combo["poi"])
                and daypart_matches(combo["daypart"])
                for combo in normalized_combos
            )

            if not location_has_exact_safe_combo:
                return False

        return True

    def _safe_location_available(self, location: str, available_locations: set[str]) -> bool:
        requested = self._match_norm(location)

        for available in available_locations:
            available_norm = self._match_norm(available)
            if requested == available_norm or requested in available_norm or available_norm in requested:
                return True

        return False

    def _safe_poi_available(self, poi: str, available_pois: set[str]) -> bool:
        requested = self._match_norm(poi)

        for available in available_pois:
            available_norm = self._match_norm(available)
            if requested == available_norm or requested in available_norm or available_norm in requested:
                return True

        return False

    def _match_norm(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        text = text.replace("-", "_").replace(" ", "_")
        text = re.sub(r"[^a-z0-9_]+", "", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text

    def _with_hybrid_metadata(
        self,
        intent: dict[str, Any],
        *,
        rag_context: dict[str, Any],
        llm_used: bool,
        resolver_mode: str,
        llm_error: str | None,
    ) -> dict[str, Any]:
        result = dict(intent)
        result["llm_used"] = llm_used
        result["resolver_mode"] = resolver_mode
        result["llm_error"] = llm_error
        result["rag_context_summary"] = {
            "safe_cohort_count": rag_context.get("safe_cohort_count", 0),
            "available_location_count": len(rag_context.get("available_locations") or []),
            "available_poi_type_count": len(rag_context.get("available_poi_types") or []),
            "available_daypart_count": len(rag_context.get("available_dayparts") or []),
        }
        return result

    def _write_result(self, result: dict[str, Any], output_dir: str | Path | None) -> None:
        if not output_dir:
            return

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "prompt_intent.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

    def _normalize_list(self, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, list):
            return []

        output = []
        for item in value:
            text = str(item).strip().lower()
            text = text.replace("-", " ").replace("_", " ")
            text = re.sub(r"[^a-z0-9\s,]+", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                output.append(text.replace(" ", "_") if self._looks_like_poi(text) else text)

        return self._dedupe(output)

    def _looks_like_poi(self, text: str) -> bool:
        poi_words = {
            "cafe",
            "coffee",
            "restaurant",
            "gym",
            "fitness",
            "coworking",
            "office",
            "clinic",
            "pharmacy",
            "salon",
            "spa",
            "tattoo",
            "store",
            "retail",
            "bar",
            "club",
        }
        return any(word in text for word in poi_words)

    def _dedupe(self, values: list[str]) -> list[str]:
        seen = set()
        output = []

        for value in values:
            item = str(value).strip().lower()
            if item and item not in seen:
                seen.add(item)
                output.append(item)

        return output

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default
