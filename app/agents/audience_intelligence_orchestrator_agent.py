from __future__ import annotations

import inspect
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.agents.privacy_layer_agent import PrivacyLayerAgent
from app.agents.synthetic_engine_agent import SyntheticEngineAgent
from app.agents.embedding_feature_store_agent import EmbeddingFeatureStoreAgent
from app.agents.cohort_management_agent import CohortManagementAgent
from app.agents.safe_export_agent import SafeExportAgent
from app.agents.autonomous_v2_swarm_review_agent import AutonomousV2SwarmReviewAgent
from app.services.autonomous_audience_intelligence_v2_service import (
    AutonomousAudienceIntelligenceV2Service,
)


class AudienceIntelligenceOrchestratorAgent:
    """
    Runs the Audience Intelligence pipeline from a business prompt.

    v1 flow remains:
    Postgres/safe input
      -> PrivacyLayerAgent
      -> prompt cohort selection
      -> SyntheticEngineAgent
      -> EmbeddingFeatureStoreAgent
      -> CohortManagementAgent
      -> SafeExportAgent

    v2 sidecar:
      -> all privacy-safe cohorts
      -> semantic prompt intent
      -> all-safe-cohort 384-dim embeddings
      -> dynamic ranking
      -> mutation/data-gap suggestions
      -> swarm review

    v2-guided selection:
      The export path can now use v2-ranked relevant cohorts so broad prompts do
      not export unrelated high-quality cohorts such as gas_station/casino/car_wash.
    """

    def run(
        self,
        prompt: str,
        output_root: str | Path = "data/prompt_runs",
        source: str = "postgres",
        safe_cohort_path: Optional[str | Path] = None,
        postgres_limit: int = 10000,
        k_min: int = 1000,
        epsilon: float = 1.0,
        synthetic_rows: int = 1000,
        max_export_cohorts: int = 25,
        min_export_quality: float = 0.25,
        approval_required: bool = True,
    ) -> Dict[str, Any]:
        run_id = self._build_run_id(prompt)
        run_dir = Path(output_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        final_summary_path = run_dir / "final_prompt_summary.json"

        print("RUN ID:", run_id)
        print("PROMPT:", prompt)
        print("RUN DIR:", run_dir)
        print()

        if source == "postgres":
            safe_raw_input = self._load_safe_rows_from_postgres(
                prompt=prompt,
                limit=postgres_limit,
            )
            source_mode = "postgres_safe_derived"
        elif source == "safe_artifact":
            if not safe_cohort_path:
                raise ValueError("--safe-cohort-path is required when source=safe_artifact")
            safe_raw_input = pd.read_csv(safe_cohort_path)
            source_mode = "existing_safe_artifact"
        else:
            raise ValueError("source must be postgres or safe_artifact")

        print("SOURCE MODE:", source_mode)
        print("SOURCE ROWS:", len(safe_raw_input))
        print("SOURCE COLUMNS:", list(safe_raw_input.columns))
        print()

        privacy_dir = run_dir / "01_privacy"
        synthetic_dir = run_dir / "02_synthetic"
        embedding_dir = run_dir / "03_embeddings"
        cohort_dir = run_dir / "04_cohort_management"
        export_dir = run_dir / "05_safe_export"
        v2_dir = run_dir / "06_v2_autonomous_preview"

        if source == "postgres":
            privacy_result = self._run_privacy_layer(
                safe_raw_input=safe_raw_input,
                output_dir=privacy_dir,
                run_id=f"{run_id}_privacy",
                k_min=k_min,
                epsilon=epsilon,
            )
            privacy_feature_path = self._resolve_privacy_feature_path(
                privacy_result=privacy_result,
                privacy_dir=privacy_dir,
            )
            privacy_cohorts = pd.read_csv(privacy_feature_path)
        else:
            privacy_dir.mkdir(parents=True, exist_ok=True)
            privacy_feature_path = privacy_dir / "clean_feature_table.csv"
            safe_raw_input.to_csv(privacy_feature_path, index=False)
            privacy_result = {
                "status": "skipped_existing_safe_artifact",
                "feature_path": str(privacy_feature_path),
            }
            privacy_cohorts = safe_raw_input.copy()

        selected_cohorts, prompt_filter_report = self._select_cohorts_for_prompt(
            prompt=prompt,
            cohorts=privacy_cohorts,
        )

        selected_path = privacy_dir / "prompt_selected_cohorts.csv"
        selected_cohorts.to_csv(selected_path, index=False)

        print("PRIVACY COHORTS:", len(privacy_cohorts))
        print("PROMPT SELECTED COHORTS:", len(selected_cohorts))
        print("PROMPT FILTER MODE:", prompt_filter_report["filter_mode"])
        print()

        try:
            v2_result = AutonomousAudienceIntelligenceV2Service().run(
                prompt=prompt,
                safe_cohorts=privacy_cohorts,
                output_dir=v2_dir,
                freshness_source_df=safe_raw_input,
            )
            print("V2 AUTONOMOUS STATUS:", v2_result.get("status"))
            print("V2 VECTOR COUNT:", v2_result.get("embedding_manifest", {}).get("vector_count"))
            print("V2 VECTOR DIM:", v2_result.get("embedding_manifest", {}).get("vector_dimension"))
            print("V2 RANKED MATCHES:", v2_result.get("ranked_match_count"))
            print("V2 MUTATION SUGGESTIONS:", v2_result.get("mutation", {}).get("suggestion_count"))
            print()
        except Exception as exc:
            v2_result = {
                "status": "failed",
                "error": str(exc),
                "approval_required": True,
                "downstream_export_enabled": False,
            }
            print("V2 AUTONOMOUS STATUS: failed")
            print("V2 ERROR:", exc)
            print()

        prompt_filter_report = self._merge_v2_intent_into_prompt_filter_report(
            prompt_filter_report=prompt_filter_report,
            v2_result=v2_result,
        )

        v2_guided_selection_report = {
            "enabled": False,
            "reason": "not_attempted",
            "rows": 0,
        }

        try:
            v2_selected_cohorts, v2_guided_selection_report = self._select_cohorts_from_v2_ranked(
                privacy_cohorts=privacy_cohorts,
                v2_result=v2_result,
                prompt_filter_report=prompt_filter_report,
                max_rows=max_export_cohorts,
            )

            if len(v2_selected_cohorts) >= 2:
                selected_cohorts = v2_selected_cohorts
                selected_cohorts.to_csv(selected_path, index=False)
                prompt_filter_report["v2_guided_selection"] = v2_guided_selection_report

                print("V2 GUIDED SELECTION:", v2_guided_selection_report)
                print("V2 GUIDED SELECTED COHORTS:", len(selected_cohorts))
                print()
            else:
                prompt_filter_report["v2_guided_selection"] = v2_guided_selection_report
                print("V2 GUIDED SELECTION SKIPPED:", v2_guided_selection_report)
                print()
        except Exception as exc:
            v2_guided_selection_report = {
                "enabled": False,
                "reason": "failed",
                "error": str(exc),
                "rows": 0,
            }
            prompt_filter_report["v2_guided_selection"] = v2_guided_selection_report
            print("V2 GUIDED SELECTION FAILED:", exc)
            print()

        # STRICT CATEGORY GUARDRAIL:
        # If user asked for a clear business/category and we could not find a
        # strong category match, do not export location/daypart-only fallbacks.
        strict_category_guardrail_report = self._strict_category_export_guardrail(
            prompt_filter_report=prompt_filter_report,
            selected_cohorts=selected_cohorts,
            v2_result=v2_result,
        )
        prompt_filter_report["strict_category_export_guardrail"] = strict_category_guardrail_report

        print("STRICT CATEGORY GUARDRAIL:", strict_category_guardrail_report)
        print()

        if strict_category_guardrail_report.get("block_export"):
            return self._build_blocked_category_gap_summary(
                run_id=run_id,
                prompt=prompt,
                source_mode=source_mode,
                safe_raw_input=safe_raw_input,
                privacy_cohorts=privacy_cohorts,
                selected_cohorts=selected_cohorts,
                prompt_filter_report=prompt_filter_report,
                coverage_warnings=strict_category_guardrail_report.get("warnings", []),
                v2_result=v2_result,
                run_dir=run_dir,
                export_dir=export_dir,
                final_summary_path=final_summary_path,
                approval_required=approval_required,
            )

        synthetic_agent = SyntheticEngineAgent()
        synthetic_result = self._call_agent_method(
            agent=synthetic_agent,
            method_names=["generate"],
            kwargs={
                "cohorts": selected_cohorts,
                "df": selected_cohorts,
                "data": selected_cohorts,
                "output_dir": synthetic_dir,
                "run_id": f"{run_id}_synthetic",
                "engine_requested": "dp_aggregate",
                "engine": "dp_aggregate",
                "production_mode": True,
                "allow_fallback": False,
                "synthetic_rows": synthetic_rows,
                "rows": synthetic_rows,
                "epsilon": epsilon,
                "k_min": k_min,
            },
        )

        print("SYNTHETIC STATUS:", synthetic_result.get("status"))
        print("SYNTHETIC ENGINE:", synthetic_result.get("engine") or synthetic_result.get("engine_used"))
        print()

        embedding_agent = EmbeddingFeatureStoreAgent()
        embedding_result = embedding_agent.build(
            cohorts=selected_cohorts,
            output_dir=embedding_dir,
            embedding_provider="sklearn_tfidf",
            run_id=f"{run_id}_embedding",
            max_features=384,
        )

        print("EMBEDDING STATUS:", embedding_result["status"])
        print("VECTOR COUNT:", embedding_result["vector_count"])
        print("VECTOR DIM:", embedding_result["vector_dimension"])
        print()

        cohort_agent = CohortManagementAgent()
        cohort_result = cohort_agent.run_from_artifacts(
            metadata_path=embedding_result["outputs"]["cohort_metadata"],
            vectors_path=embedding_result["outputs"]["cohort_vectors"],
            output_dir=cohort_dir,
            run_id=f"{run_id}_cohort_management",
            min_clusters=2,
            max_clusters=8,
            top_n=25,
            lookalike_top_k=3,
            min_export_quality=min_export_quality,
        )

        print("COHORT MANAGEMENT STATUS:", cohort_result["status"])
        print("MANAGED COHORTS:", cohort_result["managed_cohorts"])
        print("CLUSTERS:", cohort_result["cluster_count"])
        print("EXPORT READY:", cohort_result["export_ready_cohorts"])
        print()

        export_agent = SafeExportAgent()
        export_result = export_agent.run_from_artifacts(
            top_cohorts_path=cohort_result["outputs"]["top_cohorts"],
            lookalikes_path=cohort_result["outputs"]["lookalike_cohorts"],
            output_dir=export_dir,
            run_id=f"{run_id}_safe_export",
            audience_namespace="punk_audience",
            approval_required=approval_required,
            min_management_quality=min_export_quality,
            max_export_cohorts=max_export_cohorts,
        )

        print("SAFE EXPORT STATUS:", export_result["status"])
        print("APPROVAL STATUS:", export_result["approval_status"])
        print("DOWNSTREAM ENABLED:", export_result["downstream_export_enabled"])
        print("EXPORTED COHORTS:", export_result["exported_cohorts"])
        print("EXPORTED LOOKALIKE PAIRS:", export_result["exported_lookalike_pairs"])
        print()

        coverage_warnings = self._build_coverage_warnings(
            prompt=prompt,
            prompt_filter_report=prompt_filter_report,
            export_result=export_result,
        )

        if coverage_warnings:
            print("COVERAGE WARNINGS:")
            for warning in coverage_warnings:
                print("-", warning)
            print()

        final_summary = {
            "status": "completed",
            "run_id": run_id,
            "prompt": prompt,
            "source_mode": source_mode,
            "source_rows": int(len(safe_raw_input)),
            "privacy_cohorts": int(len(privacy_cohorts)),
            "prompt_selected_cohorts": int(len(selected_cohorts)),
            "prompt_filter_report": prompt_filter_report,
            "coverage_warnings": coverage_warnings,
            "v2_autonomous": self._safe_dict(v2_result),
            "synthetic": self._safe_dict(synthetic_result),
            "embedding": {
                "status": embedding_result["status"],
                "vector_count": embedding_result["vector_count"],
                "vector_dimension": embedding_result["vector_dimension"],
                "outputs": embedding_result["outputs"],
            },
            "cohort_management": self._safe_dict(cohort_result),
            "safe_export": self._safe_dict(export_result),
            "privacy_guarantees": {
                "raw_maids_exported": False,
                "hashed_identifiers_exported": False,
                "raw_observations_exported": False,
                "raw_lat_lng_exported": False,
                "raw_email_exported": False,
                "raw_phone_exported": False,
                "individual_user_data_exported": False,
                "approval_required": bool(approval_required),
                "approval_status": export_result["approval_status"],
            },
            "run_dir": str(run_dir),
            "final_summary_path": str(final_summary_path),
        }

        final_summary_path.write_text(json.dumps(final_summary, indent=2, allow_nan=False))

        try:
            v2_swarm_review = AutonomousV2SwarmReviewAgent().review_run(run_dir)
        except Exception as exc:
            v2_swarm_review = {
                "status": "failed",
                "overall_review_status": "blocked",
                "error": str(exc),
                "approval_required": True,
                "downstream_export_enabled": False,
            }

        final_summary["v2_swarm_review"] = self._safe_dict(v2_swarm_review)
        final_summary_path.write_text(json.dumps(final_summary, indent=2, allow_nan=False))

        print("V2 SWARM REVIEW:", v2_swarm_review.get("overall_review_status"))
        print("FINAL SUMMARY:", final_summary_path)
        print("SAFE EXPORT:", export_result["outputs"]["safe_export_manifest"])

        return final_summary





    def _merge_v2_intent_into_prompt_filter_report(
        self,
        *,
        prompt_filter_report: Dict[str, Any],
        v2_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(prompt_filter_report)

        intent = (
            v2_result.get("prompt_intent")
            or v2_result.get("intent")
            or v2_result.get("semantic_intent")
            or {}
        )

        if not isinstance(intent, dict):
            return merged

        def clean_list(values: Any) -> list[str]:
            if values is None:
                return []
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                return []

            output = []
            seen = set()

            for value in values:
                item = str(value).strip().lower()
                item = item.replace("-", "_").replace(" ", "_")
                item = re.sub(r"[^a-z0-9_,]+", "", item)

                if item and item not in seen:
                    seen.add(item)
                    output.append(item)

            return output

        existing_locations = clean_list(merged.get("locations_detected"))
        existing_pois = clean_list(merged.get("poi_terms_detected"))
        existing_dayparts = clean_list(merged.get("dayparts_detected"))

        ai_locations = clean_list(
            intent.get("locations")
            or intent.get("available_location_matches")
        )

        ai_pois = clean_list(
            intent.get("poi_terms")
            or intent.get("requested_categories")
            or intent.get("canonical_categories")
            or intent.get("matched_available_poi_types")
        )

        ai_dayparts = clean_list(intent.get("dayparts"))

        merged["locations_detected"] = existing_locations or ai_locations
        merged["poi_terms_detected"] = existing_pois or ai_pois
        merged["dayparts_detected"] = existing_dayparts or ai_dayparts

        if not existing_locations and ai_locations:
            merged["ai_locations_used"] = True
        if not existing_pois and ai_pois:
            merged["ai_poi_terms_used"] = True
        if not existing_dayparts and ai_dayparts:
            merged["ai_dayparts_used"] = True

        locations = merged.get("locations_detected") or []
        pois = merged.get("poi_terms_detected") or []
        dayparts = merged.get("dayparts_detected") or []

        if locations and pois and dayparts:
            merged["filter_mode"] = "location+poi+daypart"
        elif locations and pois:
            merged["filter_mode"] = "location+poi"
        elif pois and dayparts:
            merged["filter_mode"] = "poi+daypart"
        elif pois:
            merged["filter_mode"] = "poi"
        elif locations and dayparts:
            merged["filter_mode"] = "location+daypart"
        elif locations:
            merged["filter_mode"] = "location"
        elif dayparts:
            merged["filter_mode"] = "daypart"

        merged["ai_intent_merge"] = {
            "enabled": True,
            "llm_used": bool(intent.get("llm_used")),
            "resolver_mode": intent.get("resolver_mode"),
            "confidence_score": intent.get("confidence_score"),
            "data_gap_likely": bool(intent.get("data_gap_likely", False)),
        }

        return merged


    def _select_cohorts_from_v2_ranked(
        self,
        *,
        privacy_cohorts: pd.DataFrame,
        v2_result: Dict[str, Any],
        max_rows: int,
        prompt_filter_report: Optional[Dict[str, Any]] = None,
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        prompt_filter_report = prompt_filter_report or {}

        ranked_path = v2_result.get("ranked_matches_path")
        min_score = float(os.getenv("V2_EXPORT_MIN_MATCH_SCORE", "0.55"))
        strong_category_score = float(os.getenv("V2_EXPORT_MIN_CATEGORY_SCORE", "0.70"))

        allowed_match_types = {
            item.strip()
            for item in os.getenv(
                "V2_EXPORT_ALLOWED_MATCH_TYPES",
                "exact_match,adjacent_category",
            ).split(",")
            if item.strip()
        }

        requested_poi_terms = [
            str(term).strip().lower()
            for term in (prompt_filter_report.get("poi_terms_detected") or [])
            if str(term).strip()
        ]

        strict_category_requested = bool(requested_poi_terms)

        report = {
            "enabled": False,
            "reason": "no_ranked_matches",
            "ranked_path": ranked_path,
            "min_score": min_score,
            "strong_category_score": strong_category_score,
            "allowed_match_types": sorted(allowed_match_types),
            "strict_category_requested": strict_category_requested,
            "requested_poi_terms": requested_poi_terms,
            "rows": 0,
        }

        if not ranked_path or not Path(ranked_path).exists():
            return pd.DataFrame(), report

        ranked = pd.read_csv(ranked_path)
        if ranked.empty:
            report["reason"] = "empty_ranked_matches"
            return pd.DataFrame(), report

        if "final_match_score" not in ranked.columns or "match_type" not in ranked.columns:
            report["reason"] = "missing_required_v2_score_columns"
            return pd.DataFrame(), report

        candidate = ranked.copy()
        candidate["final_match_score"] = pd.to_numeric(
            candidate["final_match_score"],
            errors="coerce",
        ).fillna(0)

        candidate = candidate[
            candidate["match_type"].astype(str).isin(allowed_match_types)
            & (candidate["final_match_score"] >= min_score)
        ].copy()

        if "category_match_score" in candidate.columns:
            candidate["category_match_score"] = pd.to_numeric(
                candidate["category_match_score"],
                errors="coerce",
            ).fillna(0)

            if strict_category_requested:
                candidate = candidate[candidate["category_match_score"] >= strong_category_score].copy()
            else:
                candidate = candidate[candidate["category_match_score"] >= 0.5].copy()
        elif strict_category_requested:
            report["reason"] = "strict_category_requested_but_no_category_score"
            return pd.DataFrame(), report

        # Hard business-category export filter.
        # Example:
        # espresso/coffee prompt must export cafe/coffee cohorts only,
        # not restaurant/shawarma fallback.
        candidate, category_filter_report = self._filter_to_requested_export_category(
            candidate,
            prompt_filter_report,
        )
        report["requested_category_filter"] = category_filter_report

        if candidate.empty:
            report["reason"] = "no_safe_match_after_requested_category_filter"
            return pd.DataFrame(), report

        candidate = candidate.sort_values("final_match_score", ascending=False).head(max_rows)

        original_columns = [column for column in privacy_cohorts.columns if column in candidate.columns]
        selected = candidate[original_columns].copy().reset_index(drop=True)

        report.update(
            {
                "enabled": True,
                "reason": "selected_from_v2_ranked_matches",
                "rows": int(len(selected)),
                "top_match_type": str(candidate.iloc[0].get("match_type")),
                "top_match_score": float(candidate.iloc[0].get("final_match_score")),
                "top_category_score": float(candidate.iloc[0].get("category_match_score", 0)),
                "blocked_unrelated_low_score_rows": int(len(ranked) - len(candidate)),
            }
        )

        return selected, report



    def _filter_to_requested_export_category(
        self,
        cohorts: pd.DataFrame,
        prompt_filter_report: Dict[str, Any],
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        requested_terms = [
            str(term).strip().lower()
            for term in (prompt_filter_report.get("poi_terms_detected") or [])
            if str(term).strip()
        ]

        report = {
            "enabled": False,
            "requested_terms": requested_terms,
            "allowed_poi_terms": [],
            "input_rows": int(len(cohorts)),
            "output_rows": int(len(cohorts)),
        }

        if cohorts.empty or not requested_terms or "primary_poi_type" not in cohorts.columns:
            return cohorts, report

        allowed = self._allowed_export_poi_terms_for_request(requested_terms)

        if not allowed:
            return cohorts, report

        normalized = cohorts["primary_poi_type"].astype(str).str.lower().str.replace(" ", "_", regex=False)

        mask = pd.Series(False, index=cohorts.index)
        for allowed_term in allowed:
            allowed_norm = str(allowed_term).lower().replace(" ", "_")
            mask = mask | normalized.str.contains(re.escape(allowed_norm), na=False)

        filtered = cohorts[mask].copy()

        report.update(
            {
                "enabled": True,
                "allowed_poi_terms": sorted(allowed),
                "output_rows": int(len(filtered)),
                "blocked_rows": int(len(cohorts) - len(filtered)),
            }
        )

        return filtered, report

    def _allowed_export_poi_terms_for_request(self, requested_terms: list[str]) -> set[str]:
        normalized_terms = {
            str(term).lower().replace(" ", "_").replace("-", "_")
            for term in requested_terms
            if str(term).strip()
        }

        allowed: set[str] = set()

        cafe_terms = {"cafe", "coffee", "coffee_shop", "espresso", "snacks"}
        restaurant_terms = {
            "restaurant",
            "food",
            "fast_food",
            "quick_service_food",
            "quick_bites",
            "casual_dining",
            "dining",
        }
        office_terms = {
            "office",
            "coworking",
            "coworking_space",
            "corporate_office",
            "business",
            "professional",
            "point_of_interest",
        }
        gym_terms = {"gym", "fitness", "workout", "yoga", "health_club"}
        tattoo_terms = {"tattoo", "body_art_service", "body_ink", "piercing"}
        healthcare_terms = {"clinic", "healthcare", "hospital", "pharmacy", "dentist"}
        retail_terms = {"store", "retail", "shopping", "shopping_mall"}
        beauty_terms = {"salon", "spa", "beauty", "wellness", "grooming", "barber_shop"}

        if normalized_terms & cafe_terms:
            allowed.update({"cafe", "coffee_shop", "bakery_cafe"})

        if normalized_terms & restaurant_terms:
            allowed.update({
                "restaurant",
                "fast_food",
                "food",
                "shawarma_restaurant",
                "middle_eastern_restaurant",
                "takeaway",
            })

        if normalized_terms & office_terms:
            allowed.update({
                "coworking_space",
                "office",
                "corporate_office",
                "consultant",
                "business_center",
                "point_of_interest",
            })

        if normalized_terms & gym_terms:
            allowed.update({"gym", "fitness", "health_club", "yoga_studio"})

        if normalized_terms & tattoo_terms:
            allowed.update({"body_art_service", "tattoo", "piercing"})

        if normalized_terms & healthcare_terms:
            allowed.update({"clinic", "healthcare", "hospital", "pharmacy", "dentist"})

        if normalized_terms & retail_terms:
            allowed.update({"store", "retail", "shopping_mall", "shopping"})

        if normalized_terms & beauty_terms:
            allowed.update({"salon", "spa", "beauty", "barber_shop"})

        return allowed

    def _strict_category_export_guardrail(
        self,
        *,
        prompt_filter_report: Dict[str, Any],
        selected_cohorts: pd.DataFrame,
        v2_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        requested_poi_terms = [
            str(term).strip().lower()
            for term in (prompt_filter_report.get("poi_terms_detected") or [])
            if str(term).strip()
        ]

        requested_locations = [
            str(term).strip().lower()
            for term in (prompt_filter_report.get("locations_detected") or [])
            if str(term).strip()
        ]

        strict_category_requested = bool(requested_poi_terms)
        strict_location_requested = bool(requested_locations)
        strict_location_category_requested = bool(requested_poi_terms and requested_locations)

        filter_mode = str(prompt_filter_report.get("filter_mode") or "")

        v2_coverage_warnings = v2_result.get("coverage_warnings") or []
        mutation = v2_result.get("mutation") or {}
        data_gap_count = int(mutation.get("data_gap_count") or 0)

        warnings = []
        block_export = False
        reason = "category_location_guardrail_passed"

        def selected_has_requested_location() -> bool:
            if not requested_locations:
                return True

            if selected_cohorts.empty or "location_name" not in selected_cohorts.columns:
                return False

            selected_locations = (
                selected_cohorts["location_name"]
                .dropna()
                .astype(str)
                .str.lower()
                .str.strip()
                .tolist()
            )

            for requested in requested_locations:
                for selected in selected_locations:
                    if requested in selected or selected in requested:
                        return True

                    requested_tokens = {t for t in re.split(r"[^a-z0-9]+", requested) if len(t) >= 4}
                    selected_tokens = {t for t in re.split(r"[^a-z0-9]+", selected) if len(t) >= 4}
                    if requested_tokens and selected_tokens and requested_tokens.issubset(selected_tokens):
                        return True

            return False

        if strict_location_category_requested and selected_cohorts.empty:
            block_export = True
            reason = "strict_location_category_requested_but_no_safe_exact_match"
            warnings.append(
                "A specific location and business/category were requested, but no strong privacy-safe match exists. "
                "Cross-location or category-only fallback audiences were blocked from export."
            )

        elif strict_location_category_requested and "location" not in filter_mode:
            block_export = True
            reason = "strict_location_category_requested_but_selection_ignored_location"
            warnings.append(
                "A specific location and business/category were requested, but selected cohorts did not preserve the requested location. "
                "They were blocked from export."
            )

        elif strict_location_category_requested and not selected_has_requested_location():
            block_export = True
            reason = "strict_location_category_requested_but_selected_location_mismatch"
            warnings.append(
                "A specific location was requested, but selected cohorts came from a different market. "
                "Cross-location fallback audiences were blocked from export."
            )

        elif strict_category_requested and selected_cohorts.empty:
            block_export = True
            reason = "strict_category_requested_but_no_safe_category_match"
            warnings.append(
                "A clear business/category was requested, but no strong privacy-safe category match is available. "
                "Fallback location/daypart-only audiences were blocked from export."
            )

        elif strict_category_requested and "poi" not in filter_mode:
            block_export = True
            reason = "strict_category_requested_but_selection_is_location_or_daypart_only"
            warnings.append(
                "A clear business/category was requested, but selected cohorts were only location/daypart fallbacks. "
                "They were blocked from export to avoid misleading audience creation."
            )

        elif strict_location_requested and selected_cohorts.empty:
            block_export = True
            reason = "strict_location_requested_but_no_safe_location_match"
            warnings.append(
                "A specific location was requested, but no privacy-safe cohort matched that market. "
                "Cross-location fallback audiences were blocked from export."
            )

        elif strict_location_requested and "location" not in filter_mode:
            block_export = True
            reason = "strict_location_requested_but_selection_ignored_location"
            warnings.append(
                "A specific location was requested, but selected cohorts did not preserve that location. "
                "They were blocked from export."
            )

        elif strict_category_requested and data_gap_count > 0 and not selected_cohorts.empty:
            reason = "category_match_available_with_review_required"

        if v2_coverage_warnings:
            warnings.extend([str(item) for item in v2_coverage_warnings])

        return {
            "enabled": True,
            "strict_category_requested": strict_category_requested,
            "strict_location_requested": strict_location_requested,
            "strict_location_category_requested": strict_location_category_requested,
            "requested_poi_terms": requested_poi_terms,
            "requested_locations": requested_locations,
            "filter_mode": filter_mode,
            "selected_rows": int(len(selected_cohorts)),
            "v2_data_gap_count": data_gap_count,
            "block_export": bool(block_export),
            "reason": reason,
            "warnings": warnings,
        }


    def _build_blocked_category_gap_summary(
        self,
        *,
        run_id: str,
        prompt: str,
        source_mode: str,
        safe_raw_input: pd.DataFrame,
        privacy_cohorts: pd.DataFrame,
        selected_cohorts: pd.DataFrame,
        prompt_filter_report: Dict[str, Any],
        coverage_warnings: list[str],
        v2_result: Dict[str, Any],
        run_dir: Path,
        export_dir: Path,
        final_summary_path: Path,
        approval_required: bool,
    ) -> Dict[str, Any]:
        export_dir.mkdir(parents=True, exist_ok=True)

        empty_cohorts_path = export_dir / "safe_export_cohorts.csv"
        empty_lookalikes_path = export_dir / "safe_export_lookalike_pairs.csv"
        manifest_path = export_dir / "safe_export_manifest.json"

        pd.DataFrame(
            columns=[
                "audience_name",
                "location_name",
                "primary_poi_type",
                "created_day_part",
                "quality_score",
                "approval_status",
            ]
        ).to_csv(empty_cohorts_path, index=False)

        pd.DataFrame(
            columns=[
                "source_audience_name",
                "lookalike_audience_name",
                "similarity_score",
                "approval_status",
            ]
        ).to_csv(empty_lookalikes_path, index=False)

        safe_export_result = {
            "status": "blocked_category_data_gap",
            "approval_status": "needs_human_review",
            "downstream_export_enabled": False,
            "exported_cohorts": 0,
            "exported_lookalike_pairs": 0,
            "block_reason": "strict_category_requested_but_no_safe_category_match",
            "outputs": {
                "safe_export_manifest": str(manifest_path),
                "safe_export_cohorts": str(empty_cohorts_path),
                "safe_export_lookalike_pairs": str(empty_lookalikes_path),
            },
        }

        manifest_path.write_text(
            json.dumps(
                {
                    "status": "blocked_category_data_gap",
                    "run_id": run_id,
                    "prompt": prompt,
                    "approval_required": bool(approval_required),
                    "approval_status": "needs_human_review",
                    "downstream_export_enabled": False,
                    "exported_cohorts": 0,
                    "exported_lookalike_pairs": 0,
                    "reason": "Requested category was not available as a strong privacy-safe match. Fallback audiences were blocked.",
                    "coverage_warnings": coverage_warnings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        final_summary = {
            "status": "completed",
            "run_id": run_id,
            "prompt": prompt,
            "source_mode": source_mode,
            "source_rows": int(len(safe_raw_input)),
            "privacy_cohorts": int(len(privacy_cohorts)),
            "prompt_selected_cohorts": int(len(selected_cohorts)),
            "prompt_filter_report": prompt_filter_report,
            "coverage_warnings": coverage_warnings,
            "v2_autonomous": self._safe_dict(v2_result),
            "synthetic": {
                "status": "skipped",
                "reason": "strict_category_data_gap_blocked_export",
            },
            "embedding": {
                "status": "skipped",
                "vector_count": 0,
                "vector_dimension": 0,
                "outputs": {},
            },
            "cohort_management": {
                "status": "skipped",
                "managed_cohorts": 0,
                "cluster_count": 0,
                "export_ready_cohorts": 0,
                "reason": "strict_category_data_gap_blocked_export",
            },
            "safe_export": safe_export_result,
            "privacy_guarantees": {
                "raw_maids_exported": False,
                "hashed_identifiers_exported": False,
                "raw_observations_exported": False,
                "raw_lat_lng_exported": False,
                "raw_email_exported": False,
                "raw_phone_exported": False,
                "individual_user_data_exported": False,
                "approval_required": bool(approval_required),
                "approval_status": "needs_human_review",
            },
            "run_dir": str(run_dir),
            "final_summary_path": str(final_summary_path),
        }

        final_summary_path.write_text(json.dumps(final_summary, indent=2, allow_nan=False))

        try:
            v2_swarm_review = AutonomousV2SwarmReviewAgent().review_run(run_dir)
        except Exception as exc:
            v2_swarm_review = {
                "status": "failed",
                "overall_review_status": "needs_human_review",
                "error": str(exc),
                "approval_required": True,
                "downstream_export_enabled": False,
            }

        final_summary["v2_swarm_review"] = self._safe_dict(v2_swarm_review)
        final_summary_path.write_text(json.dumps(final_summary, indent=2, allow_nan=False))

        print("STRICT CATEGORY DATA GAP: export blocked")
        print("FINAL SUMMARY:", final_summary_path)
        print("SAFE EXPORT:", manifest_path)

        return final_summary


    def _run_privacy_layer(
        self,
        *,
        safe_raw_input: pd.DataFrame,
        output_dir: Path,
        run_id: str,
        k_min: int,
        epsilon: float,
    ) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)

        agent = PrivacyLayerAgent()

        return self._call_agent_method(
            agent=agent,
            method_names=["run", "process", "build", "execute", "transform", "anonymize"],
            kwargs={
                "raw_observations": safe_raw_input,
                "observations": safe_raw_input,
                "input_df": safe_raw_input,
                "df": safe_raw_input,
                "data": safe_raw_input,
                "cohorts": safe_raw_input,
                "output_dir": output_dir,
                "run_id": run_id,
                "identifier_column": "session_id",
                "count_column": "maid_count",
                "timestamp_column": "created_at",
                "location_column": "location_name",
                "poi_column": "primary_poi_type",
                "k_min": k_min,
                "epsilon": epsilon,
            },
        )

    def _call_agent_method(self, *, agent: Any, method_names: list[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        errors = []

        for method_name in method_names:
            if not hasattr(agent, method_name):
                continue

            method = getattr(agent, method_name)
            signature = inspect.signature(method)

            accepted_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }

            try:
                result = method(**accepted_kwargs)
                if isinstance(result, dict):
                    return result
                return {"status": "completed", "result": result}
            except TypeError as exc:
                errors.append(f"{method_name}: {exc}")

        public_methods = [
            name for name in dir(agent)
            if not name.startswith("_") and callable(getattr(agent, name))
        ]

        raise RuntimeError(
            f"Could not call agent {agent.__class__.__name__}. "
            f"Tried methods={method_names}. Public methods={public_methods}. Errors={errors}"
        )

    def _resolve_privacy_feature_path(self, *, privacy_result: Dict[str, Any], privacy_dir: Path) -> Path:
        candidates = []

        outputs = privacy_result.get("outputs")
        if isinstance(outputs, dict):
            for key in ["clean_feature_table", "feature_table", "privacy_feature", "safe_feature_table"]:
                if key in outputs:
                    candidates.append(Path(outputs[key]))

        for key in ["clean_feature_table", "feature_path", "output_path", "safe_feature_table"]:
            if key in privacy_result:
                candidates.append(Path(privacy_result[key]))

        candidates.extend(
            [
                privacy_dir / "clean_feature_table.csv",
                privacy_dir / "safe_feature_table.csv",
                privacy_dir / "privacy_feature_table.csv",
            ]
        )

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Could not find PrivacyLayerAgent clean feature table in {privacy_dir}. "
            f"Privacy result keys: {list(privacy_result.keys())}"
        )

    def _load_safe_rows_from_postgres(self, *, prompt: str, limit: int) -> pd.DataFrame:
        self._load_dotenv()

        db_url = (
            os.getenv("ECHO_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or os.getenv("POSTGRES_URL")
            or os.getenv("POSTGRES_DATABASE_URL")
            or os.getenv("SUPABASE_DB_URL")
            or os.getenv("DB_URL")
        )

        if not db_url:
            raise RuntimeError(
                "No Postgres DB URL found in env. Expected DATABASE_URL or POSTGRES_URL. "
                "Use --source safe_artifact --safe-cohort-path <clean_feature_table.csv> to run from existing safe data."
            )

        from sqlalchemy import create_engine

        engine = create_engine(db_url)

        query = """
        SELECT
            session_id,
            created_at,
            maid_count,
            pois,
            center,
            lookback_days
        FROM public.maid_extractions
        WHERE maid_count IS NOT NULL
        ORDER BY created_at DESC
        LIMIT %(limit)s
        """

        raw = pd.read_sql_query(query, engine, params={"limit": int(limit)})

        safe = pd.DataFrame(
            {
                "session_id": raw["session_id"].astype(str),
                "created_at": raw["created_at"],
                "maid_count": pd.to_numeric(raw["maid_count"], errors="coerce").fillna(0).astype(int),
                "location_name": raw["center"].apply(lambda value: self._derive_location_name(value, prompt)),
                "primary_poi_type": raw["pois"].apply(lambda value: self._derive_poi_type(value, prompt)),
            }
        )

        safe = safe[safe["maid_count"] > 0].reset_index(drop=True)

        return safe

    def _load_dotenv(self) -> None:
        env_path = Path(".env")
        if not env_path.exists():
            return

        for line in env_path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')

            if key and key not in os.environ:
                os.environ[key] = value

    def _derive_location_name(self, value: Any, prompt: str) -> str:
        obj = self._parse_jsonish(value)

        if isinstance(obj, dict):
            for key in [
                "location_name",
                "city",
                "place_name",
                "name",
                "label",
                "area",
                "region",
                "address",
                "formatted_address",
            ]:
                candidate = obj.get(key)
                if self._safe_text(candidate):
                    return self._clean_text(candidate)

        return self._location_hint_from_prompt(prompt) or "selected location"

    def _derive_poi_type(self, value: Any, prompt: str) -> str:
        obj = self._parse_jsonish(value)

        prompt_poi = self._poi_hint_from_prompt(prompt)

        if isinstance(obj, list) and obj:
            first = obj[0]
            if isinstance(first, dict):
                candidate = self._poi_from_dict(first)
                if candidate:
                    return candidate

        if isinstance(obj, dict):
            candidate = self._poi_from_dict(obj)
            if candidate:
                return candidate

        return prompt_poi or "point_of_interest"

    def _poi_from_dict(self, obj: dict) -> Optional[str]:
        for key in ["primary_poi_type", "poi_type", "place_type", "category", "type"]:
            value = obj.get(key)
            if self._safe_text(value):
                return self._clean_poi(value)

        types = obj.get("types")
        if isinstance(types, list):
            generic = {"point_of_interest", "establishment"}
            for item in types:
                cleaned = self._clean_poi(item)
                if cleaned and cleaned not in generic:
                    return cleaned
            if types:
                return self._clean_poi(types[0])

        return None

    def _parse_jsonish(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value

        if pd.isna(value):
            return None

        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return json.loads(value)
            except Exception:
                return None

        return None

    def _safe_text(self, value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return False
        return True

    def _clean_text(self, value: Any) -> str:
        text = str(value).strip().lower()
        text = re.sub(r"[^a-z0-9\s,\-]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:120] or "selected location"

    def _clean_poi(self, value: Any) -> str:
        text = str(value).strip().lower()
        text = text.replace(" ", "_").replace("-", "_")
        text = re.sub(r"[^a-z0-9_]+", "", text)
        return text[:80] or "point_of_interest"

    def _location_hint_from_prompt(self, prompt: str) -> Optional[str]:
        lower = prompt.lower()

        known = [
            "san francisco",
            "montreal",
            "montréal",
            "new york",
            "los angeles",
            "toronto",
            "quebec",
            "hyderabad",
            "bangalore",
            "chicago",
            "dhaka",
        ]

        found = [item for item in known if item in lower]
        if found:
            return found[0]

        return None

    def _poi_hint_from_prompt(self, prompt: str) -> Optional[str]:
        lower = prompt.lower().replace("-", " ")

        mapping = {
            "restaurant": "restaurant",
            "quick service food": "restaurant",
            "quick bites": "restaurant",
            "casual dining": "restaurant",
            "fast meals": "restaurant",
            "food": "food",
            "cafe": "cafe",
            "coffee": "cafe",
            "espresso": "cafe",
            "gym": "gym",
            "fitness": "gym",
            "workout": "gym",
            "yoga": "gym",
            "health": "health",
            "barber": "barber_shop",
            "tattoo": "body_art_service",
            "body ink": "body_art_service",
            "piercing": "body_art_service",
            "office": "coworking_space",
            "coworking": "coworking_space",
            "business center": "coworking_space",
            "business centers": "coworking_space",
            "business place": "coworking_space",
            "business places": "coworking_space",
            "professional": "coworking_space",
            "professionals": "coworking_space",
            "store": "store",
            "retail": "store",
            "shopping": "store",
            "clinic": "healthcare",
            "pharmacy": "healthcare",
            "healthcare": "healthcare",
            "students": "school",
            "campus": "school",
            "nightlife": "bar",
            "self care": "salon",
            "self-care": "salon",
            "wellness": "salon",
            "grooming": "salon",
        }

        for key, value in mapping.items():
            if key in lower:
                return value

        return None





    def _select_cohorts_for_prompt(self, prompt: str, cohorts):
        from app.agents.autonomous_prompt_cohort_selector_agent import AutonomousPromptCohortSelectorAgent

        return AutonomousPromptCohortSelectorAgent().select(
            prompt=prompt,
            cohorts=cohorts,
        )


    def _extract_location_terms(self, prompt: str, df: pd.DataFrame) -> list[str]:
        lower = prompt.lower()
        lower_clean = re.sub(r"[^a-z0-9\s,]+", " ", lower)
        lower_clean = re.sub(r"\s+", " ", lower_clean).strip()

        manual_locations = [
            "times square, new york",
            "times square",
            "montreal downtown",
            "montreal qc",
            "san francisco",
            "new york",
            "los angeles",
            "toronto",
            "quebec",
            "vancouver",
            "chicago",
            "montreal",
            "hyderabad",
            "bangalore",
            "mumbai",
            "delhi",
        ]

        found = []

        # First trust explicit user text only.
        for location in sorted(manual_locations, key=len, reverse=True):
            location_clean = location.lower().strip()
            location_regex = re.escape(location_clean).replace(r"\,", r",?\s*")
            if re.search(rf"\b{location_regex}\b", lower_clean):
                found.append(location_clean)

        # If prompt explicitly mentions a known city, do not infer every DB sub-area
        # containing that city. Example: "New York" should not become
        # "times square, new york" unless user actually said Times Square.
        if found:
            return self._dedupe_location_terms(found)

        # Fallback: use exact full location names from safe cohort table only.
        # Do not use loose token matching because that creates false detections.
        if "location_name" in df.columns:
            for location in df["location_name"].dropna().astype(str).str.lower().unique():
                location = re.sub(r"\s+", " ", location.strip())
                if len(location) < 3:
                    continue

                escaped = re.escape(location).replace(r"\,", r",?\s*")
                if re.search(rf"\b{escaped}\b", lower_clean):
                    found.append(location)

        return self._dedupe_location_terms(found)

    def _dedupe_location_terms(self, values: list[str]) -> list[str]:
        cleaned = []
        seen = set()

        for value in values:
            item = str(value).strip().lower()
            item = re.sub(r"\s+", " ", item)

            if not item or item in seen:
                continue

            seen.add(item)
            cleaned.append(item)

        # Remove broad duplicate when a more specific requested location exists.
        # Example: ["times square, new york", "new york"] -> ["times square, new york"]
        final = []
        for item in cleaned:
            is_parent_duplicate = any(
                item != other and item in other
                for other in cleaned
            )
            if not is_parent_duplicate:
                final.append(item)

        return final



    def _extract_poi_terms(self, prompt: str) -> list[str]:
        lower = prompt.lower().replace("-", " ")
        lower = re.sub(r"\s+", " ", lower).strip()

        terms = []

        def has_any(values: list[str]) -> bool:
            return any(value in lower for value in values)

        # Priority rule:
        # The business object wins over contextual words.
        # "caffeine break after office" means cafe + evening, not office.
        cafe_signals = [
            "cafe",
            "coffee",
            "espresso",
            "caffeine",
            "caffeine break",
            "coffee break",
            "grab coffee",
            "grab espresso",
        ]

        restaurant_signals = [
            "restaurant",
            "food street",
            "food streets",
            "quick service",
            "quick bites",
            "fast meals",
            "casual dining",
            "dining",
            "dinner",
        ]

        office_signals = [
            "coworking",
            "coworking hub",
            "coworking hubs",
            "coworking space",
            "coworking spaces",
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

        gym_signals = [
            "gym",
            "fitness",
            "workout",
            "workouts",
            "training session",
            "training sessions",
            "yoga",
        ]

        tattoo_signals = [
            "tattoo",
            "body ink",
            "body art",
            "piercing",
        ]

        healthcare_signals = [
            "clinic",
            "pharmacy",
            "pharmacies",
            "healthcare",
            "hospital",
            "doctor",
            "dentist",
        ]

        retail_signals = [
            "retail",
            "shopping",
            "shopping area",
            "shopping areas",
            "store",
            "mall",
        ]

        beauty_signals = [
            "wellness",
            "grooming",
            "self care",
            "self-care",
            "salon",
            "spa",
            "beauty",
            "barber",
        ]

        nightlife_signals = [
            "nightlife",
            "bar",
            "pub",
            "club",
            "lounge",
        ]

        education_signals = [
            "student",
            "students",
            "campus",
            "school",
            "college",
            "university",
        ]

        if has_any(cafe_signals):
            terms.extend(["cafe", "coffee"])

        if has_any(restaurant_signals):
            terms.extend(["restaurant", "food", "fast_food"])

        if has_any(office_signals):
            terms.extend(["coworking_space", "office", "corporate_office", "point_of_interest"])

        # Plain "office" should not become office category when used only as time phrase.
        # Example: "after office" means evening.
        office_temporal_only = any(
            phrase in lower
            for phrase in ["after office", "after office hours", "post office hours"]
        )
        if "office" in lower and not office_temporal_only and not has_any(cafe_signals):
            terms.extend(["coworking_space", "office", "corporate_office", "point_of_interest"])

        if has_any(gym_signals):
            terms.extend(["gym", "fitness", "health_club"])

        if has_any(tattoo_signals):
            terms.extend(["body_art_service", "tattoo"])

        if has_any(healthcare_signals):
            terms.extend(["clinic", "healthcare", "hospital", "pharmacy"])

        if has_any(retail_signals):
            terms.extend(["store", "retail", "shopping_mall"])

        if has_any(beauty_signals):
            terms.extend(["salon", "spa", "beauty", "barber_shop"])

        if has_any(nightlife_signals):
            terms.extend(["bar", "pub", "club", "lounge"])

        if has_any(education_signals):
            terms.extend(["school", "college", "university", "campus"])

        return list(dict.fromkeys(terms))


    def _build_coverage_warnings(
        self,
        *,
        prompt: str,
        prompt_filter_report: Dict[str, Any],
        export_result: Dict[str, Any],
    ) -> list[str]:
        warnings: list[str] = []

        outputs = export_result.get("outputs", {})
        cohorts_path = outputs.get("safe_export_cohorts")

        if not cohorts_path or not Path(cohorts_path).exists():
            return ["Safe export cohorts file was not found, so coverage could not be verified."]

        try:
            exported = pd.read_csv(cohorts_path)
        except Exception as exc:
            return [f"Could not read safe export cohorts for coverage verification: {exc}"]

        prompt_lower = prompt.lower().replace("café", "cafe").replace("cafés", "cafes")
        filter_mode = str(prompt_filter_report.get("filter_mode") or "")

        def clean_values(column: str) -> list[str]:
            if column not in exported.columns:
                return []
            return (
                exported[column]
                .dropna()
                .astype(str)
                .str.lower()
                .str.strip()
                .unique()
                .tolist()
            )

        def token_match(needle: str, haystack: str) -> bool:
            needle = str(needle or "").lower().replace(" ", "_").strip()
            haystack = str(haystack or "").lower().replace(" ", "_").strip()

            if not needle or not haystack:
                return False

            if needle in haystack or haystack in needle:
                return True

            needle_tokens = {t for t in re.split(r"[^a-z0-9]+", needle) if t}
            haystack_tokens = {t for t in re.split(r"[^a-z0-9]+", haystack) if t}

            if not needle_tokens or not haystack_tokens:
                return False

            return bool(needle_tokens & haystack_tokens)

        exported_locations = clean_values("location_name")
        exported_pois = clean_values("primary_poi_type")
        exported_dayparts = clean_values("created_day_part")

        requested_locations = []
        for loc in prompt_filter_report.get("locations_detected", []) or []:
            loc_clean = str(loc).lower().strip()
            if loc_clean and loc_clean in prompt_lower:
                requested_locations.append(loc_clean)
        requested_locations = list(dict.fromkeys(requested_locations))

        requested_poi_terms = [
            str(term).lower().strip()
            for term in (prompt_filter_report.get("poi_terms_detected", []) or [])
            if str(term).strip()
        ]
        requested_poi_terms = list(dict.fromkeys(requested_poi_terms))

        requested_dayparts = [
            str(daypart).lower().strip()
            for daypart in (prompt_filter_report.get("dayparts_detected", []) or [])
            if str(daypart).strip()
        ]
        requested_dayparts = list(dict.fromkeys(requested_dayparts))

        for requested in requested_locations:
            covered = any(
                requested in exported_location or exported_location in requested
                for exported_location in exported_locations
            )

            if not covered:
                warnings.append(
                    f"{requested} was requested, but no export-ready cohort for that location passed the final quality and safety filters."
                )

        if requested_poi_terms:
            exact_poi_covered = any(
                token_match(term, exported_poi)
                for term in requested_poi_terms
                for exported_poi in exported_pois
            )

            if "poi" not in filter_mode:
                warnings.append(
                    "The requested business/category terms were detected, but exact category coverage was not strong enough. "
                    "The exported cohorts are fallback/adjacent candidates and must be reviewed before approval."
                )
            elif not exact_poi_covered:
                warnings.append(
                    "The requested business/category terms were detected, but no export-ready cohort directly matched those terms."
                )

        if requested_dayparts:
            daypart_covered = any(
                requested == exported_daypart
                for requested in requested_dayparts
                for exported_daypart in exported_dayparts
            )

            if not daypart_covered:
                warnings.append(
                    "The requested time/daypart was detected, but no export-ready cohort matched that time window."
                )

        if not requested_locations and prompt_filter_report.get("locations_detected"):
            warnings.append(
                "Location terms were inferred from available cohorts, but no exact location phrase was directly confirmed from the prompt."
            )

        return warnings

    def _build_run_id(self, prompt: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:50]
        suffix = uuid.uuid4().hex[:6]
        return f"prompt_{timestamp}_{slug}_{suffix}"

    def _safe_dict(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._safe_dict(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._safe_dict(v) for v in value]

        if isinstance(value, tuple):
            return [self._safe_dict(v) for v in value]

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return None
            return value

        return value
