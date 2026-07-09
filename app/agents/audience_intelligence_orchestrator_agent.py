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

        synthetic_agent = SyntheticEngineAgent()
        # If strict prompt/v2 guardrails selected zero safe cohorts, stop here.
        # This is a valid no-export result, not a pipeline error.
        _selected_for_downstream = (
            locals().get("prompt_selected_cohorts")
            if locals().get("prompt_selected_cohorts") is not None
            else locals().get("selected_cohorts")
        )
        if _selected_for_downstream is None:
            _selected_for_downstream = locals().get("v2_selected_cohorts")

        _prompt_filter_report = locals().get("prompt_filter_report") or {}
        _v2_guided_selection_report = locals().get("v2_guided_selection_report") or {}
        _block_export = bool(
            _prompt_filter_report.get("block_export")
            or _v2_guided_selection_report.get("block_export")
            or _prompt_filter_report.get("filter_mode") in {
                "location_category_gap_no_export",
                "broad_location_no_export",
            }
            or _v2_guided_selection_report.get("filter_mode") == "location_category_gap_no_export"
        )

        _selected_empty = (
            _selected_for_downstream is None
            or getattr(_selected_for_downstream, "empty", False)
            or len(_selected_for_downstream) == 0
        )

        if _block_export or _selected_empty:
            _run_dir = Path(str(locals().get("run_dir")))
            _run_dir.mkdir(parents=True, exist_ok=True)

            _coverage_warnings = list(locals().get("coverage_warnings") or [])
            for _warning in (_prompt_filter_report.get("coverage_warnings") or []):
                if _warning not in _coverage_warnings:
                    _coverage_warnings.append(_warning)
            for _warning in (_v2_guided_selection_report.get("coverage_warnings") or []):
                if _warning not in _coverage_warnings:
                    _coverage_warnings.append(_warning)

            _final_summary_path = _run_dir / "final_summary.md"
            _final_summary = "\n".join(
                [
                    "# Audience Intelligence Result",
                    "",
                    f"Prompt: {prompt}",
                    f"Run ID: {run_id}",
                    "",
                    "## Status",
                    "",
                    "No export-ready cohort was created because the requested location/category/daypart combination has no exact safe cohort.",
                    "",
                    f"Filter mode: {_prompt_filter_report.get('filter_mode') or _v2_guided_selection_report.get('filter_mode')}",
                    f"Downstream export enabled: False",
                    "",
                    "## Coverage warnings",
                    "",
                    *[f"- {_warning}" for _warning in _coverage_warnings],
                    "",
                ]
            )
            _final_summary_path.write_text(_final_summary)

            return {
                "status": "completed",
                "run_id": run_id,
                "prompt": prompt,
                "run_dir": str(_run_dir),
                "source_mode": locals().get("source_mode"),
                "source_rows": locals().get("source_rows"),
                "source_columns": locals().get("source_columns"),
                "privacy_cohorts": int(len(locals().get("privacy_cohorts"))) if locals().get("privacy_cohorts") is not None else 0,
                "prompt_selected_cohorts": 0,
                "prompt_filter_report": _prompt_filter_report,
                "coverage_warnings": _coverage_warnings,
                "final_summary_path": str(_final_summary_path),
                "v2_autonomous": locals().get("v2_result") or {},
                "v2_swarm_review": locals().get("v2_swarm_review") or {},
                "v2_guided_selection_report": _v2_guided_selection_report,
                "safe_export": {
                    "approval_status": "blocked_no_safe_exact_match",
                    "downstream_export_enabled": False,
                    "exported_cohorts": 0,
                    "exported_lookalike_pairs": 0,
                    "outputs": {},
                },
                "privacy_guarantees": {
                    "raw_maids_exported": False,
                    "hashed_identifiers_exported": False,
                    "raw_observations_exported": False,
                    "raw_lat_lng_exported": False,
                    "raw_email_exported": False,
                    "raw_phone_exported": False,
                    "individual_user_data_exported": False,
                },
            }

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

    def _select_cohorts_from_v2_ranked(
        self,
        *,
        privacy_cohorts,
        v2_result,
        prompt_filter_report=None,
        max_rows=10,
    ):
        import os
        from pathlib import Path
        import pandas as pd

        prompt_filter_report = prompt_filter_report or {}
        v2_result = v2_result or {}

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
            for term in (
                prompt_filter_report.get("poi_terms_detected")
                or prompt_filter_report.get("poi_terms")
                or []
            )
            if str(term).strip()
        ]

        requested_locations = [
            str(term).strip().lower()
            for term in (
                prompt_filter_report.get("locations_detected")
                or prompt_filter_report.get("locations")
                or []
            )
            if str(term).strip()
        ]

        requested_dayparts = [
            str(term).strip().lower()
            for term in (
                prompt_filter_report.get("dayparts_detected")
                or prompt_filter_report.get("dayparts")
                or []
            )
            if str(term).strip()
        ]

        strict_category_requested = bool(requested_poi_terms)
        allowed_export_pois = self._allowed_export_poi_terms_for_request(requested_poi_terms)

        report = {
            "enabled": False,
            "reason": "no_ranked_matches",
            "ranked_path": ranked_path,
            "min_score": min_score,
            "strong_category_score": strong_category_score,
            "allowed_match_types": sorted(allowed_match_types),
            "strict_category_requested": strict_category_requested,
            "requested_poi_terms": requested_poi_terms,
            "requested_locations": requested_locations,
            "requested_dayparts": requested_dayparts,
            "allowed_export_poi_terms": allowed_export_pois,
            "rows": 0,
            "block_export": False,
            "downstream_export_enabled": True,
        }

        if not ranked_path:
            return pd.DataFrame(), report

        ranked_file = Path(ranked_path)
        if not ranked_file.exists():
            report["reason"] = "ranked_matches_path_missing"
            return pd.DataFrame(), report

        ranked = pd.read_csv(ranked_file)
        if ranked.empty:
            report["reason"] = "empty_ranked_matches"
            return pd.DataFrame(), report

        candidate = ranked.copy()

        if "privacy_status" in candidate.columns:
            candidate = candidate[
                candidate["privacy_status"].fillna("").astype(str).str.lower().isin(["passed", "pass", "safe", ""])
            ].copy()

        if "final_match_score" in candidate.columns:
            candidate["final_match_score"] = pd.to_numeric(candidate["final_match_score"], errors="coerce").fillna(0)
            candidate = candidate[candidate["final_match_score"] >= min_score].copy()

        if "match_type" in candidate.columns:
            candidate = candidate[candidate["match_type"].astype(str).isin(allowed_match_types)].copy()

        if requested_locations and "location_name" in candidate.columns:
            candidate = candidate[
                candidate["location_name"].apply(
                    lambda loc: any(self._vijay_location_matches_request(loc, req) for req in requested_locations)
                )
            ].copy()

        if allowed_export_pois and "primary_poi_type" in candidate.columns:
            candidate = candidate[
                candidate["primary_poi_type"].apply(
                    lambda poi: self._vijay_poi_matches_allowed(poi, allowed_export_pois)
                )
            ].copy()

        if "category_match_score" in candidate.columns:
            candidate["category_match_score"] = pd.to_numeric(candidate["category_match_score"], errors="coerce").fillna(0)
            if strict_category_requested:
                candidate = candidate[candidate["category_match_score"] >= strong_category_score].copy()
            else:
                candidate = candidate[candidate["category_match_score"] >= 0.50].copy()

        if requested_dayparts and "created_day_part" in candidate.columns:
            allowed_dayparts = {self._vijay_norm_text(x) for x in requested_dayparts}
            candidate = candidate[
                candidate["created_day_part"].apply(lambda x: self._vijay_norm_text(x) in allowed_dayparts)
            ].copy()

        if candidate.empty:
            report["enabled"] = True
            report["reason"] = "strict_location_category_requested_but_no_safe_exact_match"
            report["filter_mode"] = "location_category_gap_no_export"
            report["block_export"] = True
            report["downstream_export_enabled"] = False
            return pd.DataFrame(), report

        missing_requested_locations = []
        if requested_locations and "location_name" in candidate.columns:
            for requested_location in requested_locations:
                has_location = any(
                    self._vijay_location_matches_request(cohort_location, requested_location)
                    for cohort_location in candidate["location_name"].dropna().tolist()
                )
                if not has_location:
                    missing_requested_locations.append(str(requested_location))

        if missing_requested_locations:
            report["enabled"] = True
            report["reason"] = "strict_location_category_requested_but_no_safe_exact_match"
            report["filter_mode"] = "location_category_gap_no_export"
            report["block_export"] = True
            report["downstream_export_enabled"] = False
            report["missing_requested_locations"] = missing_requested_locations
            report["coverage_warnings"] = [
                f"{location} was requested, but no export-ready cohort for that location passed the final quality and safety filters."
                for location in missing_requested_locations
            ]
            return pd.DataFrame(), report

        sort_cols = [
            col for col in ["final_match_score", "quality_score", "total_maid_volume"]
            if col in candidate.columns
        ]
        if sort_cols:
            candidate = candidate.sort_values(sort_cols, ascending=[False] * len(sort_cols)).copy()

        candidate = candidate.head(max_rows).copy()

        selected = candidate

        if privacy_cohorts is not None and hasattr(privacy_cohorts, "columns"):
            if "cohort_id" in candidate.columns and "cohort_id" in privacy_cohorts.columns:
                ids = candidate["cohort_id"].dropna().tolist()
                selected = privacy_cohorts[privacy_cohorts["cohort_id"].isin(ids)].copy()

                if not selected.empty:
                    order = {value: idx for idx, value in enumerate(ids)}
                    selected["_vijay_order"] = selected["cohort_id"].map(order).fillna(999999)
                    selected = selected.sort_values("_vijay_order").drop(columns=["_vijay_order"]).head(max_rows).copy()

        report["enabled"] = True
        report["reason"] = "v2_ranked_safe_selection"
        report["rows"] = int(len(selected))
        report["filter_mode"] = "location+poi+daypart" if requested_locations and allowed_export_pois and requested_dayparts else "v2_ranked_safe"
        report["block_export"] = False
        report["downstream_export_enabled"] = True

        return selected, report

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

    def _select_cohorts_for_prompt(self, *, prompt: str, cohorts: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        df = cohorts.copy()
        lower = prompt.lower().replace("-", " ")

        locations = self._extract_location_terms(prompt, df)
        pois = self._extract_poi_terms(prompt)

        daypart_aliases = {
            "morning": ["morning", "breakfast", "early morning", "before work", "early in the day"],
            "afternoon": ["afternoon", "lunch", "noon", "afternoon hours"],
            "evening": [
                "evening",
                "dinner",
                "dinner time",
                "after work",
                "after office",
                "after office hours",
                "after work hours",
                "post work",
                "late evening",
            ],
            "night": ["night", "late night", "midnight", "go out late"],
            "weekend": ["weekend", "weekends", "saturday", "sunday"],
            "weekday": ["weekday", "weekdays", "monday", "tuesday", "wednesday", "thursday", "friday"],
        }

        dayparts = []
        for daypart, aliases in daypart_aliases.items():
            if any(alias in lower for alias in aliases):
                dayparts.append(daypart)

        attempts = []

        def token_match(series: pd.Series, terms: list[str]) -> pd.Series:
            term_mask = pd.Series(False, index=df.index)
            normalized = series.astype(str).str.lower().str.replace(" ", "_", regex=False)

            for term in terms:
                term_norm = str(term).lower().replace(" ", "_").replace("-", "_")
                term_tokens = [t for t in re.split(r"[^a-z0-9]+", term_norm) if t]

                direct = normalized.str.contains(re.escape(term_norm), na=False)

                token_based = pd.Series(False, index=df.index)
                for token in term_tokens:
                    if len(token) >= 4:
                        token_based = token_based | normalized.str.contains(re.escape(token), na=False)

                term_mask = term_mask | direct | token_based

            return term_mask

        def apply_filter(use_locations: bool, use_pois: bool, use_dayparts: bool) -> pd.DataFrame:
            mask = pd.Series(True, index=df.index)

            if use_locations and locations and "location_name" in df.columns:
                mask = mask & token_match(df["location_name"], locations)

            if use_pois and pois and "primary_poi_type" in df.columns:
                mask = mask & token_match(df["primary_poi_type"], pois)

            if use_dayparts and dayparts and "created_day_part" in df.columns:
                day_mask = df["created_day_part"].astype(str).str.lower().isin(dayparts)
                mask = mask & day_mask

            return df[mask].copy()

        strategies = [
            ("location+poi+daypart", True, True, True),
            ("location+poi", True, True, False),
            ("location+daypart", True, False, True),
            ("poi+daypart", False, True, True),
            ("location", True, False, False),
            ("poi", False, True, False),
            ("daypart", False, False, True),
            ("all", False, False, False),
        ]

        selected = df.copy()
        mode = "all"

        for strategy_name, use_locations, use_pois, use_dayparts in strategies:
            candidate = apply_filter(use_locations, use_pois, use_dayparts)
            attempts.append({"strategy": strategy_name, "rows": int(len(candidate))})

            if len(candidate) >= 2:
                selected = candidate
                mode = strategy_name
                break

        if "high quality" in lower or "high quality" in lower or "premium" in lower or "best" in lower:
            quality_col = "quality_score" if "quality_score" in selected.columns else None
            if quality_col:
                selected = selected.sort_values(quality_col, ascending=False)

        return selected.reset_index(drop=True), {
            "filter_mode": mode,
            "locations_detected": locations,
            "poi_terms_detected": pois,
            "dayparts_detected": dayparts,
            "attempts": attempts,
            "high_quality_requested": bool(
                "high quality" in lower or "premium" in lower or "best" in lower
            ),
        }

    def _extract_location_terms(self, prompt: str, df: pd.DataFrame) -> list[str]:
        lower = prompt.lower()
        found = []

        if "location_name" in df.columns:
            for location in df["location_name"].dropna().astype(str).str.lower().unique():
                location = location.strip()
                if len(location) < 3:
                    continue

                if location in lower:
                    found.append(location)
                    continue

                parts = [p for p in re.split(r"[\s,]+", location) if len(p) >= 4]
                if any(part in lower for part in parts):
                    found.append(location)

        manual = [
            "montreal downtown",
            "montreal qc",
            "san francisco",
            "montreal",
            "new york",
            "los angeles",
            "toronto",
            "quebec",
            "vancouver",
            "chicago",
        ]
        for item in manual:
            if item in lower and item not in found:
                found.append(item)

        return list(dict.fromkeys(found))

    def _extract_poi_terms(self, prompt: str) -> list[str]:
        lower = prompt.lower().replace("-", " ")
        terms = []

        mapping = {
            "restaurant": ["restaurant", "food", "fast_food", "quick_service_food"],
            "food": ["food", "restaurant", "fast_food"],
            "quick service": ["restaurant", "fast_food", "food"],
            "quick bites": ["restaurant", "fast_food", "food"],
            "fast meals": ["restaurant", "fast_food", "food"],
            "casual dining": ["restaurant", "food"],
            "cafe": ["cafe", "coffee"],
            "coffee": ["cafe", "coffee"],
            "espresso": ["cafe", "coffee"],
            "gym": ["gym", "fitness", "health_club"],
            "fitness": ["gym", "fitness", "health_club"],
            "workout": ["gym", "fitness", "health_club"],
            "yoga": ["gym", "fitness", "health_club"],
            "health": ["health"],
            "barber": ["barber_shop", "barber"],
            "tattoo": ["body_art_service", "tattoo"],
            "body ink": ["body_art_service", "tattoo"],
            "piercing": ["body_art_service", "tattoo"],
            "office": ["coworking_space", "office", "corporate_office", "point_of_interest"],
            "coworking": ["coworking_space", "office", "corporate_office", "point_of_interest"],
            "business": ["coworking_space", "office", "corporate_office", "point_of_interest"],
            "professional": ["coworking_space", "office", "corporate_office", "point_of_interest"],
            "store": ["store", "retail", "shopping_mall"],
            "retail": ["store", "retail", "shopping_mall"],
            "shopping": ["store", "retail", "shopping_mall"],
            "clinic": ["clinic", "healthcare", "hospital", "pharmacy"],
            "pharmacy": ["pharmacy", "healthcare", "clinic"],
            "healthcare": ["healthcare", "clinic", "hospital", "pharmacy"],
            "student": ["school", "college", "university", "campus"],
            "campus": ["school", "college", "university", "campus"],
            "nightlife": ["bar", "pub", "club", "lounge"],
            "wellness": ["salon", "spa", "beauty"],
            "grooming": ["salon", "spa", "beauty", "barber_shop"],
            "self care": ["salon", "spa", "beauty"],
            "self-care": ["salon", "spa", "beauty"],
        }

        for key, values in mapping.items():
            if key in lower:
                terms.extend(values)

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


    # === Vijay local recovery guardrails: START ===
    # Local recovery block for audience-intelligence-agents.
    # Keep these methods inside AudienceIntelligenceOrchestratorAgent.
    # Do not paste shell commands into this Python file.

    def _vijay_norm_text(self, value):
        import re

        if value is None:
            return ""
        value = str(value).lower().replace("&", " and ")
        value = value.replace("_", " ")
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    def _vijay_snake(self, value):
        return self._vijay_norm_text(value).replace(" ", "_")

    def _vijay_dedupe(self, values):
        out = []
        seen = set()
        for value in values or []:
            if value is None:
                continue
            raw = str(value).strip().lower()
            if not raw:
                continue
            if raw not in seen:
                out.append(raw)
                seen.add(raw)
        return out

    def _vijay_has_any(self, text, terms):
        norm = self._vijay_norm_text(text)
        return any(self._vijay_norm_text(term) in norm for term in terms)

    def _vijay_is_coffee_intent(self, terms_or_prompt):
        text = " ".join(terms_or_prompt) if isinstance(terms_or_prompt, (list, tuple, set)) else str(terms_or_prompt)
        return self._vijay_has_any(
            text,
            [
                "coffee",
                "caffeine",
                "espresso",
                "latte",
                "cappuccino",
                "cafe",
                "cafes",
                "café",
                "coffee shop",
                "coffee_shop",
                "snacks after work",
                "caffeine break",
            ],
        )

    def _vijay_is_restaurant_intent(self, terms_or_prompt):
        text = " ".join(terms_or_prompt) if isinstance(terms_or_prompt, (list, tuple, set)) else str(terms_or_prompt)
        return self._vijay_has_any(
            text,
            [
                "restaurant",
                "restaurants",
                "shawarma",
                "food",
                "dining",
                "eatery",
                "lunch",
                "dinner",
                "takeaway",
            ],
        )

    def _vijay_is_retail_intent(self, terms_or_prompt):
        text = " ".join(terms_or_prompt) if isinstance(terms_or_prompt, (list, tuple, set)) else str(terms_or_prompt)
        return self._vijay_has_any(
            text,
            [
                "retail",
                "shopping",
                "shopping mall",
                "mall",
                "fashion",
                "apparel",
                "clothing",
                "department store",
                "shoppers",
            ],
        )

    def _allowed_export_poi_terms_for_request(self, poi_terms):
        terms = [self._vijay_norm_text(t) for t in (poi_terms or [])]
        joined = " ".join(terms)

        coffee = self._vijay_is_coffee_intent(joined)
        restaurant = self._vijay_is_restaurant_intent(joined)
        retail = self._vijay_is_retail_intent(joined)

        # Coffee/cafe is narrow. If restaurant was also explicitly detected,
        # allow food-related POIs, but never mall/retail fallback unless mall intent is explicit.
        if coffee:
            allowed = ["cafe", "coffee", "coffee_shop", "bakery"]
            if restaurant:
                allowed += [
                    "restaurant",
                    "meal_takeaway",
                    "food",
                ]
            if retail:
                # Explicit retail/mall intent only. Plain "coffee shop" must not become shopping_mall.
                explicit_mall = any(t in joined for t in ["shopping mall", "mall", "fashion", "apparel", "clothing"])
                if explicit_mall:
                    allowed += ["shopping_mall", "retail"]
            return self._vijay_dedupe(allowed)

        if restaurant:
            return [
                "restaurant",
                "shawarma_restaurant",
                "middle_eastern_restaurant",
                "fast_food_restaurant",
                "meal_takeaway",
                "food_court",
                "food",
            ]

        if self._vijay_has_any(joined, ["coworking", "office", "business center", "business centre", "workspace"]):
            return [
                "coworking_space",
                "corporate_office",
                "office",
                "business_center",
                "business_centre",
            ]

        if self._vijay_has_any(joined, ["casino", "gaming", "gambling"]):
            return [
                "casino",
                "gaming_venue",
                "tourist_attraction",
                "entertainment",
            ]

        if retail:
            return [
                "retail",
                "store",
                "shopping_mall",
                "clothing_store",
                "shoe_store",
                "department_store",
                "fashion",
            ]

        return self._vijay_dedupe(poi_terms or [])

    def _extract_poi_terms(self, prompt):
        text = self._vijay_norm_text(prompt)

        # Priority rule: caffeine/coffee after office means cafe audience in evening,
        # not office/coworking audience.
        if self._vijay_is_coffee_intent(text):
            return ["cafe", "coffee", "coffee_shop"]

        if self._vijay_has_any(text, ["casino", "casinos", "gaming venues", "gaming venue", "tourist entertainment"]):
            return ["casino", "gaming_venue", "tourist_attraction", "entertainment"]

        if self._vijay_has_any(
            text,
            [
                "coworking",
                "coworking hubs",
                "business centers",
                "business centres",
                "flexible workspace",
                "flexible workspaces",
                "workspace",
                "workspaces",
            ],
        ):
            return ["coworking_space", "corporate_office", "office"]

        if self._vijay_is_restaurant_intent(text):
            if "shawarma" in text:
                return ["shawarma_restaurant", "middle_eastern_restaurant", "restaurant"]
            return ["restaurant", "food", "meal_takeaway"]

        if self._vijay_is_retail_intent(text):
            return ["retail", "shopping_mall", "store", "fashion"]

        return []

    def _extract_daypart_terms_for_prompt(self, prompt):
        text = self._vijay_norm_text(prompt)
        dayparts = []

        if self._vijay_has_any(text, ["morning", "breakfast", "before work"]):
            dayparts.append("morning")
        if self._vijay_has_any(text, ["afternoon", "lunch"]):
            dayparts.append("afternoon")
        if self._vijay_has_any(
            text,
            [
                "evening",
                "after work",
                "after office",
                "post work",
                "post office",
                "dinner",
                "night",
                "late night",
                "office hours",
            ],
        ):
            dayparts.append("evening")

        return self._vijay_dedupe(dayparts)

    def _extract_location_terms(self, prompt, cohorts=None):
        import re

        prompt_norm = self._vijay_norm_text(prompt)
        available = []

        if cohorts is not None and hasattr(cohorts, "columns") and "location_name" in cohorts.columns:
            try:
                available = [
                    str(x).strip().lower()
                    for x in cohorts["location_name"].dropna().unique().tolist()
                    if str(x).strip()
                ]
            except Exception:
                available = []

        # Explicit known specific locations first. This prevents
        # "Westmount Montreal" from being reduced to available broad "montreal".
        specific_known = [
            ("times square new york", "times square, new york"),
            ("times square, new york", "times square, new york"),
            ("westmount montreal", "westmount montreal"),
        ]

        for needle, canonical in specific_known:
            if self._vijay_norm_text(needle) in prompt_norm:
                return [canonical]

        # Special case: prompt says only New York, do not auto-expand to Times Square.
        if "new york" in prompt_norm and "times square" not in prompt_norm:
            for loc in available:
                if self._vijay_norm_text(loc) == "new york":
                    return [loc]
            return ["new york"]

        matched = []
        for loc in available:
            loc_norm = self._vijay_norm_text(loc)
            if not loc_norm:
                continue
            if loc_norm in prompt_norm:
                matched.append(loc)

        if matched:
            matched_sorted = sorted(
                matched,
                key=lambda x: len(self._vijay_norm_text(x).split()),
                reverse=True,
            )
            most_specific = matched_sorted[0]
            most_specific_norm = self._vijay_norm_text(most_specific)

            filtered = []
            for loc in matched_sorted:
                loc_norm = self._vijay_norm_text(loc)
                if loc_norm == most_specific_norm:
                    filtered.append(loc)
                    continue
                if loc_norm in most_specific_norm and loc_norm != most_specific_norm:
                    continue
                filtered.append(loc)

            return self._vijay_dedupe(filtered)

        known_locations = [
            "san francisco",
            "montreal",
            "quebec",
            "canada",
            "usa",
            "united states",
            "india",
            "california",
            "ontario",
        ]

        found = []
        for loc in known_locations:
            if self._vijay_norm_text(loc) in prompt_norm:
                found.append(loc)

        if found:
            return [sorted(found, key=lambda x: len(self._vijay_norm_text(x).split()), reverse=True)[0]]

        m = re.search(r"\bnear\s+([a-zA-Z][a-zA-Z\s,.-]{2,80})", str(prompt))
        if m:
            phrase = m.group(1)
            phrase = re.split(r"[.?!;:]", phrase)[0]
            phrase = re.sub(r"\b(find|people|who|visit|during|hours|for|with)\b.*$", "", phrase, flags=re.I)
            phrase = phrase.strip(" ,.-").lower()
            if phrase:
                return [phrase]

        return []

    def _run_broad_location_guardrail(self, prompt_filter_report):
        broad_locations = {
            "canada",
            "usa",
            "united states",
            "united states of america",
            "india",
            "california",
            "ontario",
            "quebec province",
            "british columbia",
            "texas",
            "new york state",
        }

        locations = [
            self._vijay_norm_text(x)
            for x in (prompt_filter_report or {}).get("locations_detected", [])
            if self._vijay_norm_text(x)
        ]

        blocked_locations = [x for x in locations if x in broad_locations]
        blocked = bool(blocked_locations)

        return {
            "enabled": True,
            "blocked": blocked,
            "block_export": blocked,
            "export_blocked": blocked,
            "should_block": blocked,
            "export_allowed": not blocked,
            "should_export": not blocked,
            "downstream_export_enabled": not blocked,
            "filter_mode": "broad_location_no_export" if blocked else "allowed",
            "reason": "country_or_broad_location_requires_city_area" if blocked else "location_scope_ok",
            "blocked_locations": blocked_locations,
            "locations_detected": locations,
        }

    def _vijay_is_broad_city_request(self, requested_location):
        loc = self._vijay_norm_text(requested_location)
        broad_city_names = {
            "montreal",
            "san francisco",
            "new york",
            "quebec",
            "chicago",
            "toronto",
            "vancouver",
            "hyderabad",
            "bangalore",
            "mumbai",
            "delhi",
        }
        return loc in broad_city_names

    def _vijay_location_matches_request(self, cohort_location, requested_location):
        cohort = self._vijay_norm_text(cohort_location)
        requested = self._vijay_norm_text(requested_location)

        if not requested:
            return True
        if not cohort:
            return False
        if cohort == requested:
            return True

        # Broad city can match subareas.
        if self._vijay_is_broad_city_request(requested):
            return cohort.startswith(requested + " ") or cohort.endswith(" " + requested) or requested in cohort

        # Specific subarea must NOT fallback to broader city.
        return False

    def _vijay_poi_matches_allowed(self, cohort_poi, allowed_terms):
        poi = self._vijay_snake(cohort_poi)
        allowed = [self._vijay_snake(x) for x in (allowed_terms or []) if self._vijay_norm_text(x)]

        if not allowed:
            return True

        for term in allowed:
            if poi == term:
                return True
            if term in {"coffee", "coffee_shop"} and poi in {"cafe", "coffee_shop"}:
                return True
            if term == "restaurant" and poi.endswith("_restaurant"):
                return True
            if term == "food" and poi in {"restaurant", "food_court", "meal_takeaway"}:
                return True

        return False

    def _filter_to_requested_export_category(self, cohorts, prompt_filter_report):
        import pandas as pd

        if cohorts is None:
            cohorts = pd.DataFrame()

        report = dict(prompt_filter_report or {})
        selected = cohorts.copy()

        locations = report.get("locations_detected") or report.get("locations") or []
        poi_terms = report.get("poi_terms_detected") or report.get("poi_terms") or []
        dayparts = report.get("dayparts_detected") or report.get("dayparts") or []

        allowed_pois = self._allowed_export_poi_terms_for_request(poi_terms)

        report["enabled"] = True
        report["block_export"] = False
        report["export_blocked"] = False

        if selected.empty:
            report["filter_mode"] = "location_category_gap_no_export"
            report["block_export"] = True
            report["export_blocked"] = True
            report["allowed_export_poi_terms"] = allowed_pois
            return selected, report

        if "privacy_status" in selected.columns:
            selected = selected[
                selected["privacy_status"].fillna("").astype(str).str.lower().isin(["passed", "pass", "safe", ""])
            ].copy()

        if locations and "location_name" in selected.columns:
            selected = selected[
                selected["location_name"].apply(
                    lambda loc: any(self._vijay_location_matches_request(loc, req) for req in locations)
                )
            ].copy()

        location_matched_count = len(selected)

        if allowed_pois and "primary_poi_type" in selected.columns:
            selected = selected[
                selected["primary_poi_type"].apply(
                    lambda poi: self._vijay_poi_matches_allowed(poi, allowed_pois)
                )
            ].copy()

        category_matched_count = len(selected)

        if dayparts and "created_day_part" in selected.columns:
            daypart_norms = {self._vijay_norm_text(x) for x in dayparts}
            selected = selected[
                selected["created_day_part"].apply(lambda x: self._vijay_norm_text(x) in daypart_norms)
            ].copy()

        if selected.empty and locations and allowed_pois:
            report["filter_mode"] = "location_category_gap_no_export"
            report["block_export"] = True
            report["export_blocked"] = True
            report["coverage_warnings"] = report.get("coverage_warnings", []) + [
                "requested location/category/daypart had no exact safe cohort; export blocked instead of falling back."
            ]
        else:
            parts = []
            if locations:
                parts.append("location")
            if allowed_pois:
                parts.append("poi")
            if dayparts:
                parts.append("daypart")
            report["filter_mode"] = "+".join(parts) if parts else "all"

        report["locations_detected"] = locations
        report["poi_terms_detected"] = poi_terms
        report["dayparts_detected"] = dayparts
        report["allowed_export_poi_terms"] = allowed_pois
        report["location_matched_count"] = int(location_matched_count)
        report["category_matched_count"] = int(category_matched_count)
        missing_requested_locations = []
        if locations and "location_name" in selected.columns:
            for requested_location in locations:
                has_location = any(
                    self._vijay_location_matches_request(cohort_location, requested_location)
                    for cohort_location in selected["location_name"].dropna().tolist()
                )
                if not has_location:
                    missing_requested_locations.append(str(requested_location))

        if missing_requested_locations:
            selected = selected.iloc[0:0].copy()
            report["filter_mode"] = "location_category_gap_no_export"
            report["block_export"] = True
            report["export_blocked"] = True
            report["downstream_export_enabled"] = False
            report["missing_requested_locations"] = missing_requested_locations
            report["coverage_warnings"] = report.get("coverage_warnings", []) + [
                f"{location} was requested, but no export-ready cohort for that location passed the final quality and safety filters."
                for location in missing_requested_locations
            ]

        report["selected_count"] = int(len(selected))

        return selected, report

    def _select_cohorts_for_prompt(self, prompt, cohorts):
        import pandas as pd

        if cohorts is None:
            cohorts = pd.DataFrame()

        locations = self._extract_location_terms(prompt, cohorts)
        poi_terms = self._extract_poi_terms(prompt)
        dayparts = self._extract_daypart_terms_for_prompt(prompt)

        report = {
            "enabled": True,
            "locations_detected": locations,
            "poi_terms_detected": poi_terms,
            "dayparts_detected": dayparts,
        }

        broad_report = self._run_broad_location_guardrail(report)
        if broad_report.get("block_export"):
            empty = cohorts.iloc[0:0].copy() if hasattr(cohorts, "iloc") else pd.DataFrame()
            report.update(broad_report)
            report["filter_mode"] = "broad_location_no_export"
            return empty, report

        selected, report = self._filter_to_requested_export_category(cohorts, report)

        if not selected.empty:
            sort_cols = [
                col for col in ["final_match_score", "quality_score", "total_maid_volume"]
                if col in selected.columns
            ]
            if sort_cols:
                selected = selected.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

        return selected, report

    def _strict_category_export_guardrail(self, prompt_filter_report, selected_cohorts, v2_result=None):
        import pandas as pd

        report = dict(prompt_filter_report or {})
        selected = selected_cohorts if selected_cohorts is not None else pd.DataFrame()
        v2_result = v2_result or {}

        locations = report.get("locations_detected") or report.get("locations") or []
        poi_terms = report.get("poi_terms_detected") or report.get("poi_terms") or []
        dayparts = report.get("dayparts_detected") or report.get("dayparts") or []
        allowed_pois = self._allowed_export_poi_terms_for_request(poi_terms)

        reasons = []

        if report.get("filter_mode") in {"location_category_gap_no_export", "broad_location_no_export"}:
            reasons.append(report.get("filter_mode"))

        if selected is None or getattr(selected, "empty", True):
            if locations and allowed_pois:
                reasons.append("empty_selected_cohorts_for_requested_location_category")

        if selected is not None and not getattr(selected, "empty", True):
            if locations and "location_name" in selected.columns:
                bad_locations = [
                    str(x)
                    for x in selected["location_name"].dropna().tolist()
                    if not any(self._vijay_location_matches_request(x, req) for req in locations)
                ]
                if bad_locations:
                    reasons.append("selected_location_mismatch")

            if allowed_pois and "primary_poi_type" in selected.columns:
                bad_pois = [
                    str(x)
                    for x in selected["primary_poi_type"].dropna().tolist()
                    if not self._vijay_poi_matches_allowed(x, allowed_pois)
                ]
                if bad_pois:
                    reasons.append("selected_category_mismatch")

            if dayparts and "created_day_part" in selected.columns:
                allowed_dayparts = {self._vijay_norm_text(x) for x in dayparts}
                bad_dayparts = [
                    str(x)
                    for x in selected["created_day_part"].dropna().tolist()
                    if self._vijay_norm_text(x) not in allowed_dayparts
                ]
                if bad_dayparts:
                    reasons.append("selected_daypart_mismatch")

        reasons = self._vijay_dedupe(reasons)
        blocked = bool(reasons)

        if not blocked:
            reason = "safe_to_export"
        elif "location_category_gap_no_export" in reasons or "empty_selected_cohorts_for_requested_location_category" in reasons:
            reason = "strict_location_category_requested_but_no_safe_exact_match"
        elif "selected_location_mismatch" in reasons:
            reason = "strict_location_category_requested_but_selected_location_mismatch"
        elif "selected_category_mismatch" in reasons:
            reason = "strict_location_category_requested_but_selected_category_mismatch"
        elif "selected_daypart_mismatch" in reasons:
            reason = "strict_location_category_requested_but_selected_daypart_mismatch"
        else:
            reason = reasons[0]

        return {
            "enabled": True,
            "blocked": blocked,
            "block_export": blocked,
            "export_blocked": blocked,
            "should_block": blocked,
            "export_allowed": not blocked,
            "should_export": not blocked,
            "reason": reason,
            "reasons": reasons,
            "filter_mode": "blocked" if blocked else "allowed",
            "locations_detected": locations,
            "poi_terms_detected": poi_terms,
            "dayparts_detected": dayparts,
            "allowed_export_poi_terms": allowed_pois,
        }

    def _build_hybrid_retrieval_intent(self, prompt, prompt_filter_report=None, privacy_cohorts=None):
        prompt_filter_report = dict(prompt_filter_report or {})

        locations = (
            prompt_filter_report.get("locations_detected")
            or prompt_filter_report.get("locations")
            or self._extract_location_terms(prompt, privacy_cohorts)
        )
        poi_terms = (
            prompt_filter_report.get("poi_terms_detected")
            or prompt_filter_report.get("poi_terms")
            or self._extract_poi_terms(prompt)
        )
        dayparts = (
            prompt_filter_report.get("dayparts_detected")
            or prompt_filter_report.get("dayparts")
            or self._extract_daypart_terms_for_prompt(prompt)
        )

        business_intent = prompt_filter_report.get("business_intent") or "unknown_business_intent"

        prompt_norm = self._vijay_norm_text(prompt)
        available_pois = []
        if privacy_cohorts is not None and hasattr(privacy_cohorts, "columns") and "primary_poi_type" in privacy_cohorts.columns:
            available_pois = [
                str(x).strip().lower()
                for x in privacy_cohorts["primary_poi_type"].dropna().unique().tolist()
                if str(x).strip()
            ]

        if business_intent == "unknown_business_intent":
            if self._vijay_has_any(prompt_norm, ["casino", "casinos", "gaming"]):
                business_intent = "casino"
                if "casino" not in poi_terms:
                    poi_terms = ["casino"] + list(poi_terms or [])
            elif self._vijay_is_coffee_intent(prompt_norm):
                business_intent = "coffee_cafe"
            elif self._vijay_is_restaurant_intent(prompt_norm):
                business_intent = "restaurant_food"
            elif self._vijay_is_retail_intent(prompt_norm):
                business_intent = "retail_shopping"

        allowed = self._allowed_export_poi_terms_for_request(poi_terms)
        matched_available = [
            poi for poi in available_pois
            if self._vijay_poi_matches_allowed(poi, allowed)
        ]

        return {
            "business_intent": business_intent,
            "locations": self._vijay_dedupe(locations),
            "locations_detected": self._vijay_dedupe(locations),
            "poi_terms": self._vijay_dedupe(poi_terms),
            "poi_terms_detected": self._vijay_dedupe(poi_terms),
            "allowed_export_poi_terms": allowed,
            "matched_available_poi_types": self._vijay_dedupe(matched_available or allowed),
            "requested_categories": self._vijay_dedupe(allowed),
            "dayparts": self._vijay_dedupe(dayparts),
            "dayparts_detected": self._vijay_dedupe(dayparts),
            "retrieval_mode": "hybrid_safe_derived",
            "confidence_score": 0.85 if business_intent != "unknown_business_intent" else 0.55,
        }

    def _merge_v2_intent_into_prompt_filter_report(self, prompt_filter_report, v2_result):
        merged = dict(prompt_filter_report or {})
        v2_result = v2_result or {}
        intent = v2_result.get("prompt_intent") or v2_result.get("intent") or {}

        def empty(value):
            return value is None or value == [] or value == "" or value == "all"

        v2_locations = intent.get("locations") or intent.get("locations_detected") or []
        v2_pois = (
            intent.get("matched_available_poi_types")
            or intent.get("poi_terms_detected")
            or intent.get("poi_terms")
            or []
        )
        v2_dayparts = intent.get("dayparts") or intent.get("dayparts_detected") or []

        if empty(merged.get("locations_detected")) and v2_locations:
            canonical_locations = [self._vijay_snake(x) for x in v2_locations]
            merged["locations_detected"] = self._vijay_dedupe(canonical_locations)
            merged["locations"] = self._vijay_dedupe(canonical_locations)

        if empty(merged.get("poi_terms_detected")) and v2_pois:
            canonical_pois = [self._vijay_snake(x) for x in v2_pois]
            merged["poi_terms_detected"] = self._vijay_dedupe(canonical_pois)
            merged["poi_terms"] = self._vijay_dedupe(canonical_pois)

        if empty(merged.get("dayparts_detected")) and v2_dayparts:
            canonical_dayparts = [self._vijay_norm_text(x) for x in v2_dayparts]
            merged["dayparts_detected"] = self._vijay_dedupe(canonical_dayparts)
            merged["dayparts"] = self._vijay_dedupe(canonical_dayparts)

        parts = []
        if merged.get("locations_detected"):
            parts.append("location")
        if merged.get("poi_terms_detected"):
            parts.append("poi")
        if merged.get("dayparts_detected"):
            parts.append("daypart")

        if parts:
            merged["filter_mode"] = "+".join(parts)

        merged["v2_intent_merged"] = bool(parts)
        merged["v2_resolver_mode"] = intent.get("resolver_mode")
        merged["v2_confidence_score"] = intent.get("confidence_score")
        merged["v2_llm_used"] = intent.get("llm_used")
        merged["ai_intent_merge"] = {
            "llm_used": intent.get("llm_used"),
            "resolver_mode": intent.get("resolver_mode"),
            "confidence_score": intent.get("confidence_score"),
            "data_gap_likely": intent.get("data_gap_likely"),
        }

        return merged

    # === Vijay local recovery guardrails: END ===


