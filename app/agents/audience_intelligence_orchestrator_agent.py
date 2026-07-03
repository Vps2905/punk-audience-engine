from __future__ import annotations

import argparse
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


class AudienceIntelligenceOrchestratorAgent:
    """
    Runs the full Audience Intelligence pipeline from a business prompt.

    Full flow:
    Postgres/safe input
      -> PrivacyLayerAgent
      -> prompt cohort selection
      -> SyntheticEngineAgent
      -> EmbeddingFeatureStoreAgent
      -> CohortManagementAgent
      -> SafeExportAgent
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

        print("FINAL SUMMARY:", final_summary_path)
        print("SAFE EXPORT:", export_result["outputs"]["safe_export_manifest"])

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
            for key in ["location_name", "city", "place_name", "name", "label", "area", "region", "address", "formatted_address"]:
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
        lower = prompt.lower()

        mapping = {
            "restaurant": "restaurant",
            "food": "food",
            "cafe": "cafe",
            "coffee": "cafe",
            "gym": "gym",
            "fitness": "gym",
            "health": "health",
            "barber": "barber_shop",
            "tattoo": "body_art_service",
            "store": "store",
            "retail": "store",
        }

        for key, value in mapping.items():
            if key in lower:
                return value

        return None

    def _select_cohorts_for_prompt(self, *, prompt: str, cohorts: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        df = cohorts.copy()
        lower = prompt.lower()

        locations = self._extract_location_terms(prompt, df)
        pois = self._extract_poi_terms(prompt)
        dayparts = [item for item in ["morning", "afternoon", "evening", "night"] if item in lower]

        attempts = []

        def apply_filter(use_locations: bool, use_pois: bool, use_dayparts: bool) -> pd.DataFrame:
            mask = pd.Series(True, index=df.index)

            if use_locations and locations:
                loc_mask = pd.Series(False, index=df.index)
                for loc in locations:
                    loc_mask = loc_mask | df["location_name"].astype(str).str.lower().str.contains(re.escape(loc), na=False)
                mask = mask & loc_mask

            if use_pois and pois:
                poi_mask = pd.Series(False, index=df.index)
                for poi in pois:
                    poi_mask = poi_mask | df["primary_poi_type"].astype(str).str.lower().str.contains(re.escape(poi), na=False)
                mask = mask & poi_mask

            if use_dayparts and dayparts:
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

        if "high quality" in lower or "high-quality" in lower or "best" in lower:
            quality_col = "quality_score" if "quality_score" in selected.columns else None
            if quality_col:
                selected = selected.sort_values(quality_col, ascending=False)

        return selected.reset_index(drop=True), {
            "filter_mode": mode,
            "locations_detected": locations,
            "poi_terms_detected": pois,
            "dayparts_detected": dayparts,
            "attempts": attempts,
            "high_quality_requested": bool("high quality" in lower or "high-quality" in lower or "best" in lower),
        }

    def _extract_location_terms(self, prompt: str, df: pd.DataFrame) -> list[str]:
        lower = prompt.lower()
        found = []

        if "location_name" not in df.columns:
            return found

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

        manual = ["montreal", "san francisco", "new york", "los angeles", "toronto", "quebec"]
        for item in manual:
            if item in lower and item not in found:
                found.append(item)

        return list(dict.fromkeys(found))

    def _extract_poi_terms(self, prompt: str) -> list[str]:
        lower = prompt.lower()
        terms = []

        mapping = {
            "restaurant": ["restaurant", "food"],
            "food": ["food", "restaurant"],
            "cafe": ["cafe", "coffee"],
            "coffee": ["cafe", "coffee"],
            "gym": ["gym", "fitness", "health"],
            "fitness": ["gym", "fitness", "health"],
            "health": ["health"],
            "barber": ["barber_shop", "barber"],
            "tattoo": ["body_art_service", "tattoo"],
            "store": ["store"],
            "retail": ["store"],
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
