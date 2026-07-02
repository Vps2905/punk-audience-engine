from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.agents.postgres_source_agent import PostgresSourceAgent


class MaidExtractionExecutionAgent:
    """
    Privacy-safe cohort generation from public.maid_extractions.

    Important:
    - Keeps latest row per session_id to avoid duplicate overcounting.
    - Never exports raw MAIDs.
    - Never exports raw observation lat/lng.
    - Uses safe aggregated traits only.
    """

    def __init__(self) -> None:
        self.source_agent = PostgresSourceAgent()
        self.output_dir = Path("data/postgres_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        schema_name: str = "public",
        table_name: str = "maid_extractions",
        limit: int = 10000,
        k_min: int = 1000,
        epsilon: float = 1.0,
    ) -> Dict[str, Any]:
        df = self.source_agent.sample_table(
            schema_name=schema_name,
            table_name=table_name,
            limit=limit,
        )

        if df.empty:
            return {"agent": "maid_extraction_execution_agent", "status": "no_data"}

        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

        raw_rows_loaded = len(df)

        # Critical: repeated session_id rows exist, so keep latest extraction per session.
        df = (
            df.sort_values("created_at")
            .drop_duplicates(subset=["session_id"], keep="last")
            .copy()
        )

        deduped_rows = len(df)

        feature_df = self._build_features(df)

        cohorts = self._aggregate_cohorts(
            feature_df=feature_df,
            k_min=k_min,
            epsilon=epsilon,
        )

        for col in ["location_name", "primary_poi_type", "created_day_part", "lookback_bucket", "search_radius_bucket"]:
            if col in cohorts.columns:
                cohorts[col] = cohorts[col].fillna("all")

        cohorts = cohorts.where(pd.notnull(cohorts), None)

        run_id = f"maid_run_{uuid.uuid4().hex[:12]}"
        output_csv = self.output_dir / f"{run_id}_safe_cohorts.csv"
        output_manifest = self.output_dir / f"{run_id}_manifest.json"

        cohorts.to_csv(output_csv, index=False)

        manifest = {
            "agent": "maid_extraction_execution_agent",
            "status": "completed",
            "run_id": run_id,
            "source": {
                "type": "postgres",
                "schema": schema_name,
                "table": table_name,
                "raw_rows_loaded": int(raw_rows_loaded),
                "deduped_sessions": int(deduped_rows),
                "feature_rows": int(len(feature_df)),
            },
            "privacy": {
                "raw_maids_exported": False,
                "raw_observations_exported": False,
                "raw_lat_lng_exported": False,
                "aggregated_only": True,
                "k_min": int(k_min),
                "epsilon": float(epsilon),
                "dp_noise": True,
            },
            "outputs": {
                "cohort_csv": str(output_csv),
                "manifest_json": str(output_manifest),
            },
            "cohort_summary": {
                "safe_cohort_count": int(len(cohorts)),
                "total_noisy_maid_volume": int(cohorts["noisy_maid_volume"].sum()) if not cohorts.empty else 0,
                "top_cohorts": cohorts.head(20).to_dict(orient="records"),
            },
            "workflow": [
                "load maid_extractions from Postgres",
                "dedupe repeated session_id rows",
                "parse center / POI / MAID JSON safely",
                "extract safe traits",
                "aggregate cohorts",
                "apply k-anonymity",
                "apply DP noise",
                "save safe cohort output",
            ],
        }

        manifest = self._json_safe(manifest)

        with open(output_manifest, "w") as f:
            json.dump(manifest, f, indent=2, default=str, allow_nan=False)

        return manifest

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            created_at = row.get("created_at")

            pois = self._safe_json(row.get("pois"))
            center = self._safe_json(row.get("center"))
            observations = self._safe_json(row.get("observations"))
            maids = self._safe_json(row.get("maids"))

            poi_traits = self._extract_poi_traits(pois)
            center_traits = self._extract_center_traits(center)

            maid_count = int(row.get("maid_count") or 0)
            parsed_maid_count = self._count_json_items(maids)
            observation_count = self._count_json_items(observations)

            maid_volume = max(maid_count, parsed_maid_count)

            hour = int(created_at.hour) if not pd.isna(created_at) else -1

            rows.append({
                "session_id": str(row.get("session_id")),
                "created_day_part": self._day_part(hour),
                "maid_volume": int(maid_volume),
                "observation_count": int(observation_count),
                "poi_count": int(poi_traits["poi_count"]),
                "primary_poi_type": poi_traits["primary_poi_type"],
                "poi_type_signature": poi_traits["poi_type_signature"],
                "location_name": center_traits["location_name"],
                "formatted_location": center_traits["formatted_location"],
                "is_city": center_traits["is_city"],
                "search_radius_bucket": self._bucket_radius(row.get("search_radius_km")),
                "lookback_bucket": self._bucket_lookback(row.get("lookback_days")),
            })

        return pd.DataFrame(rows)

    def _aggregate_cohorts(
        self,
        feature_df: pd.DataFrame,
        k_min: int,
        epsilon: float,
    ) -> pd.DataFrame:
        grouping_levels = [
            ["location_name", "primary_poi_type", "created_day_part", "lookback_bucket"],
            ["location_name", "primary_poi_type", "lookback_bucket"],
            ["location_name", "primary_poi_type"],
            ["primary_poi_type"],
        ]

        frames = []

        for level_index, group_cols in enumerate(grouping_levels, start=1):
            grouped = (
                feature_df.groupby(group_cols, dropna=False)
                .agg(
                    sessions=("session_id", "nunique"),
                    total_maid_volume=("maid_volume", "sum"),
                    total_observations=("observation_count", "sum"),
                    avg_poi_count=("poi_count", "mean"),
                )
                .reset_index()
            )

            grouped["aggregation_level"] = level_index
            grouped["grouping_columns"] = ",".join(group_cols)
            grouped["privacy_status"] = np.where(
                grouped["total_maid_volume"] >= k_min,
                "passed",
                "blocked_small_group",
            )

            grouped = grouped[grouped["privacy_status"] == "passed"].copy()

            if grouped.empty:
                continue

            grouped["noisy_maid_volume"] = grouped["total_maid_volume"].apply(
                lambda value: self._add_laplace_noise(value, epsilon)
            )

            grouped["quality_score"] = grouped.apply(
                lambda row: self._quality_score(row, k_min),
                axis=1,
            )

            frames.append(grouped)

        if not frames:
            return pd.DataFrame(columns=[
                "location_name",
                "primary_poi_type",
                "sessions",
                "total_maid_volume",
                "noisy_maid_volume",
                "privacy_status",
                "quality_score",
            ])

        result = pd.concat(frames, ignore_index=True, sort=False)
        return result.sort_values(["quality_score", "total_maid_volume"], ascending=False)


    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._json_safe(v) for v in value]

        if value is pd.NA:
            return None

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        return value

    def _safe_json(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return None

    def _count_json_items(self, value: Any) -> int:
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            return len(value)
        return 0

    def _extract_center_traits(self, center: Any) -> Dict[str, Any]:
        if not isinstance(center, dict):
            return {
                "location_name": "unknown",
                "formatted_location": "unknown",
                "is_city": False,
            }

        return {
            "location_name": str(center.get("location_name") or "unknown").lower(),
            "formatted_location": str(center.get("formatted_address") or "unknown"),
            "is_city": bool(center.get("is_city", False)),
        }

    def _extract_poi_traits(self, pois: Any) -> Dict[str, Any]:
        if not isinstance(pois, list) or not pois:
            return {
                "poi_count": 0,
                "primary_poi_type": "unknown",
                "poi_type_signature": "unknown",
            }

        all_types: List[str] = []

        for poi in pois:
            if not isinstance(poi, dict):
                continue

            for key in ["type", "category", "primary_type"]:
                value = poi.get(key)
                if isinstance(value, str) and value:
                    all_types.append(value.lower().replace(" ", "_"))

            types = poi.get("types")
            if isinstance(types, list):
                for t in types:
                    if isinstance(t, str) and t:
                        all_types.append(t.lower().replace(" ", "_"))

        if not all_types:
            return {
                "poi_count": len(pois),
                "primary_poi_type": "unknown",
                "poi_type_signature": "unknown",
            }

        counts = pd.Series(all_types).value_counts()
        return {
            "poi_count": len(pois),
            "primary_poi_type": str(counts.index[0]),
            "poi_type_signature": "|".join(counts.head(5).index.astype(str).tolist()),
        }

    def _bucket_radius(self, value: Any) -> str:
        try:
            radius = float(value)
        except Exception:
            return "unknown_radius"

        if radius <= 1:
            return "0_1km"
        if radius <= 3:
            return "1_3km"
        if radius <= 5:
            return "3_5km"
        if radius <= 10:
            return "5_10km"
        return "10km_plus"

    def _bucket_lookback(self, value: Any) -> str:
        try:
            days = int(value)
        except Exception:
            return "unknown_lookback"

        if days <= 7:
            return "0_7d"
        if days <= 30:
            return "8_30d"
        if days <= 90:
            return "31_90d"
        return "90d_plus"

    def _day_part(self, hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 22:
            return "evening"
        if hour >= 0:
            return "night"
        return "unknown"

    def _add_laplace_noise(self, value: int, epsilon: float) -> int:
        epsilon = max(float(epsilon), 0.01)
        noise = np.random.laplace(loc=0, scale=1 / epsilon)
        return max(0, int(round(value + noise)))

    def _quality_score(self, row: pd.Series, k_min: int) -> float:
        maid_volume = float(row.get("total_maid_volume", 0))
        sessions = float(row.get("sessions", 0))
        observations = float(row.get("total_observations", 0))

        size_score = min(55.0, (maid_volume / max(k_min, 1)) * 35.0)
        session_score = min(20.0, math.log1p(sessions) * 6.0)
        observation_score = min(10.0, math.log1p(observations) * 2.0)
        privacy_score = 15.0

        return round(min(100.0, size_score + session_score + observation_score + privacy_score), 2)
