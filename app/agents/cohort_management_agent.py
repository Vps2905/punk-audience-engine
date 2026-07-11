from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.audit_logger import AuditLogger
from app.core.production_config import load_production_config
from app.core.schema_validator import SafeSchemaValidator


class CohortManagementAgent:
    """
    Production CohortManagementAgent.

    Responsibilities:
    - Accept only safe cohort metadata + vectors.
    - Cluster similar cohorts.
    - Rank top cohorts.
    - Generate lookalike cohort recommendations.
    - Produce quality report and manifest.
    - Never expose raw MAIDs, observations, lat/lng, emails, phones, device IDs,
      hashed identifiers, or individual-level rows.
    """

    SAFE_METADATA_COLUMNS = [
        "location_name",
        "primary_poi_type",
        "created_day_part",
        "lookback_bucket",
        "sessions",
        "total_maid_volume",
        "noisy_maid_volume",
        "total_observations",
        "quality_score",
        "privacy_status",
        "trait_text",
    ]

    SAFE_AGGREGATE_COLUMNS = {
        "total_maid_volume",
        "noisy_maid_volume",
        "safe_maid_volume",
        "total_observations",
        "observation_count",
        "observations_count",
        "safe_observation_count",
    }

    BLOCKED_TOKENS = [
        "raw_maid",
        "device_id",
        "email",
        "phone",
        "latitude",
        "longitude",
        "raw_observation",
        "hashed",
        "lat",
        "lng",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.config = load_production_config()
        self.validator = SafeSchemaValidator()
        self.audit_logger = AuditLogger(self.config.audit_log_path)

    def run(
        self,
        metadata: pd.DataFrame,
        vectors: np.ndarray,
        output_dir: str | Path,
        run_id: Optional[str] = None,
        min_clusters: int = 2,
        max_clusters: int = 8,
        top_n: int = 25,
        lookalike_top_k: int = 3,
        min_export_quality: float = 0.25,
        persist_artifacts: bool = True,
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)

        if persist_artifacts:
            output_dir.mkdir(parents=True, exist_ok=True)

        run_id = run_id or f"cohort_management_run_{uuid.uuid4().hex[:12]}"

        self._validate_runtime_config(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            top_n=top_n,
            lookalike_top_k=lookalike_top_k,
            min_export_quality=min_export_quality,
        )

        self.audit_logger.log(
            "cohort_management_started",
            {
                "run_id": run_id,
                "metadata_rows": int(len(metadata)),
                "vector_shape": list(vectors.shape),
                "min_clusters": min_clusters,
                "max_clusters": max_clusters,
                "top_n": top_n,
                "lookalike_top_k": lookalike_top_k,
            },
        )

        self._validate_no_blocked_columns(metadata.columns, context="cohort_management_raw_metadata")
        self.validator.validate_no_secret_values(metadata, context="cohort_management_raw_metadata")

        safe_metadata = self._prepare_safe_metadata(metadata)
        self.validator.validate_safe_cohort_dataframe(
            safe_metadata,
            context="cohort_management_safe_metadata",
        )

        vectors = self._prepare_vectors(vectors, expected_rows=len(safe_metadata))

        managed = self._cluster_and_score(
            metadata=safe_metadata,
            vectors=vectors,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            min_export_quality=min_export_quality,
        )

        top_cohorts = self._build_top_cohorts(managed, top_n=top_n)
        lookalikes = self._build_lookalikes(
            managed=managed,
            vectors=vectors,
            top_seed_count=min(top_n, len(managed)),
            lookalike_top_k=lookalike_top_k,
        )

        self._validate_no_blocked_columns(managed.columns, context="cohort_management_output")
        self._validate_no_blocked_columns(lookalikes.columns, context="cohort_management_lookalikes")

        quality_report = self._build_quality_report(
            managed=managed,
            lookalikes=lookalikes,
            min_export_quality=min_export_quality,
        )

        records = {
            "top_cohorts": self._json_safe(
                top_cohorts.to_dict(orient="records")
            ),
            "lookalikes": self._json_safe(
                lookalikes.to_dict(orient="records")
            ),
        }

        if persist_artifacts:
            clusters_path = output_dir / "cohort_clusters.csv"
            top_cohorts_path = output_dir / "top_cohorts.csv"
            lookalikes_path = output_dir / "lookalike_cohorts.csv"
            quality_report_path = output_dir / "cohort_quality_report.json"
            manifest_path = output_dir / "cohort_management_manifest.json"

            managed.to_csv(clusters_path, index=False)
            top_cohorts.to_csv(top_cohorts_path, index=False)
            lookalikes.to_csv(lookalikes_path, index=False)
            self._write_json(quality_report_path, quality_report)

            outputs = {
                "cohort_clusters": str(clusters_path),
                "top_cohorts": str(top_cohorts_path),
                "lookalike_cohorts": str(lookalikes_path),
                "cohort_quality_report": str(quality_report_path),
                "cohort_management_manifest": str(manifest_path),
            }
            storage_backend = "local_files"
        else:
            base_uri = (
                "postgres://audience_run_history.final_summary"
                f"?run_id={run_id}&section=cohort_management"
            )
            outputs = {
                "cohort_clusters": base_uri + "&artifact=clusters",
                "top_cohorts": base_uri + "&artifact=top_cohorts",
                "lookalike_cohorts": base_uri + "&artifact=lookalikes",
                "cohort_quality_report": base_uri + "&artifact=quality_report",
                "cohort_management_manifest": base_uri + "&artifact=manifest",
            }
            storage_backend = "run_history_jsonb"

        manifest = {
            "module": "Cohort Management & Lookalike",
            "status": "completed",
            "run_id": run_id,
            "storage_backend": storage_backend,
            "input_rows": int(len(metadata)),
            "managed_cohorts": int(len(managed)),
            "cluster_count": int(managed["cluster_id"].nunique()),
            "top_cohorts": int(len(top_cohorts)),
            "lookalike_pairs": int(len(lookalikes)),
            "export_ready_cohorts": int(managed["export_ready"].sum()),
            "min_export_quality": float(min_export_quality),
            "raw_maids_exported": False,
            "raw_observations_exported": False,
            "raw_lat_lng_exported": False,
            "raw_email_exported": False,
            "raw_phone_exported": False,
            "individual_user_data_exported": False,
            "records": records,
            "quality_report": self._json_safe(quality_report),
            "outputs": outputs,
        }

        if persist_artifacts:
            self._write_json(manifest_path, manifest)

        self.audit_logger.log(
            "cohort_management_completed",
            {
                "run_id": run_id,
                "managed_cohorts": int(len(managed)),
                "cluster_count": int(managed["cluster_id"].nunique()),
                "export_ready_cohorts": int(managed["export_ready"].sum()),
                "manifest": manifest["outputs"]["cohort_management_manifest"],
            },
        )

        return manifest

    def run_from_artifacts(
        self,
        metadata_path: str | Path,
        vectors_path: str | Path,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        metadata = pd.read_csv(metadata_path)
        vectors = np.load(vectors_path)
        return self.run(metadata=metadata, vectors=vectors, output_dir=output_dir, **kwargs)

    def _validate_runtime_config(
        self,
        *,
        min_clusters: int,
        max_clusters: int,
        top_n: int,
        lookalike_top_k: int,
        min_export_quality: float,
    ) -> None:
        if min_clusters <= 0:
            raise ValueError("min_clusters must be greater than 0.")

        if max_clusters < min_clusters:
            raise ValueError("max_clusters must be greater than or equal to min_clusters.")

        if top_n <= 0:
            raise ValueError("top_n must be greater than 0.")

        if lookalike_top_k <= 0:
            raise ValueError("lookalike_top_k must be greater than 0.")

        if not 0 <= min_export_quality <= 1:
            raise ValueError("min_export_quality must be between 0 and 1.")

    def _validate_no_blocked_columns(self, columns, context: str) -> None:
        for col in columns:
            lower = str(col).lower()

            if lower in self.SAFE_AGGREGATE_COLUMNS:
                continue

            for token in self.BLOCKED_TOKENS:
                if lower == token or lower.startswith(f"{token}_") or lower.endswith(f"_{token}"):
                    raise ValueError(f"{context} contains blocked sensitive column: {col}")

            if lower in {
                "maid",
                "maids",
                "raw_maid",
                "device_id",
                "email",
                "phone",
                "lat",
                "lng",
                "latitude",
                "longitude",
            }:
                raise ValueError(f"{context} contains blocked sensitive column: {col}")

    def _prepare_safe_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(metadata, pd.DataFrame):
            raise TypeError("metadata must be a pandas DataFrame.")

        if metadata.empty:
            raise ValueError("metadata cannot be empty.")

        df = metadata.copy()
        safe_cols = [col for col in self.SAFE_METADATA_COLUMNS if col in df.columns]
        df = df[safe_cols].copy()

        text_defaults = {
            "location_name": "all_locations",
            "primary_poi_type": "all_poi_types",
            "created_day_part": "all_day_parts",
            "lookback_bucket": "all_lookbacks",
            "privacy_status": "passed",
            "trait_text": "",
        }

        numeric_defaults = {
            "sessions": 1,
            "total_maid_volume": 1,
            "noisy_maid_volume": 1,
            "total_observations": 1,
            "quality_score": 0.5,
        }

        for col, default in text_defaults.items():
            if col not in df.columns:
                df[col] = default

            df[col] = df[col].replace([np.nan, None], default)
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(
                {
                    "": default,
                    "nan": default,
                    "NaN": default,
                    "None": default,
                    "none": default,
                    "NULL": default,
                    "null": default,
                    "0": default,
                    "0.0": default,
                }
            )

        for col, default in numeric_defaults.items():
            if col not in df.columns:
                df[col] = default

            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(default)
            )

        max_quality = df["quality_score"].max()
        if pd.notna(max_quality) and max_quality > 1:
            df["quality_score"] = (df["quality_score"] / 100.0).clip(0, 1)
        else:
            df["quality_score"] = df["quality_score"].clip(0, 1)

        weak_trait = df["trait_text"].isin(["", "0", "0.0", "nan", "None", "none"])
        df.loc[weak_trait, "trait_text"] = df[weak_trait].apply(self._build_trait_text, axis=1)

        ordered = [
            "location_name",
            "primary_poi_type",
            "created_day_part",
            "lookback_bucket",
            "sessions",
            "total_maid_volume",
            "noisy_maid_volume",
            "total_observations",
            "quality_score",
            "privacy_status",
            "trait_text",
        ]

        return df[[c for c in ordered if c in df.columns]].reset_index(drop=True)

    def _build_trait_text(self, row: pd.Series) -> str:
        return " | ".join(
            [
                f"location {row.get('location_name', 'all_locations')}",
                f"poi {row.get('primary_poi_type', 'all_poi_types')}",
                f"daypart {row.get('created_day_part', 'all_day_parts')}",
                f"lookback {row.get('lookback_bucket', 'all_lookbacks')}",
                f"quality {round(float(row.get('quality_score', 0.5)), 3)}",
                f"sessions {int(float(row.get('sessions', 0)))}",
            ]
        )

    def _prepare_vectors(self, vectors: np.ndarray, expected_rows: int) -> np.ndarray:
        vectors = np.asarray(vectors, dtype="float32")

        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D.")

        if vectors.shape[0] != expected_rows:
            raise ValueError("vectors row count must match metadata rows.")

        if vectors.shape[1] <= 0:
            raise ValueError("vector dimension must be greater than 0.")

        if not np.isfinite(vectors).all():
            raise ValueError("vectors contain NaN or infinite values.")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)

        return vectors / norms

    def _cluster_and_score(
        self,
        *,
        metadata: pd.DataFrame,
        vectors: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        min_export_quality: float,
    ) -> pd.DataFrame:
        n_rows = len(metadata)

        if n_rows == 1:
            labels = np.array([0], dtype=int)
            cluster_count = 1
        else:
            from sklearn.cluster import KMeans

            cluster_count = min(max_clusters, n_rows)
            cluster_count = max(min_clusters, cluster_count)
            cluster_count = min(cluster_count, n_rows)

            model = KMeans(
                n_clusters=cluster_count,
                random_state=self.random_state,
                n_init=10,
            )
            labels = model.fit_predict(vectors)

        managed = metadata.copy()
        managed.insert(0, "cohort_index", range(len(managed)))
        managed["cluster_id"] = labels.astype(int)

        cluster_sizes = managed.groupby("cluster_id")["cohort_index"].transform("count")
        managed["cluster_size"] = cluster_sizes.astype(int)

        coherence_scores = self._cluster_coherence_scores(vectors, labels)
        managed["cluster_coherence_score"] = coherence_scores

        volume_col = "noisy_maid_volume" if "noisy_maid_volume" in managed.columns else "total_maid_volume"

        volume_norm = self._safe_minmax(managed[volume_col])
        session_norm = self._safe_minmax(managed["sessions"])
        base_quality = managed["quality_score"].clip(0, 1)
        coherence = managed["cluster_coherence_score"].clip(0, 1)

        managed["management_quality_score"] = (
            (base_quality * 0.40)
            + (volume_norm * 0.25)
            + (session_norm * 0.15)
            + (coherence * 0.20)
        ).clip(0, 1)

        managed["export_ready"] = (
            (managed["privacy_status"].astype(str).str.lower() == "passed")
            & (managed["management_quality_score"] >= min_export_quality)
        )

        managed["recommendation_reason"] = managed.apply(self._recommendation_reason, axis=1)

        return managed.sort_values(
            ["management_quality_score", "cluster_coherence_score", volume_col],
            ascending=False,
        ).reset_index(drop=True)

    def _cluster_coherence_scores(self, vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(vectors), dtype="float32")

        for cluster_id in sorted(set(labels.tolist())):
            idx = np.where(labels == cluster_id)[0]
            cluster_vectors = vectors[idx]

            centroid = cluster_vectors.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            cluster_scores = cluster_vectors @ centroid
            scores[idx] = np.clip(cluster_scores, 0, 1)

        return scores

    def _safe_minmax(self, series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)

        min_value = float(values.min())
        max_value = float(values.max())

        if max_value == min_value:
            return pd.Series([1.0] * len(values), index=values.index)

        return ((values - min_value) / (max_value - min_value)).clip(0, 1)

    def _recommendation_reason(self, row: pd.Series) -> str:
        parts = [
            f"cluster {int(row.get('cluster_id', -1))}",
            f"quality {round(float(row.get('management_quality_score', 0)), 3)}",
            f"coherence {round(float(row.get('cluster_coherence_score', 0)), 3)}",
            f"poi {row.get('primary_poi_type', 'unknown')}",
            f"daypart {row.get('created_day_part', 'all')}",
        ]
        return " | ".join(parts)

    def _build_top_cohorts(self, managed: pd.DataFrame, top_n: int) -> pd.DataFrame:
        cols = [
            "cohort_index",
            "cluster_id",
            "cluster_size",
            "location_name",
            "primary_poi_type",
            "created_day_part",
            "lookback_bucket",
            "sessions",
            "total_maid_volume",
            "noisy_maid_volume",
            "quality_score",
            "cluster_coherence_score",
            "management_quality_score",
            "export_ready",
            "recommendation_reason",
            "trait_text",
        ]

        available_cols = [col for col in cols if col in managed.columns]

        return managed[available_cols].head(top_n).reset_index(drop=True)

    def _build_lookalikes(
        self,
        *,
        managed: pd.DataFrame,
        vectors: np.ndarray,
        top_seed_count: int,
        lookalike_top_k: int,
    ) -> pd.DataFrame:
        if len(managed) <= 1:
            return pd.DataFrame(
                columns=[
                    "seed_cohort_index",
                    "lookalike_cohort_index",
                    "seed_cluster_id",
                    "lookalike_cluster_id",
                    "seed_location_name",
                    "seed_primary_poi_type",
                    "lookalike_location_name",
                    "lookalike_primary_poi_type",
                    "similarity_score",
                    "recommendation_reason",
                ]
            )

        rows = []

        top_seed_rows = managed.head(top_seed_count)

        for _, seed in top_seed_rows.iterrows():
            seed_original_index = int(seed["cohort_index"])

            seed_vector = vectors[seed_original_index]
            scores = vectors @ seed_vector

            ranked_indices = np.argsort(scores)[::-1]

            added = 0
            for candidate_index in ranked_indices:
                candidate_index = int(candidate_index)

                if candidate_index == seed_original_index:
                    continue

                candidate_rows = managed[managed["cohort_index"] == candidate_index]
                if candidate_rows.empty:
                    continue

                candidate = candidate_rows.iloc[0]

                rows.append(
                    {
                        "seed_cohort_index": seed_original_index,
                        "lookalike_cohort_index": candidate_index,
                        "seed_cluster_id": int(seed["cluster_id"]),
                        "lookalike_cluster_id": int(candidate["cluster_id"]),
                        "seed_location_name": seed["location_name"],
                        "seed_primary_poi_type": seed["primary_poi_type"],
                        "lookalike_location_name": candidate["location_name"],
                        "lookalike_primary_poi_type": candidate["primary_poi_type"],
                        "similarity_score": float(scores[candidate_index]),
                        "recommendation_reason": (
                            f"similar safe cohort | similarity {round(float(scores[candidate_index]), 3)} | "
                            f"seed cluster {int(seed['cluster_id'])} | lookalike cluster {int(candidate['cluster_id'])}"
                        ),
                    }
                )

                added += 1
                if added >= lookalike_top_k:
                    break

        return pd.DataFrame(rows)

    def _build_quality_report(self, managed: pd.DataFrame, lookalikes: pd.DataFrame, min_export_quality: float) -> Dict[str, Any]:
        cluster_summary = []

        for cluster_id, group in managed.groupby("cluster_id"):
            cluster_summary.append(
                {
                    "cluster_id": int(cluster_id),
                    "cohort_count": int(len(group)),
                    "avg_management_quality_score": float(group["management_quality_score"].mean()),
                    "avg_cluster_coherence_score": float(group["cluster_coherence_score"].mean()),
                    "export_ready_count": int(group["export_ready"].sum()),
                    "top_poi_types": group["primary_poi_type"].value_counts().head(5).to_dict(),
                    "top_locations": group["location_name"].value_counts().head(5).to_dict(),
                }
            )

        return {
            "module": "Cohort Quality Report",
            "status": "completed",
            "managed_cohorts": int(len(managed)),
            "cluster_count": int(managed["cluster_id"].nunique()),
            "export_ready_cohorts": int(managed["export_ready"].sum()),
            "lookalike_pairs": int(len(lookalikes)),
            "min_export_quality": float(min_export_quality),
            "avg_management_quality_score": float(managed["management_quality_score"].mean()),
            "avg_cluster_coherence_score": float(managed["cluster_coherence_score"].mean()),
            "cluster_summary": self._json_safe(cluster_summary),
            "raw_identifiers_exported": False,
            "individual_user_data_exported": False,
        }

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.write_text(json.dumps(self._json_safe(data), indent=2, allow_nan=False))

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._json_safe(v) for v in value]

        if isinstance(value, tuple):
            return [self._json_safe(v) for v in value]

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        return value
