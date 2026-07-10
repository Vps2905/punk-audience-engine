from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from app.agents.maid_extraction_execution_agent import MaidExtractionExecutionAgent
from app.agents.synthetic_engine_agent import SyntheticEngineAgent


class MaidSwarmPipelineAgent:
    """
    Full adaptive swarm pipeline for real Postgres maid_extractions data.

    Flow:
    Postgres maid_extractions
    -> privacy-safe aggregation
    -> embedding
    -> clustering
    -> synthetic safe seeds
    -> Meta-safe export package
    """

    def __init__(self) -> None:
        self.maid_agent = MaidExtractionExecutionAgent()
        self.output_dir = Path("data/swarm_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        schema_name: str = "public",
        table_name: str = "maid_extractions",
        limit: int = 10000,
        k_min: int = 1000,
        epsilon: float = 1.0,
        synthetic_rows: int = 1000,
    ) -> Dict[str, Any]:
        run_id = f"swarm_run_{uuid.uuid4().hex[:12]}"
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Privacy-safe cohort generation
        cohort_result = self.maid_agent.execute(
            schema_name=schema_name,
            table_name=table_name,
            limit=limit,
            k_min=k_min,
            epsilon=epsilon,
        )

        cohort_csv = cohort_result["outputs"]["cohort_csv"]
        cohorts = pd.read_csv(cohort_csv)

        if cohorts.empty:
            return {
                "agent": "maid_swarm_pipeline_agent",
                "status": "no_safe_cohorts",
                "message": "No cohorts passed k-anonymity.",
                "privacy_result": cohort_result,
            }

        cohorts = self._clean_dataframe(cohorts)

        # 2. Embedding
        trait_texts = cohorts.apply(self._cohort_to_text, axis=1).tolist()
        vectorizer = TfidfVectorizer(max_features=256)
        vectors = vectorizer.fit_transform(trait_texts).toarray()

        vectors_path = run_dir / "cohort_vectors.npy"
        metadata_path = run_dir / "cohort_metadata.csv"

        np.save(vectors_path, vectors)
        cohorts["trait_text"] = trait_texts

        # 3. Clustering
        cluster_count = min(8, max(1, len(cohorts)))

        if cluster_count == 1:
            cohorts["cluster_id"] = 0
        else:
            model = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
            cohorts["cluster_id"] = model.fit_predict(vectors)

        cohorts.to_csv(metadata_path, index=False)

        # 4. Synthetic safe seed generation through hardened Module 2 engine.
        synthetic_result = SyntheticEngineAgent().generate(
            cohorts=cohorts,
            output_dir=run_dir,
            run_id=f"{run_id}_synthetic",
            engine_requested="dp_aggregate",
            production_mode=True,
            allow_fallback=False,
            synthetic_rows=synthetic_rows,
            epsilon=epsilon,
            k_min=k_min,
        )

        synthetic_path = Path(synthetic_result["outputs"]["synthetic_csv"])
        synthetic_manifest_path = Path(synthetic_result["outputs"]["synthetic_manifest"])
        synthetic = pd.read_csv(synthetic_path)

        # 5. Meta-safe export package
        export_manifest = {
            "export_id": f"export_{uuid.uuid4().hex[:12]}",
            "approval_status": "pending_approval",
            "platform": "Meta Advantage+",
            "contains_raw_maids": False,
            "contains_raw_email": False,
            "contains_raw_phone": False,
            "contains_raw_lat_lng": False,
            "contains_individual_user_data": False,
            "aggregated_traits_only": True,
            "synthetic_seed_profiles": True,
            "requires_manual_approval_before_upload": True,
            "files": {
                "safe_cohort_metadata": str(metadata_path),
                "cohort_vectors": str(vectors_path),
                "synthetic_safe_seed_profiles": str(synthetic_path),
                "synthetic_manifest": str(synthetic_manifest_path),
                "synthetic_manifest": str(synthetic_manifest_path),
            },
        }

        export_manifest_path = run_dir / "meta_safe_export_manifest.json"

        with open(export_manifest_path, "w") as f:
            json.dump(export_manifest, f, indent=2)

        return {
            "agent": "maid_swarm_pipeline_agent",
            "status": "completed",
            "run_id": run_id,
            "source": cohort_result["source"],
            "privacy": cohort_result["privacy"],
            "swarm_steps": [
                "postgres_source_agent",
                "maid_extraction_execution_agent",
                "privacy_aggregation_agent",
                "embedding_agent",
                "clustering_agent",
                "synthetic_seed_agent",
                "safe_export_agent",
            ],
            "summary": {
                "safe_cohort_count": int(len(cohorts)),
                "cluster_count": int(cluster_count),
                "synthetic_rows": int(len(synthetic)),
                "synthetic_engine": synthetic_result.get("engine_used"),
                "top_clusters": cohorts.groupby("cluster_id").size().to_dict(),
                "top_cohorts": cohorts.head(10).to_dict(orient="records"),
            },
            "outputs": {
                "run_dir": str(run_dir),
                "safe_cohort_metadata": str(metadata_path),
                "cohort_vectors": str(vectors_path),
                "synthetic_safe_seed_profiles": str(synthetic_path),
                "meta_safe_export_manifest": str(export_manifest_path),
            },
            "export": export_manifest,
        }

    def _cohort_to_text(self, row: pd.Series) -> str:
        fields = [
            "location_name",
            "primary_poi_type",
            "created_day_part",
            "lookback_bucket",
            "grouping_columns",
            "quality_score",
            "sessions",
            "noisy_maid_volume",
        ]

        parts: List[str] = []

        for field in fields:
            if field in row and pd.notna(row[field]):
                parts.append(f"{field} {row[field]}")

        return " ".join(parts)

    def _generate_synthetic_seeds(
        self,
        cohorts: pd.DataFrame,
        synthetic_rows: int,
    ) -> pd.DataFrame:
        raise RuntimeError(
            "Legacy MaidSwarm synthetic sampler is disabled. "
            "Use SyntheticEngineAgent with engine_requested='dp_aggregate', "
            "production_mode=True, and allow_fallback=False."
        )

        weights = cohorts["noisy_maid_volume"].fillna(1).astype(float)
        weights = weights.clip(lower=1)
        probabilities = weights / weights.sum()

        sampled = cohorts.sample(
            n=synthetic_rows,
            replace=True,
            weights=probabilities,
            random_state=42,
        ).reset_index(drop=True)

        synthetic = pd.DataFrame({
            "synthetic_seed_id": [f"seed_{i+1:06d}" for i in range(len(sampled))],
            "location_name": sampled.get("location_name", "all"),
            "primary_poi_type": sampled.get("primary_poi_type", "unknown"),
            "created_day_part": sampled.get("created_day_part", "all"),
            "lookback_bucket": sampled.get("lookback_bucket", "all"),
            "cluster_id": sampled.get("cluster_id", 0),
            "quality_score": sampled.get("quality_score", 0),
            "privacy_mode": "synthetic_aggregated_seed",
        })

        return self._clean_dataframe(synthetic)

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)
        return df
