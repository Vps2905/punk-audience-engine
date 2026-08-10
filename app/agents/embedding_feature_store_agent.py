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


class EmbeddingFeatureStoreAgent:
    """
    Production EmbeddingFeatureStoreAgent.

    Converts privacy-safe cohort rows into searchable vectors.
    Does not accept or export raw MAIDs, observations, lat/lng, email, phone,
    device IDs, or individual-level data.
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
        "observations",
        "lat",
        "lng",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.config = load_production_config()
        self.validator = SafeSchemaValidator()
        self.audit_logger = AuditLogger(self.config.audit_log_path)

    def build(
        self,
        cohorts: pd.DataFrame,
        output_dir: str | Path,
        embedding_provider: str = "sklearn_tfidf",
        run_id: Optional[str] = None,
        max_features: int = 384,
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_id = run_id or f"embedding_run_{uuid.uuid4().hex[:12]}"

        self._validate_runtime_config(
            embedding_provider=embedding_provider,
            max_features=max_features,
        )

        self.audit_logger.log(
            "embedding_feature_store_started",
            {
                "run_id": run_id,
                "embedding_provider": embedding_provider,
                "input_rows": int(len(cohorts)),
                "max_features": max_features,
            },
        )

        self._validate_no_blocked_columns(cohorts.columns, context="embedding_raw_input")
        self.validator.validate_no_secret_values(cohorts, context="embedding_raw_input")

        safe_metadata = self._prepare_safe_metadata(cohorts)
        self.validator.validate_safe_cohort_dataframe(
            safe_metadata,
            context="embedding_safe_metadata",
        )

        trait_texts = safe_metadata["trait_text"].astype(str).fillna("").tolist()

        if embedding_provider == "sklearn_tfidf":
            vectors, provider_details = self._build_sklearn_tfidf_vectors(
                trait_texts=trait_texts,
                max_features=max_features,
            )
        elif embedding_provider == "external_embedding_service":
            raise NotImplementedError(
                "external_embedding_service is reserved for Docker/cloud embedding service integration."
            )
        else:
            raise ValueError(
                "Unsupported embedding_provider. Use sklearn_tfidf or external_embedding_service."
            )

        vectors = self._normalize_vectors(vectors)
        self._validate_vectors(vectors=vectors, expected_rows=len(safe_metadata))

        vector_path = output_dir / "cohort_vectors.npy"
        metadata_path = output_dir / "cohort_metadata.csv"
        preview_path = output_dir / "vector_preview.csv"
        manifest_path = output_dir / "embedding_manifest.json"
        similarity_demo_path = output_dir / "similarity_search_demo.json"

        np.save(vector_path, vectors)
        safe_metadata.to_csv(metadata_path, index=False)

        preview = self._build_vector_preview(safe_metadata, vectors)
        preview.to_csv(preview_path, index=False)

        similarity_demo = self.search_similar(
            query_text="restaurant evening high quality audience",
            metadata=safe_metadata,
            vectors=vectors,
            top_k=min(5, len(safe_metadata)),
        )
        self._write_json(similarity_demo_path, similarity_demo)

        manifest = {
            "module": "Embedding & Feature Store",
            "status": "completed",
            "run_id": run_id,
            "embedding_provider": embedding_provider,
            "provider_details": provider_details,
            "input_rows": int(len(cohorts)),
            "vector_count": int(vectors.shape[0]),
            "vector_dimension": int(vectors.shape[1]),
            "vectors_normalized": True,
            "metadata_rows": int(len(safe_metadata)),
            "raw_maids_exported": False,
            "raw_observations_exported": False,
            "raw_lat_lng_exported": False,
            "raw_email_exported": False,
            "raw_phone_exported": False,
            "individual_user_data_exported": False,
            "outputs": {
                "cohort_vectors": str(vector_path),
                "cohort_metadata": str(metadata_path),
                "vector_preview": str(preview_path),
                "embedding_manifest": str(manifest_path),
                "similarity_search_demo": str(similarity_demo_path),
            },
        }

        self._write_json(manifest_path, manifest)

        self.audit_logger.log(
            "embedding_feature_store_completed",
            {
                "run_id": run_id,
                "vector_count": int(vectors.shape[0]),
                "vector_dimension": int(vectors.shape[1]),
                "manifest": str(manifest_path),
            },
        )

        return manifest

    def build_in_memory(
        self,
        cohorts: pd.DataFrame,
        embedding_provider: str = "sklearn_tfidf",
        run_id: Optional[str] = None,
        max_features: int = 384,
    ) -> tuple[Dict[str, Any], pd.DataFrame, np.ndarray]:
        """Build the ordinary safe vectors without creating local artifacts."""
        run_id = run_id or f"embedding_run_{uuid.uuid4().hex[:12]}"
        self._validate_runtime_config(
            embedding_provider=embedding_provider,
            max_features=max_features,
        )
        self._validate_no_blocked_columns(
            cohorts.columns,
            context="embedding_raw_input",
        )
        self.validator.validate_no_secret_values(
            cohorts,
            context="embedding_raw_input",
        )
        safe_metadata = self._prepare_safe_metadata(cohorts)
        self.validator.validate_safe_cohort_dataframe(
            safe_metadata,
            context="embedding_safe_metadata",
        )
        trait_texts = safe_metadata["trait_text"].astype(str).fillna("").tolist()
        if embedding_provider != "sklearn_tfidf":
            raise ValueError(
                "In-memory evaluation currently requires sklearn_tfidf."
            )
        vectors, provider_details = self._build_sklearn_tfidf_vectors(
            trait_texts=trait_texts,
            max_features=max_features,
        )
        vectors = self._normalize_vectors(vectors)
        self._validate_vectors(
            vectors=vectors,
            expected_rows=len(safe_metadata),
        )
        base_uri = (
            "memory://audience_scale_evaluation/embedding"
            f"?run_id={run_id}"
        )
        manifest = {
            "module": "Embedding & Feature Store",
            "status": "completed",
            "run_id": run_id,
            "embedding_provider": embedding_provider,
            "provider_details": provider_details,
            "storage_backend": "memory",
            "input_rows": int(len(cohorts)),
            "vector_count": int(vectors.shape[0]),
            "vector_dimension": int(vectors.shape[1]),
            "vectors_normalized": True,
            "metadata_rows": int(len(safe_metadata)),
            "raw_maids_exported": False,
            "raw_observations_exported": False,
            "raw_lat_lng_exported": False,
            "raw_email_exported": False,
            "raw_phone_exported": False,
            "individual_user_data_exported": False,
            "outputs": {
                "cohort_vectors": base_uri + "&artifact=vectors",
                "cohort_metadata": base_uri + "&artifact=metadata",
                "embedding_manifest": base_uri + "&artifact=manifest",
            },
        }
        return manifest, safe_metadata, vectors

    def search_similar(
        self,
        query_text: str,
        metadata: pd.DataFrame,
        vectors: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D numpy array.")

        if len(metadata) != vectors.shape[0]:
            raise ValueError("metadata row count must match vector count.")

        query_vector = self._query_vector_from_existing_space(query_text, metadata, vectors)
        scores = vectors @ query_vector

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            row = metadata.iloc[int(idx)].to_dict()
            safe_row = self._json_safe(row)
            safe_row["similarity_score"] = float(scores[int(idx)])
            safe_row["cohort_index"] = int(idx)
            results.append(safe_row)

        return {
            "query": query_text,
            "top_k": int(top_k),
            "results": results,
            "raw_identifiers_returned": False,
        }

    def _validate_runtime_config(self, *, embedding_provider: str, max_features: int) -> None:
        if max_features <= 0:
            raise ValueError("max_features must be greater than 0.")

        allowed = {"sklearn_tfidf", "external_embedding_service"}
        if embedding_provider not in allowed:
            raise ValueError(f"embedding_provider must be one of: {sorted(allowed)}")

    def _validate_no_blocked_columns(self, columns, context: str) -> None:
        for col in columns:
            lower = str(col).lower()

            if lower in self.SAFE_AGGREGATE_COLUMNS:
                continue

            for token in self.BLOCKED_TOKENS:
                if lower == token or lower.startswith(f"{token}_") or lower.endswith(f"_{token}"):
                    raise ValueError(f"{context} contains blocked sensitive column: {col}")

            if lower in {"maid", "maids", "raw_maid", "device_id", "email", "phone", "lat", "lng"}:
                raise ValueError(f"{context} contains blocked sensitive column: {col}")

    def _prepare_safe_metadata(self, cohorts: pd.DataFrame) -> pd.DataFrame:
        df = cohorts.copy()

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

    def _build_sklearn_tfidf_vectors(
        self,
        *,
        trait_texts: List[str],
        max_features: int,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
        )

        matrix = vectorizer.fit_transform(trait_texts)
        vectors = matrix.toarray().astype("float32")

        return vectors, {
            "method": "TfidfVectorizer",
            "max_features": max_features,
            "actual_features": int(vectors.shape[1]),
            "ngram_range": [1, 2],
        }

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def _validate_vectors(self, *, vectors: np.ndarray, expected_rows: int) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D.")

        if vectors.shape[0] != expected_rows:
            raise ValueError("vector row count does not match metadata row count.")

        if vectors.shape[1] <= 0:
            raise ValueError("vector dimension must be greater than 0.")

        if not np.isfinite(vectors).all():
            raise ValueError("vectors contain NaN or infinite values.")

    def _build_vector_preview(self, metadata: pd.DataFrame, vectors: np.ndarray) -> pd.DataFrame:
        rows = []

        for idx in range(min(10, len(metadata))):
            row = metadata.iloc[idx].to_dict()
            rows.append(
                {
                    "cohort_index": idx,
                    "location_name": row.get("location_name"),
                    "primary_poi_type": row.get("primary_poi_type"),
                    "quality_score": row.get("quality_score"),
                    "vector_dimension": int(vectors.shape[1]),
                    "vector_norm": float(np.linalg.norm(vectors[idx])),
                    "privacy_status": row.get("privacy_status"),
                }
            )

        return pd.DataFrame(rows)

    def _query_vector_from_existing_space(
        self,
        query_text: str,
        metadata: pd.DataFrame,
        vectors: np.ndarray,
    ) -> np.ndarray:
        query_text = str(query_text).lower()
        trait_texts = metadata["trait_text"].astype(str).str.lower().tolist()

        query_terms = set(query_text.replace("|", " ").split())
        scores = []

        for text in trait_texts:
            text_terms = set(text.replace("|", " ").split())
            overlap = len(query_terms & text_terms)
            scores.append(overlap)

        if max(scores, default=0) == 0:
            query_vector = vectors.mean(axis=0)
        else:
            weighted = np.asarray(scores, dtype="float32")
            weighted = weighted / max(weighted.sum(), 1.0)
            query_vector = weighted @ vectors

        query_vector = np.asarray(query_vector, dtype="float32")
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            return query_vector

        return query_vector / norm

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
