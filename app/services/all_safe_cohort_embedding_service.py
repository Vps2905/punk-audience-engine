from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer


class AllSafeCohortEmbeddingService:
    """
    Builds embeddings across ALL privacy-safe cohorts.

    This fixes the old issue where only prompt-selected cohorts were embedded.
    Default:
    - embedding_scope = all_safe_cohorts
    - backend = sklearn_hashing
    - vector_dimension = 384
    """

    def __init__(
        self,
        embedding_scope: str | None = None,
        embedding_backend: str | None = None,
        max_features: int | None = None,
    ) -> None:
        self.embedding_scope = embedding_scope or os.getenv(
            "EMBEDDING_SCOPE", "all_safe_cohorts"
        )
        self.embedding_backend = embedding_backend or os.getenv(
            "EMBEDDING_BACKEND", "sklearn_hashing"
        )
        self.max_features = int(max_features or os.getenv("EMBEDDING_MAX_FEATURES", "384"))

    def build_index(
        self,
        safe_cohorts: pd.DataFrame,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        if safe_cohorts is None or safe_cohorts.empty:
            raise ValueError("safe_cohorts is empty. Cannot build embedding index.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cohort_df = safe_cohorts.copy().reset_index(drop=True)
        texts = cohort_df.apply(self._cohort_to_text, axis=1).tolist()

        vectorizer = HashingVectorizer(
            n_features=self.max_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
        )
        vectors = vectorizer.transform(texts).toarray().astype("float32")

        cohort_df["vector_index"] = list(range(len(cohort_df)))
        cohort_df["embedding_text"] = texts

        vector_path = output_path / "all_safe_cohort_vectors.npy"
        metadata_path = output_path / "all_safe_cohort_metadata.csv"
        manifest_path = output_path / "all_safe_cohort_embedding_manifest.json"

        np.save(vector_path, vectors)
        cohort_df.to_csv(metadata_path, index=False)

        manifest = {
            "status": "completed",
            "embedding_scope": self.embedding_scope,
            "embedding_backend": self.embedding_backend,
            "vector_count": int(vectors.shape[0]),
            "vector_dimension": int(vectors.shape[1]),
            "source_cohort_count": int(len(cohort_df)),
            "output_vectors": str(vector_path),
            "output_metadata": str(metadata_path),
            "privacy_note": (
                "Embeddings are built only from privacy-safe cohort metadata. "
                "No raw MAIDs, hashed IDs, raw lat/lng, email, phone, or individual rows are embedded."
            ),
        }

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def search_similar(
        self,
        query_text: str,
        vector_path: str | Path,
        metadata_path: str | Path,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        vectors = np.load(vector_path)
        metadata = pd.read_csv(metadata_path)

        if vectors.size == 0 or metadata.empty:
            return []

        vectorizer = HashingVectorizer(
            n_features=vectors.shape[1],
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
        )
        query_vector = vectorizer.transform([query_text]).toarray().astype("float32")[0]

        scores = vectors @ query_vector
        ranked_indexes = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_indexes:
            row = metadata.iloc[int(idx)].to_dict()
            row["similarity_score"] = round(float(scores[int(idx)]), 6)
            results.append(row)

        return results

    def _cohort_to_text(self, row: pd.Series) -> str:
        parts: list[str] = []

        safe_fields = [
            "location_name",
            "primary_poi_type",
            "created_day_part",
            "lookback_bucket",
            "privacy_status",
            "cohort_label",
            "audience_name",
        ]

        for field in safe_fields:
            if field in row and pd.notna(row[field]):
                value = str(row[field]).replace("_", " ").strip()
                if value:
                    parts.append(f"{field}:{value}")

        quality = self._first_number(row, ["management_quality_score", "quality_score"])
        if quality is not None:
            if quality >= 0.70:
                parts.append("quality:high")
            elif quality >= 0.35:
                parts.append("quality:medium")
            else:
                parts.append("quality:low")
            parts.append(f"quality_score:{round(quality, 3)}")

        volume = self._first_number(row, ["total_maid_volume", "noisy_maid_volume"])
        if volume is not None:
            if volume >= 100000:
                parts.append("volume:very_high")
            elif volume >= 10000:
                parts.append("volume:high")
            elif volume >= 1000:
                parts.append("volume:medium")
            else:
                parts.append("volume:low")

        return " ".join(parts)

    def _first_number(self, row: pd.Series, columns: list[str]) -> float | None:
        for column in columns:
            if column in row and pd.notna(row[column]):
                try:
                    return float(row[column])
                except Exception:
                    continue
        return None
