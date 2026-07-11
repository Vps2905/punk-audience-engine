from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from app.services.embedding_service import embed_records, search_similar_audiences


_POSTGRES_STORES = {"postgres", "postgres_array", "pg_array", "pgvector"}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "")).strip("_")
    return slug or "all_safe_cohorts"


class AllSafeCohortEmbeddingService:
    """
    Builds embeddings across ALL privacy-safe cohorts.

    Dev/default:
    - local all_safe_cohort_vectors.npy
    - local metadata CSV
    - local manifest JSON

    Production:
    - ALL_SAFE_COHORT_EMBEDDING_STORE=postgres
    - uses embed_records()
    - stores vectors through active VECTOR_BACKEND
    """

    def __init__(
        self,
        embedding_scope: str | None = None,
        embedding_backend: str | None = None,
        max_features: int | None = None,
        embedding_store: str | None = None,
    ) -> None:
        self.embedding_scope = embedding_scope or os.getenv(
            "EMBEDDING_SCOPE", "all_safe_cohorts"
        )
        self.embedding_backend = embedding_backend or os.getenv(
            "EMBEDDING_BACKEND", "sklearn_hashing"
        )
        self.max_features = int(max_features or os.getenv("EMBEDDING_MAX_FEATURES", "384"))
        self.embedding_store = (
            embedding_store
            or os.getenv("ALL_SAFE_COHORT_EMBEDDING_STORE", "local")
        ).strip().lower()

    def build_index(
        self,
        safe_cohorts: pd.DataFrame,
        output_dir: str | Path,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        if safe_cohorts is None or safe_cohorts.empty:
            raise ValueError("safe_cohorts is empty. Cannot build embedding index.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cohort_df = safe_cohorts.copy().reset_index(drop=True)
        texts = cohort_df.apply(self._cohort_to_text, axis=1).tolist()

        cohort_df["vector_index"] = list(range(len(cohort_df)))
        cohort_df["embedding_text"] = texts

        if self.embedding_store in _POSTGRES_STORES:
            return self._build_postgres_index(
                cohort_df=cohort_df,
                output_path=output_path,
                job_id=job_id,
            )

        return self._build_local_index(
            cohort_df=cohort_df,
            texts=texts,
            output_path=output_path,
        )

    def _build_postgres_index(
        self,
        cohort_df: pd.DataFrame,
        output_path: Path,
        job_id: str | None,
    ) -> dict[str, Any]:
        embedding_job_id = job_id or f"all_safe_{_slug(output_path.parent.name or output_path.name)}"

        embed_result = embed_records(
            job_id=embedding_job_id,
            records=cohort_df.to_dict(orient="records"),
        )

        manifest_path = output_path / "all_safe_cohort_embedding_manifest.json"

        manifest = {
            "status": "completed",
            "embedding_scope": self.embedding_scope,
            "embedding_backend": embed_result["embedding_backend"],
            "embedding_store": self.embedding_store,
            "vector_backend": os.getenv("VECTOR_BACKEND", ""),
            "job_id": embedding_job_id,
            "vector_count": int(embed_result["rows_embedded"]),
            "vector_dimension": int(embed_result["dimension"]),
            "source_cohort_count": int(len(cohort_df)),
            "output_vectors": embed_result.get("vectors_path"),
            "output_metadata": embed_result.get("metadata_path"),
            "output_model": embed_result.get("model_path"),
            "manifest_path": str(manifest_path),
            "privacy_note": (
                "Embeddings are built only from privacy-safe cohort metadata. "
                "No raw MAIDs, hashed IDs, raw lat/lng, email, phone, or individual rows are embedded."
            ),
        }

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def _build_local_index(
        self,
        cohort_df: pd.DataFrame,
        texts: list[str],
        output_path: Path,
    ) -> dict[str, Any]:
        vectorizer = HashingVectorizer(
            n_features=self.max_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
        )
        vectors = vectorizer.transform(texts).toarray().astype("float32")

        vector_path = output_path / "all_safe_cohort_vectors.npy"
        metadata_path = output_path / "all_safe_cohort_metadata.csv"
        manifest_path = output_path / "all_safe_cohort_embedding_manifest.json"

        np.save(vector_path, vectors)
        cohort_df.to_csv(metadata_path, index=False)

        manifest = {
            "status": "completed",
            "embedding_scope": self.embedding_scope,
            "embedding_backend": self.embedding_backend,
            "embedding_store": self.embedding_store,
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
        vector_path_text = str(vector_path)

        if vector_path_text.startswith("postgres://"):
            job_id = vector_path_text.split("job_id=", 1)[-1]
            return search_similar_audiences(
                job_id=job_id,
                query=query_text,
                top_k=top_k,
            )["results"]

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
