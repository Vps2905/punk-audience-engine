from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values


def _db_url() -> str:
    value = os.getenv("ECHO_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError("Postgres vector backend requires ECHO_DATABASE_URL or DATABASE_URL.")
    return value


def _connect():
    return psycopg2.connect(_db_url())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def ensure_postgres_vector_schema() -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS audience_vector_models (
                    job_id TEXT PRIMARY KEY,
                    model_info JSONB NOT NULL,
                    vector_count INTEGER NOT NULL DEFAULT 0,
                    vector_dimension INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS audience_vectors (
                    id BIGSERIAL PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    vector_index INTEGER NOT NULL,
                    trait_text TEXT,
                    location_name TEXT,
                    primary_poi_type TEXT,
                    created_day_part TEXT,
                    quality_score DOUBLE PRECISION,
                    embedding DOUBLE PRECISION[] NOT NULL,
                    metadata_json JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE(job_id, vector_index)
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_audience_vectors_job_id ON audience_vectors(job_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_audience_vectors_location ON audience_vectors(location_name);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_audience_vectors_poi ON audience_vectors(primary_poi_type);"
            )


def save_postgres_vector_store(
    job_id: str,
    vectors: np.ndarray,
    metadata: List[Dict[str, Any]],
    model_info: Dict[str, Any],
) -> Dict[str, str]:
    ensure_postgres_vector_schema()

    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array.")

    if len(metadata) != vectors.shape[0]:
        raise ValueError("metadata length must match vector row count.")

    rows = []
    for idx, item in enumerate(metadata):
        safe_item = _json_safe(item)
        rows.append(
            (
                job_id,
                int(item.get("vector_index", idx)),
                item.get("trait_text"),
                item.get("location_name"),
                item.get("primary_poi_type"),
                item.get("created_day_part"),
                float(item.get("quality_score", 0) or 0),
                [float(x) for x in vectors[idx].tolist()],
                Json(safe_item),
            )
        )

    safe_model_info = _json_safe(model_info)

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM audience_vectors WHERE job_id = %s;", (job_id,))
            cur.execute(
                """
                INSERT INTO audience_vector_models (
                    job_id, model_info, vector_count, vector_dimension, updated_at
                )
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT(job_id)
                DO UPDATE SET
                    model_info = EXCLUDED.model_info,
                    vector_count = EXCLUDED.vector_count,
                    vector_dimension = EXCLUDED.vector_dimension,
                    updated_at = now();
                """,
                (
                    job_id,
                    Json(safe_model_info),
                    int(vectors.shape[0]),
                    int(vectors.shape[1]),
                ),
            )
            execute_values(
                cur,
                """
                INSERT INTO audience_vectors (
                    job_id,
                    vector_index,
                    trait_text,
                    location_name,
                    primary_poi_type,
                    created_day_part,
                    quality_score,
                    embedding,
                    metadata_json
                )
                VALUES %s
                ON CONFLICT(job_id, vector_index)
                DO UPDATE SET
                    trait_text = EXCLUDED.trait_text,
                    location_name = EXCLUDED.location_name,
                    primary_poi_type = EXCLUDED.primary_poi_type,
                    created_day_part = EXCLUDED.created_day_part,
                    quality_score = EXCLUDED.quality_score,
                    embedding = EXCLUDED.embedding,
                    metadata_json = EXCLUDED.metadata_json;
                """,
                rows,
            )

    return {
        "vectors_path": f"postgres://audience_vectors?job_id={job_id}",
        "metadata_path": f"postgres://audience_vectors.metadata_json?job_id={job_id}",
        "model_path": f"postgres://audience_vector_models?job_id={job_id}",
    }


def load_postgres_vector_store(job_id: str) -> Dict[str, Any]:
    ensure_postgres_vector_schema()

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT model_info
                FROM audience_vector_models
                WHERE job_id = %s;
                """,
                (job_id,),
            )
            model_row = cur.fetchone()
            if not model_row:
                raise FileNotFoundError(f"Postgres vector model not found for job_id={job_id}")

            cur.execute(
                """
                SELECT vector_index, embedding, metadata_json
                FROM audience_vectors
                WHERE job_id = %s
                ORDER BY vector_index ASC;
                """,
                (job_id,),
            )
            rows = cur.fetchall()

    if not rows:
        raise FileNotFoundError(f"Postgres vectors not found for job_id={job_id}")

    vectors = np.asarray([row[1] for row in rows], dtype=float)
    metadata = [row[2] for row in rows]

    return {
        "vectors": vectors,
        "metadata": metadata,
        "model_info": model_row[0],
    }


def postgres_similarity_search(
    job_id: str,
    query_vector: np.ndarray,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    store = load_postgres_vector_store(job_id)
    vectors = np.asarray(store["vectors"], dtype=float)
    metadata = store["metadata"]

    if vectors.size == 0:
        return []

    query = np.asarray(query_vector, dtype=float)
    query_norm = np.linalg.norm(query)
    vector_norms = np.linalg.norm(vectors, axis=1)

    denominator = vector_norms * query_norm
    denominator[denominator == 0] = 1e-12

    scores = (vectors @ query) / denominator
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        item = dict(metadata[int(idx)])
        item["similarity_score"] = float(scores[int(idx)])
        results.append(item)

    return results


def save_postgres_cluster_output(job_id: str, clustered_df: pd.DataFrame) -> str:
    ensure_postgres_vector_schema()

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS audience_vector_clusters (
                    id BIGSERIAL PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    vector_index INTEGER,
                    cluster_id INTEGER,
                    cluster_record JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE(job_id, vector_index)
                );
                """
            )
            cur.execute("DELETE FROM audience_vector_clusters WHERE job_id = %s;", (job_id,))

            rows = []
            for idx, row in clustered_df.iterrows():
                record = _json_safe(row.to_dict())
                rows.append(
                    (
                        job_id,
                        int(record.get("vector_index", idx)),
                        int(record.get("cluster_id", 0)),
                        Json(record),
                    )
                )

            execute_values(
                cur,
                """
                INSERT INTO audience_vector_clusters (
                    job_id,
                    vector_index,
                    cluster_id,
                    cluster_record
                )
                VALUES %s
                ON CONFLICT(job_id, vector_index)
                DO UPDATE SET
                    cluster_id = EXCLUDED.cluster_id,
                    cluster_record = EXCLUDED.cluster_record;
                """,
                rows,
            )

    return f"postgres://audience_vector_clusters?job_id={job_id}"
