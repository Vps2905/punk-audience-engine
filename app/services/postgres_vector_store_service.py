from __future__ import annotations

import json
import math
import os
import re
import threading
import unicodedata
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values
from psycopg2.pool import ThreadedConnectionPool


def _db_url() -> str:
    value = os.getenv("ECHO_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError("Postgres vector backend requires ECHO_DATABASE_URL or DATABASE_URL.")
    return value


_POOL: ThreadedConnectionPool | None = None
_POOL_LOCK = threading.Lock()
_SCHEMA_READY = False
_SCHEMA_LOCK = threading.Lock()


def _pool_size() -> int:
    raw_value = os.getenv(
        "POSTGRES_VECTOR_POOL_SIZE",
        "5",
    )

    try:
        return max(2, min(int(raw_value), 20))
    except (TypeError, ValueError):
        return 5


def _get_connection_pool() -> ThreadedConnectionPool:
    global _POOL

    if _POOL is not None:
        return _POOL

    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadedConnectionPool(
                minconn=1,
                maxconn=_pool_size(),
                dsn=_db_url(),
            )

    return _POOL


@contextmanager
def _connect():
    """
    Borrow a reusable Postgres connection.

    This avoids a new network/TLS connection for every
    similarity search.
    """
    pool = _get_connection_pool()
    conn = pool.getconn()

    try:
        if conn.closed:
            pool.putconn(conn, close=True)
            conn = pool.getconn()

        yield conn
        conn.commit()

    except Exception:
        if not conn.closed:
            conn.rollback()
        raise

    finally:
        if not conn.closed:
            pool.putconn(conn)


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
                    embedding_norm DOUBLE PRECISION NOT NULL,
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

            cur.execute(
                """
                ALTER TABLE audience_vectors
                ADD COLUMN IF NOT EXISTS
                embedding_norm DOUBLE PRECISION;
                """
            )
            cur.execute(
                """
                UPDATE audience_vectors
                SET embedding_norm = SQRT(
                    (
                        SELECT COALESCE(
                            SUM(value * value),
                            0.0
                        )
                        FROM unnest(embedding)
                            AS value
                    )
                )
                WHERE embedding_norm IS NULL;
                """
            )
            cur.execute(
                """
                ALTER TABLE audience_vectors
                ALTER COLUMN embedding_norm
                SET NOT NULL;
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_audience_vectors_filter
                ON audience_vectors(
                    job_id,
                    location_name,
                    primary_poi_type,
                    created_day_part,
                    quality_score
                );
                """
            )



def _ensure_postgres_vector_schema_once() -> None:
    """
    Ensure the vector schema once per application process.

    Versioned migrations remain authoritative. This is a
    compatibility fallback for development and new database
    environments.
    """
    global _SCHEMA_READY

    if _SCHEMA_READY:
        return

    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return

        ensure_postgres_vector_schema()
        _SCHEMA_READY = True


_POI_TYPE_ALIASES = {
    "cafe": "cafe",
    "coffee": "cafe",
    "coffee_shop": "cafe",
    "coffeehouse": "cafe",
    "espresso_bar": "cafe",
    "co_working": "coworking_space",
    "coworking": "coworking_space",
    "coworking_space": "coworking_space",
    "fitness_center": "gym",
    "fitness_centre": "gym",
    "health_club": "gym",
    "gym": "gym",
    "grocery": "grocery_store",
    "grocery_store": "grocery_store",
    "supermarket": "grocery_store",
    "barbershop": "barber_shop",
    "barber": "barber_shop",
    "barber_shop": "barber_shop",
}


def canonicalize_poi_type(
    value: str | None,
) -> str | None:
    """
    Normalize user-facing POI names to the stored taxonomy.
    """
    if value is None:
        return None

    normalized = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    normalized = normalized.encode(
        "ascii",
        "ignore",
    ).decode("ascii")

    normalized = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        normalized.lower(),
    ).strip("_")

    if not normalized:
        return None

    return _POI_TYPE_ALIASES.get(
        normalized,
        normalized,
    )

def save_postgres_vector_store(
    job_id: str,
    vectors: np.ndarray,
    metadata: List[Dict[str, Any]],
    model_info: Dict[str, Any],
) -> Dict[str, str]:
    _ensure_postgres_vector_schema_once()

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
                float(np.linalg.norm(vectors[idx])),
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



def load_postgres_vector_model_info(
    job_id: str,
) -> Dict[str, Any]:
    """
    Load only embedding model metadata.

    Query encoding needs the backend and dimension but must
    not load every stored vector into application memory.
    """
    clean_job_id = str(job_id or "").strip()

    if not clean_job_id:
        raise ValueError("job_id cannot be empty.")

    _ensure_postgres_vector_schema_once()

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    model_info,
                    vector_count,
                    vector_dimension
                FROM audience_vector_models
                WHERE job_id = %s;
                """,
                (clean_job_id,),
            )

            row = cur.fetchone()

    if not row:
        raise FileNotFoundError(
            "Postgres vector model not found "
            f"for job_id={clean_job_id}"
        )

    model_info = dict(row[0] or {})
    model_info.setdefault(
        "vector_count",
        int(row[1] or 0),
    )
    model_info.setdefault(
        "dimension",
        int(row[2] or 0),
    )
    model_info.setdefault(
        "vector_dimension",
        int(row[2] or 0),
    )

    return model_info

def load_postgres_vector_store(job_id: str) -> Dict[str, Any]:
    _ensure_postgres_vector_schema_once()

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
    *,
    location_name: Optional[str] = None,
    primary_poi_type: Optional[str] = None,
    created_day_part: Optional[str] = None,
    min_quality: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Rank audience vectors inside PostgreSQL using cosine
    similarity over DOUBLE PRECISION[] embeddings.

    Only the requested top-k metadata records are returned
    to the application.
    """
    clean_job_id = str(job_id or "").strip()

    if not clean_job_id:
        raise ValueError("job_id cannot be empty.")

    query = np.asarray(
        query_vector,
        dtype=float,
    )

    if query.ndim != 1:
        raise ValueError(
            "query_vector must be one-dimensional."
        )

    if query.size == 0:
        raise ValueError(
            "query_vector cannot be empty."
        )

    if not np.isfinite(query).all():
        raise ValueError(
            "query_vector contains non-finite values."
        )

    query_dimension = int(query.shape[0])
    query_norm = float(np.linalg.norm(query))

    if query_norm <= 0:
        raise ValueError(
            "query_vector must have a non-zero norm."
        )

    top_k = max(
        1,
        min(int(top_k), 200),
    )

    _ensure_postgres_vector_schema_once()

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    model_info,
                    vector_dimension,
                    vector_count
                FROM audience_vector_models
                WHERE job_id = %s;
                """,
                (clean_job_id,),
            )

            model_row = cur.fetchone()

            if not model_row:
                raise FileNotFoundError(
                    "Postgres vector model not found "
                    f"for job_id={clean_job_id}"
                )

            stored_dimension = int(
                model_row[1] or 0
            )

            if stored_dimension != query_dimension:
                raise ValueError(
                    "Query vector dimension mismatch. "
                    f"Expected {stored_dimension}, "
                    f"received {query_dimension}."
                )

            where = [
                "av.job_id = %(job_id)s",
                (
                    "cardinality(av.embedding) = "
                    "%(query_dimension)s"
                ),
            ]

            params: Dict[str, Any] = {
                "job_id": clean_job_id,
                "query_vector": [
                    float(value)
                    for value in query.tolist()
                ],
                "query_dimension": query_dimension,
                "query_norm": query_norm,
                "top_k": top_k,
            }

            if location_name:
                where.append(
                    "LOWER(av.location_name) = "
                    "LOWER(%(location_name)s)"
                )
                params["location_name"] = str(
                    location_name
                ).strip()

            canonical_poi_type = (
                canonicalize_poi_type(
                    primary_poi_type
                )
            )

            if canonical_poi_type:
                where.append(
                    "LOWER(av.primary_poi_type) = "
                    "LOWER(%(primary_poi_type)s)"
                )
                params[
                    "primary_poi_type"
                ] = canonical_poi_type

            if created_day_part:
                where.append(
                    "LOWER(av.created_day_part) = "
                    "LOWER(%(created_day_part)s)"
                )
                params["created_day_part"] = str(
                    created_day_part
                ).strip()

            if min_quality is not None:
                params["min_quality"] = max(
                    0.0,
                    min(float(min_quality), 1.0),
                )
                where.append(
                    "COALESCE(av.quality_score, 0) "
                    ">= %(min_quality)s"
                )

            where_sql = " AND ".join(where)

            cur.execute(
                f"""
                WITH scored AS (
                    SELECT
                        av.vector_index,
                        av.metadata_json,
                        CASE
                            WHEN
                                av.embedding_norm <= 0
                                OR %(query_norm)s <= 0
                            THEN 0.0
                            ELSE (
                                SELECT COALESCE(
                                    SUM(
                                        stored.value
                                        * requested.value
                                    ),
                                    0.0
                                )
                                FROM unnest(av.embedding)
                                    WITH ORDINALITY
                                    AS stored(
                                        value,
                                        position
                                    )
                                JOIN unnest(
                                    %(query_vector)s
                                    ::DOUBLE PRECISION[]
                                )
                                    WITH ORDINALITY
                                    AS requested(
                                        value,
                                        position
                                    )
                                USING (position)
                            ) / NULLIF(
                                av.embedding_norm
                                * %(query_norm)s,
                                0.0
                            )
                        END AS similarity_score
                    FROM audience_vectors av
                    WHERE {where_sql}
                )
                SELECT
                    metadata_json,
                    similarity_score,
                    vector_index
                FROM scored
                ORDER BY
                    similarity_score DESC,
                    vector_index ASC
                LIMIT %(top_k)s;
                """,
                params,
            )

            rows = cur.fetchall()

    results: List[Dict[str, Any]] = []

    for metadata, score, vector_index in rows:
        item = dict(metadata or {})
        item.setdefault(
            "vector_index",
            int(vector_index),
        )
        item["similarity_score"] = float(
            score or 0.0
        )
        results.append(item)

    return results


def save_postgres_cluster_output(job_id: str, clustered_df: pd.DataFrame) -> str:
    _ensure_postgres_vector_schema_once()

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
