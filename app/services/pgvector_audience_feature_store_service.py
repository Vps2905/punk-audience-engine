from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.models.audience_feature_contracts import (
    CanonicalFeatureSet,
    normalize_taxonomy_value,
    validate_embedding,
)


class PgvectorAudienceFeatureStore:
    """
    Versioned, tenant-scoped feature and pgvector persistence.

    The SQL migration is authoritative. This service intentionally does not
    create or mutate schemas at request time.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: Engine | None = None,
        expected_dimension: int | None = None,
    ) -> None:
        self._database_url = database_url
        self._engine_override = engine
        self._expected_dimension = int(
            expected_dimension
            or os.getenv("PGVECTOR_DIMENSION", "384")
        )
        if self._expected_dimension != 384:
            raise ValueError(
                "Migration 0004 uses vector(384). PGVECTOR_DIMENSION must be 384."
            )

    def save_feature_set(
        self,
        feature_set: CanonicalFeatureSet,
    ) -> dict[str, Any]:
        if feature_set.embedding_dimension != self._expected_dimension:
            raise ValueError(
                "Feature-set embedding dimension does not match pgvector schema."
            )
        if feature_set.feature_count != len(feature_set.features):
            raise ValueError(
                "feature_count must match the number of canonical features."
            )

        engine = self._engine()
        with engine.begin() as connection:
            self._assert_schema_ready(connection)
            self._set_tenant_context(connection, feature_set.tenant_id)
            connection.execute(
                text(
                    """
                    INSERT INTO audience_feature_sets (
                        tenant_id,
                        feature_set_id,
                        version,
                        status,
                        source_mode,
                        data_use_mode,
                        source_ref,
                        source_version,
                        source_fingerprint,
                        source_latest_at,
                        freshness_status,
                        stale_after_hours,
                        model_backend,
                        model_name,
                        model_version,
                        embedding_dimension,
                        privacy_policy_version,
                        rights_policy_id,
                        purpose,
                        eligible_for_retrieval,
                        eligible_for_activation,
                        feature_count,
                        lineage,
                        updated_at
                    )
                    VALUES (
                        :tenant_id,
                        :feature_set_id,
                        :version,
                        :status,
                        :source_mode,
                        :data_use_mode,
                        :source_ref,
                        :source_version,
                        :source_fingerprint,
                        CAST(:source_latest_at AS timestamptz),
                        :freshness_status,
                        :stale_after_hours,
                        :model_backend,
                        :model_name,
                        :model_version,
                        :embedding_dimension,
                        :privacy_policy_version,
                        :rights_policy_id,
                        :purpose,
                        :eligible_for_retrieval,
                        :eligible_for_activation,
                        :feature_count,
                        CAST(:lineage AS jsonb),
                        now()
                    )
                    ON CONFLICT (
                        tenant_id,
                        feature_set_id,
                        version
                    )
                    DO NOTHING
                    """
                ),
                {
                    **feature_set.to_record(include_features=False),
                    "lineage": json.dumps(
                        feature_set.lineage,
                        sort_keys=True,
                        default=str,
                    ),
                },
            )

            rows = [
                self._feature_insert_record(feature)
                for feature in feature_set.features
            ]
            connection.execute(
                text(
                    """
                    INSERT INTO audience_feature_vectors (
                        tenant_id,
                        feature_set_id,
                        feature_set_version,
                        feature_id,
                        location_name,
                        primary_poi_type,
                        created_day_part,
                        lookback_bucket,
                        cohort_size,
                        quality_score,
                        privacy_status,
                        rights_status,
                        purpose,
                        source_latest_at,
                        freshness_status,
                        data_use_mode,
                        eligible_for_retrieval,
                        eligible_for_activation,
                        trait_text,
                        embedding,
                        metadata,
                        updated_at
                    )
                    VALUES (
                        :tenant_id,
                        :feature_set_id,
                        :feature_set_version,
                        :feature_id,
                        :location_name,
                        :primary_poi_type,
                        :created_day_part,
                        :lookback_bucket,
                        :cohort_size,
                        :quality_score,
                        :privacy_status,
                        :rights_status,
                        :purpose,
                        CAST(:source_latest_at AS timestamptz),
                        :freshness_status,
                        :data_use_mode,
                        :eligible_for_retrieval,
                        :eligible_for_activation,
                        :trait_text,
                        CAST(:embedding AS vector(384)),
                        CAST(:metadata AS jsonb),
                        now()
                    )
                    ON CONFLICT (
                        tenant_id,
                        feature_set_id,
                        feature_set_version,
                        feature_id
                    )
                    DO NOTHING
                    """
                ),
                rows,
            )

        return {
            "status": "completed",
            "tenant_id": feature_set.tenant_id,
            "feature_set_id": feature_set.feature_set_id,
            "feature_set_version": feature_set.version,
            "feature_count": feature_set.feature_count,
            "source_mode": feature_set.source_mode,
            "data_use_mode": feature_set.data_use_mode,
            "freshness_status": feature_set.freshness_status,
            "eligible_for_activation": feature_set.eligible_for_activation,
            "storage_backend": "postgres_pgvector",
        }

    def get_feature_set(
        self,
        *,
        tenant_id: str,
        feature_set_id: str | None = None,
        version: int | None = None,
        data_use_mode: str | None = None,
    ) -> dict[str, Any]:
        engine = self._engine()
        with engine.connect() as connection:
            self._assert_schema_ready(connection)
            self._set_tenant_context(connection, tenant_id)
            where = ["tenant_id = :tenant_id"]
            params: dict[str, Any] = {"tenant_id": tenant_id}
            if feature_set_id:
                where.append("feature_set_id = :feature_set_id")
                params["feature_set_id"] = feature_set_id
            if version is not None:
                where.append("version = :version")
                params["version"] = int(version)
            if data_use_mode:
                clean_data_use_mode = normalize_taxonomy_value(
                    data_use_mode
                )
                if clean_data_use_mode not in {
                    "historical_preview",
                    "offline_evaluation",
                    "production",
                }:
                    raise ValueError("Invalid feature-set data_use_mode.")
                where.append("data_use_mode = :data_use_mode")
                params["data_use_mode"] = clean_data_use_mode
            row = connection.execute(
                text(
                    f"""
                    SELECT
                        tenant_id,
                        feature_set_id,
                        version,
                        status,
                        source_mode,
                        data_use_mode,
                        source_ref,
                        source_version,
                        source_fingerprint,
                        source_latest_at,
                        freshness_status,
                        stale_after_hours,
                        model_backend,
                        model_name,
                        model_version,
                        embedding_dimension,
                        privacy_policy_version,
                        rights_policy_id,
                        purpose,
                        eligible_for_retrieval,
                        eligible_for_activation,
                        feature_count,
                        lineage,
                        created_at,
                        updated_at
                    FROM audience_feature_sets
                    WHERE {" AND ".join(where)}
                    ORDER BY version DESC, updated_at DESC
                    LIMIT 1
                    """
                ),
                params,
            ).mappings().first()
        if not row:
            raise FileNotFoundError(
                "No feature set is available for the requested tenant and version."
            )
        return dict(row)

    def hybrid_search(
        self,
        *,
        tenant_id: str,
        feature_set_id: str,
        feature_set_version: int,
        query_text: str,
        query_embedding: Sequence[float],
        execution_mode: str,
        locations: Sequence[str],
        categories: Sequence[str],
        dayparts: Sequence[str],
        exclusions: Sequence[str],
        top_k: int,
        search_strategy: str = "ann",
    ) -> list[dict[str, Any]]:
        vector = validate_embedding(
            query_embedding,
            expected_dimension=self._expected_dimension,
        )
        normalized_locations = self._normalize_list(locations)
        normalized_categories = self._normalize_list(categories)
        normalized_dayparts = self._normalize_list(dayparts)
        normalized_exclusions = self._normalize_list(exclusions)
        clean_mode = normalize_taxonomy_value(execution_mode)
        if clean_mode not in {"historical_preview", "production"}:
            raise ValueError(
                "execution_mode must be historical_preview or production."
            )
        clean_search_strategy = normalize_taxonomy_value(search_strategy)
        if clean_search_strategy not in {"ann", "exact"}:
            raise ValueError("search_strategy must be ann or exact.")
        top_k = max(1, min(int(top_k), 100))
        candidate_limit = min(max(top_k * 8, 40), 800)

        where = [
            "afv.tenant_id = :tenant_id",
            "afv.feature_set_id = :feature_set_id",
            "afv.feature_set_version = :feature_set_version",
            "afv.eligible_for_retrieval = TRUE",
            "afv.privacy_status IN ('safe', 'passed', 'privacy_safe')",
        ]
        if clean_mode == "production":
            where.extend(
                [
                    "afv.eligible_for_activation = TRUE",
                    "afv.freshness_status = 'fresh'",
                    "afv.data_use_mode = 'production'",
                ]
            )
        if normalized_locations:
            where.append(
                """
                EXISTS (
                    SELECT 1
                    FROM unnest(CAST(:locations AS text[])) AS requested_location
                    WHERE to_tsvector(
                        'simple',
                        replace(afv.location_name, '_', ' ')
                    ) @@ plainto_tsquery(
                        'simple',
                        replace(requested_location, '_', ' ')
                    )
                )
                """
            )
        if normalized_categories:
            where.append(
                "afv.primary_poi_type = ANY(CAST(:categories AS text[]))"
            )
        if normalized_dayparts:
            where.append(
                "afv.created_day_part = ANY(CAST(:dayparts AS text[]))"
            )
        if normalized_exclusions:
            where.append(
                "NOT (afv.primary_poi_type = ANY(CAST(:exclusions AS text[])))"
            )

        params: dict[str, Any] = {
            "tenant_id": tenant_id,
            "feature_set_id": feature_set_id,
            "feature_set_version": int(feature_set_version),
            "query_text": " ".join(str(query_text or "").split()),
            "query_embedding": self._vector_literal(vector),
            "locations": normalized_locations,
            "categories": normalized_categories,
            "dayparts": normalized_dayparts,
            "exclusions": normalized_exclusions,
            "candidate_limit": candidate_limit,
            "top_k": top_k,
        }

        sql = text(
            f"""
            WITH vector_ranked AS (
                SELECT
                    afv.feature_id,
                    1.0 - (
                        afv.embedding
                        <=> CAST(:query_embedding AS vector(384))
                    ) AS vector_score,
                    ROW_NUMBER() OVER (
                        ORDER BY
                            afv.embedding
                            <=> CAST(:query_embedding AS vector(384)),
                            afv.feature_id ASC
                    ) AS vector_rank
                FROM audience_feature_vectors afv
                WHERE {" AND ".join(where)}
                ORDER BY
                    afv.embedding
                    <=> CAST(:query_embedding AS vector(384)),
                    afv.feature_id ASC
                LIMIT :candidate_limit
            ),
            lexical_ranked AS (
                SELECT
                    afv.feature_id,
                    ts_rank_cd(
                        afv.search_document,
                        websearch_to_tsquery('simple', :query_text)
                    ) AS lexical_score,
                    ROW_NUMBER() OVER (
                        ORDER BY
                            ts_rank_cd(
                                afv.search_document,
                                websearch_to_tsquery('simple', :query_text)
                            ) DESC,
                            afv.feature_id ASC
                    ) AS lexical_rank
                FROM audience_feature_vectors afv
                WHERE {" AND ".join(where)}
                    AND afv.search_document @@
                        websearch_to_tsquery('simple', :query_text)
                ORDER BY
                    lexical_score DESC,
                    afv.feature_id ASC
                LIMIT :candidate_limit
            ),
            candidate_ids AS (
                SELECT feature_id FROM vector_ranked
                UNION
                SELECT feature_id FROM lexical_ranked
            )
            SELECT
                afv.feature_id,
                afv.location_name,
                afv.primary_poi_type,
                afv.created_day_part,
                afv.lookback_bucket,
                afv.cohort_size,
                afv.quality_score,
                afv.privacy_status,
                afv.rights_status,
                afv.purpose,
                afv.source_latest_at,
                afv.freshness_status,
                afv.data_use_mode,
                afv.eligible_for_activation,
                afv.metadata,
                GREATEST(
                    LEAST(vector_ranked.vector_score, 1.0),
                    -1.0
                ) AS vector_score,
                lexical_ranked.lexical_score,
                vector_ranked.vector_rank,
                lexical_ranked.lexical_rank,
                (
                    CASE
                        WHEN vector_ranked.vector_rank IS NULL THEN 0.0
                        ELSE 1.0 / (60.0 + vector_ranked.vector_rank)
                    END
                    +
                    CASE
                        WHEN lexical_ranked.lexical_rank IS NULL THEN 0.0
                        ELSE 1.0 / (60.0 + lexical_ranked.lexical_rank)
                    END
                ) AS fused_score
            FROM candidate_ids
            JOIN audience_feature_vectors afv
                ON afv.tenant_id = :tenant_id
                AND afv.feature_set_id = :feature_set_id
                AND afv.feature_set_version = :feature_set_version
                AND afv.feature_id = candidate_ids.feature_id
            LEFT JOIN vector_ranked
                ON vector_ranked.feature_id = candidate_ids.feature_id
            LEFT JOIN lexical_ranked
                ON lexical_ranked.feature_id = candidate_ids.feature_id
            ORDER BY
                fused_score DESC,
                vector_score DESC NULLS LAST,
                afv.quality_score DESC,
                afv.feature_id ASC
            LIMIT :top_k
            """
        )

        engine = self._engine()
        with engine.connect() as connection:
            self._assert_schema_ready(connection)
            self._set_tenant_context(connection, tenant_id)
            if clean_search_strategy == "exact":
                connection.execute(
                    text("SET LOCAL enable_indexscan = off")
                )
                connection.execute(
                    text("SET LOCAL enable_bitmapscan = off")
                )
            else:
                ef_search = max(
                    10,
                    min(
                        int(os.getenv("PGVECTOR_HNSW_EF_SEARCH", "100")),
                        1000,
                    ),
                )
                connection.execute(
                    text(
                        """
                        SELECT set_config(
                            'hnsw.ef_search',
                            :ef_search,
                            true
                        )
                        """
                    ),
                    {"ef_search": str(ef_search)},
                )
            rows = connection.execute(sql, params).mappings().all()

        return [self._result_record(row) for row in rows]

    def _feature_insert_record(self, feature: Any) -> dict[str, Any]:
        record = feature.to_record()
        record["embedding"] = self._vector_literal(record["embedding"])
        record["metadata"] = json.dumps(
            feature.metadata,
            sort_keys=True,
            default=str,
        )
        return record

    def _result_record(self, row: Any) -> dict[str, Any]:
        record = dict(row)
        for key in (
            "quality_score",
            "vector_score",
            "lexical_score",
            "fused_score",
        ):
            value = record.get(key)
            if value is not None:
                record[key] = round(float(value), 8)
        for key in ("cohort_size", "vector_rank", "lexical_rank"):
            value = record.get(key)
            if value is not None:
                record[key] = int(value)
        timestamp = record.get("source_latest_at")
        if timestamp is not None and hasattr(timestamp, "isoformat"):
            record["source_latest_at"] = timestamp.isoformat()
        record["metadata"] = dict(record.get("metadata") or {})
        return record

    def _normalize_list(self, values: Sequence[str]) -> list[str]:
        output: list[str] = []
        for value in values:
            normalized = normalize_taxonomy_value(value)
            if normalized and normalized not in output:
                output.append(normalized)
        return output

    def _vector_literal(self, values: Sequence[float]) -> str:
        vector = np.asarray(values, dtype=float)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("Embedding must be a finite one-dimensional vector.")
        if math.isclose(float(np.linalg.norm(vector)), 0.0):
            raise ValueError("Embedding must have a non-zero magnitude.")
        return "[" + ",".join(format(float(value), ".12g") for value in vector) + "]"

    def _assert_schema_ready(self, connection: Any) -> None:
        ready = connection.execute(
            text(
                """
                SELECT
                    to_regclass('public.audience_feature_sets') IS NOT NULL
                    AND
                    to_regclass('public.audience_feature_vectors') IS NOT NULL
                    AS ready
                """
            )
        ).scalar()
        if not ready:
            raise RuntimeError(
                "Phase 2 database schema is not ready. "
                "Apply migration 0004 before indexing or retrieval."
            )

    def _set_tenant_context(self, connection: Any, tenant_id: str) -> None:
        clean_tenant_id = normalize_taxonomy_value(tenant_id)
        if not clean_tenant_id:
            raise ValueError("tenant_id is required.")
        connection.execute(
            text(
                """
                SELECT set_config(
                    'app.tenant_id',
                    :tenant_id,
                    true
                )
                """
            ),
            {"tenant_id": clean_tenant_id},
        )

    def _engine(self) -> Engine:
        if self._engine_override is not None:
            return self._engine_override
        database_url = (
            self._database_url
            or os.getenv("AUDIENCE_FEATURE_DATABASE_URL")
            or ""
        ).strip()
        if not database_url:
            raise RuntimeError(
                "Phase 2 feature storage requires an explicit "
                "AUDIENCE_FEATURE_DATABASE_URL."
            )
        if database_url.startswith("postgres://"):
            database_url = "postgresql://" + database_url[len("postgres://") :]
        return create_engine(database_url, pool_pre_ping=True)
