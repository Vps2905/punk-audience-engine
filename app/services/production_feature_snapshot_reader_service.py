from __future__ import annotations

import os
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.models.audience_feature_contracts import normalize_taxonomy_value


class ProductionFeatureSnapshotReaderService:
    """Read exact aggregate feature metadata without returning embeddings.

    The reader uses the tenant-scoped feature reader role and deliberately
    excludes the pgvector embedding column. It is safe for Module 3 candidate
    generation and cannot mutate feature artifacts.
    """

    def __init__(
        self,
        *,
        database_url: str | None = None,
        engine: Engine | None = None,
    ) -> None:
        self._database_url = database_url
        self._engine_override = engine

    def read(
        self,
        *,
        tenant_id: str,
        feature_set_id: str,
        feature_set_version: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        clean_tenant = normalize_taxonomy_value(tenant_id)
        if not clean_tenant:
            raise ValueError("tenant_id is required.")
        if not str(feature_set_id or "").strip():
            raise ValueError("feature_set_id is required.")
        if int(feature_set_version) < 1:
            raise ValueError("feature_set_version must be >= 1.")

        engine = self._engine()
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.execute(
                    text(
                        "SELECT set_config('app.tenant_id', :tenant_id, true)"
                    ),
                    {"tenant_id": clean_tenant},
                )
                feature_set = connection.execute(
                    text(
                        """
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
                            privacy_policy_version,
                            rights_policy_id,
                            purpose,
                            eligible_for_retrieval,
                            eligible_for_activation,
                            feature_count
                        FROM audience_feature_sets
                        WHERE tenant_id = :tenant_id
                          AND feature_set_id = :feature_set_id
                          AND version = :feature_set_version
                        """
                    ),
                    {
                        "tenant_id": clean_tenant,
                        "feature_set_id": feature_set_id,
                        "feature_set_version": int(feature_set_version),
                    },
                ).mappings().one_or_none()
                if feature_set is None:
                    raise FileNotFoundError(
                        "Published feature set was not found."
                    )
                if not bool(feature_set["eligible_for_retrieval"]):
                    raise RuntimeError(
                        "Published feature set is not retrieval eligible."
                    )

                rows = connection.execute(
                    text(
                        """
                        SELECT
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
                            eligible_for_activation
                        FROM audience_feature_vectors
                        WHERE tenant_id = :tenant_id
                          AND feature_set_id = :feature_set_id
                          AND feature_set_version = :feature_set_version
                        ORDER BY feature_id
                        """
                    ),
                    {
                        "tenant_id": clean_tenant,
                        "feature_set_id": feature_set_id,
                        "feature_set_version": int(feature_set_version),
                    },
                ).mappings().all()
            finally:
                transaction.rollback()

        feature_record = dict(feature_set)
        feature_rows = [dict(row) for row in rows]
        if int(feature_record["feature_count"]) != len(feature_rows):
            raise RuntimeError(
                "Feature-set count does not match stored safe feature rows."
            )
        return feature_record, feature_rows

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
                "AUDIENCE_FEATURE_DATABASE_URL is required for safe feature "
                "snapshot reads."
            )
        if database_url.startswith("postgres://"):
            database_url = "postgresql://" + database_url[len("postgres://") :]
        return create_engine(database_url, pool_pre_ping=True)
