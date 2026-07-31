from __future__ import annotations

import os
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.models.audience_feature_contracts import normalize_taxonomy_value
from app.models.production_feature_build_contracts import EmbeddingModelSpec


class ProductionEmbeddingModelRegistryService:
    """
    Read-only runtime gate for tenant-approved embedding model revisions.

    Registration and approval are operator actions performed through migration
    credentials. Feature workers receive only a least-privilege runtime role.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: Engine | None = None,
    ) -> None:
        self._database_url = database_url
        self._engine_override = engine

    def require_approved(
        self,
        *,
        tenant_id: str,
        model: EmbeddingModelSpec,
    ) -> dict[str, Any]:
        clean_tenant_id = normalize_taxonomy_value(tenant_id)
        if not clean_tenant_id:
            raise ValueError("tenant_id is required.")
        engine = self._engine()
        with engine.connect() as connection:
            self._assert_schema_ready(connection)
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
            row = connection.execute(
                text(
                    """
                    SELECT
                        tenant_id,
                        model_fingerprint,
                        backend,
                        model_name,
                        model_revision,
                        embedding_dimension,
                        normalize_embeddings,
                        document_prefix,
                        query_prefix,
                        status,
                        benchmark_status,
                        benchmark_report,
                        approved_by,
                        approved_at
                    FROM audience_embedding_models
                    WHERE tenant_id = :tenant_id
                      AND model_fingerprint = :model_fingerprint
                    """
                ),
                {
                    "tenant_id": clean_tenant_id,
                    "model_fingerprint": model.fingerprint,
                },
            ).mappings().first()
        if not row:
            raise RuntimeError(
                "Embedding model revision is not registered for this tenant."
            )
        record = dict(row)
        expected = {
            "backend": model.backend,
            "model_name": model.model_name,
            "model_revision": model.model_revision,
            "embedding_dimension": model.dimension,
            "normalize_embeddings": model.normalize_embeddings,
            "document_prefix": model.document_prefix,
            "query_prefix": model.query_prefix,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise RuntimeError(
                    "Embedding model registry record conflicts with its fingerprint."
                )
        if (
            record.get("status") != "approved"
            or record.get("benchmark_status") != "passed"
            or not record.get("approved_by")
            or record.get("approved_at") is None
        ):
            raise RuntimeError(
                "Embedding model revision has not passed production approval."
            )
        return {
            "tenant_id": clean_tenant_id,
            "model_fingerprint": model.fingerprint,
            "status": "approved",
            "benchmark_status": "passed",
            "approved": True,
        }

    def _assert_schema_ready(self, connection: Any) -> None:
        ready = connection.execute(
            text(
                """
                SELECT
                    to_regclass(
                        'public.audience_embedding_models'
                    ) IS NOT NULL
                    AND
                    to_regclass(
                        'public.audience_feature_build_jobs'
                    ) IS NOT NULL
                    AS ready
                """
            )
        ).scalar()
        if not ready:
            raise RuntimeError(
                "Production feature-build registry migration 0008 is not ready."
            )

    def _engine(self) -> Engine:
        if self._engine_override is not None:
            return self._engine_override
        database_url = (
            self._database_url
            or os.getenv("AUDIENCE_FEATURE_WRITER_DATABASE_URL")
            or ""
        ).strip()
        if not database_url:
            raise RuntimeError(
                "AUDIENCE_FEATURE_WRITER_DATABASE_URL is required for "
                "production feature builds."
            )
        if database_url.startswith("postgres://"):
            database_url = (
                "postgresql://"
                + database_url[len("postgres://") :]
            )
        return create_engine(database_url, pool_pre_ping=True)
