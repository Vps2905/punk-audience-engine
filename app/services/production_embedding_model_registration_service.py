from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.models.audience_feature_contracts import normalize_taxonomy_value
from app.models.embedding_benchmark_contracts import (
    EmbeddingBenchmarkDataset,
)
from app.models.production_feature_build_contracts import EmbeddingModelSpec
from app.services.phase2_database_preflight_service import (
    Phase2DatabasePreflightService,
)
from app.services.production_embedding_benchmark_service import (
    ProductionEmbeddingBenchmarkReportValidator,
)


class ProductionEmbeddingModelRegistrationService:
    """Register one benchmark-approved immutable embedding model revision."""

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        engine_factory: Callable[..., Engine] = create_engine,
        preflight_factory: Callable[
            [], Phase2DatabasePreflightService
        ] = Phase2DatabasePreflightService,
        benchmark_validator: (
            ProductionEmbeddingBenchmarkReportValidator | None
        ) = None,
    ) -> None:
        self._environment = (
            environment if environment is not None else os.environ
        )
        self._engine_factory = engine_factory
        self._preflight_factory = preflight_factory
        self._benchmark_validator = (
            benchmark_validator
            or ProductionEmbeddingBenchmarkReportValidator()
        )

    def register_approved(
        self,
        *,
        tenant_id: str,
        model: EmbeddingModelSpec,
        benchmark_report: Mapping[str, Any],
        benchmark_dataset: EmbeddingBenchmarkDataset,
        approved_by: str,
        punk_owned_target_confirmed: bool,
        benchmark_accepted: bool,
    ) -> dict[str, Any]:
        if not punk_owned_target_confirmed:
            raise RuntimeError(
                "Explicit Punk-owned feature target confirmation is required."
            )
        if not benchmark_accepted:
            raise RuntimeError(
                "Explicit benchmark acceptance is required before approval."
            )
        clean_tenant_id = normalize_taxonomy_value(tenant_id)
        clean_approver = normalize_taxonomy_value(approved_by)
        if not clean_tenant_id or not clean_approver:
            raise ValueError("tenant_id and approved_by are required.")
        safe_benchmark = self._benchmark_validator.validate(
            benchmark_report,
            model=model,
            dataset=benchmark_dataset,
        )

        source_url = str(
            self._environment.get("ECHO_DATABASE_URL")
            or self._environment.get("DATABASE_URL")
            or ""
        ).strip()
        feature_url = str(
            self._environment.get(
                "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL"
            )
            or ""
        ).strip()
        if not source_url or not feature_url:
            raise RuntimeError(
                "Source and explicit feature migration databases are required."
            )
        preflight = self._preflight_factory().run(
            source_database_url=source_url,
            feature_database_url=feature_url,
            punk_owned_target_confirmed=True,
        )
        if (
            preflight.get("status") != "phase2_schema_ready"
            or preflight.get("same_database_as_source")
        ):
            raise RuntimeError(
                "Model registration requires the separate verified "
                "Phase 2 schema."
            )

        engine: Engine | None = None
        created = False
        try:
            engine = self._engine_factory(
                self._normalize_url(feature_url),
                pool_pre_ping=True,
            )
            if engine.dialect.name != "postgresql":
                raise RuntimeError(
                    "The production feature target must be PostgreSQL."
                )
            with engine.begin() as connection:
                self._assert_registry_ready(connection)
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
                inserted = connection.execute(
                    text(
                        """
                        INSERT INTO audience_embedding_models (
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
                        )
                        VALUES (
                            :tenant_id,
                            :model_fingerprint,
                            :backend,
                            :model_name,
                            :model_revision,
                            :embedding_dimension,
                            :normalize_embeddings,
                            :document_prefix,
                            :query_prefix,
                            'approved',
                            'passed',
                            CAST(:benchmark_report AS JSONB),
                            :approved_by,
                            now()
                        )
                        ON CONFLICT DO NOTHING
                        RETURNING model_fingerprint
                        """
                    ),
                    {
                        "tenant_id": clean_tenant_id,
                        "model_fingerprint": model.fingerprint,
                        "backend": model.backend,
                        "model_name": model.model_name,
                        "model_revision": model.model_revision,
                        "embedding_dimension": model.dimension,
                        "normalize_embeddings": (
                            model.normalize_embeddings
                        ),
                        "document_prefix": model.document_prefix,
                        "query_prefix": model.query_prefix,
                        "benchmark_report": json.dumps(
                            safe_benchmark,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "approved_by": clean_approver,
                    },
                ).mappings().first()
                created = inserted is not None
                record = connection.execute(
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
                ).mappings().one()
                self._verify_record(
                    dict(record),
                    tenant_id=clean_tenant_id,
                    model=model,
                    benchmark_report=safe_benchmark,
                    approved_by=clean_approver,
                )
        except SQLAlchemyError:
            raise RuntimeError(
                "Embedding model registration failed and was rolled back."
            ) from None
        finally:
            if engine is not None:
                engine.dispose()

        return {
            "status": (
                "approved_model_registered"
                if created
                else "approved_model_already_registered"
            ),
            "tenant_id": clean_tenant_id,
            "model_fingerprint": model.fingerprint,
            "model_name": model.model_name,
            "model_revision": model.model_revision,
            "embedding_dimension": model.dimension,
            "benchmark_status": "passed",
            "approved": True,
            "features_built": False,
            "activation_or_export_performed": False,
            "credentials_exposed": False,
        }

    def _assert_registry_ready(self, connection: Any) -> None:
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

    def _verify_record(
        self,
        record: dict[str, Any],
        *,
        tenant_id: str,
        model: EmbeddingModelSpec,
        benchmark_report: dict[str, Any],
        approved_by: str,
    ) -> None:
        expected = {
            "tenant_id": tenant_id,
            "model_fingerprint": model.fingerprint,
            "backend": model.backend,
            "model_name": model.model_name,
            "model_revision": model.model_revision,
            "embedding_dimension": model.dimension,
            "normalize_embeddings": model.normalize_embeddings,
            "document_prefix": model.document_prefix,
            "query_prefix": model.query_prefix,
            "status": "approved",
            "benchmark_status": "passed",
            "benchmark_report": benchmark_report,
            "approved_by": approved_by,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                "Existing embedding model registry record conflicts with "
                "the approved model specification."
            )
        if record.get("approved_at") is None:
            raise RuntimeError(
                "Approved embedding model is missing approval timestamp."
            )

    def _normalize_url(self, value: str) -> str:
        if value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://") :]
        return value
