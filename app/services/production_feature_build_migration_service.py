from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.services.phase2_database_preflight_service import (
    Phase2DatabasePreflightService,
)


ROOT = Path(__file__).resolve().parents[2]
MIGRATION_VERSION = "0008_production_feature_build_registry.sql"
MIGRATION_PATH = ROOT / "migrations" / MIGRATION_VERSION
MIGRATION_LOCK = "punk_production_feature_build_migration"


class ProductionFeatureBuildMigrationService:
    """Apply and verify only the operator-approved Module 2 migration."""

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        engine_factory: Callable[..., Engine] = create_engine,
        preflight_factory: Callable[
            [], Phase2DatabasePreflightService
        ] = Phase2DatabasePreflightService,
        migration_path: Path = MIGRATION_PATH,
    ) -> None:
        self._environment = (
            environment if environment is not None else os.environ
        )
        self._engine_factory = engine_factory
        self._preflight_factory = preflight_factory
        self._migration_path = migration_path

    def apply(
        self,
        *,
        punk_owned_target_confirmed: bool,
    ) -> dict[str, Any]:
        if not punk_owned_target_confirmed:
            raise RuntimeError(
                "Explicit Punk-owned feature target confirmation is required."
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
        if (
            not self._migration_path.is_file()
            or self._migration_path.name != MIGRATION_VERSION
        ):
            raise RuntimeError(
                "Only migration 0008 is permitted by this runner."
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
                "Migration 0008 requires the separate verified Phase 2 schema."
            )

        migration_sql = self._migration_path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(
            migration_sql.encode("utf-8")
        ).hexdigest()
        engine: Engine | None = None
        applied = False
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
                connection.execute(
                    text(
                        """
                        SELECT pg_advisory_xact_lock(
                            hashtext(:lock_name)
                        )
                        """
                    ),
                    {"lock_name": MIGRATION_LOCK},
                )
                connection.exec_driver_sql(
                    "SET LOCAL lock_timeout = '10s'"
                )
                connection.exec_driver_sql(
                    "SET LOCAL statement_timeout = '120s'"
                )
                existing = connection.execute(
                    text(
                        """
                        SELECT checksum
                        FROM audience_schema_migrations
                        WHERE version = :version
                        """
                    ),
                    {"version": MIGRATION_VERSION},
                ).mappings().first()
                if existing:
                    if str(existing["checksum"]) != checksum:
                        raise RuntimeError(
                            "Migration 0008 checksum does not match its ledger."
                        )
                else:
                    connection.exec_driver_sql(migration_sql)
                    connection.execute(
                        text(
                            """
                            INSERT INTO audience_schema_migrations (
                                version,
                                checksum
                            )
                            VALUES (
                                :version,
                                :checksum
                            )
                            """
                        ),
                        {
                            "version": MIGRATION_VERSION,
                            "checksum": checksum,
                        },
                    )
                    applied = True
                verification = connection.execute(
                    text(
                        """
                        SELECT
                            to_regclass(
                                'public.audience_embedding_models'
                            ) IS NOT NULL AS model_registry_present,
                            to_regclass(
                                'public.audience_feature_build_jobs'
                            ) IS NOT NULL AS build_ledger_present,
                            EXISTS (
                                SELECT 1
                                FROM pg_trigger
                                WHERE tgname =
                                    'trg_approved_embedding_model_immutable'
                                  AND NOT tgisinternal
                            ) AS approved_model_immutability_present,
                            EXISTS (
                                SELECT 1
                                FROM pg_trigger
                                WHERE tgname =
                                    'trg_audience_feature_sets_immutable'
                                  AND NOT tgisinternal
                            ) AS feature_set_immutability_present,
                            EXISTS (
                                SELECT 1
                                FROM pg_trigger
                                WHERE tgname =
                                    'trg_audience_feature_vectors_immutable'
                                  AND NOT tgisinternal
                            ) AS feature_vector_immutability_present,
                            (
                                SELECT relforcerowsecurity
                                FROM pg_class
                                WHERE oid = to_regclass(
                                    'public.audience_embedding_models'
                                )
                            ) AS model_registry_rls_forced,
                            (
                                SELECT relforcerowsecurity
                                FROM pg_class
                                WHERE oid = to_regclass(
                                    'public.audience_feature_build_jobs'
                                )
                            ) AS build_ledger_rls_forced
                        """
                    )
                ).mappings().one()
                if not all(bool(value) for value in verification.values()):
                    raise RuntimeError(
                        "Migration 0008 schema verification failed."
                    )
        except SQLAlchemyError:
            raise RuntimeError(
                "Migration 0008 failed and was rolled back."
            ) from None
        finally:
            if engine is not None:
                engine.dispose()

        return {
            "status": (
                "applied_and_verified"
                if applied
                else "already_applied_and_verified"
            ),
            "migration_version": MIGRATION_VERSION,
            "migration_checksum_sha256": checksum,
            "model_registry_present": True,
            "build_ledger_present": True,
            "approved_models_immutable": True,
            "feature_artifacts_immutable": True,
            "tenant_rls_forced": True,
            "source_database_modified": False,
            "features_built": False,
            "activation_or_export_performed": False,
            "credentials_exposed": False,
        }

    def _normalize_url(self, value: str) -> str:
        if value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://") :]
        return value
