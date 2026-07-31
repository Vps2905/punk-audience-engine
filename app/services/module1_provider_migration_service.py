from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.services.module1_provider_database_preflight_service import (
    Module1ProviderDatabasePreflightService,
)


ROOT = Path(__file__).resolve().parents[2]
MIGRATION_VERSIONS = (
    "0003_provider_ingestion_gateway.sql",
    "0006_provider_distributed_scale_dispatch.sql",
    "0007_provider_distributed_privacy_releases.sql",
    "0009_provider_privacy_windows_and_rights.sql",
    "0010_provider_control_plane_rls.sql",
)


class Module1ProviderMigrationService:
    """Apply only approved Module 1 migrations to a separate Punk database."""

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        engine_factory: Callable[..., Engine] = create_engine,
        migrations_dir: Path = ROOT / "migrations",
    ) -> None:
        self._environment = environment if environment is not None else os.environ
        self._engine_factory = engine_factory
        self._migrations_dir = migrations_dir

    def apply(
        self,
        *,
        punk_owned_target_confirmed: bool,
    ) -> dict[str, Any]:
        if not punk_owned_target_confirmed:
            raise RuntimeError(
                "Explicit Punk-owned provider target confirmation is required."
            )
        target_url = str(
            self._environment.get(
                "PROVIDER_INGESTION_MIGRATION_DATABASE_URL"
            )
            or ""
        ).strip()
        source_url = str(
            self._environment.get("ECHO_DATABASE_URL")
            or self._environment.get("DATABASE_URL")
            or ""
        ).strip()
        if not target_url:
            raise RuntimeError(
                "PROVIDER_INGESTION_MIGRATION_DATABASE_URL must be "
                "configured explicitly."
            )

        preflight_service = Module1ProviderDatabasePreflightService(
            environment=self._environment,
            engine_factory=self._engine_factory,
        )
        before = preflight_service.run(
            target_database_url=target_url,
            historical_database_url=source_url,
            punk_owned_target_confirmed=True,
        )
        if before["same_database_as_historical_source"]:
            raise RuntimeError(
                "Provider control database must be separate from the "
                "historical source."
            )
        if before["status"] not in {
            "ready_for_approved_migration",
            "module1_provider_schema_ready",
        }:
            raise RuntimeError(
                "Module 1 provider database preflight is not migration-ready."
            )

        migrations = []
        for version in MIGRATION_VERSIONS:
            path = self._migrations_dir / version
            if not path.is_file():
                raise RuntimeError(f"Approved migration is missing: {version}")
            sql = path.read_text(encoding="utf-8")
            migrations.append(
                (
                    version,
                    sql,
                    hashlib.sha256(sql.encode("utf-8")).hexdigest(),
                )
            )

        engine: Engine | None = None
        applied: list[str] = []
        try:
            engine = self._engine_factory(
                self._normalize(target_url),
                pool_pre_ping=True,
            )
            if engine.dialect.name != "postgresql":
                raise RuntimeError("Module 1 control database must be PostgreSQL")
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "SELECT pg_advisory_xact_lock("
                        "hashtext('punk_module1_provider_migrations'))"
                    )
                )
                connection.exec_driver_sql("SET LOCAL lock_timeout = '10s'")
                connection.exec_driver_sql(
                    "SET LOCAL statement_timeout = '180s'"
                )
                connection.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS audience_schema_migrations (
                            version TEXT PRIMARY KEY,
                            checksum TEXT NOT NULL,
                            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                        """
                    )
                )
                for version, sql, checksum in migrations:
                    existing = connection.execute(
                        text(
                            """
                            SELECT checksum
                            FROM audience_schema_migrations
                            WHERE version = :version
                            """
                        ),
                        {"version": version},
                    ).mappings().first()
                    if existing:
                        if str(existing["checksum"]) != checksum:
                            raise RuntimeError(
                                f"Migration checksum changed: {version}"
                            )
                        continue
                    connection.exec_driver_sql(sql)
                    connection.execute(
                        text(
                            """
                            INSERT INTO audience_schema_migrations (
                                version, checksum
                            ) VALUES (:version, :checksum)
                            """
                        ),
                        {"version": version, "checksum": checksum},
                    )
                    applied.append(version)
        except SQLAlchemyError:
            raise RuntimeError(
                "Module 1 provider migration failed and was rolled back."
            ) from None
        finally:
            if engine is not None:
                engine.dispose()

        after = preflight_service.run(
            target_database_url=target_url,
            historical_database_url=source_url,
            punk_owned_target_confirmed=True,
        )
        if after["status"] != "module1_provider_schema_ready":
            raise RuntimeError(
                "Module 1 migrations completed but schema verification failed."
            )
        return {
            "status": (
                "applied_and_verified"
                if applied
                else "already_applied_and_verified"
            ),
            "migration_versions": list(MIGRATION_VERSIONS),
            "applied_versions": applied,
            "module1_provider_schema_ready": True,
            "tenant_rls_ready": True,
            "same_database_as_historical_source": False,
            "source_database_modified": False,
            "provider_objects_processed": False,
            "activation_or_export_performed": False,
            "credentials_exposed": False,
        }

    def _normalize(self, value: str) -> str:
        if value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://") :]
        return value
