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
PHASE2_MIGRATION_VERSION = (
    "0004_versioned_audience_features_pgvector.sql"
)
PHASE2_MIGRATION_PATH = ROOT / "migrations" / PHASE2_MIGRATION_VERSION
MIGRATION_LOCK_NAME = "punk_audience_phase2_feature_migration"


class Phase2FeatureMigrationService:
    """
    Apply only the versioned Phase 2 feature-store migration.

    This operator-only service requires explicit target ownership approval,
    refuses a target that resolves to the historical source, records a
    checksum, serializes concurrent attempts and verifies the final schema.
    """

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        engine_factory: Callable[..., Engine] = create_engine,
        preflight_factory: Callable[
            [], Phase2DatabasePreflightService
        ] = Phase2DatabasePreflightService,
        migration_path: Path = PHASE2_MIGRATION_PATH,
    ) -> None:
        self._environment = environment if environment is not None else os.environ
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
        if not source_url:
            raise RuntimeError(
                "A read-only historical source database is required."
            )
        if not feature_url:
            raise RuntimeError(
                "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL must be configured "
                "explicitly."
            )
        if not self._migration_path.is_file():
            raise RuntimeError(
                "The approved Phase 2 migration file is missing."
            )
        if self._migration_path.name != PHASE2_MIGRATION_VERSION:
            raise RuntimeError(
                "Only migration 0004 is permitted by this runner."
            )

        preflight_before = self._preflight_factory().run(
            source_database_url=source_url,
            feature_database_url=feature_url,
            punk_owned_target_confirmed=True,
        )
        if preflight_before.get("same_database_as_source"):
            raise RuntimeError(
                "The Phase 2 feature target must be separate from the "
                "historical source."
            )
        allowed_preflight_statuses = {
            "ready_for_approved_migration",
            "phase2_schema_ready",
        }
        if preflight_before.get("status") not in allowed_preflight_statuses:
            raise RuntimeError(
                "The Phase 2 database preflight is not migration-ready."
            )

        migration_sql = self._migration_path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(
            migration_sql.encode("utf-8")
        ).hexdigest()
        engine: Engine | None = None
        migration_applied = False
        try:
            engine = self._engine_factory(
                self._normalize_url(feature_url),
                pool_pre_ping=True,
            )
            if engine.dialect.name != "postgresql":
                raise RuntimeError(
                    "The Phase 2 feature target must be PostgreSQL."
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
                    {"lock_name": MIGRATION_LOCK_NAME},
                )
                connection.exec_driver_sql(
                    "SET LOCAL lock_timeout = '10s'"
                )
                connection.exec_driver_sql(
                    "SET LOCAL statement_timeout = '120s'"
                )
                connection.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS
                        public.audience_schema_migrations (
                            version TEXT PRIMARY KEY,
                            checksum TEXT NOT NULL,
                            applied_at TIMESTAMPTZ
                                NOT NULL DEFAULT now()
                        )
                        """
                    )
                )
                existing = connection.execute(
                    text(
                        """
                        SELECT checksum
                        FROM public.audience_schema_migrations
                        WHERE version = :version
                        """
                    ),
                    {"version": PHASE2_MIGRATION_VERSION},
                ).mappings().first()

                if existing:
                    if str(existing["checksum"]) != checksum:
                        raise RuntimeError(
                            "The recorded Phase 2 migration checksum does "
                            "not match the approved migration file."
                        )
                else:
                    connection.exec_driver_sql(migration_sql)
                    connection.execute(
                        text(
                            """
                            INSERT INTO
                            public.audience_schema_migrations (
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
                            "version": PHASE2_MIGRATION_VERSION,
                            "checksum": checksum,
                        },
                    )
                    migration_applied = True
        except SQLAlchemyError:
            raise RuntimeError(
                "Phase 2 migration failed and its transaction was rolled back."
            ) from None
        finally:
            if engine is not None:
                engine.dispose()

        preflight_after = self._preflight_factory().run(
            source_database_url=source_url,
            feature_database_url=feature_url,
            punk_owned_target_confirmed=True,
        )
        if (
            preflight_after.get("status") != "phase2_schema_ready"
            or preflight_after.get("same_database_as_source")
        ):
            raise RuntimeError(
                "Phase 2 migration completed but schema verification failed. "
                "No historical snapshot was imported."
            )

        return {
            "status": (
                "applied_and_verified"
                if migration_applied
                else "already_applied_and_verified"
            ),
            "migration_version": PHASE2_MIGRATION_VERSION,
            "migration_checksum_sha256": checksum,
            "preflight_before": preflight_before.get("status"),
            "preflight_after": preflight_after.get("status"),
            "phase2_schema_ready": True,
            "same_database_as_source": False,
            "source_database_modified": False,
            "historical_snapshot_imported": False,
            "activation_or_export_performed": False,
            "credentials_exposed": False,
        }

    def _normalize_url(self, value: str) -> str:
        if value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://") :]
        return value
