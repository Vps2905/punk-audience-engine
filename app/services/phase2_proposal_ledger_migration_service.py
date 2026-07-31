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
PROPOSAL_LEDGER_MIGRATION_VERSION = (
    "0005_punk_ai_audience_proposal_ledger.sql"
)
PROPOSAL_LEDGER_MIGRATION_PATH = (
    ROOT / "migrations" / PROPOSAL_LEDGER_MIGRATION_VERSION
)
PROPOSAL_LEDGER_MIGRATION_LOCK = (
    "punk_audience_phase2_proposal_ledger_migration"
)


class Phase2ProposalLedgerMigrationService:
    """Apply and verify only the immutable proposal-ledger migration."""

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        engine_factory: Callable[..., Engine] = create_engine,
        preflight_factory: Callable[
            [], Phase2DatabasePreflightService
        ] = Phase2DatabasePreflightService,
        migration_path: Path = PROPOSAL_LEDGER_MIGRATION_PATH,
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
        migration_url = str(
            self._environment.get(
                "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL"
            )
            or ""
        ).strip()
        if not source_url:
            raise RuntimeError(
                "A read-only historical source database is required."
            )
        if not migration_url:
            raise RuntimeError(
                "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL must be configured."
            )
        if (
            not self._migration_path.is_file()
            or self._migration_path.name
            != PROPOSAL_LEDGER_MIGRATION_VERSION
        ):
            raise RuntimeError(
                "Only the approved proposal-ledger migration is permitted."
            )

        preflight = self._preflight_factory().run(
            source_database_url=source_url,
            feature_database_url=migration_url,
            punk_owned_target_confirmed=True,
        )
        if (
            preflight.get("status") != "phase2_schema_ready"
            or preflight.get("same_database_as_source")
        ):
            raise RuntimeError(
                "Migration 0004 must be verified on a separate Punk-owned "
                "feature database before proposal-ledger migration."
            )

        migration_sql = self._migration_path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(
            migration_sql.encode("utf-8")
        ).hexdigest()
        engine: Engine | None = None
        migration_applied = False
        try:
            engine = self._engine_factory(
                self._normalize_url(migration_url),
                pool_pre_ping=True,
            )
            if engine.dialect.name != "postgresql":
                raise RuntimeError(
                    "The proposal ledger target must be PostgreSQL."
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
                    {"lock_name": PROPOSAL_LEDGER_MIGRATION_LOCK},
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
                        FROM public.audience_schema_migrations
                        WHERE version = :version
                        """
                    ),
                    {"version": PROPOSAL_LEDGER_MIGRATION_VERSION},
                ).mappings().first()
                if existing:
                    if str(existing["checksum"]) != checksum:
                        raise RuntimeError(
                            "The recorded proposal-ledger migration checksum "
                            "does not match the approved file."
                        )
                else:
                    connection.exec_driver_sql(migration_sql)
                    connection.execute(
                        text(
                            """
                            INSERT INTO public.audience_schema_migrations (
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
                            "version": PROPOSAL_LEDGER_MIGRATION_VERSION,
                            "checksum": checksum,
                        },
                    )
                    migration_applied = True

            verification = self._verify(engine)
        except SQLAlchemyError:
            raise RuntimeError(
                "Proposal-ledger migration failed and was rolled back."
            ) from None
        finally:
            if engine is not None:
                engine.dispose()

        if not all(verification.values()):
            raise RuntimeError(
                "Proposal-ledger migration completed but verification failed."
            )
        return {
            "status": (
                "applied_and_verified"
                if migration_applied
                else "already_applied_and_verified"
            ),
            "migration_version": PROPOSAL_LEDGER_MIGRATION_VERSION,
            "migration_checksum_sha256": checksum,
            "proposal_table_present": True,
            "proposal_rls_forced": True,
            "proposal_tenant_policy_present": True,
            "proposal_immutability_trigger_present": True,
            "proposal_lookup_index_present": True,
            "source_database_modified": False,
            "feature_rows_modified": False,
            "activation_or_export_performed": False,
            "credentials_exposed": False,
        }

    def _verify(self, engine: Engine) -> dict[str, bool]:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.exec_driver_sql(
                    "SET TRANSACTION READ ONLY"
                )
                row = connection.execute(
                    text(
                        """
                        SELECT
                            to_regclass(
                                'public.punk_ai_audience_proposals'
                            ) IS NOT NULL AS proposal_table_present,
                            proposal.relrowsecurity
                                AND proposal.relforcerowsecurity
                                AS proposal_rls_forced,
                            EXISTS (
                                SELECT 1
                                FROM pg_policies
                                WHERE
                                    schemaname = 'public'
                                    AND tablename =
                                        'punk_ai_audience_proposals'
                                    AND policyname =
                                        'punk_ai_audience_proposals_tenant_policy'
                            ) AS proposal_tenant_policy_present,
                            EXISTS (
                                SELECT 1
                                FROM pg_trigger
                                WHERE
                                    tgrelid =
                                        'public.punk_ai_audience_proposals'
                                        ::regclass
                                    AND tgname =
                                        'trg_reject_punk_ai_audience_proposal_mutation'
                                    AND NOT tgisinternal
                            ) AS proposal_immutability_trigger_present,
                            to_regclass(
                                'public.idx_punk_ai_audience_proposals_lookup'
                            ) IS NOT NULL
                                AS proposal_lookup_index_present
                        FROM pg_class proposal
                        WHERE proposal.oid =
                            'public.punk_ai_audience_proposals'::regclass
                        """
                    )
                ).mappings().one()
            finally:
                transaction.rollback()
        return {
            key: bool(value)
            for key, value in row.items()
        }

    def _normalize_url(self, value: str) -> str:
        if value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://") :]
        return value
