from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


def _normalize_postgres_url(value: str) -> str:
    if value.startswith("postgres://"):
        return "postgresql://" + value[len("postgres://") :]
    return value


class ProductionFreshDataWorkflowReadinessService:
    """Read-only deployment verification for the Module 1 -> 2 -> 3 bridge.

    The probe verifies the actual database migration and least-privilege role
    boundary. It never returns credentials, database URLs, or exception text.
    """

    MIGRATION_VERSION = "0014_production_fresh_data_workflows.sql"

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        engine_factory: Callable[..., Engine] = create_engine,
        root: Path | None = None,
    ) -> None:
        self._environment = environment if environment is not None else os.environ
        self._engine_factory = engine_factory
        self._root = root or Path.cwd()

    def inspect(self) -> dict[str, Any]:
        provider = self._inspect_provider_runtime()
        reader = self._inspect_feature_role(role="reader")
        writer = self._inspect_feature_role(role="writer")
        migration = self._inspect_migration_state()

        components = {
            "provider_runtime_ready": bool(
                provider["connection_ok"]
                and provider["ingestion_table_present"]
                and provider["ingestion_select"]
            ),
            "feature_reader_boundary_ready": bool(
                reader["connection_ok"]
                and reader["feature_select"]
                and not reader["feature_write"]
                and not reader["workflow_access"]
            ),
            "feature_writer_boundary_ready": bool(
                writer["connection_ok"]
                and writer["feature_write"]
                and writer["workflow_select"]
                and writer["workflow_insert"]
                and writer["workflow_update"]
                and not writer["workflow_delete"]
                and writer["event_select"]
                and writer["event_insert"]
                and not writer["event_update"]
                and not writer["event_delete"]
                and writer["event_sequence_usage"]
            ),
            "workflow_migration_ready": bool(
                migration["connection_ok"]
                and migration["migration_file_present"]
                and migration["schema_verified"]
            ),
        }
        blockers = sorted(
            name for name, ready in components.items() if not ready
        )
        return {
            "ready": not blockers,
            "read_only": True,
            "credentials_exposed": False,
            "components": components,
            "provider_runtime": provider,
            "feature_reader": reader,
            "feature_writer": writer,
            "migration": migration,
            "blockers": blockers,
        }

    def _inspect_provider_runtime(self) -> dict[str, Any]:
        result = {
            "configured": False,
            "connection_ok": False,
            "postgresql": False,
            "ingestion_table_present": False,
            "ingestion_select": False,
            "error_code": "database_not_configured",
        }
        url = self._value("PROVIDER_INGESTION_DATABASE_URL")
        if not url:
            return result
        result["configured"] = True
        result["error_code"] = None
        engine: Engine | None = None
        try:
            engine = self._engine(url)
            if engine.dialect.name != "postgresql":
                result["error_code"] = "postgresql_required"
                return result
            result["postgresql"] = True
            with engine.connect() as connection:
                transaction = connection.begin()
                try:
                    connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                    row = connection.execute(
                        text(
                            """
                            SELECT
                                to_regclass(
                                    'public.provider_ingestion_objects'
                                ) IS NOT NULL AS ingestion_table_present,
                                CASE WHEN to_regclass(
                                    'public.provider_ingestion_objects'
                                ) IS NULL THEN FALSE ELSE
                                    has_table_privilege(
                                        current_user,
                                        'public.provider_ingestion_objects',
                                        'SELECT'
                                    )
                                END AS ingestion_select
                            """
                        )
                    ).mappings().one()
                    result.update(
                        connection_ok=True,
                        ingestion_table_present=bool(
                            row["ingestion_table_present"]
                        ),
                        ingestion_select=bool(row["ingestion_select"]),
                    )
                finally:
                    transaction.rollback()
        except (SQLAlchemyError, ValueError, TypeError):
            result["error_code"] = "provider_runtime_preflight_failed"
        finally:
            if engine is not None:
                engine.dispose()
        return result

    def _inspect_feature_role(self, *, role: str) -> dict[str, Any]:
        if role not in {"reader", "writer"}:
            raise ValueError("Unsupported feature role readiness probe.")
        result = {
            "configured": False,
            "connection_ok": False,
            "postgresql": False,
            "feature_select": False,
            "feature_write": False,
            "workflow_access": False,
            "workflow_select": False,
            "workflow_insert": False,
            "workflow_update": False,
            "workflow_delete": False,
            "event_select": False,
            "event_insert": False,
            "event_update": False,
            "event_delete": False,
            "event_sequence_usage": False,
            "error_code": "database_not_configured",
        }
        key = (
            "AUDIENCE_FEATURE_DATABASE_URL"
            if role == "reader"
            else "AUDIENCE_FEATURE_WRITER_DATABASE_URL"
        )
        url = self._value(key)
        if not url:
            return result
        result["configured"] = True
        result["error_code"] = None
        engine: Engine | None = None
        try:
            engine = self._engine(url)
            if engine.dialect.name != "postgresql":
                result["error_code"] = "postgresql_required"
                return result
            result["postgresql"] = True
            with engine.connect() as connection:
                transaction = connection.begin()
                try:
                    connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                    row = connection.execute(
                        text(
                            """
                            SELECT
                                has_table_privilege(
                                    current_user,
                                    'public.audience_feature_sets',
                                    'SELECT'
                                ) AND has_table_privilege(
                                    current_user,
                                    'public.audience_feature_vectors',
                                    'SELECT'
                                ) AS feature_select,
                                has_table_privilege(
                                    current_user,
                                    'public.audience_feature_sets',
                                    'INSERT,UPDATE'
                                ) AND has_table_privilege(
                                    current_user,
                                    'public.audience_feature_vectors',
                                    'INSERT,UPDATE'
                                ) AS feature_write,
                                has_table_privilege(
                                    current_user,
                                    'public.audience_fresh_data_workflows',
                                    'SELECT'
                                ) AS workflow_select,
                                has_table_privilege(
                                    current_user,
                                    'public.audience_fresh_data_workflows',
                                    'INSERT'
                                ) AS workflow_insert,
                                has_table_privilege(
                                    current_user,
                                    'public.audience_fresh_data_workflows',
                                    'UPDATE'
                                ) AS workflow_update,
                                has_table_privilege(
                                    current_user,
                                    'public.audience_fresh_data_workflows',
                                    'DELETE'
                                ) AS workflow_delete,
                                has_table_privilege(
                                    current_user,
                                    'public.audience_fresh_data_workflow_events',
                                    'SELECT'
                                ) AS event_select,
                                has_table_privilege(
                                    current_user,
                                    'public.audience_fresh_data_workflow_events',
                                    'INSERT'
                                ) AS event_insert,
                                has_table_privilege(
                                    current_user,
                                    'public.audience_fresh_data_workflow_events',
                                    'UPDATE'
                                ) AS event_update,
                                has_table_privilege(
                                    current_user,
                                    'public.audience_fresh_data_workflow_events',
                                    'DELETE'
                                ) AS event_delete,
                                has_sequence_privilege(
                                    current_user,
                                    'public.audience_fresh_data_workflow_events_event_id_seq',
                                    'USAGE'
                                ) AS event_sequence_usage
                            """
                        )
                    ).mappings().one()
                    workflow_access = any(
                        bool(row[name])
                        for name in (
                            "workflow_select",
                            "workflow_insert",
                            "workflow_update",
                            "workflow_delete",
                            "event_select",
                            "event_insert",
                            "event_update",
                            "event_delete",
                            "event_sequence_usage",
                        )
                    )
                    result.update(
                        connection_ok=True,
                        feature_select=bool(row["feature_select"]),
                        feature_write=bool(row["feature_write"]),
                        workflow_access=workflow_access,
                        workflow_select=bool(row["workflow_select"]),
                        workflow_insert=bool(row["workflow_insert"]),
                        workflow_update=bool(row["workflow_update"]),
                        workflow_delete=bool(row["workflow_delete"]),
                        event_select=bool(row["event_select"]),
                        event_insert=bool(row["event_insert"]),
                        event_update=bool(row["event_update"]),
                        event_delete=bool(row["event_delete"]),
                        event_sequence_usage=bool(
                            row["event_sequence_usage"]
                        ),
                    )
                finally:
                    transaction.rollback()
        except (SQLAlchemyError, ValueError, TypeError):
            result["error_code"] = f"feature_{role}_preflight_failed"
        finally:
            if engine is not None:
                engine.dispose()
        return result

    def _inspect_migration_state(self) -> dict[str, Any]:
        """Verify the operational workflow schema through the runtime writer.

        Deployment-time migration-ledger and checksum attestation is performed
        by the migration command and its receipt. The application runtime must
        never require or retain the migration/admin database credential.
        """

        result = {
            "configured": False,
            "connection_ok": False,
            "postgresql": False,
            "migration_file_present": False,
            "deployment_ledger_checked": False,
            "deployment_ledger_required_at_runtime": False,
            "admin_credentials_required": False,
            "schema_verified": False,
            "error_code": "database_not_configured",
        }
        migration_path = (
            self._root / "migrations" / self.MIGRATION_VERSION
        )
        result["migration_file_present"] = migration_path.is_file()
        url = self._value("AUDIENCE_FEATURE_WRITER_DATABASE_URL")
        if not url:
            return result
        result["configured"] = True
        result["error_code"] = None
        if not migration_path.is_file():
            result["error_code"] = "migration_file_missing"
            return result
        engine: Engine | None = None
        try:
            engine = self._engine(url)
            if engine.dialect.name != "postgresql":
                result["error_code"] = "postgresql_required"
                return result
            result["postgresql"] = True
            with engine.connect() as connection:
                transaction = connection.begin()
                try:
                    connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                    row = connection.execute(
                        text(
                            """
                            SELECT
                                to_regclass(
                                    'public.audience_fresh_data_workflows'
                                ) IS NOT NULL AS workflows_table_present,
                                to_regclass(
                                    'public.audience_fresh_data_workflow_events'
                                ) IS NOT NULL AS events_table_present,
                                COALESCE((
                                    SELECT relforcerowsecurity
                                    FROM pg_class
                                    WHERE oid = to_regclass(
                                        'public.audience_fresh_data_workflows'
                                    )
                                ), FALSE) AS workflows_rls_forced,
                                COALESCE((
                                    SELECT relforcerowsecurity
                                    FROM pg_class
                                    WHERE oid = to_regclass(
                                        'public.audience_fresh_data_workflow_events'
                                    )
                                ), FALSE) AS events_rls_forced,
                                EXISTS (
                                    SELECT 1 FROM pg_policies
                                    WHERE schemaname = 'public'
                                      AND tablename =
                                          'audience_fresh_data_workflows'
                                      AND policyname =
                                          'audience_fresh_data_workflows_tenant_policy'
                                ) AS workflows_policy_present,
                                EXISTS (
                                    SELECT 1 FROM pg_policies
                                    WHERE schemaname = 'public'
                                      AND tablename =
                                          'audience_fresh_data_workflow_events'
                                      AND policyname =
                                          'audience_fresh_data_workflow_events_tenant_policy'
                                ) AS events_policy_present,
                                EXISTS (
                                    SELECT 1 FROM pg_trigger
                                    WHERE tgname =
                                        'trg_fresh_data_workflow_identity_immutable'
                                      AND NOT tgisinternal
                                ) AS identity_trigger_present,
                                EXISTS (
                                    SELECT 1 FROM pg_trigger
                                    WHERE tgname =
                                        'trg_fresh_data_workflow_events_immutable'
                                      AND NOT tgisinternal
                                ) AS events_trigger_present,
                                to_regclass(
                                    'public.idx_fresh_data_workflows_status'
                                ) IS NOT NULL AS status_index_present,
                                to_regclass(
                                    'public.idx_fresh_data_workflows_ingestion'
                                ) IS NOT NULL AS ingestion_index_present,
                                to_regclass(
                                    'public.idx_fresh_data_workflows_lease'
                                ) IS NOT NULL AS lease_index_present,
                                to_regclass(
                                    'public.idx_fresh_data_workflow_events_lookup'
                                ) IS NOT NULL AS events_index_present
                            """
                        )
                    ).mappings().one()
                    schema_keys = (
                        "workflows_table_present",
                        "events_table_present",
                        "workflows_rls_forced",
                        "events_rls_forced",
                        "workflows_policy_present",
                        "events_policy_present",
                        "identity_trigger_present",
                        "events_trigger_present",
                        "status_index_present",
                        "ingestion_index_present",
                        "lease_index_present",
                        "events_index_present",
                    )
                    result.update(
                        connection_ok=True,
                        schema_verified=all(
                            bool(row[key]) for key in schema_keys
                        ),
                    )
                finally:
                    transaction.rollback()
        except (SQLAlchemyError, ValueError, TypeError, OSError):
            result["error_code"] = "workflow_schema_preflight_failed"
        finally:
            if engine is not None:
                engine.dispose()
        return result

    def _engine(self, database_url: str) -> Engine:
        return self._engine_factory(
            _normalize_postgres_url(database_url),
            pool_pre_ping=True,
            connect_args={"connect_timeout": 3},
        )

    def _value(self, key: str) -> str:
        return str(self._environment.get(key) or "").strip()
