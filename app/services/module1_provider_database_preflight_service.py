from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError


TARGET_ENV_KEYS = (
    "PROVIDER_INGESTION_MIGRATION_DATABASE_URL",
    "PROVIDER_INGESTION_DATABASE_URL",
)
SOURCE_ENV_KEYS = ("ECHO_DATABASE_URL", "DATABASE_URL")


def _first(environment: Mapping[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(environment.get(key) or "").strip()
        if value:
            return value
    return ""


def _normalize(value: str) -> str:
    if value.startswith("postgres://"):
        return "postgresql://" + value[len("postgres://") :]
    return value


class Module1ProviderDatabasePreflightService:
    """Read-only verification of the Punk-owned provider control database."""

    REQUIRED_TABLES = (
        "provider_dataset_contracts",
        "provider_ingestion_objects",
        "provider_ingestion_replays",
        "provider_privacy_releases",
        "provider_privacy_windows",
        "provider_privacy_window_partitions",
        "provider_canonical_partitions",
        "provider_data_rights_requests",
        "provider_scale_acceptance_runs",
    )
    REQUIRED_POLICIES = (
        "provider_dataset_contracts_tenant",
        "provider_ingestion_objects_tenant",
        "provider_ingestion_replays_tenant",
        "provider_privacy_releases_tenant",
        "provider_privacy_windows_tenant",
        "provider_privacy_window_partitions_tenant",
        "provider_canonical_partitions_tenant",
        "provider_data_rights_requests_tenant",
        "provider_scale_acceptance_runs_tenant",
    )

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        engine_factory: Callable[..., Engine] = create_engine,
    ) -> None:
        self._environment = environment if environment is not None else os.environ
        self._engine_factory = engine_factory

    def run(
        self,
        *,
        target_database_url: str | None = None,
        historical_database_url: str | None = None,
        punk_owned_target_confirmed: bool = False,
    ) -> dict[str, Any]:
        target_url = str(
            target_database_url
            or _first(self._environment, TARGET_ENV_KEYS)
        ).strip()
        historical_url = str(
            historical_database_url
            or _first(self._environment, SOURCE_ENV_KEYS)
        ).strip()
        same_database = self._same_database(target_url, historical_url)
        target = self._inspect(target_url)

        tables_ready = all(
            target["tables"].get(name, False)
            for name in self.REQUIRED_TABLES
        )
        rls_ready = all(
            target["rls_forced"].get(name, False)
            for name in self.REQUIRED_TABLES
        )
        policies_ready = all(
            name in target["policies"] for name in self.REQUIRED_POLICIES
        )
        schema_ready = bool(
            target["connection_ok"]
            and target["migration_0010_recorded"]
            and target["runtime_tenant_mapping_present"]
            and target["runtime_tenant_function_present"]
            and tables_ready
            and rls_ready
            and policies_ready
        )

        blockers: list[str] = []
        if not target_url:
            blockers.append("explicit_provider_database_not_configured")
        elif not target["connection_ok"]:
            blockers.append("provider_database_preflight_failed")
        elif not schema_ready and not target["public_schema_create_privilege"]:
            blockers.append("provider_schema_create_privilege_missing")
        if same_database:
            blockers.append("provider_target_matches_historical_source")
        if not punk_owned_target_confirmed:
            blockers.append("punk_owned_provider_database_not_confirmed")

        if blockers:
            status = "blocked"
            next_action = (
                "configure_separate_punk_owned_provider_database"
                if "provider_target_matches_historical_source" in blockers
                or "explicit_provider_database_not_configured" in blockers
                else "resolve_provider_database_preflight_blockers"
            )
        elif schema_ready:
            status = "module1_provider_schema_ready"
            next_action = "provision_tenant_pinned_runtime_role"
        else:
            status = "ready_for_approved_migration"
            next_action = "apply_module1_provider_migrations"

        return {
            "status": status,
            "read_only": True,
            "migration_attempted": False,
            "credentials_exposed": False,
            "punk_owned_target_confirmed": bool(
                punk_owned_target_confirmed
            ),
            "same_database_as_historical_source": same_database,
            "provider_target": target,
            "required_table_count": len(self.REQUIRED_TABLES),
            "required_policy_count": len(self.REQUIRED_POLICIES),
            "tables_ready": tables_ready,
            "tenant_rls_ready": rls_ready and policies_ready,
            "module1_provider_schema_ready": schema_ready,
            "blockers": sorted(set(blockers)),
            "next_action": next_action,
        }

    def _inspect(self, database_url: str) -> dict[str, Any]:
        base: dict[str, Any] = {
            "configured": bool(database_url),
            "connection_ok": False,
            "postgresql": False,
            "read_only_transaction_verified": False,
            "server_version_num": None,
            "database_name": None,
            "current_role_superuser": False,
            "current_role_bypass_rls": False,
            "public_schema_create_privilege": False,
            "migration_0010_recorded": False,
            "runtime_tenant_mapping_present": False,
            "runtime_tenant_function_present": False,
            "tables": {name: False for name in self.REQUIRED_TABLES},
            "rls_forced": {name: False for name in self.REQUIRED_TABLES},
            "policies": [],
            "error_code": (
                None if database_url else "database_not_configured"
            ),
        }
        if not database_url:
            return base

        engine: Engine | None = None
        try:
            engine = self._engine_factory(
                _normalize(database_url),
                pool_pre_ping=True,
            )
            if engine.dialect.name != "postgresql":
                base["error_code"] = "postgresql_required"
                return base
            with engine.connect() as connection:
                transaction = connection.begin()
                try:
                    connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                    common = connection.execute(
                        text(
                            """
                            SELECT
                                current_setting('transaction_read_only')
                                    AS transaction_read_only,
                                current_setting('server_version_num')::INTEGER
                                    AS server_version_num,
                                current_database() AS database_name,
                                COALESCE(role.rolsuper, FALSE)
                                    AS current_role_superuser,
                                COALESCE(role.rolbypassrls, FALSE)
                                    AS current_role_bypass_rls,
                                has_schema_privilege(
                                    current_user, 'public', 'CREATE'
                                ) AS public_schema_create_privilege
                            FROM pg_roles role
                            WHERE role.rolname = current_user
                            """
                        )
                    ).mappings().one()
                    table_rows = connection.execute(
                        text(
                            """
                            SELECT
                                class.relname AS table_name,
                                class.relforcerowsecurity AS rls_forced
                            FROM pg_class class
                            JOIN pg_namespace namespace
                              ON namespace.oid = class.relnamespace
                            WHERE namespace.nspname = 'public'
                              AND class.relkind = 'r'
                              AND class.relname = ANY(:table_names)
                            """
                        ),
                        {"table_names": list(self.REQUIRED_TABLES)},
                    ).mappings().all()
                    policies = connection.execute(
                        text(
                            """
                            SELECT policyname
                            FROM pg_policies
                            WHERE schemaname = 'public'
                              AND policyname = ANY(:policy_names)
                            """
                        ),
                        {"policy_names": list(self.REQUIRED_POLICIES)},
                    ).scalars().all()
                    migrations_table_present = bool(
                        connection.execute(
                            text(
                                "SELECT to_regclass("
                                "'public.audience_schema_migrations') "
                                "IS NOT NULL"
                            )
                        ).scalar()
                    )
                    migration_recorded = False
                    if migrations_table_present:
                        migration_recorded = bool(
                            connection.execute(
                                text(
                                    """
                                    SELECT EXISTS (
                                        SELECT 1
                                        FROM audience_schema_migrations
                                        WHERE version =
                                            '0010_provider_control_plane_rls.sql'
                                    )
                                    """
                                )
                            ).scalar()
                        )
                    runtime_mapping_present = bool(
                        connection.execute(
                            text(
                                "SELECT to_regclass("
                                "'public.provider_runtime_tenants') "
                                "IS NOT NULL"
                            )
                        ).scalar()
                    )
                    runtime_function_present = bool(
                        connection.execute(
                            text(
                                "SELECT to_regprocedure("
                                "'public.provider_runtime_tenant()') "
                                "IS NOT NULL"
                            )
                        ).scalar()
                    )
                    base.update(
                        {
                            "connection_ok": True,
                            "postgresql": True,
                            "read_only_transaction_verified": str(
                                common["transaction_read_only"]
                            ).lower() == "on",
                            "server_version_num": int(
                                common["server_version_num"]
                            ),
                            "database_name": common["database_name"],
                            "current_role_superuser": bool(
                                common["current_role_superuser"]
                            ),
                            "current_role_bypass_rls": bool(
                                common["current_role_bypass_rls"]
                            ),
                            "public_schema_create_privilege": bool(
                                common["public_schema_create_privilege"]
                            ),
                            "migration_0010_recorded": migration_recorded,
                            "runtime_tenant_mapping_present": (
                                runtime_mapping_present
                            ),
                            "runtime_tenant_function_present": (
                                runtime_function_present
                            ),
                            "error_code": None,
                        }
                    )
                    for row in table_rows:
                        name = str(row["table_name"])
                        base["tables"][name] = True
                        base["rls_forced"][name] = bool(
                            row["rls_forced"]
                        )
                    base["policies"] = sorted(str(value) for value in policies)
                finally:
                    transaction.rollback()
        except (SQLAlchemyError, ValueError, TypeError):
            base["error_code"] = "provider_database_preflight_failed"
        finally:
            if engine is not None:
                engine.dispose()
        return base

    def _same_database(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        try:
            left_url = make_url(_normalize(left))
            right_url = make_url(_normalize(right))
        except Exception:
            return False
        return (
            (left_url.host or "").lower(),
            left_url.port or 5432,
            (left_url.database or "").lower(),
        ) == (
            (right_url.host or "").lower(),
            right_url.port or 5432,
            (right_url.database or "").lower(),
        )
