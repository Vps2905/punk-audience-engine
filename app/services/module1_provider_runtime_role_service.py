from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.services.module1_provider_database_preflight_service import (
    Module1ProviderDatabasePreflightService,
)


_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class Module1ProviderRuntimeRoleService:
    """Provision one least-privilege, database-authenticated tenant role."""

    TABLES = (
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

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        engine_factory: Callable[..., Engine] = create_engine,
    ) -> None:
        self._environment = environment if environment is not None else os.environ
        self._engine_factory = engine_factory

    def provision(
        self,
        *,
        tenant_id: str,
        punk_owned_target_confirmed: bool,
    ) -> dict[str, Any]:
        if not punk_owned_target_confirmed:
            raise RuntimeError("Explicit Punk-owned target confirmation required")
        tenant = str(tenant_id or "").strip().lower()
        if not _TENANT_RE.fullmatch(tenant):
            raise ValueError("tenant_id is invalid")
        role = str(
            self._environment.get("MODULE1_PROVIDER_RUNTIME_USER") or ""
        ).strip().lower()
        password = str(
            self._environment.get("MODULE1_PROVIDER_RUNTIME_PASSWORD") or ""
        )
        database_url = str(
            self._environment.get(
                "PROVIDER_INGESTION_MIGRATION_DATABASE_URL"
            )
            or ""
        ).strip()
        if not _ROLE_RE.fullmatch(role):
            raise ValueError("MODULE1_PROVIDER_RUNTIME_USER is invalid")
        if len(password) < 32:
            raise ValueError(
                "MODULE1_PROVIDER_RUNTIME_PASSWORD must contain at least "
                "32 characters"
            )
        if not database_url:
            raise RuntimeError(
                "PROVIDER_INGESTION_MIGRATION_DATABASE_URL is required"
            )

        preflight = Module1ProviderDatabasePreflightService(
            environment=self._environment,
            engine_factory=self._engine_factory,
        ).run(
            target_database_url=database_url,
            punk_owned_target_confirmed=True,
        )
        if preflight["status"] != "module1_provider_schema_ready":
            raise RuntimeError("Module 1 provider schema is not ready")

        engine = self._engine_factory(
            self._normalize(database_url), pool_pre_ping=True
        )
        role_sql = self._quote_identifier(role)
        with engine.begin() as connection:
            database_name = str(
                connection.execute(text("SELECT current_database()"))
                .scalar_one()
            )
            exists = bool(
                connection.execute(
                    text("SELECT 1 FROM pg_roles WHERE rolname = :role"),
                    {"role": role},
                ).first()
            )
            cursor = connection.connection.driver_connection.cursor()
            try:
                if exists:
                    cursor.execute(
                        f"ALTER ROLE {role_sql} LOGIN NOSUPERUSER "
                        "NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS "
                        "PASSWORD %s",
                        (password,),
                    )
                else:
                    cursor.execute(
                        f"CREATE ROLE {role_sql} LOGIN NOSUPERUSER "
                        "NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS "
                        "PASSWORD %s",
                        (password,),
                    )
            finally:
                cursor.close()

            connection.execute(
                text(
                    """
                    INSERT INTO provider_runtime_tenants (
                        role_name, tenant_id, updated_at
                    ) VALUES (:role, :tenant_id, now())
                    ON CONFLICT (role_name) DO UPDATE
                    SET tenant_id = EXCLUDED.tenant_id,
                        updated_at = now()
                    """
                ),
                {"role": role, "tenant_id": tenant},
            )
            database_sql = self._quote_identifier(database_name)
            connection.exec_driver_sql(
                f"GRANT CONNECT ON DATABASE {database_sql} TO {role_sql}"
            )
            connection.exec_driver_sql(
                f"GRANT USAGE ON SCHEMA public TO {role_sql}"
            )
            connection.exec_driver_sql(
                f"REVOKE CREATE ON SCHEMA public FROM {role_sql}"
            )
            tables = ", ".join(
                f"public.{self._quote_identifier(name)}" for name in self.TABLES
            )
            connection.exec_driver_sql(
                f"GRANT SELECT, INSERT, UPDATE ON {tables} TO {role_sql}"
            )
            connection.exec_driver_sql(
                f"REVOKE DELETE, TRUNCATE, REFERENCES, TRIGGER "
                f"ON {tables} FROM {role_sql}"
            )
            connection.exec_driver_sql(
                f"GRANT EXECUTE ON FUNCTION "
                f"public.provider_runtime_tenant() TO {role_sql}"
            )
            connection.exec_driver_sql(
                f"REVOKE ALL ON TABLE public.provider_runtime_tenants "
                f"FROM {role_sql}"
            )

            verification = connection.execute(
                text(
                    """
                    SELECT
                        role.rolsuper,
                        role.rolcreatedb,
                        role.rolcreaterole,
                        role.rolbypassrls,
                        mapping.tenant_id,
                        has_schema_privilege(
                            :role, 'public', 'CREATE'
                        ) AS can_create_schema_objects,
                        has_table_privilege(
                            :role,
                            'public.provider_ingestion_objects',
                            'INSERT'
                        ) AS can_insert,
                        has_table_privilege(
                            :role,
                            'public.provider_ingestion_objects',
                            'DELETE'
                        ) AS can_delete,
                        has_table_privilege(
                            :role,
                            'public.provider_runtime_tenants',
                            'SELECT'
                        ) AS can_read_tenant_mapping
                    FROM pg_roles role
                    JOIN provider_runtime_tenants mapping
                      ON mapping.role_name = role.rolname
                    WHERE role.rolname = :role
                    """
                ),
                {"role": role},
            ).mappings().one()
        engine.dispose()

        if (
            verification["rolsuper"]
            or verification["rolcreatedb"]
            or verification["rolcreaterole"]
            or verification["rolbypassrls"]
            or verification["can_create_schema_objects"]
            or verification["can_delete"]
            or verification["can_read_tenant_mapping"]
            or not verification["can_insert"]
            or str(verification["tenant_id"]) != tenant
        ):
            raise RuntimeError("Provider runtime role verification failed")

        return {
            "status": "module1_runtime_role_provisioned_and_verified",
            "runtime_role": role,
            "tenant_id": tenant,
            "runtime_superuser": False,
            "runtime_bypass_rls": False,
            "runtime_can_create_schema_objects": False,
            "runtime_can_insert_update": True,
            "runtime_can_delete": False,
            "runtime_can_read_tenant_mapping": False,
            "credentials_exposed": False,
            "activation_or_export_performed": False,
        }

    def _quote_identifier(self, value: str) -> str:
        return '"' + str(value).replace('"', '""') + '"'

    def _normalize(self, value: str) -> str:
        if value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://") :]
        return value
