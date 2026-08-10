from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError

from app.models.production_module3_cohort_contracts import required_slug


ROLE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,62}$")
ROLE_PROVISION_LOCK = "punk_preproduction_runtime_role_provision_v1"


class PreproductionDatabaseRoleProvisioningService:
    """Provision separate non-admin API and worker identities in staging."""

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        engine_factory: Callable[..., Engine] = create_engine,
    ) -> None:
        self._environment = environment if environment is not None else os.environ
        self._engine_factory = engine_factory

    def provision(self) -> dict[str, Any]:
        self._validate_gate()
        tenant_id = required_slug(
            self._required("PREPRODUCTION_TENANT_ID"),
            label="PREPRODUCTION_TENANT_ID",
        )
        api_role = self._role_name(self._required("API_DATABASE_USER"))
        api_password = self._required("API_DATABASE_PASSWORD")
        worker_role = self._role_name(self._required("WORKER_DATABASE_USER"))
        worker_password = self._required("WORKER_DATABASE_PASSWORD")
        if api_role == worker_role:
            raise ValueError("API and worker database roles must be different.")

        engine: Engine | None = None
        try:
            engine = self._engine_factory(
                self._normalize_url(self._required("DATABASE_URL")),
                pool_pre_ping=True,
            )
            if engine.dialect.name != "postgresql":
                raise RuntimeError(
                    "Preproduction role provisioning requires PostgreSQL."
                )
            with engine.begin() as connection:
                admin = connection.execute(
                    text(
                        """
                        SELECT current_user AS role_name, rolsuper, rolcreaterole
                        FROM pg_roles
                        WHERE rolname = current_user
                        """
                    )
                ).mappings().one()
                if not (admin["rolsuper"] or admin["rolcreaterole"]):
                    raise RuntimeError(
                        "The migration identity cannot provision runtime roles."
                    )
                if admin["role_name"] in {api_role, worker_role}:
                    raise RuntimeError(
                        "The database administrator cannot be a runtime role."
                    )
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:lock_name))"),
                    {"lock_name": ROLE_PROVISION_LOCK},
                )
                connection.exec_driver_sql("SET LOCAL lock_timeout = '10s'")
                self._upsert_role(
                    connection,
                    engine,
                    role_name=api_role,
                    password=api_password,
                    connection_limit=40,
                )
                self._upsert_role(
                    connection,
                    engine,
                    role_name=worker_role,
                    password=worker_password,
                    connection_limit=20,
                )
                self._remove_memberships(connection, engine, api_role)
                self._remove_memberships(connection, engine, worker_role)
                self._grant_runtime_permissions(
                    connection,
                    engine,
                    role_name=api_role,
                )
                self._grant_runtime_permissions(
                    connection,
                    engine,
                    role_name=worker_role,
                )
                self._bind_worker_tenant(
                    connection,
                    role_name=worker_role,
                    tenant_id=tenant_id,
                )
                self._verify_roles(
                    connection,
                    api_role=api_role,
                    worker_role=worker_role,
                    tenant_id=tenant_id,
                )
        except SQLAlchemyError:
            raise RuntimeError(
                "Preproduction database role provisioning failed and was "
                "rolled back."
            ) from None
        finally:
            if engine is not None:
                engine.dispose()

        return {
            "status": "preproduction_database_roles_provisioned",
            "tenant_id": tenant_id,
            "api_and_worker_roles_distinct": True,
            "runtime_roles_are_superusers": False,
            "runtime_roles_can_create_database": False,
            "runtime_roles_can_create_roles": False,
            "runtime_roles_bypass_rls": False,
            "runtime_roles_can_modify_schema": False,
            "runtime_roles_can_delete_rows": False,
            "worker_tenant_binding_verified": True,
            "administrator_role_used_by_runtime": False,
            "credentials_returned": False,
            "audience_data_read": False,
            "activation_or_export_performed": False,
        }

    def _validate_gate(self) -> None:
        if str(self._environment.get("APP_ENV") or "").strip().lower() not in {
            "preproduction",
            "staging",
        }:
            raise RuntimeError(
                "Role provisioning requires APP_ENV=preproduction or staging."
            )
        if str(
            self._environment.get("PREPRODUCTION_DEPLOYMENT_PHASE") or ""
        ).strip().lower() != "foundation":
            raise RuntimeError(
                "Role provisioning is allowed only during the foundation phase."
            )
        if str(
            self._environment.get("PREPRODUCTION_DATABASE_BOOTSTRAP_CONFIRMED")
            or ""
        ).strip().lower() not in {"1", "true", "yes", "on"}:
            raise RuntimeError(
                "Explicit preproduction database bootstrap confirmation is "
                "required."
            )

    def _required(self, name: str) -> str:
        value = str(self._environment.get(name) or "").strip()
        if not value:
            raise RuntimeError(f"{name} is required for database bootstrap.")
        return value

    @staticmethod
    def _role_name(value: str) -> str:
        normalized = str(value or "").strip().lower()
        if not ROLE_NAME_PATTERN.fullmatch(normalized):
            raise ValueError("Database runtime role name is invalid.")
        return normalized

    @staticmethod
    def _normalize_url(value: str) -> str:
        url = make_url(value)
        if url.drivername == "postgres":
            url = url.set(drivername="postgresql")
        return url.render_as_string(hide_password=False)

    def _upsert_role(
        self,
        connection: Any,
        engine: Engine,
        *,
        role_name: str,
        password: str,
        connection_limit: int,
    ) -> None:
        quoted_role = engine.dialect.identifier_preparer.quote(role_name)
        exists = connection.execute(
            text("SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname=:name)"),
            {"name": role_name},
        ).scalar_one()
        statement = "ALTER ROLE" if exists else "CREATE ROLE"
        connection.execute(
            text(
                f"""
                {statement} {quoted_role} WITH
                    LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT
                    NOREPLICATION NOBYPASSRLS
                    CONNECTION LIMIT {int(connection_limit)}
                    PASSWORD :password
                """
            ),
            {"password": password},
        )
        connection.exec_driver_sql(
            f"ALTER ROLE {quoted_role} SET statement_timeout = '30s'"
        )
        connection.exec_driver_sql(
            f"ALTER ROLE {quoted_role} SET lock_timeout = '5s'"
        )
        connection.exec_driver_sql(
            f"ALTER ROLE {quoted_role} SET idle_in_transaction_session_timeout = '30s'"
        )

    def _remove_memberships(
        self,
        connection: Any,
        engine: Engine,
        role_name: str,
    ) -> None:
        memberships = connection.execute(
            text(
                """
                SELECT parent.rolname AS parent_role
                FROM pg_auth_members membership
                JOIN pg_roles parent ON parent.oid = membership.roleid
                JOIN pg_roles child ON child.oid = membership.member
                WHERE child.rolname = :role_name
                """
            ),
            {"role_name": role_name},
        ).scalars()
        quoted_child = engine.dialect.identifier_preparer.quote(role_name)
        for parent in memberships:
            quoted_parent = engine.dialect.identifier_preparer.quote(parent)
            connection.exec_driver_sql(
                f"REVOKE {quoted_parent} FROM {quoted_child}"
            )

    def _grant_runtime_permissions(
        self,
        connection: Any,
        engine: Engine,
        *,
        role_name: str,
    ) -> None:
        quoted_role = engine.dialect.identifier_preparer.quote(role_name)
        database_name = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()
        quoted_database = engine.dialect.identifier_preparer.quote(database_name)
        statements = (
            f"GRANT CONNECT ON DATABASE {quoted_database} TO {quoted_role}",
            f"GRANT USAGE ON SCHEMA public TO {quoted_role}",
            f"REVOKE CREATE ON SCHEMA public FROM {quoted_role}",
            f"GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO {quoted_role}",
            f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {quoted_role}",
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE ON TABLES TO {quoted_role}",
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO {quoted_role}",
            f"REVOKE ALL ON audience_schema_migrations FROM {quoted_role}",
            f"REVOKE ALL ON provider_runtime_tenants FROM {quoted_role}",
        )
        for statement in statements:
            connection.exec_driver_sql(statement)

    @staticmethod
    def _bind_worker_tenant(
        connection: Any,
        *,
        role_name: str,
        tenant_id: str,
    ) -> None:
        connection.execute(
            text(
                """
                INSERT INTO provider_runtime_tenants (role_name, tenant_id)
                VALUES (:role_name, :tenant_id)
                ON CONFLICT (role_name) DO UPDATE
                SET tenant_id = EXCLUDED.tenant_id, updated_at = now()
                """
            ),
            {"role_name": role_name, "tenant_id": tenant_id},
        )

    @staticmethod
    def _verify_roles(
        connection: Any,
        *,
        api_role: str,
        worker_role: str,
        tenant_id: str,
    ) -> None:
        rows = connection.execute(
            text(
                """
                SELECT rolname, rolsuper, rolcreatedb, rolcreaterole,
                       rolreplication, rolbypassrls
                FROM pg_roles
                WHERE rolname IN (:api_role, :worker_role)
                """
            ),
            {"api_role": api_role, "worker_role": worker_role},
        ).mappings()
        records = {row["rolname"]: row for row in rows}
        if set(records) != {api_role, worker_role}:
            raise RuntimeError("Runtime database roles were not created.")
        for record in records.values():
            if any(
                record[field]
                for field in (
                    "rolsuper",
                    "rolcreatedb",
                    "rolcreaterole",
                    "rolreplication",
                    "rolbypassrls",
                )
            ):
                raise RuntimeError("Runtime database role is overprivileged.")
        binding = connection.execute(
            text(
                """
                SELECT tenant_id
                FROM provider_runtime_tenants
                WHERE role_name = :worker_role
                """
            ),
            {"worker_role": worker_role},
        ).scalar_one_or_none()
        if binding != tenant_id:
            raise RuntimeError("Worker tenant binding verification failed.")
