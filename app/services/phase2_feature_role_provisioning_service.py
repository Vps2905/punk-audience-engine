from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError

from app.services.phase2_database_preflight_service import (
    Phase2DatabasePreflightService,
)

ROLE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,62}$")
ROLE_PROVISION_LOCK = "punk_audience_phase2_role_provision"


class Phase2FeatureRoleProvisioningService:
    """
    Provision separate non-superuser reader and writer database identities.

    The migration identity is used only for this operator action. Passwords are
    accepted through environment-backed configuration and never returned.
    """

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        engine_factory: Callable[..., Engine] = create_engine,
        preflight_factory: Callable[
            [], Phase2DatabasePreflightService
        ] = Phase2DatabasePreflightService,
    ) -> None:
        self._environment = environment if environment is not None else os.environ
        self._engine_factory = engine_factory
        self._preflight_factory = preflight_factory

    def provision(
        self,
        *,
        tenant_id: str,
        expected_feature_count: int,
        punk_owned_target_confirmed: bool,
    ) -> dict[str, Any]:
        if not punk_owned_target_confirmed:
            raise RuntimeError(
                "Explicit Punk-owned feature target confirmation is required."
            )
        clean_tenant_id = self._required_slug(tenant_id, "tenant_id")
        expected_count = int(expected_feature_count)
        if expected_count < 0:
            raise ValueError("expected_feature_count cannot be negative.")

        source_url = self._required_environment("ECHO_DATABASE_URL")
        migration_url = self._required_environment(
            "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL"
        )
        reader_user = self._role_name(
            self._required_environment("PHASE2_FEATURE_READER_USER")
        )
        reader_password = self._required_environment(
            "PHASE2_FEATURE_READER_PASSWORD"
        )
        writer_user = self._role_name(
            self._required_environment("PHASE2_FEATURE_WRITER_USER")
        )
        writer_password = self._required_environment(
            "PHASE2_FEATURE_WRITER_PASSWORD"
        )
        if reader_user == writer_user:
            raise ValueError("Reader and writer roles must be different.")

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
                "Role provisioning requires a verified, separate Phase 2 "
                "feature schema."
            )

        admin_engine: Engine | None = None
        try:
            admin_engine = self._engine_factory(
                self._normalize_url(migration_url),
                pool_pre_ping=True,
            )
            if admin_engine.dialect.name != "postgresql":
                raise RuntimeError(
                    "The Phase 2 migration target must be PostgreSQL."
                )
            with admin_engine.begin() as connection:
                admin = connection.execute(
                    text(
                        """
                        SELECT
                            current_user AS role_name,
                            rolsuper,
                            rolcreaterole
                        FROM pg_roles
                        WHERE rolname = current_user
                        """
                    )
                ).mappings().one()
                if not (
                    admin["rolsuper"] or admin["rolcreaterole"]
                ):
                    raise RuntimeError(
                        "The migration identity cannot manage database roles."
                    )
                if admin["role_name"] in {reader_user, writer_user}:
                    raise RuntimeError(
                        "The migration identity cannot also be the reader "
                        "or writer identity."
                    )

                connection.execute(
                    text(
                        """
                        SELECT pg_advisory_xact_lock(
                            hashtext(:lock_name)
                        )
                        """
                    ),
                    {"lock_name": ROLE_PROVISION_LOCK},
                )
                connection.exec_driver_sql(
                    "SET LOCAL lock_timeout = '10s'"
                )
                self._upsert_login_role(
                    connection,
                    admin_engine,
                    role_name=reader_user,
                    password=reader_password,
                    connection_limit=20,
                )
                self._upsert_login_role(
                    connection,
                    admin_engine,
                    role_name=writer_user,
                    password=writer_password,
                    connection_limit=5,
                )
                self._revoke_role_memberships(
                    connection,
                    admin_engine,
                    role_name=reader_user,
                )
                self._revoke_role_memberships(
                    connection,
                    admin_engine,
                    role_name=writer_user,
                )
                self._apply_grants(
                    connection,
                    admin_engine,
                    reader_user=reader_user,
                    writer_user=writer_user,
                )
        except SQLAlchemyError:
            raise RuntimeError(
                "Phase 2 role provisioning failed and was rolled back."
            ) from None
        finally:
            if admin_engine is not None:
                admin_engine.dispose()

        reader_url = self._role_url(
            migration_url,
            username=reader_user,
            password=reader_password,
        )
        writer_url = self._role_url(
            migration_url,
            username=writer_user,
            password=writer_password,
        )
        reader_verification = self._verify_reader(
            reader_url,
            expected_role=reader_user,
            tenant_id=clean_tenant_id,
            expected_feature_count=expected_count,
        )
        writer_verification = self._verify_writer(
            writer_url,
            expected_role=writer_user,
            tenant_id=clean_tenant_id,
            expected_feature_count=expected_count,
        )

        return {
            "status": "roles_provisioned_and_verified",
            "tenant_id": clean_tenant_id,
            "reader_role": reader_user,
            "writer_role": writer_user,
            "reader_superuser": False,
            "reader_bypass_rls": False,
            "writer_superuser": False,
            "writer_bypass_rls": False,
            "reader_visible_feature_count": reader_verification[
                "own_tenant_count"
            ],
            "reader_other_tenant_visible_feature_count": (
                reader_verification["other_tenant_count"]
            ),
            "writer_visible_feature_count": writer_verification[
                "own_tenant_count"
            ],
            "reader_can_write": False,
            "writer_can_insert_update": True,
            "writer_can_delete": False,
            "reader_can_create_schema_objects": False,
            "writer_can_create_schema_objects": False,
            "credentials_exposed": False,
            "activation_or_export_performed": False,
        }

    def _upsert_login_role(
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
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_roles
                    WHERE rolname = :role_name
                ) AS role_exists
                """
            ),
            {"role_name": role_name},
        ).mappings().one()
        if not exists["role_exists"]:
            connection.execute(
                text(
                    f"""
                    CREATE ROLE {quoted_role}
                    WITH
                        LOGIN
                        NOSUPERUSER
                        NOCREATEDB
                        NOCREATEROLE
                        NOINHERIT
                        NOREPLICATION
                        NOBYPASSRLS
                        CONNECTION LIMIT {int(connection_limit)}
                        PASSWORD :password
                    """
                ),
                {"password": password},
            )
        else:
            connection.execute(
                text(
                    f"""
                    ALTER ROLE {quoted_role}
                    WITH
                        LOGIN
                        NOSUPERUSER
                        NOCREATEDB
                        NOCREATEROLE
                        NOINHERIT
                        NOREPLICATION
                        NOBYPASSRLS
                        CONNECTION LIMIT {int(connection_limit)}
                        PASSWORD :password
                    """
                ),
                {"password": password},
            )
        connection.exec_driver_sql(
            f"ALTER ROLE {quoted_role} SET row_security = on"
        )

    def _revoke_role_memberships(
        self,
        connection: Any,
        engine: Engine,
        *,
        role_name: str,
    ) -> None:
        quoted_member = engine.dialect.identifier_preparer.quote(role_name)
        memberships = connection.execute(
            text(
                """
                SELECT granted_role.rolname AS granted_role
                FROM pg_auth_members membership
                JOIN pg_roles granted_role
                  ON granted_role.oid = membership.roleid
                JOIN pg_roles member_role
                  ON member_role.oid = membership.member
                WHERE member_role.rolname = :role_name
                """
            ),
            {"role_name": role_name},
        ).mappings().all()
        for membership in memberships:
            granted_role = engine.dialect.identifier_preparer.quote(
                str(membership["granted_role"])
            )
            connection.exec_driver_sql(
                f"REVOKE {granted_role} FROM {quoted_member}"
            )

    def _apply_grants(
        self,
        connection: Any,
        engine: Engine,
        *,
        reader_user: str,
        writer_user: str,
    ) -> None:
        reader = engine.dialect.identifier_preparer.quote(reader_user)
        writer = engine.dialect.identifier_preparer.quote(writer_user)
        database_name = connection.execute(
            text("SELECT current_database() AS database_name")
        ).mappings().one()["database_name"]
        database = engine.dialect.identifier_preparer.quote(
            str(database_name)
        )
        for role in (reader, writer):
            connection.exec_driver_sql(
                f"GRANT CONNECT ON DATABASE {database} TO {role}"
            )
            connection.exec_driver_sql(
                f"REVOKE CREATE ON DATABASE {database} FROM {role}"
            )
        connection.exec_driver_sql(
            "REVOKE ALL ON public.audience_feature_sets FROM PUBLIC"
        )
        connection.exec_driver_sql(
            "REVOKE ALL ON public.audience_feature_vectors FROM PUBLIC"
        )
        connection.exec_driver_sql(
            "REVOKE ALL ON public.audience_schema_migrations FROM PUBLIC"
        )
        connection.exec_driver_sql(
            "REVOKE CREATE ON SCHEMA public FROM PUBLIC"
        )
        for role in (reader, writer):
            connection.exec_driver_sql(
                f"GRANT USAGE ON SCHEMA public TO {role}"
            )
            connection.exec_driver_sql(
                "REVOKE ALL ON public.audience_schema_migrations "
                f"FROM {role}"
            )
        connection.exec_driver_sql(
            "GRANT SELECT ON public.audience_feature_sets, "
            f"public.audience_feature_vectors TO {reader}"
        )
        connection.exec_driver_sql(
            "GRANT SELECT, INSERT, UPDATE ON "
            "public.audience_feature_sets, "
            f"public.audience_feature_vectors TO {writer}"
        )
        connection.exec_driver_sql(
            "REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER "
            "ON public.audience_feature_sets, "
            f"public.audience_feature_vectors FROM {reader}"
        )
        connection.exec_driver_sql(
            "REVOKE DELETE, TRUNCATE, REFERENCES, TRIGGER "
            "ON public.audience_feature_sets, "
            f"public.audience_feature_vectors FROM {writer}"
        )
        registry_ready = bool(
            connection.execute(
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
                        AS registry_ready
                    """
                )
            ).mappings().one()["registry_ready"]
        )
        if registry_ready:
            for table in (
                "audience_embedding_models",
                "audience_feature_build_jobs",
            ):
                connection.exec_driver_sql(
                    f"REVOKE ALL ON public.{table} FROM PUBLIC"
                )
                connection.exec_driver_sql(
                    f"REVOKE ALL ON public.{table} FROM {reader}"
                )
            connection.exec_driver_sql(
                "GRANT SELECT ON public.audience_embedding_models "
                f"TO {writer}"
            )
            connection.exec_driver_sql(
                "GRANT SELECT, INSERT, UPDATE ON "
                "public.audience_feature_build_jobs "
                f"TO {writer}"
            )
            connection.exec_driver_sql(
                "REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, "
                "TRIGGER ON public.audience_embedding_models "
                f"FROM {writer}"
            )
            connection.exec_driver_sql(
                "REVOKE DELETE, TRUNCATE, REFERENCES, TRIGGER "
                "ON public.audience_feature_build_jobs "
                f"FROM {writer}"
            )

    def _verify_reader(
        self,
        database_url: str,
        *,
        expected_role: str,
        tenant_id: str,
        expected_feature_count: int,
    ) -> dict[str, int]:
        engine = self._engine_factory(database_url, pool_pre_ping=True)
        try:
            with engine.connect() as connection:
                transaction = connection.begin()
                try:
                    connection.exec_driver_sql(
                        "SET TRANSACTION READ ONLY"
                    )
                    role = self._role_and_privilege_report(connection)
                    self._assert_expected_role(role, expected_role)
                    self._assert_reader_privileges(role)
                    own_count = self._tenant_count(
                        connection,
                        tenant_id,
                    )
                    other_count = self._tenant_count(
                        connection,
                        f"{tenant_id}_isolation_probe",
                    )
                finally:
                    transaction.rollback()
        except SQLAlchemyError:
            raise RuntimeError(
                "The Phase 2 reader role verification failed."
            ) from None
        finally:
            engine.dispose()
        if own_count != expected_feature_count or other_count != 0:
            raise RuntimeError(
                "The Phase 2 reader role failed tenant-isolation verification."
            )
        return {
            "own_tenant_count": own_count,
            "other_tenant_count": other_count,
        }

    def _verify_writer(
        self,
        database_url: str,
        *,
        expected_role: str,
        tenant_id: str,
        expected_feature_count: int,
    ) -> dict[str, int]:
        engine = self._engine_factory(database_url, pool_pre_ping=True)
        try:
            with engine.connect() as connection:
                transaction = connection.begin()
                try:
                    connection.exec_driver_sql(
                        "SET TRANSACTION READ ONLY"
                    )
                    role = self._role_and_privilege_report(connection)
                    self._assert_expected_role(role, expected_role)
                    self._assert_writer_privileges(role)
                    own_count = self._tenant_count(
                        connection,
                        tenant_id,
                    )
                finally:
                    transaction.rollback()
        except SQLAlchemyError:
            raise RuntimeError(
                "The Phase 2 writer role verification failed."
            ) from None
        finally:
            engine.dispose()
        if own_count != expected_feature_count:
            raise RuntimeError(
                "The Phase 2 writer role cannot read the expected tenant."
            )
        return {"own_tenant_count": own_count}

    def _role_and_privilege_report(
        self,
        connection: Any,
    ) -> Mapping[str, Any]:
        return connection.execute(
            text(
                """
                SELECT
                    current_user AS role_name,
                    role.rolsuper,
                    role.rolcreatedb,
                    role.rolcreaterole,
                    role.rolreplication,
                    role.rolbypassrls,
                    role.rolinherit,
                    role.rolcanlogin,
                    current_setting('row_security') AS row_security,
                    has_database_privilege(
                        current_user,
                        current_database(),
                        'CREATE'
                    ) AS can_create_database_objects,
                    has_schema_privilege(
                        current_user,
                        'public',
                        'CREATE'
                    ) AS can_create_schema_objects,
                    has_table_privilege(
                        current_user,
                        'public.audience_feature_sets',
                        'SELECT'
                    ) AS can_select_feature_sets,
                    has_table_privilege(
                        current_user,
                        'public.audience_feature_vectors',
                        'SELECT'
                    ) AS can_select_feature_vectors,
                    has_table_privilege(
                        current_user,
                        'public.audience_feature_sets',
                        'INSERT'
                    ) AS can_insert_feature_sets,
                    has_table_privilege(
                        current_user,
                        'public.audience_feature_vectors',
                        'INSERT'
                    ) AS can_insert_feature_vectors,
                    has_table_privilege(
                        current_user,
                        'public.audience_feature_sets',
                        'UPDATE'
                    ) AS can_update_feature_sets,
                    has_table_privilege(
                        current_user,
                        'public.audience_feature_vectors',
                        'UPDATE'
                    ) AS can_update_feature_vectors,
                    has_table_privilege(
                        current_user,
                        'public.audience_feature_sets',
                        'DELETE'
                    ) AS can_delete_feature_sets,
                    has_table_privilege(
                        current_user,
                        'public.audience_feature_vectors',
                        'DELETE'
                    ) AS can_delete_feature_vectors,
                    has_table_privilege(
                        current_user,
                        'public.audience_schema_migrations',
                        'SELECT'
                    ) AS can_read_migration_ledger
                FROM pg_roles role
                WHERE role.rolname = current_user
                """
            )
        ).mappings().one()

    def _assert_expected_role(
        self,
        role: Mapping[str, Any],
        expected_role: str,
    ) -> None:
        if role["role_name"] != expected_role:
            raise RuntimeError(
                "Phase 2 role verification used an unexpected identity."
            )

    def _assert_reader_privileges(
        self,
        role: Mapping[str, Any],
    ) -> None:
        if (
            role["rolsuper"]
            or role["rolcreatedb"]
            or role["rolcreaterole"]
            or role["rolreplication"]
            or role["rolbypassrls"]
            or role["rolinherit"]
            or not role["rolcanlogin"]
            or str(role["row_security"]).lower() != "on"
            or role["can_create_database_objects"]
            or role["can_create_schema_objects"]
            or not role["can_select_feature_sets"]
            or not role["can_select_feature_vectors"]
            or role["can_insert_feature_sets"]
            or role["can_insert_feature_vectors"]
            or role["can_update_feature_sets"]
            or role["can_update_feature_vectors"]
            or role["can_delete_feature_sets"]
            or role["can_delete_feature_vectors"]
            or role["can_read_migration_ledger"]
        ):
            raise RuntimeError(
                "The Phase 2 reader role has unsafe privileges."
            )

    def _assert_writer_privileges(
        self,
        role: Mapping[str, Any],
    ) -> None:
        if (
            role["rolsuper"]
            or role["rolcreatedb"]
            or role["rolcreaterole"]
            or role["rolreplication"]
            or role["rolbypassrls"]
            or role["rolinherit"]
            or not role["rolcanlogin"]
            or str(role["row_security"]).lower() != "on"
            or role["can_create_database_objects"]
            or role["can_create_schema_objects"]
            or not role["can_select_feature_sets"]
            or not role["can_select_feature_vectors"]
            or not role["can_insert_feature_sets"]
            or not role["can_insert_feature_vectors"]
            or not role["can_update_feature_sets"]
            or not role["can_update_feature_vectors"]
            or role["can_delete_feature_sets"]
            or role["can_delete_feature_vectors"]
            or role["can_read_migration_ledger"]
        ):
            raise RuntimeError(
                "The Phase 2 writer role has unsafe privileges."
            )

    def _tenant_count(
        self,
        connection: Any,
        tenant_id: str,
    ) -> int:
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
            {"tenant_id": tenant_id},
        )
        return int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM public.audience_feature_vectors
                    """
                )
            ).scalar_one()
        )

    def _role_url(
        self,
        migration_url: str,
        *,
        username: str,
        password: str,
    ) -> str:
        return make_url(
            self._normalize_url(migration_url)
        ).set(
            username=username,
            password=password,
        ).render_as_string(hide_password=False)

    def _required_environment(self, key: str) -> str:
        value = str(self._environment.get(key) or "").strip()
        if not value:
            raise RuntimeError(f"{key} must be configured.")
        return value

    def _role_name(self, value: str) -> str:
        clean = value.strip().lower()
        if not ROLE_NAME_PATTERN.fullmatch(clean):
            raise ValueError(
                "Database role names must use lowercase letters, numbers "
                "and underscores."
            )
        return clean

    def _required_slug(self, value: str, field: str) -> str:
        clean = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip(
            "_"
        )
        if not clean:
            raise ValueError(f"{field} is required.")
        return clean

    def _normalize_url(self, value: str) -> str:
        if value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://") :]
        return value
