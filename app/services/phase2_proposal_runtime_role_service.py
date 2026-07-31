from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError

from app.services.phase2_database_preflight_service import (
    Phase2DatabasePreflightService,
)

ROLE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,62}$")
PROPOSAL_ROLE_LOCK = "punk_audience_phase2_proposal_runtime_role"


class Phase2ProposalRuntimeRoleService:
    """Provision a runtime role that can only read/insert proposal records."""

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
        feature_set_id: str,
        feature_set_version: int,
        punk_owned_target_confirmed: bool,
    ) -> dict[str, Any]:
        if not punk_owned_target_confirmed:
            raise RuntimeError(
                "Explicit Punk-owned feature target confirmation is required."
            )
        clean_tenant_id = self._required_slug(tenant_id, "tenant_id")
        clean_feature_set_id = self._required_text(
            feature_set_id,
            "feature_set_id",
        )
        clean_feature_set_version = int(feature_set_version)
        if clean_feature_set_version < 1:
            raise ValueError("feature_set_version must be positive.")

        source_url = self._required_environment("ECHO_DATABASE_URL")
        migration_url = self._required_environment(
            "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL"
        )
        runtime_user = self._role_name(
            self._required_environment("PHASE2_PROPOSAL_RUNTIME_USER")
        )
        runtime_password = self._required_environment(
            "PHASE2_PROPOSAL_RUNTIME_PASSWORD"
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
                "Proposal runtime provisioning requires a verified, "
                "separate Phase 2 feature database."
            )

        admin_engine: Engine | None = None
        try:
            admin_engine = self._engine_factory(
                self._normalize_url(migration_url),
                pool_pre_ping=True,
            )
            if admin_engine.dialect.name != "postgresql":
                raise RuntimeError(
                    "The proposal runtime target must be PostgreSQL."
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
                if admin["role_name"] == runtime_user:
                    raise RuntimeError(
                        "The migration and proposal runtime identities "
                        "must be different."
                    )
                ledger_present = connection.execute(
                    text(
                        """
                        SELECT to_regclass(
                            'public.punk_ai_audience_proposals'
                        ) IS NOT NULL
                        """
                    )
                ).scalar_one()
                if not ledger_present:
                    raise RuntimeError(
                        "Apply proposal-ledger migration 0005 first."
                    )
                connection.execute(
                    text(
                        """
                        SELECT pg_advisory_xact_lock(
                            hashtext(:lock_name)
                        )
                        """
                    ),
                    {"lock_name": PROPOSAL_ROLE_LOCK},
                )
                connection.exec_driver_sql(
                    "SET LOCAL lock_timeout = '10s'"
                )
                self._upsert_role(
                    connection,
                    admin_engine,
                    role_name=runtime_user,
                    password=runtime_password,
                )
                self._revoke_memberships(
                    connection,
                    admin_engine,
                    role_name=runtime_user,
                )
                self._apply_grants(
                    connection,
                    admin_engine,
                    role_name=runtime_user,
                )
        except SQLAlchemyError:
            raise RuntimeError(
                "Proposal runtime role provisioning failed and rolled back."
            ) from None
        finally:
            if admin_engine is not None:
                admin_engine.dispose()

        runtime_url = make_url(
            self._normalize_url(migration_url)
        ).set(
            username=runtime_user,
            password=runtime_password,
        ).render_as_string(hide_password=False)
        verification = self._verify_runtime(
            runtime_url,
            expected_role=runtime_user,
            tenant_id=clean_tenant_id,
            feature_set_id=clean_feature_set_id,
            feature_set_version=clean_feature_set_version,
        )
        return {
            "status": "proposal_runtime_role_provisioned_and_verified",
            "tenant_id": clean_tenant_id,
            "runtime_role": runtime_user,
            "runtime_superuser": False,
            "runtime_bypass_rls": False,
            "runtime_can_read_proposals": True,
            "runtime_can_insert_proposals": True,
            "runtime_can_update_proposals": False,
            "runtime_can_delete_proposals": False,
            "runtime_can_read_feature_tables": False,
            "runtime_can_write_feature_tables": False,
            "runtime_can_read_migration_ledger": False,
            "own_tenant_probe_visible": verification["own_tenant_count"],
            "other_tenant_probe_visible": verification[
                "other_tenant_count"
            ],
            "probe_rolled_back": True,
            "credentials_exposed": False,
            "activation_or_export_performed": False,
        }

    def _upsert_role(
        self,
        connection: Any,
        engine: Engine,
        *,
        role_name: str,
        password: str,
    ) -> None:
        role = engine.dialect.identifier_preparer.quote(role_name)
        exists = connection.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_roles
                    WHERE rolname = :role_name
                )
                """
            ),
            {"role_name": role_name},
        ).scalar_one()
        statement = "ALTER ROLE" if exists else "CREATE ROLE"
        connection.execute(
            text(
                f"""
                {statement} {role}
                WITH
                    LOGIN
                    NOSUPERUSER
                    NOCREATEDB
                    NOCREATEROLE
                    NOINHERIT
                    NOREPLICATION
                    NOBYPASSRLS
                    CONNECTION LIMIT 20
                    PASSWORD :password
                """
            ),
            {"password": password},
        )
        connection.exec_driver_sql(
            f"ALTER ROLE {role} SET row_security = on"
        )

    def _revoke_memberships(
        self,
        connection: Any,
        engine: Engine,
        *,
        role_name: str,
    ) -> None:
        member = engine.dialect.identifier_preparer.quote(role_name)
        rows = connection.execute(
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
        for row in rows:
            granted = engine.dialect.identifier_preparer.quote(
                str(row["granted_role"])
            )
            connection.exec_driver_sql(
                f"REVOKE {granted} FROM {member}"
            )

    def _apply_grants(
        self,
        connection: Any,
        engine: Engine,
        *,
        role_name: str,
    ) -> None:
        role = engine.dialect.identifier_preparer.quote(role_name)
        database_name = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()
        database = engine.dialect.identifier_preparer.quote(
            str(database_name)
        )
        connection.exec_driver_sql(
            f"GRANT CONNECT ON DATABASE {database} TO {role}"
        )
        connection.exec_driver_sql(
            f"REVOKE CREATE ON DATABASE {database} FROM {role}"
        )
        connection.exec_driver_sql(
            f"GRANT USAGE ON SCHEMA public TO {role}"
        )
        connection.exec_driver_sql(
            "REVOKE ALL ON public.punk_ai_audience_proposals FROM PUBLIC"
        )
        connection.exec_driver_sql(
            f"REVOKE ALL ON public.audience_feature_sets FROM {role}"
        )
        connection.exec_driver_sql(
            f"REVOKE ALL ON public.audience_feature_vectors FROM {role}"
        )
        connection.exec_driver_sql(
            f"REVOKE ALL ON public.audience_schema_migrations FROM {role}"
        )
        connection.exec_driver_sql(
            "GRANT SELECT, INSERT ON "
            f"public.punk_ai_audience_proposals TO {role}"
        )
        connection.exec_driver_sql(
            "REVOKE UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON "
            f"public.punk_ai_audience_proposals FROM {role}"
        )

    def _verify_runtime(
        self,
        database_url: str,
        *,
        expected_role: str,
        tenant_id: str,
        feature_set_id: str,
        feature_set_version: int,
    ) -> dict[str, int]:
        engine = self._engine_factory(database_url, pool_pre_ping=True)
        probe_id = "role_probe_" + uuid.uuid4().hex
        try:
            with engine.connect() as connection:
                transaction = connection.begin()
                try:
                    role = connection.execute(
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
                                current_setting('row_security')
                                    AS row_security,
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
                                    'public.punk_ai_audience_proposals',
                                    'SELECT'
                                ) AS can_select_proposals,
                                has_table_privilege(
                                    current_user,
                                    'public.punk_ai_audience_proposals',
                                    'INSERT'
                                ) AS can_insert_proposals,
                                has_table_privilege(
                                    current_user,
                                    'public.punk_ai_audience_proposals',
                                    'UPDATE'
                                ) AS can_update_proposals,
                                has_table_privilege(
                                    current_user,
                                    'public.punk_ai_audience_proposals',
                                    'DELETE'
                                ) AS can_delete_proposals,
                                (
                                    has_table_privilege(
                                        current_user,
                                        'public.audience_feature_sets',
                                        'SELECT'
                                    )
                                    OR has_table_privilege(
                                        current_user,
                                        'public.audience_feature_sets',
                                        'INSERT'
                                    )
                                    OR has_table_privilege(
                                        current_user,
                                        'public.audience_feature_sets',
                                        'UPDATE'
                                    )
                                    OR has_table_privilege(
                                        current_user,
                                        'public.audience_feature_sets',
                                        'DELETE'
                                    )
                                ) AS has_feature_set_privileges,
                                (
                                    has_table_privilege(
                                        current_user,
                                        'public.audience_feature_vectors',
                                        'SELECT'
                                    )
                                    OR has_table_privilege(
                                        current_user,
                                        'public.audience_feature_vectors',
                                        'INSERT'
                                    )
                                    OR has_table_privilege(
                                        current_user,
                                        'public.audience_feature_vectors',
                                        'UPDATE'
                                    )
                                    OR has_table_privilege(
                                        current_user,
                                        'public.audience_feature_vectors',
                                        'DELETE'
                                    )
                                ) AS has_feature_vector_privileges,
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
                    self._assert_role(role, expected_role)
                    self._set_tenant_context(connection, tenant_id)
                    connection.execute(
                        text(
                            """
                            INSERT INTO public.punk_ai_audience_proposals (
                                tenant_id,
                                proposal_id,
                                campaign_id,
                                idempotency_key,
                                request_fingerprint,
                                contract_version,
                                feature_set_id,
                                feature_set_version,
                                execution_mode,
                                status,
                                approval_status,
                                response_document
                            )
                            VALUES (
                                :tenant_id,
                                :probe_id,
                                :probe_id,
                                :probe_id,
                                :request_fingerprint,
                                'role-verification',
                                :feature_set_id,
                                :feature_set_version,
                                'historical_preview',
                                'verification',
                                'blocked_historical_source',
                                CAST(:response_document AS JSONB)
                            )
                            """
                        ),
                        {
                            "tenant_id": tenant_id,
                            "probe_id": probe_id,
                            "request_fingerprint": "0" * 64,
                            "feature_set_id": feature_set_id,
                            "feature_set_version": feature_set_version,
                            "response_document": json.dumps(
                                {"verification_probe": True}
                            ),
                        },
                    )
                    own_count = self._probe_count(
                        connection,
                        tenant_id=tenant_id,
                        probe_id=probe_id,
                    )
                    other_count = self._probe_count(
                        connection,
                        tenant_id=f"{tenant_id}_isolation_probe",
                        probe_id=probe_id,
                    )
                finally:
                    transaction.rollback()
        except SQLAlchemyError:
            raise RuntimeError(
                "Proposal runtime role verification failed."
            ) from None
        finally:
            engine.dispose()
        if own_count != 1 or other_count != 0:
            raise RuntimeError(
                "Proposal runtime role failed tenant-isolation verification."
            )
        return {
            "own_tenant_count": own_count,
            "other_tenant_count": other_count,
        }

    def _assert_role(
        self,
        role: Mapping[str, Any],
        expected_role: str,
    ) -> None:
        if (
            role["role_name"] != expected_role
            or role["rolsuper"]
            or role["rolcreatedb"]
            or role["rolcreaterole"]
            or role["rolreplication"]
            or role["rolbypassrls"]
            or role["rolinherit"]
            or not role["rolcanlogin"]
            or str(role["row_security"]).lower() != "on"
            or role["can_create_database_objects"]
            or role["can_create_schema_objects"]
            or not role["can_select_proposals"]
            or not role["can_insert_proposals"]
            or role["can_update_proposals"]
            or role["can_delete_proposals"]
            or role["has_feature_set_privileges"]
            or role["has_feature_vector_privileges"]
            or role["can_read_migration_ledger"]
        ):
            raise RuntimeError(
                "The proposal runtime role has unsafe privileges."
            )

    def _probe_count(
        self,
        connection: Any,
        *,
        tenant_id: str,
        probe_id: str,
    ) -> int:
        self._set_tenant_context(connection, tenant_id)
        return int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM public.punk_ai_audience_proposals
                    WHERE proposal_id = :probe_id
                    """
                ),
                {"probe_id": probe_id},
            ).scalar_one()
        )

    def _set_tenant_context(
        self,
        connection: Any,
        tenant_id: str,
    ) -> None:
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

    def _required_slug(self, value: str, label: str) -> str:
        clean = re.sub(
            r"[^a-z0-9_]+",
            "_",
            value.strip().lower(),
        ).strip("_")
        if not clean:
            raise ValueError(f"{label} is required.")
        return clean

    def _required_text(self, value: str, label: str) -> str:
        clean = " ".join(str(value or "").split())
        if not clean:
            raise ValueError(f"{label} is required.")
        return clean

    def _normalize_url(self, value: str) -> str:
        if value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://") :]
        return value
