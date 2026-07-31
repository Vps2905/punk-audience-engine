from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

SOURCE_DATABASE_ENV_KEYS = (
    "ECHO_DATABASE_URL",
    "DATABASE_URL",
)
FEATURE_DATABASE_ENV_KEY = "AUDIENCE_FEATURE_DATABASE_URL"


def _first_configured(
    environment: Mapping[str, str],
    keys: tuple[str, ...],
) -> str:
    for key in keys:
        value = str(environment.get(key) or "").strip()
        if value:
            return value
    return ""


def _normalize_postgres_url(value: str) -> str:
    if value.startswith("postgres://"):
        return "postgresql://" + value[len("postgres://") :]
    return value


class Phase2DatabasePreflightService:
    """
    Inspect Phase 2 database prerequisites without changing database state.

    The source and feature target are deliberately resolved independently.
    The feature target never falls back to the historical source because that
    could make an operator apply product-owned tables to a provider database.
    """

    def __init__(
        self,
        *,
        engine_factory: Callable[..., Engine] = create_engine,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._engine_factory = engine_factory
        self._environment = environment if environment is not None else os.environ

    def run(
        self,
        *,
        source_database_url: str | None = None,
        feature_database_url: str | None = None,
        punk_owned_target_confirmed: bool = False,
    ) -> dict[str, Any]:
        source_url = str(
            source_database_url
            or _first_configured(
                self._environment,
                SOURCE_DATABASE_ENV_KEYS,
            )
        ).strip()
        feature_url = str(
            feature_database_url
            or self._environment.get(FEATURE_DATABASE_ENV_KEY)
            or ""
        ).strip()

        source = self._inspect_configured_database(
            source_url,
            role="historical_source",
        )
        feature_target = self._inspect_configured_database(
            feature_url,
            role="feature_target",
        )

        same_database = self._same_database(source, feature_target)
        source.pop("_database_identity", None)
        feature_target.pop("_database_identity", None)

        source_ready = bool(
            source.get("connection_ok")
            and source.get("legacy_vector_models_table_present")
            and source.get("legacy_vectors_table_present")
            and source.get("legacy_snapshot_present")
            and source.get("legacy_vector_dimension") == 384
        )
        target_schema_ready = bool(
            feature_target.get("connection_ok")
            and feature_target.get("public_schema_create_privilege")
        )
        vector_ready = bool(feature_target.get("pgvector_installed"))
        migration_already_applied = bool(
            feature_target.get("phase2_migration_recorded")
            and feature_target.get("feature_sets_table_present")
            and feature_target.get("feature_vectors_table_present")
            and feature_target.get("feature_embedding_type") == "vector(384)"
            and feature_target.get("feature_sets_rls_forced")
            and feature_target.get("feature_vectors_rls_forced")
            and feature_target.get("feature_sets_tenant_policy_present")
            and feature_target.get("feature_vectors_tenant_policy_present")
            and feature_target.get("feature_set_lookup_index_present")
            and feature_target.get("feature_hard_filter_index_present")
            and feature_target.get("feature_lexical_index_present")
            and feature_target.get("feature_hnsw_index_present")
        )
        recorded_schema_incomplete = bool(
            feature_target.get("phase2_migration_recorded")
            and not migration_already_applied
        )

        blockers: list[str] = []
        warnings: list[str] = []
        if not source_url:
            blockers.append("historical_source_database_not_configured")
        elif not source.get("connection_ok"):
            blockers.append("historical_source_preflight_failed")
        elif not source_ready:
            blockers.append("compatible_legacy_snapshot_not_found")

        if not feature_url:
            blockers.append("explicit_feature_database_not_configured")
        elif not feature_target.get("connection_ok"):
            blockers.append("feature_database_preflight_failed")

        if feature_target.get("connection_ok"):
            if recorded_schema_incomplete:
                blockers.append("phase2_recorded_schema_incomplete")
            if not feature_target.get("pgvector_available"):
                blockers.append("pgvector_not_available")
            elif not vector_ready and not feature_target.get(
                "current_role_superuser"
            ):
                blockers.append(
                    "pgvector_requires_database_admin_installation"
                )
            if not migration_already_applied and not target_schema_ready:
                blockers.append("feature_schema_create_privilege_missing")

        if not punk_owned_target_confirmed:
            blockers.append("punk_owned_feature_database_not_confirmed")
        if same_database:
            warnings.append("source_and_feature_target_are_the_same_database")

        if blockers:
            status = "blocked"
            next_action = self._next_action(blockers)
        elif migration_already_applied:
            status = "phase2_schema_ready"
            next_action = "index_historical_snapshot_after_operator_approval"
        else:
            status = "ready_for_approved_migration"
            next_action = "apply_phase2_migration_after_explicit_approval"

        return {
            "status": status,
            "read_only": True,
            "migration_attempted": False,
            "indexing_attempted": False,
            "credentials_exposed": False,
            "source": source,
            "feature_target": feature_target,
            "same_database_as_source": same_database,
            "punk_owned_target_confirmed": bool(
                punk_owned_target_confirmed
            ),
            "source_snapshot_ready": source_ready,
            "phase2_schema_ready": migration_already_applied,
            "blockers": sorted(set(blockers)),
            "warnings": sorted(set(warnings)),
            "next_action": next_action,
        }

    def _inspect_configured_database(
        self,
        database_url: str,
        *,
        role: str,
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "role": role,
            "configured": bool(database_url),
            "connection_ok": False,
            "read_only_transaction_verified": False,
            "postgresql": False,
            "server_version_num": None,
            "pgvector_available": False,
            "pgvector_installed": False,
            "pgvector_version": None,
            "current_role_superuser": False,
            "database_create_privilege": False,
            "public_schema_create_privilege": False,
            "legacy_vector_models_table_present": False,
            "legacy_vectors_table_present": False,
            "legacy_snapshot_present": False,
            "legacy_vector_count": 0,
            "legacy_vector_dimension": None,
            "latest_source_timestamp": None,
            "source_events_table_present": False,
            "feature_sets_table_present": False,
            "feature_vectors_table_present": False,
            "feature_embedding_type": None,
            "feature_sets_rls_forced": False,
            "feature_vectors_rls_forced": False,
            "feature_sets_tenant_policy_present": False,
            "feature_vectors_tenant_policy_present": False,
            "feature_set_lookup_index_present": False,
            "feature_hard_filter_index_present": False,
            "feature_lexical_index_present": False,
            "feature_hnsw_index_present": False,
            "phase2_migration_recorded": False,
            "error_code": (
                None if database_url else "database_not_configured"
            ),
        }
        if not database_url:
            return base

        engine: Engine | None = None
        try:
            engine = self._engine_factory(
                _normalize_postgres_url(database_url),
                pool_pre_ping=True,
            )
            if engine.dialect.name != "postgresql":
                base["error_code"] = "postgresql_required"
                return base

            with engine.connect() as connection:
                transaction = connection.begin()
                try:
                    connection.exec_driver_sql(
                        "SET TRANSACTION READ ONLY"
                    )
                    read_only = connection.execute(
                        text(
                            """
                            SELECT current_setting(
                                'transaction_read_only'
                            ) AS transaction_read_only
                            """
                        )
                    ).mappings().one()
                    if str(
                        read_only["transaction_read_only"]
                    ).lower() != "on":
                        base["error_code"] = (
                            "read_only_transaction_not_verified"
                        )
                        return base

                    common = connection.execute(
                        text(
                            """
                            SELECT
                                current_setting(
                                    'server_version_num'
                                )::INTEGER AS server_version_num,
                                EXISTS (
                                    SELECT 1
                                    FROM pg_available_extensions
                                    WHERE name = 'vector'
                                ) AS pgvector_available,
                                (
                                    SELECT installed_version
                                    FROM pg_available_extensions
                                    WHERE name = 'vector'
                                ) AS pgvector_version,
                                COALESCE(
                                    (
                                        SELECT rolsuper
                                        FROM pg_roles
                                        WHERE rolname = current_user
                                    ),
                                    FALSE
                                ) AS current_role_superuser,
                                has_database_privilege(
                                    current_user,
                                    current_database(),
                                    'CREATE'
                                ) AS database_create_privilege,
                                has_schema_privilege(
                                    current_user,
                                    'public',
                                    'CREATE'
                                ) AS public_schema_create_privilege,
                                (
                                    to_regclass(
                                        'public.audience_vector_models'
                                    ) IS NOT NULL
                                ) AS legacy_models_present,
                                (
                                    to_regclass(
                                        'public.audience_vectors'
                                    ) IS NOT NULL
                                ) AS legacy_vectors_present,
                                (
                                    to_regclass(
                                        'public.audience_feature_sets'
                                    ) IS NOT NULL
                                ) AS feature_sets_present,
                                (
                                    to_regclass(
                                        'public.maid_extractions'
                                    ) IS NOT NULL
                                ) AS source_events_present,
                                (
                                    to_regclass(
                                        'public.audience_feature_vectors'
                                    ) IS NOT NULL
                                ) AS feature_vectors_present,
                                (
                                    to_regclass(
                                        'public.audience_schema_migrations'
                                    ) IS NOT NULL
                                ) AS migrations_present,
                                current_database() AS database_name,
                                COALESCE(
                                    inet_server_addr()::TEXT,
                                    'local'
                                ) AS server_address,
                                COALESCE(
                                    inet_server_port(),
                                    0
                                ) AS server_port
                            """
                        )
                    ).mappings().one()

                    base.update(
                        {
                            "connection_ok": True,
                            "read_only_transaction_verified": True,
                            "postgresql": True,
                            "server_version_num": int(
                                common["server_version_num"]
                            ),
                            "pgvector_available": bool(
                                common["pgvector_available"]
                            ),
                            "pgvector_installed": bool(
                                common["pgvector_version"]
                            ),
                            "pgvector_version": common[
                                "pgvector_version"
                            ],
                            "current_role_superuser": bool(
                                common["current_role_superuser"]
                            ),
                            "database_create_privilege": bool(
                                common["database_create_privilege"]
                            ),
                            "public_schema_create_privilege": bool(
                                common[
                                    "public_schema_create_privilege"
                                ]
                            ),
                            "legacy_vector_models_table_present": bool(
                                common["legacy_models_present"]
                            ),
                            "legacy_vectors_table_present": bool(
                                common["legacy_vectors_present"]
                            ),
                            "feature_sets_table_present": bool(
                                common["feature_sets_present"]
                            ),
                            "source_events_table_present": bool(
                                common["source_events_present"]
                            ),
                            "feature_vectors_table_present": bool(
                                common["feature_vectors_present"]
                            ),
                            "error_code": None,
                            "_database_identity": self._identity_digest(
                                common
                            ),
                        }
                    )

                    if (
                        base["legacy_vector_models_table_present"]
                        and base["legacy_vectors_table_present"]
                    ):
                        self._inspect_legacy_snapshot(
                            connection,
                            base,
                        )
                    if (
                        base["feature_sets_table_present"]
                        and base["feature_vectors_table_present"]
                    ):
                        self._inspect_feature_schema(
                            connection,
                            base,
                        )
                    if common["migrations_present"]:
                        migration = connection.execute(
                            text(
                                """
                                SELECT EXISTS (
                                    SELECT 1
                                    FROM public.audience_schema_migrations
                                    WHERE version =
                                        '0004_versioned_audience_features_pgvector.sql'
                                ) AS migration_recorded
                                """
                            )
                        ).mappings().one()
                        base["phase2_migration_recorded"] = bool(
                            migration["migration_recorded"]
                        )
                finally:
                    transaction.rollback()
        except (SQLAlchemyError, TypeError, ValueError):
            base["connection_ok"] = False
            base["read_only_transaction_verified"] = False
            base["error_code"] = "database_preflight_failed"
        finally:
            if engine is not None:
                engine.dispose()

        return base

    def _inspect_legacy_snapshot(
        self,
        connection: Any,
        result: dict[str, Any],
    ) -> None:
        snapshot = connection.execute(
            text(
                """
                SELECT
                    m.vector_count,
                    m.vector_dimension,
                    EXISTS (
                        SELECT 1
                        FROM public.audience_vectors v
                        WHERE v.job_id = m.job_id
                    ) AS has_vectors
                FROM public.audience_vector_models m
                ORDER BY m.updated_at DESC, m.job_id ASC
                LIMIT 1
                """
            )
        ).mappings().first()
        if snapshot:
            result["legacy_snapshot_present"] = bool(
                snapshot["has_vectors"]
            )
            result["legacy_vector_count"] = int(
                snapshot["vector_count"] or 0
            )
            result["legacy_vector_dimension"] = int(
                snapshot["vector_dimension"] or 0
            )

        if result["source_events_table_present"]:
            source_timestamp = connection.execute(
                text(
                    """
                    SELECT MAX(created_at) AS latest_source_timestamp
                    FROM public.maid_extractions
                    """
                )
            ).mappings().one()
            value = source_timestamp["latest_source_timestamp"]
            result["latest_source_timestamp"] = (
                value.isoformat() if value is not None else None
            )

    def _inspect_feature_schema(
        self,
        connection: Any,
        result: dict[str, Any],
    ) -> None:
        schema = connection.execute(
            text(
                """
                SELECT
                    format_type(
                        attribute.atttypid,
                        attribute.atttypmod
                    ) AS embedding_type,
                    feature_sets.relrowsecurity
                        AND feature_sets.relforcerowsecurity
                        AS feature_sets_rls_forced,
                    feature_vectors.relrowsecurity
                        AND feature_vectors.relforcerowsecurity
                        AS feature_vectors_rls_forced,
                    EXISTS (
                        SELECT 1
                        FROM pg_policies
                        WHERE
                            schemaname = 'public'
                            AND tablename = 'audience_feature_sets'
                            AND policyname =
                                'audience_feature_sets_tenant_policy'
                    ) AS feature_sets_tenant_policy_present,
                    EXISTS (
                        SELECT 1
                        FROM pg_policies
                        WHERE
                            schemaname = 'public'
                            AND tablename = 'audience_feature_vectors'
                            AND policyname =
                                'audience_feature_vectors_tenant_policy'
                    ) AS feature_vectors_tenant_policy_present,
                    (
                        to_regclass(
                            'public.idx_audience_feature_sets_lookup'
                        ) IS NOT NULL
                    ) AS feature_set_lookup_index_present,
                    (
                        to_regclass(
                            'public.idx_audience_feature_vectors_hard_filters'
                        ) IS NOT NULL
                    ) AS feature_hard_filter_index_present,
                    (
                        to_regclass(
                            'public.idx_audience_feature_vectors_search'
                        ) IS NOT NULL
                    ) AS feature_lexical_index_present,
                    (
                        to_regclass(
                            'public.idx_audience_feature_vectors_embedding_hnsw'
                        ) IS NOT NULL
                    ) AS feature_hnsw_index_present
                FROM pg_attribute attribute
                JOIN pg_class feature_vectors
                    ON feature_vectors.oid = attribute.attrelid
                JOIN pg_namespace feature_vectors_namespace
                    ON feature_vectors_namespace.oid =
                        feature_vectors.relnamespace
                CROSS JOIN pg_class feature_sets
                JOIN pg_namespace feature_sets_namespace
                    ON feature_sets_namespace.oid =
                        feature_sets.relnamespace
                WHERE
                    feature_vectors_namespace.nspname = 'public'
                    AND feature_vectors.relname =
                        'audience_feature_vectors'
                    AND attribute.attname = 'embedding'
                    AND feature_sets_namespace.nspname = 'public'
                    AND feature_sets.relname =
                        'audience_feature_sets'
                LIMIT 1
                """
            )
        ).mappings().first()
        if not schema:
            return
        result["feature_embedding_type"] = schema["embedding_type"]
        result["feature_sets_rls_forced"] = bool(
            schema["feature_sets_rls_forced"]
        )
        result["feature_vectors_rls_forced"] = bool(
            schema["feature_vectors_rls_forced"]
        )
        result["feature_sets_tenant_policy_present"] = bool(
            schema["feature_sets_tenant_policy_present"]
        )
        result["feature_vectors_tenant_policy_present"] = bool(
            schema["feature_vectors_tenant_policy_present"]
        )
        result["feature_set_lookup_index_present"] = bool(
            schema["feature_set_lookup_index_present"]
        )
        result["feature_hard_filter_index_present"] = bool(
            schema["feature_hard_filter_index_present"]
        )
        result["feature_lexical_index_present"] = bool(
            schema["feature_lexical_index_present"]
        )
        result["feature_hnsw_index_present"] = bool(
            schema["feature_hnsw_index_present"]
        )

    def _identity_digest(self, row: Mapping[str, Any]) -> str:
        identity = "|".join(
            [
                str(row["server_address"]),
                str(row["server_port"]),
                str(row["database_name"]),
            ]
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _same_database(
        self,
        source: Mapping[str, Any],
        feature_target: Mapping[str, Any],
    ) -> bool:
        source_identity = source.get("_database_identity")
        feature_identity = feature_target.get("_database_identity")
        return bool(
            source_identity
            and feature_identity
            and source_identity == feature_identity
        )

    def _next_action(self, blockers: list[str]) -> str:
        priority = (
            (
                "phase2_recorded_schema_incomplete",
                "repair_phase2_schema_before_use",
            ),
            (
                "explicit_feature_database_not_configured",
                "configure_punk_owned_feature_database",
            ),
            (
                "feature_database_preflight_failed",
                "verify_feature_database_connectivity",
            ),
            (
                "pgvector_not_available",
                "provision_postgresql_with_pgvector",
            ),
            (
                "pgvector_requires_database_admin_installation",
                "ask_database_admin_to_install_pgvector",
            ),
            (
                "feature_schema_create_privilege_missing",
                "grant_migration_role_schema_create_privilege",
            ),
            (
                "punk_owned_feature_database_not_confirmed",
                "confirm_punk_owned_feature_database",
            ),
            (
                "historical_source_database_not_configured",
                "configure_read_only_historical_source",
            ),
            (
                "historical_source_preflight_failed",
                "verify_historical_source_connectivity",
            ),
            (
                "compatible_legacy_snapshot_not_found",
                "verify_legacy_privacy_safe_vector_snapshot",
            ),
        )
        for blocker, action in priority:
            if blocker in blockers:
                return action
        return "resolve_preflight_blockers"
