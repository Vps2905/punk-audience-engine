from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from app.services.phase2_feature_role_provisioning_service import (
    Phase2FeatureRoleProvisioningService,
)


class FakeResult:
    def __init__(self, *, row=None, scalar=None):
        self._row = row
        self._scalar = scalar

    def mappings(self):
        return self

    def one(self):
        return self._row

    def all(self):
        return self._row or []

    def scalar_one(self):
        return self._scalar


class FakeIdentifierPreparer:
    def quote(self, value):
        return f'"{value}"'


class FakeDialect:
    name = "postgresql"
    identifier_preparer = FakeIdentifierPreparer()


class FakeTransaction:
    def __init__(self):
        self.rolled_back = False

    def rollback(self):
        self.rolled_back = True


class FakeConnection:
    def __init__(self, role_type):
        self.role_type = role_type
        self.calls = []
        self.current_tenant = None
        self.transaction = FakeTransaction()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def begin(self):
        return self.transaction

    def execute(self, statement, params=None):
        sql = str(statement)
        params = params or {}
        self.calls.append((sql, params))
        if (
            "current_user AS role_name" in sql
            and "role.rolsuper" not in sql
        ):
            return FakeResult(
                row={
                    "role_name": "punk_feature_admin",
                    "rolsuper": True,
                    "rolcreaterole": True,
                }
            )
        if "SELECT EXISTS" in sql and "FROM pg_roles" in sql:
            return FakeResult(row={"role_exists": False})
        if "FROM pg_auth_members" in sql:
            return FakeResult(row=[])
        if "SELECT current_database()" in sql:
            return FakeResult(
                row={"database_name": "punk_audience_features"}
            )
        if "AS registry_ready" in sql:
            return FakeResult(row={"registry_ready": True})
        if "role.rolsuper" in sql:
            writer = self.role_type == "writer"
            return FakeResult(
                row={
                    "role_name": (
                        "punk_feature_writer"
                        if writer
                        else "punk_feature_reader"
                    ),
                    "rolsuper": False,
                    "rolcreatedb": False,
                    "rolcreaterole": False,
                    "rolreplication": False,
                    "rolbypassrls": False,
                    "rolinherit": False,
                    "rolcanlogin": True,
                    "row_security": "on",
                    "can_create_database_objects": False,
                    "can_create_schema_objects": False,
                    "can_select_feature_sets": True,
                    "can_select_feature_vectors": True,
                    "can_insert_feature_sets": writer,
                    "can_insert_feature_vectors": writer,
                    "can_update_feature_sets": writer,
                    "can_update_feature_vectors": writer,
                    "can_delete_feature_sets": False,
                    "can_delete_feature_vectors": False,
                    "can_read_migration_ledger": False,
                }
            )
        if "set_config" in sql:
            self.current_tenant = params["tenant_id"]
            return FakeResult()
        if "COUNT(*)" in sql:
            count = (
                86
                if self.current_tenant == "punk_internal"
                else 0
            )
            return FakeResult(scalar=count)
        return FakeResult()

    def exec_driver_sql(self, statement):
        self.calls.append((str(statement), {}))
        return FakeResult()


class FakeContext:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, *_args):
        return False


class FakeEngine:
    dialect = FakeDialect()

    def __init__(self, role_type):
        self.connection = FakeConnection(role_type)
        self.disposed = False

    def begin(self):
        return FakeContext(self.connection)

    def connect(self):
        return self.connection

    def dispose(self):
        self.disposed = True


class FakePreflight:
    def run(self, **_kwargs):
        return {
            "status": "phase2_schema_ready",
            "same_database_as_source": False,
        }


def _environment():
    return {
        "ECHO_DATABASE_URL": "postgresql://source@source/source",
        "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL": (
            "postgresql://admin@feature/punk_audience_features"
        ),
        "PHASE2_FEATURE_READER_USER": "punk_feature_reader",
        "PHASE2_FEATURE_READER_PASSWORD": "reader-test-value",
        "PHASE2_FEATURE_WRITER_USER": "punk_feature_writer",
        "PHASE2_FEATURE_WRITER_PASSWORD": "writer-test-value",
    }


def _engine_factory(
    engines: dict[str, FakeEngine],
) -> Callable[..., FakeEngine]:
    def factory(database_url, **_kwargs):
        value = str(database_url)
        if "punk_feature_reader" in value:
            return engines["reader"]
        if "punk_feature_writer" in value:
            return engines["writer"]
        return engines["admin"]

    return factory


def test_role_provisioning_requires_explicit_approval():
    service = Phase2FeatureRoleProvisioningService(
        environment=_environment()
    )

    with pytest.raises(RuntimeError, match="confirmation"):
        service.provision(
            tenant_id="punk_internal",
            expected_feature_count=86,
            punk_owned_target_confirmed=False,
        )


def test_role_names_are_validated_and_must_be_distinct():
    environment = _environment()
    environment["PHASE2_FEATURE_WRITER_USER"] = (
        environment["PHASE2_FEATURE_READER_USER"]
    )
    service = Phase2FeatureRoleProvisioningService(
        environment=environment
    )

    with pytest.raises(ValueError, match="different"):
        service.provision(
            tenant_id="punk_internal",
            expected_feature_count=86,
            punk_owned_target_confirmed=True,
        )


def test_roles_are_least_privilege_and_rls_isolation_is_verified():
    engines = {
        "admin": FakeEngine("admin"),
        "reader": FakeEngine("reader"),
        "writer": FakeEngine("writer"),
    }
    service = Phase2FeatureRoleProvisioningService(
        environment=_environment(),
        engine_factory=_engine_factory(engines),
        preflight_factory=FakePreflight,
    )

    result = service.provision(
        tenant_id="punk-internal",
        expected_feature_count=86,
        punk_owned_target_confirmed=True,
    )

    assert result["status"] == "roles_provisioned_and_verified"
    assert result["tenant_id"] == "punk_internal"
    assert result["reader_visible_feature_count"] == 86
    assert result["reader_other_tenant_visible_feature_count"] == 0
    assert result["writer_visible_feature_count"] == 86
    assert result["reader_superuser"] is False
    assert result["reader_bypass_rls"] is False
    assert result["reader_can_write"] is False
    assert result["writer_can_insert_update"] is True
    assert result["writer_can_delete"] is False
    assert result["credentials_exposed"] is False

    admin_sql = "\n".join(
        sql for sql, _ in engines["admin"].connection.calls
    )
    assert "NOSUPERUSER" in admin_sql
    assert "NOBYPASSRLS" in admin_sql
    assert "NOCREATEDB" in admin_sql
    assert "NOCREATEROLE" in admin_sql
    assert "REVOKE CREATE ON SCHEMA public FROM PUBLIC" in admin_sql
    assert "REVOKE ALL ON public.audience_feature_sets FROM PUBLIC" in (
        admin_sql
    )
    assert "GRANT SELECT ON public.audience_embedding_models" in admin_sql
    assert (
        "GRANT SELECT, INSERT, UPDATE ON "
        "public.audience_feature_build_jobs"
    ) in admin_sql
    assert "FROM pg_auth_members" in admin_sql
    assert "reader-test-value" not in str(result)
    assert "writer-test-value" not in str(result)


def test_source_and_feature_target_must_pass_preflight():
    class BlockedPreflight:
        def run(self, **_kwargs):
            return {
                "status": "blocked",
                "same_database_as_source": False,
            }

    service = Phase2FeatureRoleProvisioningService(
        environment=_environment(),
        preflight_factory=BlockedPreflight,
    )

    with pytest.raises(RuntimeError, match="verified"):
        service.provision(
            tenant_id="punk_internal",
            expected_feature_count=86,
            punk_owned_target_confirmed=True,
        )


def test_store_and_api_have_no_historical_source_fallback():
    store = (
        "app/services/pgvector_audience_feature_store_service.py"
    )
    api = "app/api/punk_ai_audience_proposals.py"

    store_source = Path(store).read_text(encoding="utf-8")
    api_source = Path(api).read_text(encoding="utf-8")

    store_engine_source = store_source[
        store_source.index("    def _engine(") :
    ]
    api_builder_source = api_source[
        api_source.index("def build_audience_feature_store") :
        api_source.index(
            "def build_audience_feature_proposal_service"
        )
    ]
    assert "ECHO_DATABASE_URL" not in store_engine_source
    assert 'os.getenv("DATABASE_URL")' not in store_engine_source
    assert "ECHO_DATABASE_URL" not in api_builder_source
    assert 'os.getenv("DATABASE_URL")' not in api_builder_source
