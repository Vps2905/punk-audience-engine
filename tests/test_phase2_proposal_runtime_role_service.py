from __future__ import annotations

from app.services.phase2_proposal_runtime_role_service import (
    Phase2ProposalRuntimeRoleService,
)


class FakeResult:
    def __init__(self, *, row=None, scalar=None):
        self.row = row
        self.scalar = scalar

    def mappings(self):
        return self

    def one(self):
        return self.row

    def all(self):
        return self.row or []

    def scalar_one(self):
        return self.scalar


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
        if "current_user AS role_name" in sql and "role.rolsuper" not in sql:
            return FakeResult(
                row={
                    "role_name": "punk_feature_admin",
                    "rolsuper": True,
                    "rolcreaterole": True,
                }
            )
        if "to_regclass" in sql:
            return FakeResult(scalar=True)
        if "SELECT EXISTS" in sql and "FROM pg_roles" in sql:
            return FakeResult(scalar=False)
        if "FROM pg_auth_members" in sql:
            return FakeResult(row=[])
        if "SELECT current_database()" in sql:
            return FakeResult(scalar="punk_audience_features")
        if "role.rolsuper" in sql:
            return FakeResult(
                row={
                    "role_name": "punk_proposal_runtime",
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
                    "can_select_proposals": True,
                    "can_insert_proposals": True,
                    "can_update_proposals": False,
                    "can_delete_proposals": False,
                    "has_feature_set_privileges": False,
                    "has_feature_vector_privileges": False,
                    "can_read_migration_ledger": False,
                }
            )
        if "set_config" in sql:
            self.current_tenant = params["tenant_id"]
            return FakeResult()
        if "SELECT COUNT(*)" in sql:
            return FakeResult(
                scalar=(
                    1
                    if self.current_tenant == "punk_internal"
                    else 0
                )
            )
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

    def begin(self):
        return FakeContext(self.connection)

    def connect(self):
        return self.connection

    def dispose(self):
        return None


class ReadyPreflight:
    def run(self, **_kwargs):
        return {
            "status": "phase2_schema_ready",
            "same_database_as_source": False,
        }


def _environment():
    return {
        "ECHO_DATABASE_URL": "postgresql://source",
        "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL": (
            "postgresql://admin@feature/punk_audience_features"
        ),
        "PHASE2_PROPOSAL_RUNTIME_USER": "punk_proposal_runtime",
        "PHASE2_PROPOSAL_RUNTIME_PASSWORD": "runtime-test-value",
    }


def test_proposal_runtime_role_is_minimal_and_rls_verified():
    engines = {
        "admin": FakeEngine("admin"),
        "runtime": FakeEngine("runtime"),
    }

    def factory(database_url, **_kwargs):
        if "punk_proposal_runtime" in str(database_url):
            return engines["runtime"]
        return engines["admin"]

    service = Phase2ProposalRuntimeRoleService(
        environment=_environment(),
        engine_factory=factory,
        preflight_factory=ReadyPreflight,
    )

    result = service.provision(
        tenant_id="punk-internal",
        feature_set_id="feature_set_1",
        feature_set_version=1,
        punk_owned_target_confirmed=True,
    )

    assert result["status"] == (
        "proposal_runtime_role_provisioned_and_verified"
    )
    assert result["runtime_superuser"] is False
    assert result["runtime_bypass_rls"] is False
    assert result["runtime_can_read_proposals"] is True
    assert result["runtime_can_insert_proposals"] is True
    assert result["runtime_can_update_proposals"] is False
    assert result["runtime_can_delete_proposals"] is False
    assert result["runtime_can_read_feature_tables"] is False
    assert result["own_tenant_probe_visible"] == 1
    assert result["other_tenant_probe_visible"] == 0
    assert result["probe_rolled_back"] is True
    assert result["credentials_exposed"] is False
    assert "runtime-test-value" not in str(result)

    admin_sql = "\n".join(
        sql for sql, _params in engines["admin"].connection.calls
    )
    assert "NOSUPERUSER" in admin_sql
    assert "NOBYPASSRLS" in admin_sql
    assert "GRANT SELECT, INSERT" in admin_sql
    assert "REVOKE UPDATE, DELETE" in admin_sql
