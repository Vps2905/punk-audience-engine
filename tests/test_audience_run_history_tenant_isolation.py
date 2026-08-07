from pathlib import Path

from fastapi.testclient import TestClient

from app.api import audience_intelligence_modules as modules_api
from app.api import audience_intelligence_run_history as history_api
from app.main import app
from app.services import audience_run_history_service as history_module
from app.services.audience_run_history_service import (
    AudienceRunHistoryService,
)


class _FakeResult:
    def fetchone(self):
        return None


class _FakeConnection:
    def __init__(self):
        self.calls: list[tuple[str, dict | None]] = []

    def execute(self, statement, parameters=None):
        self.calls.append((str(statement), parameters))
        return _FakeResult()


class _FakeTransaction:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeEngine:
    def __init__(self):
        self.connection = _FakeConnection()

    def begin(self):
        return _FakeTransaction(self.connection)


class _TenantOwnedHistory:
    def _result(self, tenant_id, run_id):
        if tenant_id != "tenant-a":
            return {
                "enabled": True,
                "status": "not_found",
                "run_id": run_id,
            }
        return {
            "enabled": True,
            "status": "ok",
            "run": {"run_id": run_id},
        }

    def get_run(self, run_id, *, tenant_id):
        return self._result(tenant_id, run_id)

    def list_cohorts(self, *, tenant_id, run_id, **kwargs):
        result = self._result(tenant_id, run_id)
        result["cohorts"] = []
        return result

    def get_audit(self, run_id, *, tenant_id):
        result = self._result(tenant_id, run_id)
        result["events"] = []
        result["approvals"] = []
        return result

    def approve_run(self, *, tenant_id, run_id, **kwargs):
        return self._result(tenant_id, run_id)

    def reject_run(self, *, tenant_id, run_id, **kwargs):
        return self._result(tenant_id, run_id)


def _headers(tenant_id: str) -> dict[str, str]:
    return {
        "X-Audience-API-Key": "test-key",
        "X-Audience-Tenant-Id": tenant_id,
    }


def _configure_auth(monkeypatch) -> None:
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setenv("PRODUCTION_MODE", "false")
    monkeypatch.setenv("APP_ENV", "local")


def test_run_and_approval_routes_hide_another_tenants_run(
    monkeypatch,
):
    _configure_auth(monkeypatch)
    history = _TenantOwnedHistory()
    monkeypatch.setattr(
        history_api,
        "AudienceRunHistoryService",
        lambda: history,
    )
    monkeypatch.setattr(
        modules_api,
        "AudienceRunHistoryService",
        lambda: history,
    )
    client = TestClient(app)
    headers = _headers("tenant-b")

    responses = [
        client.get(
            "/api/audience-intelligence/runs/shared-run",
            headers=headers,
        ),
        client.get(
            "/api/audience-intelligence/runs/shared-run/cohorts",
            headers=headers,
        ),
        client.get(
            "/api/audience-intelligence/runs/shared-run/audit",
            headers=headers,
        ),
        client.post(
            "/api/audience-intelligence/runs/shared-run/approve",
            headers=headers,
            json={"actor": "reviewer"},
        ),
        client.post(
            "/api/audience-intelligence/runs/shared-run/reject",
            headers=headers,
            json={"actor": "reviewer"},
        ),
        client.post(
            "/api/audience-intelligence/modules/export/meta/shared-cohort",
            headers=headers,
            json={"run_id": "shared-run", "actor": "reviewer"},
        ),
    ]

    assert [response.status_code for response in responses] == [
        404,
        404,
        404,
        404,
        404,
        404,
    ]


def test_postgres_run_read_sets_context_and_tenant_predicate(
    monkeypatch,
):
    engine = _FakeEngine()
    monkeypatch.setattr(
        history_module,
        "create_engine",
        lambda _url: engine,
    )
    service = AudienceRunHistoryService(
        db_url="postgresql://service:unused@database/audience"
    )

    result = service.get_run("shared-run", tenant_id="Tenant-A")

    assert result["status"] == "not_found"
    context_calls = [
        call
        for call in engine.connection.calls
        if "set_config" in call[0]
    ]
    select_calls = [
        call
        for call in engine.connection.calls
        if "SELECT * FROM public.audience_run_history" in call[0]
    ]

    assert context_calls[0][1] == {"tenant_id": "tenant-a"}
    assert select_calls[0][1] == {
        "tenant_id": "tenant-a",
        "run_id": "shared-run",
    }
    assert "WHERE tenant_id = :tenant_id" in select_calls[0][0]


def test_migration_covers_all_tenant_owned_run_state():
    sql = Path(
        "migrations/0017_audience_run_history_tenant_isolation.sql"
    ).read_text(encoding="utf-8")

    tables = [
        "audience_run_history",
        "audience_run_cohorts",
        "audience_run_artifacts",
        "audience_run_warnings",
        "audience_run_events",
        "audience_run_approvals",
        "audience_privacy_budget_ledger",
    ]

    for table in tables:
        assert table in sql

    assert "UNIQUE (tenant_id, run_id)" in sql
    assert "FOREIGN KEY (tenant_id, run_id)" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "FORCE ROW LEVEL SECURITY" in sql
    assert "current_setting(''app.tenant_id'', true)" in sql
    assert "explicit tenant assignment" in sql
    assert "append-only" in sql
    assert "tenant_id TEXT DEFAULT" not in sql


def test_schema_reference_has_tenant_keys_and_forced_rls():
    sql = Path("docs/module1_postgres_schema.sql").read_text(
        encoding="utf-8"
    )

    assert "UNIQUE (tenant_id, run_id)" in sql
    assert "FOREIGN KEY (tenant_id, run_id)" in sql
    assert sql.count("FORCE ROW LEVEL SECURITY") >= 7
    assert "current_setting('app.tenant_id', true)" in sql
