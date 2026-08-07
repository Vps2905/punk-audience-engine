from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import audience_intelligence_jobs as jobs
from app.core.audience_job_store import AudienceJobStore
from app.main import app


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
    def __init__(self, connection: _FakeConnection):
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


def test_job_routes_hide_another_tenants_job(
    monkeypatch,
    tmp_path: Path,
):
    _configure_auth(monkeypatch)
    store = AudienceJobStore(root_dir=tmp_path / "jobs")
    record = store.create_job(
        {"prompt": "privacy-safe audience", "source": "postgres"},
        tenant_id="tenant-a",
    )
    monkeypatch.setattr(jobs, "job_store", store)

    client = TestClient(app)
    status = client.get(
        f"/api/audience-intelligence/jobs/status/{record['job_id']}",
        headers=_headers("tenant-b"),
    )
    result = client.get(
        f"/api/audience-intelligence/jobs/result/{record['job_id']}",
        headers=_headers("tenant-b"),
    )
    approval = client.post(
        f"/api/audience-intelligence/jobs/approve/{record['job_id']}",
        headers=_headers("tenant-b"),
        json={"approver": "reviewer"},
    )

    assert status.status_code == 404
    assert result.status_code == 404
    assert approval.status_code == 404


def test_run_route_propagates_verified_tenant_to_background_job(
    monkeypatch,
    tmp_path: Path,
):
    _configure_auth(monkeypatch)
    store = AudienceJobStore(root_dir=tmp_path / "jobs")
    background_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(jobs, "job_store", store)
    monkeypatch.setattr(
        jobs,
        "_run_job_background",
        lambda job_id, tenant_id: background_calls.append(
            (job_id, tenant_id)
        ),
    )

    response = TestClient(app).post(
        "/api/audience-intelligence/jobs/run",
        headers=_headers("Tenant-A"),
        json={
            "prompt": "privacy-safe global audience",
            "source": "postgres",
        },
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert background_calls == [(job_id, "tenant-a")]
    assert store.get(job_id, tenant_id="tenant-a")["tenant_id"] == (
        "tenant-a"
    )

    hidden = TestClient(app).get(
        f"/api/audience-intelligence/jobs/status/{job_id}",
        headers=_headers("tenant-b"),
    )
    assert hidden.status_code == 404


def test_migration_enforces_tenant_key_rls_and_no_default():
    sql = Path(
        "migrations/0015_audience_job_tenant_isolation.sql"
    ).read_text(encoding="utf-8")

    assert "PRIMARY KEY (\n    tenant_id,\n    job_id\n)" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "FORCE ROW LEVEL SECURITY" in sql
    assert "current_setting('app.tenant_id', true)" in sql
    assert "explicit tenant assignment" in sql
    assert "tenant_id TEXT DEFAULT" not in sql


def test_postgres_job_write_sets_context_and_uses_tenant_key():
    store = AudienceJobStore(
        backend="postgres",
        db_url="postgresql://service:unused@database/audience",
    )
    engine = _FakeEngine()
    store._engine_instance = engine

    record = store.create_job(
        {"prompt": "privacy-safe audience"},
        tenant_id="Tenant-A",
    )

    context_calls = [
        call
        for call in engine.connection.calls
        if "set_config" in call[0]
    ]
    insert_calls = [
        call
        for call in engine.connection.calls
        if "INSERT INTO public.audience_jobs" in call[0]
    ]

    assert record["tenant_id"] == "tenant-a"
    assert context_calls[0][1] == {"tenant_id": "tenant-a"}
    assert insert_calls[0][1]["tenant_id"] == "tenant-a"
    assert "ON CONFLICT (tenant_id, job_id)" in insert_calls[0][0]


def test_postgres_job_read_uses_context_and_tenant_predicate():
    store = AudienceJobStore(
        backend="postgres",
        db_url="postgresql://service:unused@database/audience",
    )
    engine = _FakeEngine()
    store._engine_instance = engine

    with pytest.raises(FileNotFoundError):
        store.get("job_missing", tenant_id="tenant-a")

    select_calls = [
        call
        for call in engine.connection.calls
        if "SELECT tenant_id, job_id" in call[0]
    ]

    assert select_calls[0][1] == {
        "tenant_id": "tenant-a",
        "job_id": "job_missing",
    }
    assert "WHERE tenant_id = :tenant_id" in select_calls[0][0]
