from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.core.audience_request_context import (
    require_authenticated_audience_request,
)
from app.core.tenant_request_auth import tenant_signature
from app.main import app


def _headers(*, request_id: str | None = None) -> dict[str, str]:
    headers = {
        "X-Audience-API-Key": "test-key",
        "X-Audience-Tenant-Id": "tenant-a",
    }
    if request_id:
        headers["X-Request-Id"] = request_id
    return headers


def test_every_business_api_route_has_shared_request_boundary():
    public_paths = {
        "/",
        "/health",
        "/ready",
        "/ui/audience-agents",
        "/api/audience-intelligence/prompt/ui",
    }
    unprotected = []

    for route in app.routes:
        if not isinstance(route, APIRoute) or route.path in public_paths:
            continue
        dependency_calls = {
            dependency.call
            for dependency in route.dependant.dependencies
        }
        if require_authenticated_audience_request not in dependency_calls:
            unprotected.append(route.path)

    assert unprotected == []


def test_shared_boundary_rejects_missing_api_key_before_business_logic(
    monkeypatch,
):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")

    response = TestClient(app).post(
        "/chat",
        headers={"X-Audience-Tenant-Id": "tenant-a"},
        json={"message": "help"},
    )

    assert response.status_code == 401


def test_shared_boundary_requires_tenant_identity(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")

    response = TestClient(app).post(
        "/chat",
        headers={"X-Audience-API-Key": "test-key"},
        json={"message": "help"},
    )

    assert response.status_code == 422


def test_shared_boundary_propagates_caller_request_id(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setenv("PRODUCTION_MODE", "false")
    monkeypatch.setenv("APP_ENV", "local")

    response = TestClient(app).post(
        "/chat",
        headers=_headers(request_id="request-123"),
        json={"message": "help"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "request-123"


def test_shared_boundary_rejects_log_unsafe_request_id(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")

    response = TestClient(app).post(
        "/chat",
        headers=_headers(request_id="unsafe request id"),
        json={"message": "help"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid request identifier."


def test_production_hides_legacy_and_agent_routes(monkeypatch):
    secret = "test-tenant-secret"
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("PUNK_AI_TENANT_AUTH_SECRET", secret)

    headers = _headers()
    headers["X-Audience-Tenant-Signature"] = tenant_signature(
        "tenant-a",
        secret=secret,
    )
    client = TestClient(app)

    legacy = client.post(
        "/chat",
        headers=headers,
        json={"message": "help"},
    )
    agent = client.get(
        "/agents/postgres/tables",
        headers=headers,
    )

    assert legacy.status_code == 404
    assert agent.status_code == 404


def test_production_rejects_invalid_tenant_signature(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("PUNK_AI_TENANT_AUTH_SECRET", "test-tenant-secret")

    headers = _headers()
    headers["X-Audience-Tenant-Signature"] = "wrong-signature"
    response = TestClient(app).post(
        "/chat",
        headers=headers,
        json={"message": "help"},
    )

    assert response.status_code == 403


def test_health_remains_public_when_business_auth_is_required(monkeypatch):
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
