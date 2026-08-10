from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

import app.api.audience_intelligence_prompt as prompt_api
from app.main import app


def _enable_local_workspace(monkeypatch) -> None:
    monkeypatch.setenv("AUDIENCE_LOCAL_UI_ENABLED", "true")
    monkeypatch.setenv(
        "AUDIENCE_LOCAL_UI_TENANT_ID",
        "punk_internal",
    )
    monkeypatch.setenv("PRODUCTION_MODE", "false")
    monkeypatch.setenv("APP_ENV", "local")


def test_workspace_is_polished_external_asset_ui(monkeypatch):
    _enable_local_workspace(monkeypatch)
    client = TestClient(app, client=("127.0.0.1", 50000))

    response = client.get("/audience-workspace")

    assert response.status_code == 200
    assert "Campaign strategy workspace" in response.text
    assert "Five-module analysis" in response.text
    assert "Audience API Key" not in response.text
    assert "Tenant Signature" not in response.text
    assert "safe_cohort_path" not in response.text
    assert "<style" not in response.text
    assert "<script>" not in response.text
    assert "style-src 'self'" in response.headers[
        "Content-Security-Policy"
    ]
    assert "script-src 'self'" in response.headers[
        "Content-Security-Policy"
    ]
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]

    styles = client.get(
        "/audience-workspace/assets/styles.css"
    )
    script = client.get("/audience-workspace/assets/app.js")

    assert styles.status_code == 200
    assert styles.headers["content-type"].startswith("text/css")
    assert script.status_code == 200
    assert "X-Punk-Local-UI" in script.text


def test_legacy_prompt_ui_redirects_to_workspace(monkeypatch):
    _enable_local_workspace(monkeypatch)
    client = TestClient(
        app,
        client=("127.0.0.1", 50000),
        follow_redirects=False,
    )

    response = client.get(
        "/api/audience-intelligence/prompt/ui"
    )

    assert response.status_code == 307
    assert response.headers["location"] == "/audience-workspace"


def test_workspace_is_hidden_in_production(monkeypatch):
    monkeypatch.setenv("AUDIENCE_LOCAL_UI_ENABLED", "true")
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")

    response = TestClient(
        app,
        client=("127.0.0.1", 50000),
    ).get("/audience-workspace")

    assert response.status_code == 404


def test_workspace_is_hidden_from_non_loopback_clients(monkeypatch):
    _enable_local_workspace(monkeypatch)

    response = TestClient(
        app,
        client=("10.42.0.25", 50000),
    ).get("/audience-workspace")

    assert response.status_code == 404


def test_local_run_requires_session_and_csrf_header(monkeypatch):
    _enable_local_workspace(monkeypatch)
    no_session = TestClient(
        app,
        client=("127.0.0.1", 50000),
    )

    missing_header = no_session.post(
        "/api/audience-intelligence/prompt/ui/run",
        json={"prompt": "Explain available audience coverage"},
    )
    missing_session = no_session.post(
        "/api/audience-intelligence/prompt/ui/run",
        headers={"X-Punk-Local-UI": "1"},
        json={"prompt": "Explain available audience coverage"},
    )

    assert missing_header.status_code == 403
    assert missing_session.status_code == 401


def test_local_run_forces_server_side_safe_defaults(monkeypatch):
    _enable_local_workspace(monkeypatch)
    captured: dict[str, Any] = {}

    def fake_execute(request, context):
        captured["request"] = request
        captured["context"] = context
        return {"status": "captured"}

    monkeypatch.setattr(
        prompt_api,
        "_execute_audience_prompt",
        fake_execute,
    )
    client = TestClient(app, client=("127.0.0.1", 50000))
    assert client.get("/audience-workspace").status_code == 200

    response = client.post(
        "/api/audience-intelligence/prompt/ui/run",
        headers={"X-Punk-Local-UI": "1"},
        json={"prompt": "Compare safe audience strategies"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "captured"}
    request = captured["request"]
    context = captured["context"]
    assert request.source == "postgres"
    assert request.safe_cohort_path is None
    assert request.output_root == "data/prompt_runs"
    assert request.approval_required is True
    assert request.k_min == 1000
    assert context.tenant_id == "punk_internal"
    assert context.request_id.startswith("local_ui_")
