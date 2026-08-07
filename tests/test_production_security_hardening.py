import json
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.http_security import production_api_docs_enabled
from app.main import app
from app.models.production_security_contracts import (
    ProductionSecurityAssessmentRequest,
)
from app.services.production_security_hardening_service import (
    ProductionSecurityManualReviewService,
    ProductionSecurityPostureService,
)


TENANT = "tenant_a"
API_SECRET = "Aa1!Bb2@Cc3#Dd4$Ee5%Ff6^Gg7&Hh8*"
TENANT_SECRET = "Zz9!Yy8@Xx7#Ww6$Vv5%Uu4^Tt3&Ss2*"


def complete_environment():
    return {
        "APP_ENV": "production",
        "PRODUCTION_MODE": "true",
        "REQUIRE_AUDIENCE_API_KEY": "true",
        "AUDIENCE_API_KEY": API_SECRET,
        "REQUIRE_PUNK_AI_TENANT_SIGNATURE": "true",
        "PUNK_AI_TENANT_AUTH_SECRET": TENANT_SECRET,
        "ALLOW_LOCAL_FILE_STORAGE": "false",
        "ALLOW_DEMO_ROUTES": "false",
        "EXPOSE_API_DOCS": "false",
        "SECURITY_HEADERS_ENABLED": "true",
        "PRODUCTION_ALLOWED_HOSTS": "api.example.com",
        "PRODUCTION_CORS_ORIGINS": "https://app.example.com",
        "MAX_REQUEST_BODY_BYTES": "10485760",
        "REQUIRE_DATABASE_TLS": "true",
        "REQUIRE_OUTBOUND_TLS": "true",
        "SECRETS_INJECTED_BY_SECRET_MANAGER": "true",
        "CONTAINER_RUNTIME_NON_ROOT": "true",
        "CONTAINER_ROOT_FILESYSTEM_READ_ONLY": "true",
        "CONTAINER_NO_NEW_PRIVILEGES": "true",
        "DEBUG": "false",
    }


def request():
    return ProductionSecurityAssessmentRequest(
        tenant_id=TENANT,
        assessment_id="security-assessment-1",
        execution_mode="production",
    )


def full_evidence():
    posture = ProductionSecurityPostureService().assess(
        request=request(),
        environment=complete_environment(),
    )
    review = ProductionSecurityManualReviewService().review(
        posture_report=posture,
        manual_review={
            "decision": "approved_for_preproduction_review",
            "review_reference": "security-review-1",
        },
    )
    return posture, review


def test_complete_security_posture_passes_without_serializing_secrets():
    posture, _ = full_evidence()
    serialized = json.dumps(posture)
    assert posture["posture_status"] == "pass"
    assert posture["failed_control_count"] == 0
    assert API_SECRET not in serialized
    assert TENANT_SECRET not in serialized
    assert posture["safety"]["secret_values_stored"] is False
    assert posture["safety"]["production_release_authorized"] is False


def test_weak_secrets_fail_closed():
    environment = complete_environment()
    environment["AUDIENCE_API_KEY"] = "change_me"
    environment["PUNK_AI_TENANT_AUTH_SECRET"] = "short"
    posture = ProductionSecurityPostureService().assess(
        request=request(), environment=environment
    )
    assert posture["posture_status"] == "fail_closed"
    assert "api_key_strength_attested" in posture["failed_control_codes"]
    assert "tenant_signing_secret_strength_attested" in posture[
        "failed_control_codes"
    ]


def test_wildcard_host_or_cors_and_release_flags_fail_closed():
    environment = complete_environment()
    environment["PRODUCTION_ALLOWED_HOSTS"] = "*"
    environment["PRODUCTION_CORS_ORIGINS"] = "*"
    environment["MODULE4_PRODUCTION_ROUTING_ENABLED"] = "true"
    posture = ProductionSecurityPostureService().assess(
        request=request(), environment=environment
    )
    assert posture["posture_status"] == "fail_closed"
    assert set(posture["failed_control_codes"]) >= {
        "cors_origins_restricted",
        "production_release_flags_disabled",
        "trusted_hosts_restricted",
    }


def test_failed_posture_cannot_receive_preproduction_approval():
    posture = ProductionSecurityPostureService().assess(
        request=request(), environment={}
    )
    with pytest.raises(ValueError, match="cannot be approved"):
        ProductionSecurityManualReviewService().review(
            posture_report=posture,
            manual_review={
                "decision": "approved_for_preproduction_review",
                "review_reference": "invalid-approval",
            },
        )


def test_manual_review_is_preproduction_only_and_tamper_evident():
    _, review = full_evidence()
    assert review["manual_review"]["production_release_authorized"] is False
    assert review["safety"]["production_release_authorized"] is False
    tampered = deepcopy(review)
    tampered["manual_review"]["production_release_authorized"] = True
    with pytest.raises(ValueError, match="cannot set"):
        ProductionSecurityManualReviewService().validate_report(tampered)


def test_security_status_accepts_complete_safe_evidence_chain(tmp_path):
    from app.services.production_security_status_service import (
        ProductionSecurityStatusService,
    )

    posture, review = full_evidence()
    environment = {
        "SECURITY_HARDENING_ENABLED": "true",
    }
    for key, value in {
        "SECURITY_POSTURE_EVIDENCE_PATH": posture,
        "SECURITY_REVIEW_EVIDENCE_PATH": review,
    }.items():
        path = tmp_path / f"{key.lower()}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        environment[key] = str(path)
    status = ProductionSecurityStatusService(environment=environment).status()
    assert status["status"] == "security_engineering_evidence_ready"
    assert status["security_engineering_evidence_ready"] is True
    assert status["live_production_certified"] is False
    assert all(status["components"].values())


def test_security_status_blocks_production_release_flag():
    from app.services.production_security_status_service import (
        ProductionSecurityStatusService,
    )

    status = ProductionSecurityStatusService(environment={
        "SECURITY_PRODUCTION_RELEASE_ENABLED": "true",
    }).status()
    assert status["status"] == (
        "unsafe_configuration_production_release_blocked"
    )
    assert status["live_production_certified"] is False


def test_production_http_headers_and_host_allowlist(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("PRODUCTION_ALLOWED_HOSTS", "testserver")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "max-age=31536000" in response.headers["strict-transport-security"]
    blocked = client.get("/health", headers={"host": "evil.example"})
    assert blocked.status_code == 400


def test_declared_oversized_request_is_rejected(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("PRODUCTION_ALLOWED_HOSTS", "testserver")
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "1024")
    response = TestClient(app).post(
        "/health",
        content=b"x" * 2048,
        headers={"content-length": "2048"},
    )
    assert response.status_code == 413


def test_actual_body_limit_rejects_false_small_content_length(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("PRODUCTION_ALLOWED_HOSTS", "testserver")
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "1024")
    response = TestClient(app).post(
        "/health",
        content=b"x" * 2048,
        headers={"content-length": "10"},
    )
    assert response.status_code == 413


def test_api_docs_default_to_hidden_in_production(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.delenv("EXPOSE_API_DOCS", raising=False)
    assert production_api_docs_enabled() is False
    monkeypatch.setenv("EXPOSE_API_DOCS", "true")
    assert production_api_docs_enabled() is True


def test_container_configuration_is_non_root_and_restricted():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "USER 10001:10001" in dockerfile
    assert "useradd --system --uid 10001" in dockerfile
    assert "read_only: true" in compose
    assert "no-new-privileges:true" in compose
    assert "cap_drop:" in compose
    assert "- ALL" in compose


def test_security_migration_is_immutable_tenant_scoped_and_non_secret():
    sql = Path(
        "migrations/0025_production_security_hardening_evidence.sql"
    ).read_text(encoding="utf-8")
    for fragment in (
        "secret_values_stored = FALSE",
        "credentials_returned = FALSE",
        "environment_values_returned = FALSE",
        "security_configuration_mutated = FALSE",
        "automatic_remediation_performed = FALSE",
        "production_release_authorized = FALSE",
        "manual_approval_required = TRUE",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "current_setting(''app.tenant_id'', true)",
        "BEFORE UPDATE OR DELETE",
        "REVOKE ALL",
    ):
        assert fragment in sql


def test_security_router_and_safe_defaults_are_registered():
    main = Path("app/main.py").read_text(encoding="utf-8")
    env = Path(".env.example").read_text(encoding="utf-8")
    assert "audience_intelligence_security_status_router" in main
    assert "AUDIENCE_API_KEY=\n" in env
    assert "SECURITY_PRODUCTION_RELEASE_ENABLED=false" in env
    assert "SECRETS_INJECTED_BY_SECRET_MANAGER=false" in env
    assert "EXPOSE_API_DOCS=false" in env
