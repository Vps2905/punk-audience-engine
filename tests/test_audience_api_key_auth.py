from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.core.api_key_auth import require_audience_api_key


def test_api_key_accepts_matching_header(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setenv("AUDIENCE_API_KEY", "secret-key")

    assert require_audience_api_key(x_audience_api_key="secret-key", authorization=None) is True


def test_api_key_accepts_bearer_token(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setenv("AUDIENCE_API_KEY", "secret-key")

    assert require_audience_api_key(
        x_audience_api_key=None,
        authorization="Bearer secret-key",
    ) is True


def test_api_key_rejects_invalid_key(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setenv("AUDIENCE_API_KEY", "secret-key")

    with pytest.raises(HTTPException) as exc:
        require_audience_api_key(x_audience_api_key="bad-key", authorization=None)

    assert exc.value.status_code == 401


def test_api_key_fails_closed_when_missing_in_production(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("AUDIENCE_API_KEY", raising=False)
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setenv("AUDIENCE_API_KEY_CONFIG_FILE", str(tmp_path / "missing.env"))

    with pytest.raises(HTTPException) as exc:
        require_audience_api_key(x_audience_api_key="any-key", authorization=None)

    assert exc.value.status_code == 503


def test_api_key_can_be_disabled_only_when_configured(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("AUDIENCE_API_KEY", raising=False)
    monkeypatch.setenv("PRODUCTION_MODE", "false")
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "false")
    monkeypatch.setenv("AUDIENCE_API_KEY_CONFIG_FILE", str(tmp_path / "missing.env"))

    assert require_audience_api_key(x_audience_api_key=None, authorization=None) is True
