import pytest
from fastapi import HTTPException

from app.core.api_key_auth import require_audience_api_key


def test_accepts_x_audience_api_key(monkeypatch):
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")

    assert require_audience_api_key(x_audience_api_key="test-key") is True


def test_accepts_x_api_key_alias(monkeypatch):
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")

    assert require_audience_api_key(x_api_key="test-key") is True


def test_rejects_wrong_api_key(monkeypatch):
    monkeypatch.setenv("REQUIRE_AUDIENCE_API_KEY", "true")
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")

    with pytest.raises(HTTPException):
        require_audience_api_key(x_api_key="wrong-key")
