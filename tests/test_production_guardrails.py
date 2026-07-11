import numpy as np
import pytest

from app.core.production_guardrails import (
    demo_routes_allowed,
    local_file_storage_allowed,
    require_local_file_storage_allowed,
)
from app.services.vector_store_service import save_vector_store


def test_local_file_storage_allowed_in_local_mode(monkeypatch):
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.delenv("PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("ALLOW_LOCAL_FILE_STORAGE", raising=False)

    assert local_file_storage_allowed() is True
    require_local_file_storage_allowed("unit test")


def test_local_file_storage_blocked_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("ALLOW_LOCAL_FILE_STORAGE", raising=False)

    assert local_file_storage_allowed() is False

    with pytest.raises(RuntimeError, match="Production guardrail blocked local file storage"):
        require_local_file_storage_allowed("unit test")


def test_local_vector_store_blocked_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("VECTOR_BACKEND", "local")
    monkeypatch.delenv("PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("ALLOW_LOCAL_FILE_STORAGE", raising=False)

    with pytest.raises(RuntimeError, match="Production guardrail blocked local file storage"):
        save_vector_store(
            job_id="prod_guardrail_block_test",
            vectors=np.zeros((1, 2)),
            metadata=[{"safe": "ok"}],
            model_info={"backend": "unit_test"},
        )


def test_demo_routes_blocked_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("ALLOW_DEMO_ROUTES", raising=False)

    assert demo_routes_allowed() is False


def test_demo_routes_can_be_explicitly_allowed(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("ALLOW_DEMO_ROUTES", "true")

    assert demo_routes_allowed() is True
