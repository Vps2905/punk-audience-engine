from app.core import audience_job_store as module
from app.core.audience_job_store import AudienceJobStore


def _store():
    return AudienceJobStore(
        backend="local",
        root_dir="/tmp/test_audience_job_driver",
    )


def test_connection_url_prefers_psycopg(monkeypatch):
    def fake_find_spec(name):
        if name == "psycopg":
            return object()
        return None

    monkeypatch.setattr(module, "find_spec", fake_find_spec)

    url = _store()._connection_url(
        "postgresql+asyncpg://user:pass@host/db"
    )

    assert url == "postgresql+psycopg://user:pass@host/db"


def test_connection_url_falls_back_to_psycopg2(monkeypatch):
    def fake_find_spec(name):
        if name == "psycopg2":
            return object()
        return None

    monkeypatch.setattr(module, "find_spec", fake_find_spec)

    url = _store()._connection_url(
        "postgres://user:pass@host/db"
    )

    assert url == "postgresql+psycopg2://user:pass@host/db"
