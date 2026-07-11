from app.core.environment_validation import validate_environment


def test_local_environment_does_not_require_production_secrets(monkeypatch):
    monkeypatch.delenv("PRODUCTION_MODE", raising=False)
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.delenv("AUDIENCE_API_KEY", raising=False)
    monkeypatch.delenv("ECHO_DATABASE_URL", raising=False)
    monkeypatch.delenv("HASH_SECRET", raising=False)
    monkeypatch.delenv("AUDIENCE_HASH_SALT", raising=False)

    result = validate_environment()

    assert result.ok is True
    assert result.missing_required == []


def test_production_requires_api_key_echo_db_and_hash_secrets(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("AUDIENCE_API_KEY", raising=False)
    monkeypatch.delenv("ECHO_DATABASE_URL", raising=False)
    monkeypatch.delenv("HASH_SECRET", raising=False)
    monkeypatch.delenv("AUDIENCE_HASH_SALT", raising=False)

    result = validate_environment()

    assert result.ok is False
    assert "AUDIENCE_API_KEY" in result.missing_required
    assert "ECHO_DATABASE_URL" in result.missing_required
    assert "HASH_SECRET" in result.missing_required
    assert "AUDIENCE_HASH_SALT" in result.missing_required


def test_production_passes_when_required_envs_exist(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("ECHO_DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("HASH_SECRET", "test-hash-secret")
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "test-audience-salt")

    result = validate_environment()

    assert result.ok is True
    assert result.missing_required == []
