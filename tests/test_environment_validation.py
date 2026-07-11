from app.core.environment_validation import validate_environment


def test_local_environment_does_not_require_production_secrets(monkeypatch):
    monkeypatch.delenv("PRODUCTION_MODE", raising=False)
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.delenv("AUDIENCE_API_KEY", raising=False)
    monkeypatch.delenv("ECHO_DATABASE_URL", raising=False)
    monkeypatch.delenv("HASH_SECRET", raising=False)
    monkeypatch.delenv("AUDIENCE_HASH_SALT", raising=False)
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("ALL_SAFE_COHORT_EMBEDDING_STORE", raising=False)

    result = validate_environment()

    assert result.ok is True
    assert result.missing_required == []


def test_production_requires_api_key_echo_db_hash_and_embedding_envs(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("AUDIENCE_API_KEY", raising=False)
    monkeypatch.delenv("ECHO_DATABASE_URL", raising=False)
    monkeypatch.delenv("HASH_SECRET", raising=False)
    monkeypatch.delenv("AUDIENCE_HASH_SALT", raising=False)
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("ALL_SAFE_COHORT_EMBEDDING_STORE", raising=False)

    result = validate_environment()

    assert result.ok is False
    assert "AUDIENCE_API_KEY" in result.missing_required
    assert "ECHO_DATABASE_URL" in result.missing_required
    assert "HASH_SECRET" in result.missing_required
    assert "AUDIENCE_HASH_SALT" in result.missing_required
    assert "VECTOR_BACKEND" in result.missing_required
    assert "EMBEDDING_BACKEND" in result.missing_required
    assert "ALL_SAFE_COHORT_EMBEDDING_STORE" in result.missing_required


def test_production_passes_when_required_envs_exist(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("ECHO_DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("HASH_SECRET", "test-hash-secret")
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "test-audience-salt")
    monkeypatch.setenv("VECTOR_BACKEND", "postgres_array")
    monkeypatch.setenv("EMBEDDING_BACKEND", "sklearn_hashing")
    monkeypatch.setenv("ALL_SAFE_COHORT_EMBEDDING_STORE", "postgres")

    result = validate_environment()

    assert result.ok is True
    assert result.missing_required == []


def test_production_rejects_local_vector_backend(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("ECHO_DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("HASH_SECRET", "test-hash-secret")
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "test-audience-salt")
    monkeypatch.setenv("VECTOR_BACKEND", "local")
    monkeypatch.setenv("EMBEDDING_BACKEND", "sklearn_hashing")
    monkeypatch.setenv("ALL_SAFE_COHORT_EMBEDDING_STORE", "postgres")

    result = validate_environment()

    assert result.ok is False
    assert "VECTOR_BACKEND(postgres_array|pgvector)" in result.missing_required


def test_production_rejects_tfidf_embedding_backend(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("ECHO_DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("HASH_SECRET", "test-hash-secret")
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "test-audience-salt")
    monkeypatch.setenv("VECTOR_BACKEND", "postgres_array")
    monkeypatch.setenv("EMBEDDING_BACKEND", "tfidf")
    monkeypatch.setenv("ALL_SAFE_COHORT_EMBEDDING_STORE", "postgres")

    result = validate_environment()

    assert result.ok is False
    assert "EMBEDDING_BACKEND(sklearn_hashing|external_embedding_service)" in result.missing_required


def test_production_rejects_local_all_safe_embedding_store(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("ECHO_DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("HASH_SECRET", "test-hash-secret")
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "test-audience-salt")
    monkeypatch.setenv("VECTOR_BACKEND", "postgres_array")
    monkeypatch.setenv("EMBEDDING_BACKEND", "sklearn_hashing")
    monkeypatch.setenv("ALL_SAFE_COHORT_EMBEDDING_STORE", "local")

    result = validate_environment()

    assert result.ok is False
    assert "ALL_SAFE_COHORT_EMBEDDING_STORE(postgres)" in result.missing_required
