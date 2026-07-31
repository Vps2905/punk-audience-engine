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


def test_enabled_provider_gateway_requires_production_object_store_config(
    monkeypatch,
):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("ECHO_DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("HASH_SECRET", "test-hash-secret")
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "test-audience-salt")
    monkeypatch.setenv("VECTOR_BACKEND", "postgres_array")
    monkeypatch.setenv("EMBEDDING_BACKEND", "sklearn_hashing")
    monkeypatch.setenv("ALL_SAFE_COHORT_EMBEDDING_STORE", "postgres")
    monkeypatch.setenv("PROVIDER_GATEWAY_ENABLED", "true")
    monkeypatch.delenv("PROVIDER_INGESTION_DATABASE_URL", raising=False)
    monkeypatch.delenv("PROVIDER_S3_REGION", raising=False)
    monkeypatch.delenv("PROVIDER_SQS_QUEUE_URL", raising=False)
    monkeypatch.delenv("PROVIDER_SQS_DLQ_URL", raising=False)
    monkeypatch.delenv("CANONICAL_S3_BUCKET", raising=False)
    monkeypatch.delenv("CANONICAL_S3_SERVER_SIDE_ENCRYPTION", raising=False)
    monkeypatch.delenv("AUDIENCE_DP_SEED_SECRET", raising=False)
    monkeypatch.delenv("AUDIENCE_TOKENIZATION_HMAC_KEY", raising=False)

    result = validate_environment()

    assert result.ok is False
    assert "PROVIDER_INGESTION_DATABASE_URL" in result.missing_required
    assert "PROVIDER_S3_REGION" in result.missing_required
    assert "PROVIDER_SQS_QUEUE_URL" in result.missing_required
    assert "PROVIDER_SQS_DLQ_URL" in result.missing_required
    assert "CANONICAL_S3_BUCKET" in result.missing_required
    assert "CANONICAL_S3_SERVER_SIDE_ENCRYPTION" in result.missing_required
    assert "AUDIENCE_DP_SEED_SECRET" in result.missing_required
    assert "AUDIENCE_TOKENIZATION_HMAC_KEY" in result.missing_required


def test_enabled_provider_gateway_accepts_complete_production_config(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv("ECHO_DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("HASH_SECRET", "test-hash-secret")
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "test-audience-salt")
    monkeypatch.setenv("VECTOR_BACKEND", "postgres_array")
    monkeypatch.setenv("EMBEDDING_BACKEND", "sklearn_hashing")
    monkeypatch.setenv("ALL_SAFE_COHORT_EMBEDDING_STORE", "postgres")
    monkeypatch.setenv("PROVIDER_GATEWAY_ENABLED", "true")
    monkeypatch.setenv(
        "PROVIDER_INGESTION_DATABASE_URL",
        "postgresql://user:pass@localhost:5432/db",
    )
    monkeypatch.setenv("PROVIDER_S3_REGION", "test-region-1")
    monkeypatch.setenv(
        "PROVIDER_SQS_QUEUE_URL",
        "https://sqs.test/queue",
    )
    monkeypatch.setenv(
        "PROVIDER_SQS_DLQ_URL",
        "https://sqs.test/dlq",
    )
    monkeypatch.setenv("CANONICAL_S3_BUCKET", "canonical-test-bucket")
    monkeypatch.setenv("CANONICAL_S3_SERVER_SIDE_ENCRYPTION", "AES256")
    monkeypatch.setenv("AUDIENCE_DP_SEED_SECRET", "test-dp-seed-secret")
    monkeypatch.setenv("AUDIENCE_TOKENIZATION_HMAC_KEY", "test-hmac-key")

    result = validate_environment()

    assert result.ok is True
    assert result.missing_required == []


def test_distributed_provider_processing_requires_state_machine(
    monkeypatch,
):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv(
        "ECHO_DATABASE_URL",
        "postgresql://user:pass@localhost:5432/db",
    )
    monkeypatch.setenv("HASH_SECRET", "test-hash-secret")
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "test-audience-salt")
    monkeypatch.setenv("VECTOR_BACKEND", "postgres_array")
    monkeypatch.setenv("EMBEDDING_BACKEND", "sklearn_hashing")
    monkeypatch.setenv("ALL_SAFE_COHORT_EMBEDDING_STORE", "postgres")
    monkeypatch.setenv("PROVIDER_GATEWAY_ENABLED", "true")
    monkeypatch.setenv(
        "PROVIDER_INGESTION_DATABASE_URL",
        "postgresql://user:pass@localhost:5432/db",
    )
    monkeypatch.setenv("PROVIDER_S3_REGION", "test-region-1")
    monkeypatch.setenv("PROVIDER_SQS_QUEUE_URL", "https://sqs.test/queue")
    monkeypatch.setenv("PROVIDER_SQS_DLQ_URL", "https://sqs.test/dlq")
    monkeypatch.setenv("CANONICAL_S3_BUCKET", "canonical-test-bucket")
    monkeypatch.setenv("CANONICAL_S3_SERVER_SIDE_ENCRYPTION", "AES256")
    monkeypatch.setenv("AUDIENCE_DP_SEED_SECRET", "test-dp-seed-secret")
    monkeypatch.setenv("AUDIENCE_TOKENIZATION_HMAC_KEY", "test-hmac-key")
    monkeypatch.setenv(
        "PROVIDER_DISTRIBUTED_PROCESSING_ENABLED",
        "true",
    )
    monkeypatch.delenv(
        "PROVIDER_DISTRIBUTED_STATE_MACHINE_ARN",
        raising=False,
    )

    result = validate_environment()

    assert result.ok is False
    assert (
        "PROVIDER_DISTRIBUTED_STATE_MACHINE_ARN"
        in result.missing_required
    )


def test_distributed_provider_processing_accepts_complete_data_plane_config(
    monkeypatch,
):
    values = {
        "PRODUCTION_MODE": "true",
        "APP_ENV": "production",
        "AUDIENCE_API_KEY": "test-key",
        "ECHO_DATABASE_URL": (
            "postgresql://user:pass@localhost:5432/db"
        ),
        "HASH_SECRET": "test-hash-secret",
        "AUDIENCE_HASH_SALT": "test-audience-salt",
        "VECTOR_BACKEND": "postgres_array",
        "EMBEDDING_BACKEND": "sklearn_hashing",
        "ALL_SAFE_COHORT_EMBEDDING_STORE": "postgres",
        "PROVIDER_GATEWAY_ENABLED": "true",
        "PROVIDER_INGESTION_DATABASE_URL": (
            "postgresql://user:pass@localhost:5432/db"
        ),
        "PROVIDER_S3_REGION": "test-region-1",
        "PROVIDER_SQS_QUEUE_URL": "https://sqs.test/queue",
        "PROVIDER_SQS_DLQ_URL": "https://sqs.test/dlq",
        "CANONICAL_S3_BUCKET": "canonical-test-bucket",
        "CANONICAL_S3_SERVER_SIDE_ENCRYPTION": "AES256",
        "AUDIENCE_DP_SEED_SECRET": "test-dp-seed-secret",
        "AUDIENCE_TOKENIZATION_HMAC_KEY": "test-hmac-key",
        "PROVIDER_DISTRIBUTED_PROCESSING_ENABLED": "true",
        "PROVIDER_DISTRIBUTED_STATE_MACHINE_ARN": (
            "arn:aws:states:test:123:stateMachine:provider"
        ),
        "PROVIDER_DISTRIBUTED_CONTROL_PLANE_LAMBDA_ARN": (
            "arn:aws:lambda:test:123:function:provider-control"
        ),
        "PROVIDER_DISTRIBUTED_GLUE_JOB_NAME": "provider-privacy",
        "PROVIDER_DISTRIBUTED_STAGING_BUCKET": "staging-bucket",
        "PROVIDER_DISTRIBUTED_STAGING_ENCRYPTION": "aws:kms",
        "PROVIDER_DISTRIBUTED_STAGING_KMS_KEY_ID": (
            "arn:aws:kms:test:123:key/test"
        ),
        "PROVIDER_INGESTION_DATABASE_SECRET_ARN": (
            "arn:aws:secretsmanager:test:123:secret:database"
        ),
        "PROVIDER_DISTRIBUTED_TOKENIZATION_SECRET_ARN": (
            "arn:aws:secretsmanager:test:123:secret:token"
        ),
        "PROVIDER_DISTRIBUTED_DP_SEED_SECRET_ARN": (
            "arn:aws:secretsmanager:test:123:secret:dp"
        ),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    result = validate_environment()

    assert result.ok is True
    assert result.missing_required == []


def test_enabled_phase2_requires_feature_database_and_real_pgvector(
    monkeypatch,
):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv(
        "ECHO_DATABASE_URL",
        "postgresql://user:pass@localhost:5432/db",
    )
    monkeypatch.setenv("HASH_SECRET", "test-hash-secret")
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "test-audience-salt")
    monkeypatch.setenv("VECTOR_BACKEND", "postgres_array")
    monkeypatch.setenv("EMBEDDING_BACKEND", "sklearn_hashing")
    monkeypatch.setenv("ALL_SAFE_COHORT_EMBEDDING_STORE", "postgres")
    monkeypatch.setenv("PHASE2_FEATURES_ENABLED", "true")
    monkeypatch.delenv("AUDIENCE_FEATURE_DATABASE_URL", raising=False)
    monkeypatch.delenv("AUDIENCE_PROPOSAL_DATABASE_URL", raising=False)
    monkeypatch.setenv("PGVECTOR_DIMENSION", "128")
    monkeypatch.setenv("PHASE2_RETRIEVAL_BACKEND", "postgres_array")
    monkeypatch.delenv("PUNK_AI_TENANT_AUTH_SECRET", raising=False)

    result = validate_environment()

    assert result.ok is False
    assert "AUDIENCE_FEATURE_DATABASE_URL" in result.missing_required
    assert "AUDIENCE_PROPOSAL_DATABASE_URL" in result.missing_required
    assert "PGVECTOR_DIMENSION(384)" in result.missing_required
    assert (
        "PHASE2_RETRIEVAL_BACKEND(pgvector)"
        in result.missing_required
    )
    assert "PUNK_AI_TENANT_AUTH_SECRET" in result.missing_required


def test_enabled_phase2_accepts_complete_production_config(monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUDIENCE_API_KEY", "test-key")
    monkeypatch.setenv(
        "ECHO_DATABASE_URL",
        "postgresql://user:pass@localhost:5432/db",
    )
    monkeypatch.setenv("HASH_SECRET", "test-hash-secret")
    monkeypatch.setenv("AUDIENCE_HASH_SALT", "test-audience-salt")
    monkeypatch.setenv("VECTOR_BACKEND", "postgres_array")
    monkeypatch.setenv("EMBEDDING_BACKEND", "sklearn_hashing")
    monkeypatch.setenv("ALL_SAFE_COHORT_EMBEDDING_STORE", "pgvector")
    monkeypatch.setenv("PHASE2_FEATURES_ENABLED", "true")
    monkeypatch.setenv(
        "AUDIENCE_FEATURE_DATABASE_URL",
        "postgresql://user:pass@localhost:5432/db",
    )
    monkeypatch.setenv(
        "AUDIENCE_PROPOSAL_DATABASE_URL",
        "postgresql://proposal:pass@localhost:5432/db",
    )
    monkeypatch.setenv("PGVECTOR_DIMENSION", "384")
    monkeypatch.setenv("PHASE2_RETRIEVAL_BACKEND", "pgvector")
    monkeypatch.setenv(
        "PUNK_AI_TENANT_AUTH_SECRET",
        "test-tenant-auth-secret",
    )

    result = validate_environment()

    assert result.ok is True
    assert result.missing_required == []


def _set_production_feature_build_environment(monkeypatch):
    values = {
        "PRODUCTION_MODE": "true",
        "APP_ENV": "production",
        "AUDIENCE_API_KEY": "test-key",
        "ECHO_DATABASE_URL": (
            "postgresql://user:pass@localhost:5432/db"
        ),
        "HASH_SECRET": "test-hash-secret",
        "AUDIENCE_HASH_SALT": "test-audience-salt",
        "VECTOR_BACKEND": "pgvector",
        "EMBEDDING_BACKEND": "sklearn_hashing",
        "ALL_SAFE_COHORT_EMBEDDING_STORE": "pgvector",
        "PHASE2_FEATURES_ENABLED": "true",
        "AUDIENCE_FEATURE_DATABASE_URL": (
            "postgresql://reader:pass@localhost:5432/features"
        ),
        "AUDIENCE_FEATURE_WRITER_DATABASE_URL": (
            "postgresql://writer:pass@localhost:5432/features"
        ),
        "AUDIENCE_PROPOSAL_DATABASE_URL": (
            "postgresql://proposal:pass@localhost:5432/features"
        ),
        "PGVECTOR_DIMENSION": "384",
        "PHASE2_RETRIEVAL_BACKEND": "pgvector",
        "PUNK_AI_TENANT_AUTH_SECRET": "test-tenant-secret",
        "CANONICAL_S3_BUCKET": "canonical-test-bucket",
        "PRODUCTION_FEATURE_BUILDS_ENABLED": "true",
        "FEATURE_EMBEDDING_BACKEND": "sentence-transformers",
        "FEATURE_EMBEDDING_MODEL_NAME": "organization/model",
        "FEATURE_EMBEDDING_MODEL_REVISION": "immutable-revision",
        "FEATURE_EMBEDDING_DIMENSION": "384",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_production_feature_builds_require_pinned_model_and_writer(
    monkeypatch,
):
    _set_production_feature_build_environment(monkeypatch)
    monkeypatch.delenv(
        "AUDIENCE_FEATURE_WRITER_DATABASE_URL",
        raising=False,
    )
    monkeypatch.setenv("FEATURE_EMBEDDING_MODEL_REVISION", "main")
    monkeypatch.setenv("FEATURE_EMBEDDING_DIMENSION", "768")

    result = validate_environment()

    assert result.ok is False
    assert (
        "AUDIENCE_FEATURE_WRITER_DATABASE_URL"
        in result.missing_required
    )
    assert (
        "FEATURE_EMBEDDING_MODEL_REVISION(immutable)"
        in result.missing_required
    )
    assert (
        "FEATURE_EMBEDDING_DIMENSION(384)"
        in result.missing_required
    )


def test_production_feature_builds_accept_complete_model_config(
    monkeypatch,
):
    _set_production_feature_build_environment(monkeypatch)

    result = validate_environment()

    assert result.ok is True
    assert result.missing_required == []
    assert result.env_presence["PRODUCTION_FEATURE_BUILDS_ENABLED"]
