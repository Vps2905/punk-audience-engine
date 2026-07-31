from __future__ import annotations

import os
from dataclasses import dataclass


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_feature_backend(value: str | None) -> str:
    return str(value or "").strip().lower().replace("-", "_")


@dataclass(frozen=True)
class EnvironmentValidationResult:
    ok: bool
    mode: str
    missing_required: list[str]
    warnings: list[str]
    env_presence: dict[str, bool]


def is_production_mode() -> bool:
    return _truthy(os.getenv("PRODUCTION_MODE")) or os.getenv("APP_ENV", "").strip().lower() == "production"


def validate_environment() -> EnvironmentValidationResult:
    production = is_production_mode()
    mode = "production" if production else os.getenv("APP_ENV", "local")

    required_in_production = [
        "AUDIENCE_API_KEY",
        "ECHO_DATABASE_URL",
        "HASH_SECRET",
        "AUDIENCE_HASH_SALT",
        "VECTOR_BACKEND",
        "EMBEDDING_BACKEND",
        "ALL_SAFE_COHORT_EMBEDDING_STORE",
    ]

    optional_but_recommended = [
        "OPENROUTER_API_KEY",
    ]

    missing_required: list[str] = []
    warnings: list[str] = []

    for key in required_in_production:
        if production and not os.getenv(key):
            missing_required.append(key)

    for key in optional_but_recommended:
        if production and not os.getenv(key):
            warnings.append(f"{key} is not configured.")

    provider_gateway_enabled = _truthy(os.getenv("PROVIDER_GATEWAY_ENABLED"))
    distributed_processing_enabled = _truthy(
        os.getenv("PROVIDER_DISTRIBUTED_PROCESSING_ENABLED")
    )
    if production and distributed_processing_enabled:
        if not provider_gateway_enabled:
            missing_required.append("PROVIDER_GATEWAY_ENABLED(true)")
        for key in [
            "PROVIDER_DISTRIBUTED_STATE_MACHINE_ARN",
            "PROVIDER_DISTRIBUTED_CONTROL_PLANE_LAMBDA_ARN",
            "PROVIDER_DISTRIBUTED_GLUE_JOB_NAME",
            "PROVIDER_DISTRIBUTED_STAGING_BUCKET",
            "PROVIDER_DISTRIBUTED_STAGING_ENCRYPTION",
            "PROVIDER_INGESTION_DATABASE_SECRET_ARN",
            "PROVIDER_DISTRIBUTED_TOKENIZATION_SECRET_ARN",
            "PROVIDER_DISTRIBUTED_DP_SEED_SECRET_ARN",
        ]:
            if not os.getenv(key):
                missing_required.append(key)
        staging_encryption = os.getenv(
            "PROVIDER_DISTRIBUTED_STAGING_ENCRYPTION",
            "",
        ).strip()
        if staging_encryption not in {
            "AES256",
            "aws:kms",
            "aws:kms:dsse",
        }:
            missing_required.append(
                "PROVIDER_DISTRIBUTED_STAGING_ENCRYPTION"
                "(AES256|aws:kms|aws:kms:dsse)"
            )
        if (
            staging_encryption in {"aws:kms", "aws:kms:dsse"}
            and not os.getenv(
                "PROVIDER_DISTRIBUTED_STAGING_KMS_KEY_ID"
            )
        ):
            missing_required.append(
                "PROVIDER_DISTRIBUTED_STAGING_KMS_KEY_ID"
            )
    if production and provider_gateway_enabled:
        for key in [
            "PROVIDER_INGESTION_DATABASE_URL",
            "PROVIDER_S3_REGION",
            "PROVIDER_SQS_QUEUE_URL",
            "PROVIDER_SQS_DLQ_URL",
            "CANONICAL_S3_BUCKET",
            "CANONICAL_S3_SERVER_SIDE_ENCRYPTION",
            "AUDIENCE_DP_SEED_SECRET",
            "AUDIENCE_TOKENIZATION_HMAC_KEY",
        ]:
            if not os.getenv(key):
                missing_required.append(key)

        canonical_encryption = os.getenv(
            "CANONICAL_S3_SERVER_SIDE_ENCRYPTION",
            "",
        ).strip()
        if canonical_encryption not in {"AES256", "aws:kms", "aws:kms:dsse"}:
            missing_required.append(
                "CANONICAL_S3_SERVER_SIDE_ENCRYPTION(AES256|aws:kms|aws:kms:dsse)"
            )
        if (
            canonical_encryption in {"aws:kms", "aws:kms:dsse"}
            and not os.getenv("CANONICAL_S3_KMS_KEY_ID")
        ):
            missing_required.append("CANONICAL_S3_KMS_KEY_ID")

    phase2_features_enabled = _truthy(
        os.getenv("PHASE2_FEATURES_ENABLED")
    )
    if production and phase2_features_enabled:
        if not os.getenv("AUDIENCE_FEATURE_DATABASE_URL"):
            missing_required.append("AUDIENCE_FEATURE_DATABASE_URL")
        if not os.getenv("AUDIENCE_PROPOSAL_DATABASE_URL"):
            missing_required.append("AUDIENCE_PROPOSAL_DATABASE_URL")
        pgvector_dimension = os.getenv(
            "PGVECTOR_DIMENSION",
            "",
        ).strip()
        if pgvector_dimension != "384":
            missing_required.append("PGVECTOR_DIMENSION(384)")
        phase2_retrieval_backend = os.getenv(
            "PHASE2_RETRIEVAL_BACKEND",
            "",
        ).strip().lower()
        if phase2_retrieval_backend != "pgvector":
            missing_required.append("PHASE2_RETRIEVAL_BACKEND(pgvector)")
        if not os.getenv("PUNK_AI_TENANT_AUTH_SECRET"):
            missing_required.append("PUNK_AI_TENANT_AUTH_SECRET")

    production_feature_builds_enabled = _truthy(
        os.getenv("PRODUCTION_FEATURE_BUILDS_ENABLED")
    )
    if production and production_feature_builds_enabled:
        if not phase2_features_enabled:
            missing_required.append("PHASE2_FEATURES_ENABLED(true)")
        for key in [
            "AUDIENCE_FEATURE_DATABASE_URL",
            "AUDIENCE_FEATURE_WRITER_DATABASE_URL",
            "CANONICAL_S3_BUCKET",
            "FEATURE_EMBEDDING_BACKEND",
            "FEATURE_EMBEDDING_MODEL_NAME",
            "FEATURE_EMBEDDING_MODEL_REVISION",
        ]:
            if not os.getenv(key):
                missing_required.append(key)
        feature_embedding_backend = normalize_feature_backend(
            os.getenv("FEATURE_EMBEDDING_BACKEND")
        )
        if feature_embedding_backend not in {
            "sentence_transformers",
            "external_embedding_service",
        }:
            missing_required.append(
                "FEATURE_EMBEDDING_BACKEND"
                "(sentence_transformers|external_embedding_service)"
            )
        feature_model_revision = os.getenv(
            "FEATURE_EMBEDDING_MODEL_REVISION",
            "",
        ).strip()
        if feature_model_revision.lower() in {"latest", "main", "master"}:
            missing_required.append(
                "FEATURE_EMBEDDING_MODEL_REVISION(immutable)"
            )
        if os.getenv("FEATURE_EMBEDDING_DIMENSION", "").strip() != "384":
            missing_required.append("FEATURE_EMBEDDING_DIMENSION(384)")
        if feature_embedding_backend == "external_embedding_service":
            for key in [
                "FEATURE_EMBEDDING_EXTERNAL_ENDPOINT",
                "FEATURE_EMBEDDING_EXTERNAL_AUTH_SECRET_ARN",
            ]:
                if not os.getenv(key):
                    missing_required.append(key)

    if production and _truthy(os.getenv("ALLOW_LOCAL_FILE_STORAGE")):
        warnings.append("ALLOW_LOCAL_FILE_STORAGE is enabled in production.")

    if production and _truthy(os.getenv("ALLOW_DEMO_ROUTES")):
        warnings.append("ALLOW_DEMO_ROUTES is enabled in production.")

    vector_backend = os.getenv("VECTOR_BACKEND", "").strip().lower()
    allowed_production_vector_backends = {"postgres_array", "postgres", "pg_array", "pgvector"}

    if production and vector_backend and vector_backend not in allowed_production_vector_backends:
        missing_required.append("VECTOR_BACKEND(postgres_array|pgvector)")

    embedding_backend = os.getenv("EMBEDDING_BACKEND", "").strip().lower()
    allowed_production_embedding_backends = {
        "sklearn_hashing",
        "hashing",
        "sentence-transformers",
        "sentence_transformers",
        "external_embedding_service",
    }

    if production and embedding_backend and embedding_backend not in allowed_production_embedding_backends:
        missing_required.append("EMBEDDING_BACKEND(sklearn_hashing|external_embedding_service)")

    all_safe_embedding_store = os.getenv("ALL_SAFE_COHORT_EMBEDDING_STORE", "").strip().lower()
    allowed_all_safe_embedding_stores = {"postgres", "postgres_array", "pg_array", "pgvector"}

    if production and all_safe_embedding_store and all_safe_embedding_store not in allowed_all_safe_embedding_stores:
        missing_required.append("ALL_SAFE_COHORT_EMBEDDING_STORE(postgres)")

    env_presence = {
        "AUDIENCE_API_KEY": bool(os.getenv("AUDIENCE_API_KEY")),
        "ECHO_DATABASE_URL": bool(os.getenv("ECHO_DATABASE_URL")),
        "HASH_SECRET": bool(os.getenv("HASH_SECRET")),
        "AUDIENCE_HASH_SALT": bool(os.getenv("AUDIENCE_HASH_SALT")),
        "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY")),
        "PRODUCTION_MODE": bool(os.getenv("PRODUCTION_MODE")),
        "APP_ENV": bool(os.getenv("APP_ENV")),
        "ALLOW_LOCAL_FILE_STORAGE": bool(os.getenv("ALLOW_LOCAL_FILE_STORAGE")),
        "ALLOW_DEMO_ROUTES": bool(os.getenv("ALLOW_DEMO_ROUTES")),
        "VECTOR_BACKEND": bool(os.getenv("VECTOR_BACKEND")),
        "EMBEDDING_BACKEND": bool(os.getenv("EMBEDDING_BACKEND")),
        "ALL_SAFE_COHORT_EMBEDDING_STORE": bool(os.getenv("ALL_SAFE_COHORT_EMBEDDING_STORE")),
        "PROVIDER_GATEWAY_ENABLED": provider_gateway_enabled,
        "PROVIDER_DISTRIBUTED_PROCESSING_ENABLED": (
            distributed_processing_enabled
        ),
        "PROVIDER_DISTRIBUTED_STATE_MACHINE_ARN": bool(
            os.getenv("PROVIDER_DISTRIBUTED_STATE_MACHINE_ARN")
        ),
        "PROVIDER_DISTRIBUTED_CONTROL_PLANE_LAMBDA_ARN": bool(
            os.getenv(
                "PROVIDER_DISTRIBUTED_CONTROL_PLANE_LAMBDA_ARN"
            )
        ),
        "PROVIDER_DISTRIBUTED_GLUE_JOB_NAME": bool(
            os.getenv("PROVIDER_DISTRIBUTED_GLUE_JOB_NAME")
        ),
        "PROVIDER_DISTRIBUTED_STAGING_BUCKET": bool(
            os.getenv("PROVIDER_DISTRIBUTED_STAGING_BUCKET")
        ),
        "PROVIDER_DISTRIBUTED_STAGING_ENCRYPTION": bool(
            os.getenv("PROVIDER_DISTRIBUTED_STAGING_ENCRYPTION")
        ),
        "PROVIDER_DISTRIBUTED_STAGING_KMS_KEY_ID": bool(
            os.getenv("PROVIDER_DISTRIBUTED_STAGING_KMS_KEY_ID")
        ),
        "PROVIDER_INGESTION_DATABASE_SECRET_ARN": bool(
            os.getenv("PROVIDER_INGESTION_DATABASE_SECRET_ARN")
        ),
        "PROVIDER_DISTRIBUTED_TOKENIZATION_SECRET_ARN": bool(
            os.getenv(
                "PROVIDER_DISTRIBUTED_TOKENIZATION_SECRET_ARN"
            )
        ),
        "PROVIDER_DISTRIBUTED_DP_SEED_SECRET_ARN": bool(
            os.getenv("PROVIDER_DISTRIBUTED_DP_SEED_SECRET_ARN")
        ),
        "PROVIDER_INGESTION_DATABASE_URL": bool(
            os.getenv("PROVIDER_INGESTION_DATABASE_URL")
        ),
        "PROVIDER_S3_REGION": bool(os.getenv("PROVIDER_S3_REGION")),
        "PROVIDER_SQS_QUEUE_URL": bool(os.getenv("PROVIDER_SQS_QUEUE_URL")),
        "PROVIDER_SQS_DLQ_URL": bool(os.getenv("PROVIDER_SQS_DLQ_URL")),
        "CANONICAL_S3_BUCKET": bool(os.getenv("CANONICAL_S3_BUCKET")),
        "CANONICAL_S3_SERVER_SIDE_ENCRYPTION": bool(
            os.getenv("CANONICAL_S3_SERVER_SIDE_ENCRYPTION")
        ),
        "CANONICAL_S3_KMS_KEY_ID": bool(os.getenv("CANONICAL_S3_KMS_KEY_ID")),
        "AUDIENCE_DP_SEED_SECRET": bool(
            os.getenv("AUDIENCE_DP_SEED_SECRET")
        ),
        "AUDIENCE_TOKENIZATION_HMAC_KEY": bool(
            os.getenv("AUDIENCE_TOKENIZATION_HMAC_KEY")
        ),
        "PHASE2_FEATURES_ENABLED": phase2_features_enabled,
        "PRODUCTION_FEATURE_BUILDS_ENABLED": (
            production_feature_builds_enabled
        ),
        "AUDIENCE_FEATURE_DATABASE_URL": bool(
            os.getenv("AUDIENCE_FEATURE_DATABASE_URL")
        ),
        "AUDIENCE_FEATURE_WRITER_DATABASE_URL": bool(
            os.getenv("AUDIENCE_FEATURE_WRITER_DATABASE_URL")
        ),
        "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL": bool(
            os.getenv("AUDIENCE_FEATURE_MIGRATION_DATABASE_URL")
        ),
        "AUDIENCE_PROPOSAL_DATABASE_URL": bool(
            os.getenv("AUDIENCE_PROPOSAL_DATABASE_URL")
        ),
        "PHASE2_FEATURE_READER_USER": bool(
            os.getenv("PHASE2_FEATURE_READER_USER")
        ),
        "PHASE2_FEATURE_READER_PASSWORD": bool(
            os.getenv("PHASE2_FEATURE_READER_PASSWORD")
        ),
        "PHASE2_FEATURE_WRITER_USER": bool(
            os.getenv("PHASE2_FEATURE_WRITER_USER")
        ),
        "PHASE2_FEATURE_WRITER_PASSWORD": bool(
            os.getenv("PHASE2_FEATURE_WRITER_PASSWORD")
        ),
        "PHASE2_PROPOSAL_RUNTIME_USER": bool(
            os.getenv("PHASE2_PROPOSAL_RUNTIME_USER")
        ),
        "PHASE2_PROPOSAL_RUNTIME_PASSWORD": bool(
            os.getenv("PHASE2_PROPOSAL_RUNTIME_PASSWORD")
        ),
        "PGVECTOR_DIMENSION": bool(os.getenv("PGVECTOR_DIMENSION")),
        "PHASE2_RETRIEVAL_BACKEND": bool(
            os.getenv("PHASE2_RETRIEVAL_BACKEND")
        ),
        "PUNK_AI_TENANT_AUTH_SECRET": bool(
            os.getenv("PUNK_AI_TENANT_AUTH_SECRET")
        ),
        "FEATURE_EMBEDDING_BACKEND": bool(
            os.getenv("FEATURE_EMBEDDING_BACKEND")
        ),
        "FEATURE_EMBEDDING_MODEL_NAME": bool(
            os.getenv("FEATURE_EMBEDDING_MODEL_NAME")
        ),
        "FEATURE_EMBEDDING_MODEL_REVISION": bool(
            os.getenv("FEATURE_EMBEDDING_MODEL_REVISION")
        ),
        "FEATURE_EMBEDDING_DIMENSION": bool(
            os.getenv("FEATURE_EMBEDDING_DIMENSION")
        ),
        "FEATURE_EMBEDDING_EXTERNAL_ENDPOINT": bool(
            os.getenv("FEATURE_EMBEDDING_EXTERNAL_ENDPOINT")
        ),
        "FEATURE_EMBEDDING_EXTERNAL_AUTH_SECRET_ARN": bool(
            os.getenv("FEATURE_EMBEDDING_EXTERNAL_AUTH_SECRET_ARN")
        ),
    }

    return EnvironmentValidationResult(
        ok=not missing_required,
        mode=mode,
        missing_required=missing_required,
        warnings=warnings,
        env_presence=env_presence,
    )


def assert_environment_ready() -> None:
    result = validate_environment()

    if not result.ok:
        missing = ", ".join(result.missing_required)
        raise RuntimeError(f"Production environment is not ready. Missing: {missing}")
