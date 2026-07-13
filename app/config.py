from pathlib import Path

from app.core import production_guardrails


class Settings:
    APP_NAME = "Audience Intelligence Engine"
    APP_VERSION = "1.0.0"

    DEFAULT_K_ANONYMITY = 1000
    DEFAULT_EPSILON = 1.0

    DATA_DIR = Path("data")
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    SYNTHETIC_DIR = DATA_DIR / "synthetic"
    VECTOR_DIR = DATA_DIR / "vectors"
    COHORT_DIR = DATA_DIR / "cohorts"
    EXPORT_DIR = DATA_DIR / "exports"
    LINEAGE_DIR = DATA_DIR / "lineage"


settings = Settings()


def ensure_data_dirs() -> list[str]:
    """
    Create legacy local-storage directories only when local file storage
    is explicitly allowed.

    Production deployments use Postgres-backed persistence and must not
    create local runtime data directories during application startup.
    """
    if not production_guardrails.local_file_storage_allowed():
        return []

    created: list[str] = []

    for directory in [
        settings.RAW_DIR,
        settings.PROCESSED_DIR,
        settings.SYNTHETIC_DIR,
        settings.VECTOR_DIR,
        settings.COHORT_DIR,
        settings.EXPORT_DIR,
        settings.LINEAGE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
        created.append(str(directory))

    return created
