from __future__ import annotations

import hashlib
import os
from pathlib import Path

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = ROOT / "migrations"
OPERATOR_ONLY_MIGRATIONS = {
    "0003_provider_ingestion_gateway.sql",
    "0004_versioned_audience_features_pgvector.sql",
    "0005_punk_ai_audience_proposal_ledger.sql",
    "0006_provider_distributed_scale_dispatch.sql",
    "0007_provider_distributed_privacy_releases.sql",
    "0008_production_feature_build_registry.sql",
    "0009_provider_privacy_windows_and_rights.sql",
    "0010_provider_control_plane_rls.sql",
}


def database_url() -> str:
    value = (
        os.getenv("PROVIDER_INGESTION_DATABASE_URL")
        or os.getenv("AUDIENCE_HISTORY_DATABASE_URL")
        or os.getenv("ECHO_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("POSTGRES_DATABASE_URL")
        or os.getenv("SUPABASE_DB_URL")
        or os.getenv("DB_URL")
        or ""
    ).strip()

    if not value:
        raise RuntimeError(
            "No database URL configured for migrations."
        )

    if value.startswith("postgres://"):
        value = (
            "postgresql://"
            + value[len("postgres://") :]
        )

    return value


def migration_files() -> list[Path]:
    return sorted(
        migration
        for migration in MIGRATIONS_DIR.glob("*.sql")
        if migration.name not in OPERATOR_ONLY_MIGRATIONS
    )


def apply_migrations() -> None:
    files = migration_files()

    if not files:
        print("NO_MIGRATIONS_FOUND")
        return

    engine = create_engine(database_url())

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS
                audience_schema_migrations (
                    version TEXT PRIMARY KEY,
                    checksum TEXT NOT NULL,
                    applied_at TIMESTAMPTZ
                        NOT NULL DEFAULT now()
                )
                """
            )
        )

        existing = {
            row.version: row.checksum
            for row in conn.execute(
                text(
                    """
                    SELECT version, checksum
                    FROM audience_schema_migrations
                    """
                )
            )
        }

        for migration in files:
            version = migration.name
            sql = migration.read_text(
                encoding="utf-8"
            )
            checksum = hashlib.sha256(
                sql.encode("utf-8")
            ).hexdigest()

            previous = existing.get(version)

            if previous:
                if previous != checksum:
                    raise RuntimeError(
                        "Migration checksum changed: "
                        f"{version}"
                    )

                print(f"SKIPPED {version}")
                continue

            driver_connection = (
                conn.connection.driver_connection
            )
            cursor = driver_connection.cursor()

            try:
                cursor.execute(sql)
            finally:
                cursor.close()

            conn.execute(
                text(
                    """
                    INSERT INTO
                    audience_schema_migrations (
                        version,
                        checksum
                    )
                    VALUES (
                        :version,
                        :checksum
                    )
                    """
                ),
                {
                    "version": version,
                    "checksum": checksum,
                },
            )

            print(f"APPLIED {version}")


if __name__ == "__main__":
    apply_migrations()
