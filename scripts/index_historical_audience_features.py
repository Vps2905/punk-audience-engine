from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.services.pgvector_audience_feature_store_service import (
    PgvectorAudienceFeatureStore,
)
from app.services.phase2_database_preflight_service import (
    Phase2DatabasePreflightService,
)
from app.services.postgres_canonical_feature_adapter_service import (
    LegacyPostgresSafeVectorReader,
    PostgresCanonicalFeatureAdapterService,
)

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import an existing privacy-safe Postgres vector snapshot into "
            "the versioned Phase 2 pgvector feature registry."
        )
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--rights-policy-id", required=True)
    parser.add_argument("--privacy-policy-version", required=True)
    parser.add_argument(
        "--job-id",
        required=True,
        help=(
            "Exact compatible snapshot selected from the read-only "
            "snapshot inventory."
        ),
    )
    parser.add_argument(
        "--source-ref",
        default="postgres://legacy_safe_audience_vectors",
    )
    parser.add_argument(
        "--data-use-mode",
        choices=["historical_preview", "offline_evaluation"],
        default="historical_preview",
    )
    return parser


def index_historical_features(args: argparse.Namespace) -> dict[str, Any]:
    source_database_url = (
        os.getenv("ECHO_DATABASE_URL")
        or os.getenv("DATABASE_URL")
    )
    feature_writer_database_url = os.getenv(
        "AUDIENCE_FEATURE_WRITER_DATABASE_URL"
    )
    feature_migration_database_url = os.getenv(
        "AUDIENCE_FEATURE_MIGRATION_DATABASE_URL"
    )
    if not source_database_url:
        raise RuntimeError(
            "Set ECHO_DATABASE_URL or DATABASE_URL for the read-only "
            "historical source."
        )
    if not feature_writer_database_url:
        raise RuntimeError(
            "Set AUDIENCE_FEATURE_WRITER_DATABASE_URL for Phase 2 imports."
        )
    if not feature_migration_database_url:
        raise RuntimeError(
            "Set AUDIENCE_FEATURE_MIGRATION_DATABASE_URL for operator "
            "schema verification before historical import."
        )
    preflight = Phase2DatabasePreflightService().run(
        source_database_url=source_database_url,
        feature_database_url=feature_migration_database_url,
        punk_owned_target_confirmed=True,
    )
    if preflight.get("same_database_as_source"):
        raise RuntimeError(
            "Historical import requires a feature database that is "
            "separate from the historical source."
        )
    if preflight.get("status") != "phase2_schema_ready":
        raise RuntimeError(
            "Historical import requires a preflight-verified Phase 2 "
            "schema. No feature rows were written."
        )
    snapshot = LegacyPostgresSafeVectorReader(
        database_url=source_database_url
    ).load_snapshot(job_id=args.job_id)
    feature_set = PostgresCanonicalFeatureAdapterService().adapt(
        snapshot,
        tenant_id=args.tenant_id,
        purpose=args.purpose,
        rights_policy_id=args.rights_policy_id,
        privacy_policy_version=args.privacy_policy_version,
        source_ref=args.source_ref,
        data_use_mode=args.data_use_mode,
    )
    receipt = PgvectorAudienceFeatureStore(
        database_url=feature_writer_database_url
    ).save_feature_set(feature_set)
    return {
        **receipt,
        "source_job_id": snapshot.job_id,
        "source_latest_timestamp": (
            snapshot.latest_source_timestamp.isoformat()
            if snapshot.latest_source_timestamp
            else None
        ),
        "raw_identifiers_read": False,
        "raw_identifiers_stored": False,
        "activation_blocked": True,
    }


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    receipt = index_historical_features(_parser().parse_args())
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
