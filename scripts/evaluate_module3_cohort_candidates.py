#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from app.models.production_module3_cohort_contracts import (
    Module3CohortCandidatePolicy,
    Module3CohortGenerationRequest,
)
from app.services.production_module3_cohort_candidate_service import (
    ProductionModule3CohortCandidateService,
)


ROOT = Path(__file__).resolve().parents[1]


def load_environment() -> None:
    load_dotenv(ROOT / ".env", override=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a no-write Module 3.1-3.2 cohort candidate evidence report "
            "from aggregate feature rows."
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--tenant-id", default="punk_internal")
    parser.add_argument("--feature-set-id")
    parser.add_argument("--feature-set-version", type=int)
    parser.add_argument("--min-cohort-size", type=int, default=1000)
    parser.add_argument("--max-candidates", type=int, default=250)
    parser.add_argument("--confirm-engineering-preview-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_environment()
    args = parse_args()
    if not args.confirm_engineering_preview_only:
        raise SystemExit("--confirm-engineering-preview-only is required")

    database_url = str(os.getenv("AUDIENCE_FEATURE_DATABASE_URL") or "").strip()
    if not database_url:
        raise SystemExit("AUDIENCE_FEATURE_DATABASE_URL is required")

    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                connection.execute(
                    text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
                    {"tenant_id": args.tenant_id},
                )
                feature_set = _read_feature_set(
                    connection,
                    tenant_id=args.tenant_id,
                    feature_set_id=args.feature_set_id,
                    feature_set_version=args.feature_set_version,
                )
                rows = _read_safe_feature_rows(
                    connection,
                    tenant_id=args.tenant_id,
                    feature_set_id=str(feature_set["feature_set_id"]),
                    feature_set_version=int(feature_set["version"]),
                )
            finally:
                transaction.rollback()
    finally:
        engine.dispose()

    request = Module3CohortGenerationRequest(
        tenant_id=args.tenant_id,
        feature_set_id=str(feature_set["feature_set_id"]),
        feature_set_version=int(feature_set["version"]),
        execution_mode=str(feature_set["data_use_mode"]),
        purpose="internal_audience_evaluation",
    )
    policy = Module3CohortCandidatePolicy(
        min_cohort_size=args.min_cohort_size,
        max_candidates=args.max_candidates,
    )
    report = ProductionModule3CohortCandidateService(policy=policy).generate(
        request=request,
        feature_set=feature_set,
        feature_rows=rows,
    ).to_record()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("STATUS=engineering_preview_ready")
    print("SOURCE_FEATURE_COUNT=", report["source_feature_count"])
    print("GENERATED_CANDIDATE_COUNT=", report["generated_candidate_count"])
    print("TRUNCATED_CANDIDATE_COUNT=", report["truncated_candidate_count"])
    print("EXCLUDED_BELOW_K_COUNT=", report["excluded_below_k_count"])
    print("BLOCKED_SENSITIVE_COUNT=", report["blocked_sensitive_count"])
    print(
        "REVIEW_REQUIRED_SENSITIVE_COUNT=",
        report["review_required_sensitive_count"],
    )
    print("RAW_IDENTIFIERS_READ=False")
    print("DATABASE_CANDIDATE_WRITE_PERFORMED=False")
    print("LOOKALIKE_GENERATION_PERFORMED=False")
    print("ACTIVATION_OR_EXPORT_PERFORMED=False")
    print("OUTPUT=", output)
    return 0


def _read_feature_set(
    connection: Any,
    *,
    tenant_id: str,
    feature_set_id: str | None,
    feature_set_version: int | None,
) -> dict[str, Any]:
    filters = [
        "tenant_id = :tenant_id",
        "eligible_for_retrieval = TRUE",
    ]
    params: dict[str, Any] = {"tenant_id": tenant_id}
    if feature_set_id:
        filters.append("feature_set_id = :feature_set_id")
        params["feature_set_id"] = feature_set_id
    if feature_set_version is not None:
        filters.append("version = :feature_set_version")
        params["feature_set_version"] = int(feature_set_version)

    row = connection.execute(
        text(
            f"""
            SELECT
                tenant_id,
                feature_set_id,
                version,
                status,
                source_mode,
                data_use_mode,
                source_ref,
                source_version,
                source_fingerprint,
                source_latest_at,
                freshness_status,
                privacy_policy_version,
                rights_policy_id,
                purpose,
                eligible_for_retrieval,
                eligible_for_activation,
                feature_count
            FROM audience_feature_sets
            WHERE {' AND '.join(filters)}
            ORDER BY version DESC, updated_at DESC
            LIMIT 1
            """
        ),
        params,
    ).mappings().one_or_none()
    if row is None:
        raise RuntimeError("No retrieval-eligible feature set matched the request.")
    return dict(row)


def _read_safe_feature_rows(
    connection: Any,
    *,
    tenant_id: str,
    feature_set_id: str,
    feature_set_version: int,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        text(
            """
            SELECT
                tenant_id,
                feature_set_id,
                feature_set_version,
                feature_id,
                location_name,
                primary_poi_type,
                created_day_part,
                lookback_bucket,
                cohort_size,
                quality_score,
                privacy_status,
                rights_status,
                purpose,
                source_latest_at,
                freshness_status,
                data_use_mode,
                eligible_for_retrieval,
                eligible_for_activation
            FROM audience_feature_vectors
            WHERE tenant_id = :tenant_id
              AND feature_set_id = :feature_set_id
              AND feature_set_version = :feature_set_version
            ORDER BY feature_id
            """
        ),
        {
            "tenant_id": tenant_id,
            "feature_set_id": feature_set_id,
            "feature_set_version": feature_set_version,
        },
    ).mappings().all()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    raise SystemExit(main())
