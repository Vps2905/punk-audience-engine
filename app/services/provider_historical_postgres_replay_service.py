from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.models.provider_historical_replay_contracts import (
    HistoricalReplayConfig,
)
from app.services.sensitive_poi_privacy_risk_service import (
    SensitivePOIPrivacyRiskService,
)
from app.services.provider_geography_normalization_service import (
    ProviderGeographyNormalizationService,
)
from app.utils.dp_noise import add_gaussian_noise


class ProviderHistoricalPostgresReplayService:
    """
    Read-only compatibility replay for the legacy maid_extractions snapshot.

    Raw identifiers are expanded and contribution-bounded inside PostgreSQL.
    Only k-safe aggregate rows cross the database boundary. The adapter never
    selects, logs, stores, returns, hashes, or exports an identifier value.

    This is deliberately an offline validation path. It cannot publish a
    canonical provider release because it does not reserve durable privacy
    budget or exercise the production S3/SQS/distributed data plane.
    """

    REQUIRED_COLUMNS = {
        "id",
        "session_id",
        "created_at",
        "maid_count",
        "maids",
        "pois",
        "center",
    }

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: Engine | None = None,
        now_fn: Any = None,
        rng: random.Random | None = None,
    ) -> None:
        self._database_url = database_url
        self._engine_override = engine
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._rng = rng
        self._risk = SensitivePOIPrivacyRiskService()
        self._geography = ProviderGeographyNormalizationService()

    def replay(
        self,
        config: HistoricalReplayConfig,
        *,
        confirm_restricted_identifier_processing: bool = False,
    ) -> dict[str, Any]:
        if not confirm_restricted_identifier_processing:
            raise PermissionError(
                "Historical replay requires explicit confirmation of "
                "restricted in-database identifier processing"
            )

        engine = self._engine()
        source_profile: dict[str, Any]
        internal_rows: list[dict[str, Any]]

        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                connection.exec_driver_sql(
                    f"SET LOCAL statement_timeout = "
                    f"'{config.statement_timeout_ms}ms'"
                )
                self._validate_source_schema(connection, config)
                source_profile = self._source_profile(connection, config)
                internal_rows = [
                    dict(row)
                    for row in connection.execute(
                        text(self._bounded_cohort_query(config)),
                        {"min_cohort_size": config.min_cohort_size},
                    ).mappings().all()
                ]
            finally:
                transaction.rollback()

        fingerprint = self._fingerprint(config, source_profile)
        replay_rng = self._rng or self._deterministic_rng(fingerprint)
        safe_rows = []
        blocked_sensitive = 0
        review_required = 0
        for internal in internal_rows:
            geography = self._geography.normalize(
                internal.get("location_name")
            )
            candidate = {
                "location_name": geography.canonical_name,
                "geo_id": geography.geo_id,
                "geo_resolution_status": geography.resolution_status,
                "geo_normalization_version": (
                    geography.normalization_version
                ),
                "primary_poi_type": self._normalize_trait(
                    internal.get("primary_poi_type")
                ),
                "created_day_part": "unknown",
            }
            assessment = self._risk.assess(
                prompt="",
                selected_cohorts=[candidate],
            )
            decision = assessment["overall_decision"]
            if decision == "block_export":
                blocked_sensitive += 1
                continue
            if decision == "review_required":
                review_required += 1

            private = add_gaussian_noise(
                int(internal["bounded_count"]),
                epsilon=config.epsilon,
                delta=config.delta,
                sensitivity=config.sensitivity,
                rng=replay_rng,
            )
            safe_rows.append(
                {
                    **candidate,
                    "dp_noisy_count": private["private_count"],
                    "dp_epsilon": config.epsilon,
                    "dp_delta": config.delta,
                    "dp_mechanism": "gaussian",
                    "privacy_status": "k_safe_dp_protected",
                    "sensitive_poi_status": decision,
                    "freshness_status": "stale",
                    "data_use_mode": config.data_use_mode,
                    "eligible_for_activation": False,
                }
            )

        return {
            "status": "historical_privacy_replay_completed",
            "replay_id": f"historical_replay_{fingerprint[:24]}",
            "tenant_id": config.tenant_id,
            "provider_id": config.provider_id,
            "dataset_id": config.dataset_id,
            "source_fingerprint": fingerprint,
            "privacy_policy_version": config.privacy_policy_version,
            "source_row_count": source_profile["source_row_count"],
            "declared_maid_count": source_profile["declared_maid_count"],
            "latest_source_timestamp": source_profile[
                "latest_source_timestamp"
            ],
            "freshness_status": "stale",
            "safe_cohort_count": len(safe_rows),
            "blocked_sensitive_cohort_count": blocked_sensitive,
            "review_required_cohort_count": review_required,
            "safe_feature_rows": safe_rows,
            "privacy_controls_exercised": [
                "read_only_source_transaction",
                "latest_session_deduplication",
                "in_database_identifier_expansion",
                "cross_batch_unique_entity_contribution_bounding",
                "k_anonymity_minimum_1000",
                "sensitive_poi_filtering",
                "non_coordinate_geography_normalization",
                "gaussian_differential_privacy",
                "raw_identifier_non_disclosure",
            ],
            "controls_not_exercised": [
                "s3_object_version_and_checksum_validation",
                "sqs_retry_and_dead_letter_queue",
                "distributed_entity_hash_partitioning",
                "durable_privacy_budget_reservation",
                "canonical_candidate_to_active_publication",
            ],
            "source_modified": False,
            "raw_identifier_values_returned": False,
            "raw_identifier_values_logged": False,
            "raw_observations_read": False,
            "exact_coordinates_read": False,
            "temporal_fidelity": "unavailable_in_historical_snapshot",
            "privacy_budget_persisted": False,
            "dp_release_replay_stable": True,
            "canonical_publication_performed": False,
            "eligible_for_activation": False,
            "activation_or_export_performed": False,
            "next_action": (
                "run_production_s3_replay_after_provider_control_database_"
                "and_live_or_mock_object_feed_are_configured"
            ),
        }

    def _validate_source_schema(
        self,
        connection: Any,
        config: HistoricalReplayConfig,
    ) -> None:
        columns = set(
            connection.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema_name
                      AND table_name = :table_name
                    """
                ),
                {
                    "schema_name": config.schema_name,
                    "table_name": config.table_name,
                },
            ).scalars().all()
        )
        missing = sorted(self.REQUIRED_COLUMNS - columns)
        if missing:
            raise RuntimeError(
                "Historical source schema is incompatible; missing: "
                + ", ".join(missing)
            )

    def _source_profile(
        self,
        connection: Any,
        config: HistoricalReplayConfig,
    ) -> dict[str, Any]:
        table = self._qualified_table(config)
        row = connection.execute(
            text(
                f"""
                SELECT
                    COUNT(*) AS source_row_count,
                    COALESCE(SUM(maid_count), 0) AS declared_maid_count,
                    MIN(created_at) AS earliest_source_timestamp,
                    MAX(created_at) AS latest_source_timestamp
                FROM {table}
                """
            )
        ).mappings().one()
        return {
            "source_row_count": int(row["source_row_count"] or 0),
            "declared_maid_count": int(row["declared_maid_count"] or 0),
            "earliest_source_timestamp": self._iso(
                row["earliest_source_timestamp"]
            ),
            "latest_source_timestamp": self._iso(
                row["latest_source_timestamp"]
            ),
        }

    def _bounded_cohort_query(self, config: HistoricalReplayConfig) -> str:
        table = self._qualified_table(config)
        return f"""
            WITH latest AS MATERIALIZED (
                SELECT DISTINCT ON (session_id)
                    id,
                    session_id,
                    created_at,
                    maids::jsonb AS maids,
                    pois::jsonb AS pois,
                    center::jsonb AS center
                FROM {table}
                WHERE session_id IS NOT NULL
                  AND created_at IS NOT NULL
                  AND maids IS NOT NULL
                  AND jsonb_typeof(maids::jsonb) = 'array'
                ORDER BY session_id, created_at DESC, id DESC
            ),
            classified AS MATERIALIZED (
                SELECT
                    session_id,
                    maids,
                    CASE
                        WHEN COALESCE((center ->> 'is_city')::boolean, false)
                        THEN COALESCE(
                            NULLIF(lower(trim(center ->> 'location_name')), ''),
                            'unknown'
                        )
                        ELSE 'unknown'
                    END AS location_name,
                    COALESCE(
                        (
                            SELECT regexp_replace(
                                lower(trim(poi_type.value)),
                                '[^a-z0-9]+',
                                '_',
                                'g'
                            )
                            FROM jsonb_array_elements(
                                CASE
                                    WHEN jsonb_typeof(pois) = 'array'
                                    THEN pois
                                    ELSE '[]'::jsonb
                                END
                            ) AS poi(item)
                            CROSS JOIN LATERAL jsonb_array_elements_text(
                                CASE
                                    WHEN jsonb_typeof(poi.item -> 'types') = 'array'
                                    THEN poi.item -> 'types'
                                    ELSE '[]'::jsonb
                                END
                            ) AS poi_type(value)
                            WHERE trim(poi_type.value) <> ''
                              AND regexp_replace(
                                    lower(trim(poi_type.value)),
                                    '[^a-z0-9]+',
                                    '_',
                                    'g'
                                  ) NOT IN (
                                    'establishment',
                                    'point_of_interest',
                                    'premise',
                                    'business'
                                  )
                            GROUP BY poi_type.value
                            ORDER BY COUNT(*) DESC, poi_type.value ASC
                            LIMIT 1
                        ),
                        'unknown'
                    ) AS primary_poi_type
                FROM latest
            ),
            bounded AS MATERIALIZED (
                SELECT DISTINCT
                    entity.value AS entity_id,
                    classified.location_name,
                    classified.primary_poi_type
                FROM classified
                CROSS JOIN LATERAL jsonb_array_elements_text(
                    classified.maids
                ) AS entity(value)
                WHERE trim(entity.value) <> ''
                  AND classified.location_name <> 'unknown'
                  AND classified.primary_poi_type <> 'unknown'
            ),
            cohorts AS (
                SELECT
                    location_name,
                    primary_poi_type,
                    COUNT(*) AS bounded_count
                FROM bounded
                GROUP BY location_name, primary_poi_type
            )
            SELECT
                location_name,
                primary_poi_type,
                bounded_count
            FROM cohorts
            WHERE bounded_count >= :min_cohort_size
            ORDER BY location_name, primary_poi_type
        """

    def _qualified_table(self, config: HistoricalReplayConfig) -> str:
        # Config validation makes this interpolation identifier-safe.
        return f'"{config.schema_name}"."{config.table_name}"'

    def _engine(self) -> Engine:
        if self._engine_override is not None:
            return self._engine_override
        database_url = (
            self._database_url
            or os.getenv("ECHO_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or ""
        ).strip()
        if not database_url:
            raise RuntimeError("Historical database is not configured")
        if database_url.startswith("postgres://"):
            database_url = "postgresql://" + database_url[len("postgres://") :]
        return create_engine(database_url, pool_pre_ping=True)

    def _fingerprint(
        self,
        config: HistoricalReplayConfig,
        profile: dict[str, Any],
    ) -> str:
        payload = {
            "tenant_id": config.tenant_id,
            "provider_id": config.provider_id,
            "dataset_id": config.dataset_id,
            "schema_name": config.schema_name,
            "table_name": config.table_name,
            "min_cohort_size": config.min_cohort_size,
            "epsilon": config.epsilon,
            "delta": config.delta,
            "sensitivity": config.sensitivity,
            "privacy_policy_version": config.privacy_policy_version,
            "profile": profile,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def _deterministic_rng(self, fingerprint: str) -> random.Random:
        secret = str(os.getenv("AUDIENCE_DP_SEED_SECRET") or "")
        if len(secret) < 32:
            raise RuntimeError(
                "Historical DP replay requires AUDIENCE_DP_SEED_SECRET "
                "with at least 32 characters"
            )
        digest = hmac.new(
            secret.encode("utf-8"),
            (
                "historical-provider-replay\x1f" + fingerprint
            ).encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return random.Random(int.from_bytes(digest, "big"))

    def _normalize_trait(self, value: Any) -> str:
        return str(value or "unknown").strip().lower() or "unknown"

    def _iso(self, value: Any) -> str | None:
        if value is None:
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)
