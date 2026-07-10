from __future__ import annotations

import hashlib
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from app.services.contribution_bounding_service import (
    ContributionBoundingConfig,
    ContributionBoundingService,
)
from app.services.ingestion_job_service import (
    IngestionJobCreate,
    IngestionJobService,
)
from app.services.ingestion_lineage_service import (
    IngestionLineageService,
    LineageEvent,
)


@dataclass(frozen=True)
class PrivacyIngestionConfig:
    """
    Production ingestion privacy config.

    This pipeline converts raw/event-level signals into safe aggregated
    feature rows.

    Default production k-anonymity threshold is 1000.
    Tests can override it with a smaller value.
    """

    source_type: str = "api"
    source_ref: Optional[str] = None
    entity_id_column: str = "entity_id"
    timestamp_column: str = "created_at"
    cohort_columns: Sequence[str] = (
        "location_name",
        "primary_poi_type",
        "created_day_part",
    )
    min_cohort_size: int = 1000
    epsilon: float = 1.0
    delta: float = 1e-5
    sensitivity: float = 1.0
    mechanism: str = "gaussian"
    hash_salt: Optional[str] = None
    random_seed: Optional[int] = None


class PrivacyIngestionPipelineService:
    """
    Module 1 production pipeline.

    Flow:
        source rows
        -> hash entity identifiers
        -> contribution bounding
        -> aggregate into cohorts
        -> enforce k-anonymity
        -> add DP noise
        -> record lineage
        -> update ingestion job status

    Output rule:
        No raw entity IDs are returned.
        Output is cohort-level aggregated/DP-safe feature rows only.
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._database_url = database_url
        self._jobs = IngestionJobService(database_url=database_url)
        self._lineage = IngestionLineageService(database_url=database_url)
        self._bounding = ContributionBoundingService()

    def process_events(
        self,
        events: List[Dict[str, Any]],
        *,
        config: Optional[PrivacyIngestionConfig] = None,
        run_id: Optional[str] = None,
        actor: str = "system",
    ) -> Dict[str, Any]:
        config = config or PrivacyIngestionConfig()

        self._validate_config(config)

        job = self._jobs.create_job(
            IngestionJobCreate(
                source_type=config.source_type,
                source_ref=config.source_ref,
                run_id=run_id,
                actor=actor,
                metadata={
                    "module": "module_1_ingestion_privacy_layer",
                    "min_cohort_size": config.min_cohort_size,
                    "epsilon": config.epsilon,
                    "delta": config.delta,
                    "mechanism": config.mechanism,
                },
            )
        )

        job_id = job.get("job_id") or f"local_job_{random.randint(1000, 9999)}"

        self._jobs.mark_running(job_id)

        input_rows = len(events or [])

        self._record_lineage(
            job_id=job_id,
            run_id=run_id,
            stage="source",
            transformation="source_feed_received",
            input_rows=input_rows,
            output_rows=input_rows,
            dropped_rows=0,
            actor=actor,
            details={"source_type": config.source_type, "source_ref": config.source_ref},
        )

        if input_rows == 0:
            self._jobs.mark_blocked(job_id=job_id, reason="No input rows supplied.")
            return {
                "status": "blocked",
                "reason": "No input rows supplied.",
                "job_id": job_id,
                "safe_feature_rows": [],
            }

        df = pd.DataFrame(events)

        required = [config.entity_id_column, config.timestamp_column, *config.cohort_columns]
        missing = [col for col in required if col not in df.columns]

        if missing:
            reason = "Missing required columns for raw-event privacy processing."
            self._record_lineage(
                job_id=job_id,
                run_id=run_id,
                stage="blocked",
                transformation="required_column_validation",
                input_rows=input_rows,
                output_rows=0,
                dropped_rows=input_rows,
                actor=actor,
                details={"missing_columns": missing},
            )
            self._jobs.mark_blocked(job_id=job_id, reason=f"{reason} Missing: {missing}")
            return {
                "status": "blocked",
                "reason": reason,
                "missing_columns": missing,
                "job_id": job_id,
                "safe_feature_rows": [],
            }

        hashed_events = self._hash_entity_ids(df, config)

        self._record_lineage(
            job_id=job_id,
            run_id=run_id,
            stage="hashing",
            transformation="sha256_salted_entity_hash",
            input_rows=input_rows,
            output_rows=len(hashed_events),
            dropped_rows=0,
            actor=actor,
            details={
                "raw_entity_id_removed": True,
                "hash_algorithm": "sha256",
            },
        )

        bounded = self._bounding.bound_events(
            hashed_events,
            ContributionBoundingConfig(
                entity_id_column=config.entity_id_column,
                timestamp_column=config.timestamp_column,
                cohort_columns=config.cohort_columns,
                max_contributions_per_entity_per_window=1,
                drop_entity_id_from_output=True,
            ),
        )

        if not bounded.get("bounded"):
            reason = bounded.get("reason") or "Contribution bounding failed."
            self._jobs.mark_blocked(job_id=job_id, reason=reason)
            return {
                "status": "blocked",
                "reason": reason,
                "job_id": job_id,
                "bounding": bounded,
                "safe_feature_rows": [],
            }

        bounded_rows = bounded.get("bounded_rows") or []

        self._record_lineage(
            job_id=job_id,
            run_id=run_id,
            stage="contribution_bounding",
            transformation="max_one_contribution_per_entity_per_cohort_per_day",
            input_rows=bounded.get("input_rows"),
            output_rows=bounded.get("output_rows"),
            dropped_rows=bounded.get("dropped_rows"),
            actor=actor,
            details=bounded.get("lineage") or {},
        )

        aggregated_rows = self._aggregate_bounded_rows(bounded_rows, config)

        self._record_lineage(
            job_id=job_id,
            run_id=run_id,
            stage="aggregation",
            transformation="cohort_level_count_aggregation",
            input_rows=len(bounded_rows),
            output_rows=len(aggregated_rows),
            dropped_rows=0,
            actor=actor,
            details={"cohort_columns": list(config.cohort_columns)},
        )

        k_safe_rows = [
            row for row in aggregated_rows if int(row["bounded_count"]) >= config.min_cohort_size
        ]

        k_dropped = len(aggregated_rows) - len(k_safe_rows)

        self._record_lineage(
            job_id=job_id,
            run_id=run_id,
            stage="k_anonymity",
            transformation="minimum_cohort_size_filter",
            input_rows=len(aggregated_rows),
            output_rows=len(k_safe_rows),
            dropped_rows=k_dropped,
            actor=actor,
            details={
                "k_anonymity_enforced": True,
                "min_cohort_size": config.min_cohort_size,
            },
        )

        if not k_safe_rows:
            reason = "All cohorts failed k-anonymity threshold."
            self._jobs.mark_blocked(job_id=job_id, reason=reason)
            return {
                "status": "blocked",
                "reason": reason,
                "job_id": job_id,
                "min_cohort_size": config.min_cohort_size,
                "safe_feature_rows": [],
                "input_rows": input_rows,
                "bounded_rows": len(bounded_rows),
                "aggregated_rows": len(aggregated_rows),
            }

        dp_rows = self._apply_dp_noise(k_safe_rows, config)

        self._record_lineage(
            job_id=job_id,
            run_id=run_id,
            stage="dp_safe",
            transformation=f"{config.mechanism}_differential_privacy_noise",
            input_rows=len(k_safe_rows),
            output_rows=len(dp_rows),
            dropped_rows=0,
            actor=actor,
            details={
                "epsilon": config.epsilon,
                "delta": config.delta,
                "sensitivity": config.sensitivity,
                "mechanism": config.mechanism,
                "no_raw_entity_ids": True,
            },
        )

        self._record_lineage(
            job_id=job_id,
            run_id=run_id,
            stage="output",
            transformation="safe_feature_table_ready",
            input_rows=len(dp_rows),
            output_rows=len(dp_rows),
            dropped_rows=0,
            actor=actor,
            details={
                "output_type": "aggregated_dp_safe_feature_table",
                "contains_raw_entity_id": False,
            },
        )

        dropped_rows = input_rows - len(bounded_rows)

        self._jobs.mark_completed(
            job_id=job_id,
            input_rows=input_rows,
            output_rows=len(dp_rows),
            dropped_rows=dropped_rows,
            metadata_update={
                "privacy_pipeline_completed": True,
                "safe_feature_rows": len(dp_rows),
            },
        )

        return {
            "status": "completed",
            "job_id": job_id,
            "run_id": run_id,
            "input_rows": input_rows,
            "bounded_rows": len(bounded_rows),
            "aggregated_rows": len(aggregated_rows),
            "safe_feature_rows_count": len(dp_rows),
            "safe_feature_rows": dp_rows,
            "privacy_controls": [
                "salted_hashing",
                "contribution_bounding",
                "k_anonymity",
                "differential_privacy_noise",
                "lineage_logging",
                "job_status_tracking",
            ],
            "privacy_note": (
                "Output contains cohort-level aggregated/DP-safe rows only. "
                "Raw entity IDs are not returned."
            ),
        }

    def _hash_entity_ids(
        self,
        df: pd.DataFrame,
        config: PrivacyIngestionConfig,
    ) -> List[Dict[str, Any]]:
        salt = config.hash_salt or os.getenv("AUDIENCE_HASH_SALT") or "dev_only_change_me"

        working = df.copy()

        def _hash(value: Any) -> str:
            raw = str(value).encode("utf-8")
            salted = salt.encode("utf-8") + b":" + raw
            return hashlib.sha256(salted).hexdigest()

        working[config.entity_id_column] = working[config.entity_id_column].map(_hash)

        return working.to_dict(orient="records")

    def _aggregate_bounded_rows(
        self,
        rows: List[Dict[str, Any]],
        config: PrivacyIngestionConfig,
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []

        df = pd.DataFrame(rows)
        grouped = (
            df.groupby(list(config.cohort_columns), dropna=False)
            .size()
            .reset_index(name="bounded_count")
        )

        output: List[Dict[str, Any]] = []
        for row in grouped.to_dict(orient="records"):
            cohort_key = "|".join(str(row.get(col, "unknown")) for col in config.cohort_columns)
            item = {str(key): value for key, value in row.items()}
            item["cohort_key"] = cohort_key
            output.append(item)

        return output

    def _apply_dp_noise(
        self,
        rows: List[Dict[str, Any]],
        config: PrivacyIngestionConfig,
    ) -> List[Dict[str, Any]]:
        rng = random.Random(config.random_seed)

        if config.mechanism.lower() != "gaussian":
            raise ValueError("Only gaussian mechanism is currently supported.")

        sigma = config.sensitivity * math.sqrt(2 * math.log(1.25 / config.delta)) / config.epsilon

        output: List[Dict[str, Any]] = []

        for row in rows:
            bounded_count = float(row["bounded_count"])
            noisy_count = max(0, int(round(bounded_count + rng.gauss(0, sigma))))

            safe_row = dict(row)
            safe_row["dp_noisy_count"] = noisy_count
            safe_row["dp_epsilon"] = config.epsilon
            safe_row["dp_delta"] = config.delta
            safe_row["dp_mechanism"] = config.mechanism
            safe_row["dp_sigma"] = sigma

            output.append(safe_row)

        return output

    def _record_lineage(
        self,
        *,
        job_id: str,
        run_id: Optional[str],
        stage: str,
        transformation: str,
        input_rows: Optional[int],
        output_rows: Optional[int],
        dropped_rows: Optional[int],
        actor: str,
        details: Dict[str, Any],
    ) -> None:
        self._lineage.record_event(
            LineageEvent(
                job_id=job_id,
                run_id=run_id,
                stage=stage,
                transformation=transformation,
                input_rows=input_rows,
                output_rows=output_rows,
                dropped_rows=dropped_rows,
                actor=actor,
                details=details,
            )
        )

    def _validate_config(self, config: PrivacyIngestionConfig) -> None:
        if config.min_cohort_size < 1:
            raise ValueError("min_cohort_size must be >= 1")

        if config.epsilon <= 0:
            raise ValueError("epsilon must be > 0")

        if config.delta <= 0 or config.delta >= 1:
            raise ValueError("delta must be between 0 and 1")

        if config.sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")

        if not config.cohort_columns:
            raise ValueError("cohort_columns is required")
