from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


@dataclass(frozen=True)
class ContributionBoundingConfig:
    """
    Production privacy rule:

    One entity can contribute only a limited number of times per
    cohort + time window.

    This is required to make DP sensitivity defensible.

    Example:
        If max_contributions_per_entity_per_window = 1,
        then one user/device can affect one cohort/day count by max 1.
    """

    entity_id_column: str = "entity_id"
    timestamp_column: str = "created_at"
    cohort_columns: Sequence[str] = (
        "location_name",
        "primary_poi_type",
        "created_day_part",
    )
    window: str = "1D"
    max_contributions_per_entity_per_window: int = 1
    drop_entity_id_from_output: bool = True


class ContributionBoundingService:
    """
    Enforces contribution bounding before aggregation/DP.

    Why this exists:
        DP sensitivity = 1 is only valid when one entity contributes
        at most once to a cohort/time-window count.

    This service:
        - accepts event-level rows
        - groups by entity + cohort + time window
        - keeps only allowed contributions
        - removes entity_id from output by default
        - returns lineage summary for audit
    """

    def bound_events(
        self,
        events: pd.DataFrame | List[Dict[str, Any]],
        config: ContributionBoundingConfig | None = None,
    ) -> Dict[str, Any]:
        config = config or ContributionBoundingConfig()

        df = self._to_dataframe(events)
        input_rows = len(df)

        if input_rows == 0:
            return {
                "status": "completed",
                "bounded": True,
                "input_rows": 0,
                "output_rows": 0,
                "dropped_rows": 0,
                "bounded_rows": [],
                "lineage": self._lineage(config=config, input_rows=0, output_rows=0, dropped_rows=0),
            }

        required = [config.entity_id_column, config.timestamp_column, *config.cohort_columns]
        missing = [col for col in required if col not in df.columns]

        if missing:
            return {
                "status": "needs_upstream_bounding",
                "bounded": False,
                "reason": "Cannot enforce contribution bounding because required columns are missing.",
                "missing_columns": missing,
                "input_rows": input_rows,
                "output_rows": 0,
                "dropped_rows": 0,
                "bounded_rows": [],
                "lineage": {
                    "stage": "contribution_bounding",
                    "status": "needs_upstream_bounding",
                    "missing_columns": missing,
                    "privacy_note": (
                        "If data is already aggregated/bought, upstream provider must certify "
                        "contribution limits before DP sensitivity can be assumed."
                    ),
                },
            }

        if config.max_contributions_per_entity_per_window < 1:
            raise ValueError("max_contributions_per_entity_per_window must be >= 1")

        working = df.copy()

        working["_bounded_entity_id"] = working[config.entity_id_column].astype(str)
        working["_bounded_timestamp"] = pd.to_datetime(
            working[config.timestamp_column],
            errors="coerce",
            utc=True,
        )

        missing_ts = working["_bounded_timestamp"].isna()
        if missing_ts.any():
            working.loc[missing_ts, "_bounded_timestamp"] = pd.Timestamp.utcnow()

        working["_privacy_window"] = working["_bounded_timestamp"].dt.floor(config.window)

        group_cols = [
            "_bounded_entity_id",
            "_privacy_window",
            *list(config.cohort_columns),
        ]

        working = working.sort_values("_bounded_timestamp")
        working["_contribution_rank"] = working.groupby(group_cols, dropna=False).cumcount() + 1

        bounded = working[
            working["_contribution_rank"] <= config.max_contributions_per_entity_per_window
        ].copy()

        dropped_rows = int(input_rows - len(bounded))

        internal_cols = [
            "_bounded_entity_id",
            "_bounded_timestamp",
            "_privacy_window",
            "_contribution_rank",
        ]

        bounded = bounded.drop(columns=[col for col in internal_cols if col in bounded.columns])

        if config.drop_entity_id_from_output and config.entity_id_column in bounded.columns:
            bounded = bounded.drop(columns=[config.entity_id_column])

        bounded_rows = self._clean_records(bounded.to_dict(orient="records"))

        return {
            "status": "completed",
            "bounded": True,
            "input_rows": input_rows,
            "output_rows": len(bounded_rows),
            "dropped_rows": dropped_rows,
            "max_contributions_per_entity_per_window": config.max_contributions_per_entity_per_window,
            "cohort_columns": list(config.cohort_columns),
            "window": config.window,
            "bounded_rows": bounded_rows,
            "lineage": self._lineage(
                config=config,
                input_rows=input_rows,
                output_rows=len(bounded_rows),
                dropped_rows=dropped_rows,
            ),
            "privacy_note": (
                "Contribution bounding applied before aggregation/DP. "
                "Entity identifiers are removed from output by default."
            ),
        }

    def _to_dataframe(self, events: pd.DataFrame | List[Dict[str, Any]]) -> pd.DataFrame:
        if isinstance(events, pd.DataFrame):
            return events.copy()

        if isinstance(events, list):
            return pd.DataFrame(events)

        raise TypeError("events must be a pandas DataFrame or list of dictionaries")

    def _lineage(
        self,
        *,
        config: ContributionBoundingConfig,
        input_rows: int,
        output_rows: int,
        dropped_rows: int,
    ) -> Dict[str, Any]:
        return {
            "stage": "contribution_bounding",
            "status": "completed",
            "input_rows": input_rows,
            "output_rows": output_rows,
            "dropped_rows": dropped_rows,
            "entity_id_removed_from_output": config.drop_entity_id_from_output,
            "max_contributions_per_entity_per_window": config.max_contributions_per_entity_per_window,
            "window": config.window,
            "cohort_columns": list(config.cohort_columns),
            "sensitivity_claim": (
                "Sensitivity can be treated as 1 for bounded count queries when max contribution is 1."
            ),
        }

    def _clean_records(self, rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []

        for row in rows:
            item: Dict[str, Any] = {}
            for key, value in row.items():
                if pd.isna(value):
                    item[str(key)] = None
                elif hasattr(value, "isoformat"):
                    item[str(key)] = value.isoformat()
                else:
                    item[str(key)] = value
            cleaned.append(item)

        return cleaned
