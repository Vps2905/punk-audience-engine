from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DataFreshnessReport:
    status: str
    freshness_status: str
    timestamp_column: str | None
    latest_source_timestamp: str | None
    previous_latest_source_timestamp: str | None
    source_rows_checked: int
    new_rows_since_last_run: int
    stale_data_warning: bool
    hours_since_latest_source: float | None
    reason: str


class DataFreshnessAgent:
    """
    Fresh Echo/Postgres data checker.

    This agent checks whether latest source data is fresh, stale, unchanged,
    or unknown. It only uses metadata like timestamps and row counts.
    It does not export raw rows or identifiers.
    """

    DEFAULT_TIMESTAMP_COLUMNS = (
        "ingested_at",
        "created_at",
        "updated_at",
        "event_timestamp",
        "timestamp",
        "date",
    )

    def __init__(self, stale_after_hours: int | None = None) -> None:
        self.stale_after_hours = stale_after_hours or int(
            os.getenv("DATA_FRESHNESS_STALE_HOURS", "48")
        )

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        run_dir: str | Path | None = None,
        previous_report_path: str | Path | None = None,
        timestamp_columns: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        timestamp_columns = timestamp_columns or self.DEFAULT_TIMESTAMP_COLUMNS

        report = self._build_report(
            df=df,
            previous_report_path=previous_report_path,
            timestamp_columns=timestamp_columns,
        )

        result = asdict(report)

        if run_dir:
            run_path = Path(run_dir)
            run_path.mkdir(parents=True, exist_ok=True)
            (run_path / "data_freshness_report.json").write_text(
                json.dumps(result, indent=2),
                encoding="utf-8",
            )

        return result

    def _build_report(
        self,
        df: pd.DataFrame,
        previous_report_path: str | Path | None,
        timestamp_columns: tuple[str, ...],
    ) -> DataFreshnessReport:
        if df is None or df.empty:
            return DataFreshnessReport(
                status="completed",
                freshness_status="unknown",
                timestamp_column=None,
                latest_source_timestamp=None,
                previous_latest_source_timestamp=None,
                source_rows_checked=0,
                new_rows_since_last_run=0,
                stale_data_warning=True,
                hours_since_latest_source=None,
                reason="No source rows were available for freshness analysis.",
            )

        timestamp_column = self._find_timestamp_column(df, timestamp_columns)

        if not timestamp_column:
            return DataFreshnessReport(
                status="completed",
                freshness_status="unknown",
                timestamp_column=None,
                latest_source_timestamp=None,
                previous_latest_source_timestamp=None,
                source_rows_checked=len(df),
                new_rows_since_last_run=0,
                stale_data_warning=False,
                hours_since_latest_source=None,
                reason="No usable timestamp column was found. Freshness cannot be verified.",
            )

        timestamps = pd.to_datetime(df[timestamp_column], errors="coerce", utc=True)
        valid_timestamps = timestamps.dropna()

        if valid_timestamps.empty:
            return DataFreshnessReport(
                status="completed",
                freshness_status="unknown",
                timestamp_column=timestamp_column,
                latest_source_timestamp=None,
                previous_latest_source_timestamp=None,
                source_rows_checked=len(df),
                new_rows_since_last_run=0,
                stale_data_warning=True,
                hours_since_latest_source=None,
                reason=f"Timestamp column {timestamp_column} exists, but values could not be parsed.",
            )

        latest_ts = valid_timestamps.max()
        previous_ts = self._read_previous_latest_timestamp(previous_report_path)

        now = datetime.now(timezone.utc)
        hours_since_latest = max(
            (now - latest_ts.to_pydatetime()).total_seconds() / 3600,
            0,
        )

        freshness_status = "fresh"
        stale_warning = hours_since_latest > self.stale_after_hours
        new_rows = 0
        reason = "Fresh source data is available."

        if previous_ts is not None:
            new_rows = int((valid_timestamps > previous_ts).sum())

            if latest_ts <= previous_ts:
                freshness_status = "unchanged"
                reason = "No newer source timestamp was found compared with the previous run."
            elif new_rows > 0:
                freshness_status = "fresh"
                reason = f"{new_rows} newer source rows were found compared with the previous run."

        if stale_warning:
            freshness_status = "stale"
            reason = f"Latest source timestamp is older than {self.stale_after_hours} hours."

        return DataFreshnessReport(
            status="completed",
            freshness_status=freshness_status,
            timestamp_column=timestamp_column,
            latest_source_timestamp=latest_ts.isoformat(),
            previous_latest_source_timestamp=previous_ts.isoformat()
            if previous_ts is not None
            else None,
            source_rows_checked=len(df),
            new_rows_since_last_run=new_rows,
            stale_data_warning=stale_warning,
            hours_since_latest_source=round(hours_since_latest, 3),
            reason=reason,
        )

    def _find_timestamp_column(
        self,
        df: pd.DataFrame,
        timestamp_columns: tuple[str, ...],
    ) -> str | None:
        lower_map = {column.lower(): column for column in df.columns}

        for candidate in timestamp_columns:
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]

        return None

    def _read_previous_latest_timestamp(
        self,
        previous_report_path: str | Path | None,
    ) -> pd.Timestamp | None:
        if not previous_report_path:
            return None

        path = Path(previous_report_path)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(errors="ignore"))
            raw_ts = data.get("latest_source_timestamp")
            if not raw_ts:
                return None

            parsed = pd.to_datetime(raw_ts, errors="coerce", utc=True)
            if pd.isna(parsed):
                return None
            return parsed
        except Exception:
            return None
