from datetime import datetime, timedelta, timezone

import pandas as pd

from app.agents.data_freshness_agent import DataFreshnessAgent


def test_data_freshness_agent_detects_fresh_data():
    now = datetime.now(timezone.utc)

    df = pd.DataFrame(
        {
            "created_at": [now.isoformat(), (now - timedelta(hours=1)).isoformat()],
            "location_name": ["montreal", "san francisco"],
            "primary_poi_type": ["restaurant", "coworking_space"],
        }
    )

    result = DataFreshnessAgent(stale_after_hours=48).analyze_dataframe(df)

    assert result["status"] == "completed"
    assert result["freshness_status"] == "fresh"
    assert result["timestamp_column"] == "created_at"
    assert result["source_rows_checked"] == 2
    assert result["latest_source_timestamp"] is not None
    assert result["stale_data_warning"] is False


def test_data_freshness_agent_detects_stale_data():
    old_time = datetime.now(timezone.utc) - timedelta(days=5)

    df = pd.DataFrame(
        {
            "created_at": [old_time.isoformat()],
            "location_name": ["montreal"],
            "primary_poi_type": ["restaurant"],
        }
    )

    result = DataFreshnessAgent(stale_after_hours=48).analyze_dataframe(df)

    assert result["status"] == "completed"
    assert result["freshness_status"] == "stale"
    assert result["stale_data_warning"] is True


def test_data_freshness_agent_handles_missing_timestamp_column():
    df = pd.DataFrame(
        {
            "location_name": ["montreal"],
            "primary_poi_type": ["restaurant"],
        }
    )

    result = DataFreshnessAgent().analyze_dataframe(df)

    assert result["status"] == "completed"
    assert result["freshness_status"] == "unknown"
    assert result["timestamp_column"] is None
    assert result["source_rows_checked"] == 1


def test_data_freshness_agent_detects_new_rows_since_previous_report(tmp_path):
    previous_report = tmp_path / "previous.json"
    previous_report.write_text(
        '{"latest_source_timestamp": "2026-07-01T10:00:00+00:00"}',
        encoding="utf-8",
    )

    df = pd.DataFrame(
        {
            "created_at": [
                "2026-07-01T09:00:00+00:00",
                "2026-07-01T11:00:00+00:00",
                "2026-07-01T12:00:00+00:00",
            ],
            "location_name": ["a", "b", "c"],
        }
    )

    result = DataFreshnessAgent(stale_after_hours=999999).analyze_dataframe(
        df,
        previous_report_path=previous_report,
    )

    assert result["status"] == "completed"
    assert result["freshness_status"] == "fresh"
    assert result["new_rows_since_last_run"] == 2
    assert result["previous_latest_source_timestamp"] is not None
