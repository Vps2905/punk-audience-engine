from datetime import datetime
from pathlib import Path

import pytest

from app.services.ingestion_lineage_service import (
    IngestionLineageService,
    LineageEvent,
)


def test_lineage_records_and_lists_events_in_order(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'lineage.db'}"
    service = IngestionLineageService(database_url=db_url)

    service.record_event(
        LineageEvent(
            job_id="job_1",
            run_id="run_1",
            stage="source",
            transformation="csv_ingest",
            input_ref="source.csv",
            output_ref="raw_table",
            input_rows=100,
            output_rows=100,
            dropped_rows=0,
            details={"source_type": "csv"},
        )
    )

    service.record_event(
        LineageEvent(
            job_id="job_1",
            run_id="run_1",
            stage="contribution_bounding",
            transformation="max_one_contribution_per_entity_per_day",
            input_ref="raw_table",
            output_ref="bounded_table",
            input_rows=100,
            output_rows=80,
            dropped_rows=20,
            details={"entity_id_removed": True},
        )
    )

    result = service.list_events(job_id="job_1")

    assert result["enabled"] is True
    assert result["status"] == "ok"
    assert result["event_count"] == 2
    assert [event["stage"] for event in result["events"]] == [
        "source",
        "contribution_bounding",
    ]
    assert result["events"][1]["dropped_rows"] == 20
    assert result["events"][1]["details"]["entity_id_removed"] is True


def test_lineage_chain_summary_detects_privacy_controls(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'lineage.db'}"
    service = IngestionLineageService(database_url=db_url)

    events = [
        LineageEvent(
            job_id="job_2",
            stage="source",
            transformation="api_ingest",
            input_rows=1000,
            output_rows=1000,
        ),
        LineageEvent(
            job_id="job_2",
            stage="contribution_bounding",
            transformation="max_one_contribution_per_entity_per_day",
            input_rows=1000,
            output_rows=850,
            dropped_rows=150,
        ),
        LineageEvent(
            job_id="job_2",
            stage="dp_safe",
            transformation="gaussian_differential_privacy_noise",
            input_rows=850,
            output_rows=850,
            details={"k_anonymity_enforced": True},
        ),
        LineageEvent(
            job_id="job_2",
            stage="synthetic_export",
            transformation="sdv_synthetic_generation",
            input_rows=850,
            output_rows=5000,
        ),
    ]

    for event in events:
        service.record_event(event)

    summary = service.build_chain_summary(job_id="job_2")

    assert summary["enabled"] is True
    assert summary["event_count"] == 4
    assert summary["has_source"] is True
    assert summary["has_privacy_step"] is True
    assert summary["has_output"] is True
    assert "contribution_bounding" in summary["privacy_controls_detected"]
    assert "differential_privacy" in summary["privacy_controls_detected"]
    assert "k_anonymity" in summary["privacy_controls_detected"]
    assert "synthetic_generation" in summary["privacy_controls_detected"]


def test_lineage_event_validation_rejects_bad_event(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'lineage.db'}"
    service = IngestionLineageService(database_url=db_url)

    with pytest.raises(ValueError, match="job_id is required"):
        service.record_event(
            LineageEvent(
                job_id="",
                stage="source",
                transformation="csv_ingest",
            )
        )

    with pytest.raises(ValueError, match="input_rows must be >= 0"):
        service.record_event(
            LineageEvent(
                job_id="job_3",
                stage="source",
                transformation="csv_ingest",
                input_rows=-1,
            )
        )


def test_lineage_sanitizes_non_json_details(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'lineage.db'}"
    service = IngestionLineageService(database_url=db_url)

    service.record_event(
        LineageEvent(
            job_id="job_4",
            stage="source",
            transformation="csv_ingest",
            details={"loaded_at": datetime(2026, 7, 10, 10, 0, 0)},
        )
    )

    result = service.list_events(job_id="job_4")
    loaded_at = result["events"][0]["details"]["loaded_at"]

    assert "2026-07-10" in loaded_at
