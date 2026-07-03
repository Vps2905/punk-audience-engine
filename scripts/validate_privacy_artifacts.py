from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


BLOCKED_COLUMNS = [
    "maid",
    "raw_maid",
    "device_id",
    "email",
    "phone",
    "lat",
    "lng",
    "latitude",
    "longitude",
    "raw_observation",
    "observations",
    "hashed",
]


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_privacy_artifacts.py <privacy_output_dir>")
        return 2

    base = Path(sys.argv[1])
    feature_path = base / "clean_feature_table.csv"
    privacy_report_path = base / "privacy_report.json"
    lineage_path = base / "lineage_report.json"
    schema_path = base / "privacy_input_schema.json"

    for path in [feature_path, privacy_report_path, lineage_path, schema_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    df = pd.read_csv(feature_path)
    report = json.loads(privacy_report_path.read_text())
    lineage = json.loads(lineage_path.read_text())
    schema = json.loads(schema_path.read_text())

    assert report["status"] == "completed"
    assert report["export_safety"]["raw_maids_exported"] is False
    assert report["export_safety"]["hashed_real_maids_exported"] is False
    assert report["export_safety"]["raw_observations_exported"] is False
    assert report["export_safety"]["raw_lat_lng_exported"] is False
    assert report["export_safety"]["raw_email_exported"] is False
    assert report["export_safety"]["raw_phone_exported"] is False
    assert report["export_safety"]["individual_user_data_exported"] is False
    assert report["export_safety"]["aggregated_only"] is True

    assert report["differential_privacy"]["enabled"] is True
    assert report["differential_privacy"]["privacy_budget_recorded"] is True
    assert report["k_anonymity"]["enabled"] is True

    assert schema["output_is_aggregated_only"] is True
    assert lineage["privacy_controls"]["raw_identifiers_exported"] is False
    assert lineage["privacy_controls"]["raw_coordinates_exported"] is False

    safe_aggregate_columns = {
        "total_maid_volume",
        "noisy_maid_volume",
        "safe_maid_volume",
        "total_observations",
        "observation_count",
        "observations_count",
        "safe_observation_count",
    }

    for col in df.columns:
        lower = col.lower()

        if lower in safe_aggregate_columns:
            continue

        for blocked in BLOCKED_COLUMNS:
            assert blocked not in lower, f"Blocked column leaked: {col}"

    assert len(df) == report["output_safe_cohorts"]
    assert df["privacy_status"].eq("passed").all()
    assert df["quality_score"].between(0, 1).all()
    assert (df["noisy_maid_volume"] >= report["k_anonymity"]["k_min"]).all()

    print("Privacy artifact validation passed ✅")
    print("Safe cohorts:", len(df))
    print("Blocked cohorts:", report["blocked_cohorts"])
    print("k_min:", report["k_anonymity"]["k_min"])
    print("epsilon:", report["differential_privacy"]["epsilon"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
