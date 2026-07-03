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
    "user_id",
    "client_id",
]

SAFE_AGGREGATE_COLUMNS = {
    "total_maid_volume",
    "noisy_maid_volume",
    "safe_maid_volume",
    "total_observations",
    "observation_count",
    "observations_count",
    "safe_observation_count",
}


def assert_safe_columns(df: pd.DataFrame, name: str) -> None:
    for col in df.columns:
        lower = col.lower()

        if lower in SAFE_AGGREGATE_COLUMNS:
            continue

        for blocked in BLOCKED_COLUMNS:
            if lower == blocked or lower.startswith(f"{blocked}_") or lower.endswith(f"_{blocked}"):
                raise AssertionError(f"{name} leaked blocked column: {col}")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_safe_export_artifacts.py <safe_export_output_dir>")
        return 2

    base = Path(sys.argv[1])

    cohorts_path = base / "safe_export_cohorts.csv"
    lookalikes_path = base / "safe_export_lookalikes.csv"
    payload_path = base / "safe_export_payload.json"
    approval_path = base / "export_approval_request.json"
    manifest_path = base / "safe_export_manifest.json"

    for path in [cohorts_path, lookalikes_path, payload_path, approval_path, manifest_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    cohorts = pd.read_csv(cohorts_path)
    lookalikes = pd.read_csv(lookalikes_path)
    payload = json.loads(payload_path.read_text())
    approval = json.loads(approval_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    assert manifest["status"] == "completed"
    assert manifest["approval_required"] is True
    assert manifest["approval_status"] == "pending_approval"
    assert manifest["downstream_export_enabled"] is False
    assert manifest["export_blocked_until_approved"] is True

    assert approval["approval_status"] == "pending_approval"
    assert approval["export_blocked_until_approved"] is True
    assert approval["review_required_before_downstream_delivery"] is True

    assert payload["approval_status"] == "pending_approval"
    assert payload["downstream_export_enabled"] is False
    assert payload["privacy_guarantees"]["raw_maids_exported"] is False
    assert payload["privacy_guarantees"]["hashed_identifiers_exported"] is False
    assert payload["privacy_guarantees"]["raw_observations_exported"] is False
    assert payload["privacy_guarantees"]["raw_lat_lng_exported"] is False
    assert payload["privacy_guarantees"]["individual_user_data_exported"] is False

    assert len(cohorts) == manifest["exported_cohorts"]
    assert len(lookalikes) == manifest["exported_lookalike_pairs"]
    assert len(cohorts) > 0

    assert manifest["raw_maids_exported"] is False
    assert manifest["hashed_identifiers_exported"] is False
    assert manifest["raw_observations_exported"] is False
    assert manifest["raw_lat_lng_exported"] is False
    assert manifest["raw_email_exported"] is False
    assert manifest["raw_phone_exported"] is False
    assert manifest["individual_user_data_exported"] is False

    assert_safe_columns(cohorts, "safe_export_cohorts")
    assert_safe_columns(lookalikes, "safe_export_lookalikes")

    required_cohort_cols = [
        "export_cohort_id",
        "audience_name",
        "cohort_index",
        "cluster_id",
        "location_name",
        "primary_poi_type",
        "management_quality_score",
        "privacy_mode",
        "data_safety_status",
        "export_status",
    ]

    for col in required_cohort_cols:
        assert col in cohorts.columns, f"Missing safe export cohort column: {col}"

    assert cohorts["export_status"].eq("pending_approval").all()
    assert cohorts["privacy_mode"].eq("aggregated_dp_safe").all()
    assert cohorts["data_safety_status"].eq("safe_aggregated_no_raw_identifiers").all()
    assert cohorts["management_quality_score"].between(0, 1).all()

    assert cohorts["export_cohort_id"].astype(str).str.startswith("punk_audience_").all()

    if len(lookalikes) > 0:
        assert "similarity_score" in lookalikes.columns
        assert lookalikes["similarity_score"].between(-1, 1).all()
        assert lookalikes["data_safety_status"].eq("safe_aggregated_no_raw_identifiers").all()

    assert "checksums_sha256" in manifest
    assert len(manifest["checksums_sha256"]) >= 4

    print("Safe export artifact validation passed ✅")
    print("Exported cohorts:", len(cohorts))
    print("Exported lookalike pairs:", len(lookalikes))
    print("Approval status:", manifest["approval_status"])
    print("Downstream enabled:", manifest["downstream_export_enabled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
