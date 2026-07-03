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
    "hashed",
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
        print("Usage: python scripts/validate_cohort_management_artifacts.py <cohort_management_output_dir>")
        return 2

    base = Path(sys.argv[1])

    clusters_path = base / "cohort_clusters.csv"
    top_path = base / "top_cohorts.csv"
    lookalikes_path = base / "lookalike_cohorts.csv"
    report_path = base / "cohort_quality_report.json"
    manifest_path = base / "cohort_management_manifest.json"

    for path in [clusters_path, top_path, lookalikes_path, report_path, manifest_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    clusters = pd.read_csv(clusters_path)
    top = pd.read_csv(top_path)
    lookalikes = pd.read_csv(lookalikes_path)
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    assert manifest["status"] == "completed"
    assert report["status"] == "completed"

    assert len(clusters) == manifest["managed_cohorts"]
    assert len(top) == manifest["top_cohorts"]
    assert len(lookalikes) == manifest["lookalike_pairs"]

    assert manifest["cluster_count"] == clusters["cluster_id"].nunique()
    assert manifest["export_ready_cohorts"] == int(clusters["export_ready"].sum())

    assert manifest["raw_maids_exported"] is False
    assert manifest["raw_observations_exported"] is False
    assert manifest["raw_lat_lng_exported"] is False
    assert manifest["raw_email_exported"] is False
    assert manifest["raw_phone_exported"] is False
    assert manifest["individual_user_data_exported"] is False

    assert report["raw_identifiers_exported"] is False
    assert report["individual_user_data_exported"] is False

    assert_safe_columns(clusters, "cohort_clusters")
    assert_safe_columns(top, "top_cohorts")
    assert_safe_columns(lookalikes, "lookalike_cohorts")

    required_cluster_cols = [
        "cohort_index",
        "cluster_id",
        "cluster_size",
        "management_quality_score",
        "cluster_coherence_score",
        "export_ready",
    ]

    for col in required_cluster_cols:
        assert col in clusters.columns, f"Missing cluster column: {col}"

    assert clusters["management_quality_score"].between(0, 1).all()
    assert clusters["cluster_coherence_score"].between(0, 1).all()
    assert clusters["privacy_status"].eq("passed").all()

    if len(lookalikes) > 0:
        assert "similarity_score" in lookalikes.columns
        assert lookalikes["similarity_score"].between(-1, 1).all()

    print("Cohort management artifact validation passed ✅")
    print("Managed cohorts:", len(clusters))
    print("Clusters:", manifest["cluster_count"])
    print("Top cohorts:", len(top))
    print("Lookalike pairs:", len(lookalikes))
    print("Export ready:", manifest["export_ready_cohorts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
