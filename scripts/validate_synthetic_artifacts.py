from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


BLOCKED_TOKENS = [
    "raw_maid",
    "device_id",
    "email",
    "phone",
    "latitude",
    "longitude",
    "observation",
    "observations",
]


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_synthetic_artifacts.py <synthetic_output_dir>")
        return 2

    base = Path(sys.argv[1])
    csv_path = base / "synthetic_safe_seed_profiles.csv"
    manifest_path = base / "synthetic_manifest.json"
    approval_path = base / "approval_request.json"
    schema_path = base / "synthetic_safe_input_schema.json"

    for path in [csv_path, manifest_path, approval_path, schema_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    df = pd.read_csv(csv_path)
    manifest = json.loads(manifest_path.read_text())
    approval = json.loads(approval_path.read_text())
    schema = json.loads(schema_path.read_text())

    assert len(df) == manifest["rows_generated"], "CSV row count does not match manifest"
    assert manifest["status"] == "completed"
    assert manifest["approval_status"] == "pending_approval"
    assert approval["status"] == "pending_approval"
    assert schema["production_safe"] is True
    assert schema["raw_identifiers_present"] is False
    assert schema["raw_coordinates_present"] is False
    assert schema["individual_rows_present"] is False

    safety_flags = [
        "raw_maids_exported",
        "raw_observations_exported",
        "raw_lat_lng_exported",
        "raw_email_exported",
        "raw_phone_exported",
        "individual_user_data_exported",
    ]

    for flag in safety_flags:
        assert manifest[flag] is False, f"Unsafe manifest flag: {flag}"

    for col in df.columns:
        lower = col.lower()
        for token in BLOCKED_TOKENS:
            assert token not in lower, f"Blocked column leaked: {col}"

    if "trait_text" in df.columns:
        assert not df["trait_text"].astype(str).str.contains("sdv-id", case=False, na=False).any()

    if "quality_score" in df.columns:
        assert df["quality_score"].between(0, 1).all()

    print("Synthetic artifact validation passed ✅")
    print("Rows:", len(df))
    print("Engine:", manifest["engine_used"])
    print("Approval:", manifest["approval_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
