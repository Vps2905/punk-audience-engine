import json

import pandas as pd

from app.services.audience_run_history_service import (
    AudienceRunHistoryService,
)


def _summary():
    return {
        "run_id": "run_module_4",
        "approval_status": "blocked_stale_source",
        "safe_export": {
            "approval_status": "blocked_stale_source",
            "downstream_export_enabled": False,
            "package": {
                "cohorts": [
                    {
                        "export_cohort_id": "cohort_1",
                        "audience_name": "Montreal Cafe",
                        "location_name": "montreal downtown",
                        "primary_poi_type": "cafe",
                        "created_day_part": "evening",
                        "lookback_bucket": "31_90d",
                        "quality_score": 0.64,
                        "management_quality_score": 0.82,
                        "privacy_mode": "aggregated_dp_safe",
                        "data_safety_status": (
                            "safe_aggregated_no_raw_identifiers"
                        ),
                        "export_status": "pending_approval",
                    },
                    {
                        "export_cohort_id": "cohort_2",
                        "audience_name": "Montreal Coffee",
                        "location_name": "montreal downtown",
                        "primary_poi_type": "coffee_shop",
                        "created_day_part": "evening",
                        "lookback_bucket": "0_30d",
                        "quality_score": 0.61,
                        "management_quality_score": 0.79,
                        "privacy_mode": "aggregated_dp_safe",
                        "data_safety_status": (
                            "safe_aggregated_no_raw_identifiers"
                        ),
                        "export_status": "pending_approval",
                    },
                ]
            },
        },
    }


def test_package_cohorts_are_preferred_over_selected_rows():
    service = AudienceRunHistoryService()

    selected = pd.DataFrame(
        [
            {
                "audience_name": "Pre-export row",
                "location_name": "wrong source",
            }
        ]
    )

    rows = service._build_cohort_rows(
        run_id="run_module_4",
        final_summary=_summary(),
        selected_cohorts=selected,
    )

    assert len(rows) == 2
    assert rows[0]["export_cohort_id"] == "cohort_1"
    assert rows[0]["_normalized_source"] == (
        "safe_export.package.cohorts"
    )
    assert rows[0][
        "_normalized_approval_status"
    ] == "blocked_stale_source"


class FakeConn:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        self.calls.append(
            {
                "sql": str(statement),
                "params": params or {},
            }
        )


def test_normalized_cohort_columns_are_inserted():
    service = AudienceRunHistoryService()
    conn = FakeConn()

    rows = service._build_cohort_rows(
        run_id="run_module_4",
        final_summary=_summary(),
    )

    service._replace_cohorts(
        conn,
        "tenant-a",
        "run_module_4",
        rows,
    )

    insert_calls = [
        call
        for call in conn.calls
        if "INSERT INTO public.audience_run_cohorts"
        in call["sql"]
    ]

    assert len(insert_calls) == 2

    params = insert_calls[0]["params"]

    assert params["export_cohort_id"] == "cohort_1"
    assert params["tenant_id"] == "tenant-a"
    assert params["lookback_bucket"] == "31_90d"
    assert params["management_quality_score"] == 0.82
    assert params["approval_status"] == (
        "blocked_stale_source"
    )
    assert params["privacy_mode"] == (
        "aggregated_dp_safe"
    )

    metadata = json.loads(params["metadata"])

    assert metadata["normalized_source"] == (
        "safe_export.package.cohorts"
    )
    assert "_normalized_source" not in metadata
    assert (
        "_normalized_approval_status"
        not in metadata
    )
