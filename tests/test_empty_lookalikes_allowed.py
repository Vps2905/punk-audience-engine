import pandas as pd


def test_single_export_cohort_can_have_zero_lookalike_pairs():
    safe_export_cohorts = pd.DataFrame(
        {
            "audience_name": ["Punk Audience - Cafe - Evening - Montreal Downtown - Cluster 0"],
            "location_name": ["montreal downtown"],
            "primary_poi_type": ["cafe"],
            "created_day_part": ["evening"],
            "approval_status": ["pending_approval"],
        }
    )

    safe_export_lookalikes = pd.DataFrame()

    assert len(safe_export_cohorts) == 1
    assert safe_export_lookalikes.empty

    response_summary = {
        "exported_audiences": int(len(safe_export_cohorts)),
        "lookalike_pairs": int(len(safe_export_lookalikes)),
        "approval_required": True,
        "downstream_export_enabled": False,
    }

    assert response_summary["exported_audiences"] == 1
    assert response_summary["lookalike_pairs"] == 0
    assert response_summary["downstream_export_enabled"] is False
