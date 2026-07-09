from app.services.sensitive_poi_privacy_risk_service import SensitivePOIPrivacyRiskService


def test_restaurant_audience_is_low_risk_but_still_approval_gated():
    report = SensitivePOIPrivacyRiskService().assess(
        prompt="Build restaurant evening audience in Montreal",
        selected_cohorts=[
            {
                "audience_name": "Restaurant - Evening - Montreal",
                "location_name": "montreal",
                "primary_poi_type": "restaurant",
                "created_day_part": "evening",
            }
        ],
    )

    assert report["overall_decision"] == "allow_approval_gated_export"
    assert report["safe_to_export"] is False
    assert report["approval_required"] is True
    assert report["downstream_export_enabled"] is False
    assert report["assessed_audiences"][0]["risk_level"] == "low"


def test_healthcare_sensitive_poi_blocks_export():
    report = SensitivePOIPrivacyRiskService().assess(
        prompt="Promote a healthcare checkup campaign",
        selected_cohorts=[
            {
                "audience_name": "Hospital - Afternoon - Montreal",
                "location_name": "montreal",
                "primary_poi_type": "hospital",
                "created_day_part": "afternoon",
            }
        ],
    )

    assert report["overall_decision"] == "block_export"
    assert report["blocked_sensitive_count"] == 1
    assert report["assessed_audiences"][0]["decision"] == "block_export"


def test_casino_requires_review_not_direct_export():
    report = SensitivePOIPrivacyRiskService().assess(
        prompt="Casino entertainment offer in Quebec",
        selected_cohorts=[
            {
                "audience_name": "Casino - Evening - Quebec",
                "location_name": "quebec",
                "primary_poi_type": "casino",
                "created_day_part": "evening",
            }
        ],
    )

    assert report["overall_decision"] == "review_required"
    assert report["review_required_count"] == 1
    assert report["safe_to_export"] is False
    assert report["downstream_export_enabled"] is False


def test_sensitive_prompt_blocks_even_if_poi_is_generic():
    report = SensitivePOIPrivacyRiskService().assess(
        prompt="Find people likely dealing with mental health or addiction problems",
        selected_cohorts=[
            {
                "audience_name": "Point Of Interest - Evening - City",
                "location_name": "city",
                "primary_poi_type": "point_of_interest",
                "created_day_part": "evening",
            }
        ],
    )

    assert report["overall_decision"] == "block_export"
    assert report["prompt_risk"]["decision"] == "block_export"


def test_healthcare_prompt_blocks_even_when_no_selected_rows():
    report = SensitivePOIPrivacyRiskService().assess(
        prompt="Build an audience of people visiting hospitals and clinics in San Francisco during afternoon.",
        selected_cohorts=[],
    )

    assert report["overall_decision"] == "block_export"
    assert report["prompt_risk"]["decision"] == "block_export"
    assert report["safe_to_export"] is False
    assert report["downstream_export_enabled"] is False
