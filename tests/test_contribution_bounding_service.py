from app.services.contribution_bounding_service import (
    ContributionBoundingConfig,
    ContributionBoundingService,
)


def test_contribution_bounding_keeps_one_entity_per_cohort_per_day():
    rows = [
        {
            "entity_id": "user_1",
            "created_at": "2026-07-10T10:00:00Z",
            "location_name": "montreal",
            "primary_poi_type": "cafe",
            "created_day_part": "morning",
            "signal": "visit_1",
        },
        {
            "entity_id": "user_1",
            "created_at": "2026-07-10T11:00:00Z",
            "location_name": "montreal",
            "primary_poi_type": "cafe",
            "created_day_part": "morning",
            "signal": "visit_2",
        },
        {
            "entity_id": "user_2",
            "created_at": "2026-07-10T11:30:00Z",
            "location_name": "montreal",
            "primary_poi_type": "cafe",
            "created_day_part": "morning",
            "signal": "visit_3",
        },
    ]

    result = ContributionBoundingService().bound_events(rows)

    assert result["status"] == "completed"
    assert result["bounded"] is True
    assert result["input_rows"] == 3
    assert result["output_rows"] == 2
    assert result["dropped_rows"] == 1
    assert result["lineage"]["sensitivity_claim"]

    for row in result["bounded_rows"]:
        assert "entity_id" not in row


def test_contribution_bounding_allows_same_entity_in_different_cohort():
    rows = [
        {
            "entity_id": "user_1",
            "created_at": "2026-07-10T10:00:00Z",
            "location_name": "montreal",
            "primary_poi_type": "cafe",
            "created_day_part": "morning",
        },
        {
            "entity_id": "user_1",
            "created_at": "2026-07-10T10:30:00Z",
            "location_name": "montreal",
            "primary_poi_type": "restaurant",
            "created_day_part": "morning",
        },
    ]

    result = ContributionBoundingService().bound_events(rows)

    assert result["status"] == "completed"
    assert result["output_rows"] == 2
    assert result["dropped_rows"] == 0


def test_contribution_bounding_requires_entity_id_for_raw_events():
    rows = [
        {
            "created_at": "2026-07-10T10:00:00Z",
            "location_name": "montreal",
            "primary_poi_type": "cafe",
            "created_day_part": "morning",
        }
    ]

    result = ContributionBoundingService().bound_events(rows)

    assert result["status"] == "needs_upstream_bounding"
    assert result["bounded"] is False
    assert "entity_id" in result["missing_columns"]
