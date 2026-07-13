from app.api import audience_intelligence_run_history as api


class FakeService:
    def list_cohorts(self, **kwargs):
        return {
            "enabled": True,
            "status": "ok",
            "run_id": kwargs["run_id"],
            "count": 1,
            "cohorts": [
                {
                    "export_cohort_id": "cohort_1",
                    "location_name": kwargs["location"],
                    "primary_poi_type": kwargs["poi_type"],
                }
            ],
        }


def test_cohort_query_api_delegates_filters(monkeypatch):
    monkeypatch.setattr(
        api,
        "AudienceRunHistoryService",
        lambda: FakeService(),
    )

    result = api.list_run_cohorts(
        run_id="run_1",
        location="montreal downtown",
        poi_type="cafe",
        daypart="evening",
        approval_status="approved",
        min_quality=0.5,
        limit=25,
        offset=0,
    )

    assert result["status"] == "ok"
    assert result["count"] == 1
    assert (
        result["cohorts"][0]["export_cohort_id"]
        == "cohort_1"
    )
