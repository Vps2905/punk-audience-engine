from app.services.audience_run_history_service import (
    AudienceRunHistoryService,
)


def _summary(cohorts):
    return {
        "run_id": "run_module4d",
        "safe_export": {
            "approval_status": "pending_approval",
            "package": {
                "cohorts": cohorts,
            },
        },
    }


def test_duplicate_export_ids_are_collapsed():
    service = AudienceRunHistoryService()

    rows = service._build_cohort_rows(
        run_id="run_module4d",
        final_summary=_summary(
            [
                {
                    "export_cohort_id": "cohort_1",
                    "audience_name": "Lower quality",
                    "management_quality_score": 0.3,
                },
                {
                    "export_cohort_id": "cohort_1",
                    "audience_name": "Higher quality",
                    "management_quality_score": 0.8,
                },
            ]
        ),
    )

    assert len(rows) == 1
    assert rows[0]["export_cohort_id"] == "cohort_1"
    assert rows[0]["audience_name"] == "Higher quality"


def test_missing_export_id_is_deterministic():
    service = AudienceRunHistoryService()

    summary = _summary(
        [
            {
                "audience_name": "Montreal Cafe",
                "location_name": "montreal",
                "primary_poi_type": "cafe",
                "created_day_part": "evening",
                "cluster_id": 1,
            }
        ]
    )

    first = service._build_cohort_rows(
        run_id="run_module4d",
        final_summary=summary,
    )
    second = service._build_cohort_rows(
        run_id="run_module4d",
        final_summary=summary,
    )

    assert len(first) == 1
    assert first[0]["export_cohort_id"].startswith(
        "punk_audience_"
    )
    assert (
        first[0]["export_cohort_id"]
        == second[0]["export_cohort_id"]
    )


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


def test_cohort_persistence_is_conflict_safe():
    service = AudienceRunHistoryService()
    conn = FakeConn()

    rows = service._build_cohort_rows(
        run_id="run_module4d",
        final_summary=_summary(
            [
                {
                    "export_cohort_id": "cohort_1",
                    "audience_name": "Cafe",
                }
            ]
        ),
    )

    service._replace_cohorts(
        conn,
        "run_module4d",
        rows,
    )

    all_sql = "\n".join(
        call["sql"] for call in conn.calls
    )

    assert "pg_advisory_xact_lock" in all_sql
    assert "ON CONFLICT" in all_sql
    assert "run_id" in all_sql
    assert "export_cohort_id" in all_sql
