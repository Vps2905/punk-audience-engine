import pytest
from fastapi import HTTPException

from app.api import audience_intelligence_modules as module
from app.core.audience_request_context import AudienceRequestContext


CONTEXT = AudienceRequestContext(
    tenant_id="tenant-a",
    request_id="request-meta",
)


class FakeHistory:
    def __init__(self, run):
        self.run = run
        self.events = []

    def get_run(self, run_id, *, tenant_id):
        assert run_id == "run_1"
        assert tenant_id == "tenant-a"
        return {
            "enabled": True,
            "status": "ok",
            "run": self.run,
        }

    def record_event(
        self,
        *,
        tenant_id,
        run_id,
        event_type,
        actor,
        details,
    ):
        self.events.append(
            {
                "tenant_id": tenant_id,
                "run_id": run_id,
                "event_type": event_type,
                "actor": actor,
                "details": details,
            }
        )
        return {
            "enabled": True,
            "status": "recorded",
            "run_id": run_id,
            "event_type": event_type,
        }


def _approved_run():
    return {
        "run_id": "run_1",
        "approval_status": "approved",
        "downstream_export_enabled": True,
        "final_summary": {
            "safe_export": {
                "approval_status": "approved",
                "downstream_export_enabled": True,
                "storage_backend": "run_history_jsonb",
                "package": {
                    "cohorts": [
                        {
                            "export_cohort_id": "cohort_1",
                            "audience_name": "Montreal Cafe",
                            "location_name": "montreal",
                            "primary_poi_type": "cafe",
                            "created_day_part": "evening",
                            "lookback_bucket": "31_90d",
                            "management_quality_score": 0.82,
                            "privacy_mode": "aggregated_dp_safe",
                            "data_safety_status": (
                                "safe_aggregated_no_raw_identifiers"
                            ),
                        }
                    ]
                },
            }
        },
    }


def _blocked_run():
    run = _approved_run()
    run["approval_status"] = "blocked_stale_source"
    run["downstream_export_enabled"] = False

    safe_export = run["final_summary"]["safe_export"]
    safe_export["approval_status"] = "blocked_stale_source"
    safe_export["downstream_export_enabled"] = False

    return run


def test_db_meta_seed_requires_approved_run(monkeypatch):
    history = FakeHistory(_blocked_run())

    monkeypatch.setattr(
        module,
        "AudienceRunHistoryService",
        lambda: history,
    )

    with pytest.raises(HTTPException) as exc_info:
        module.export_meta_seed_payload(
            "cohort_1",
            module.MetaExportRequest(
                run_id="run_1",
                actor="reviewer",
            ),
            CONTEXT,
        )

    assert exc_info.value.status_code == 403
    assert history.events[-1]["event_type"] == (
        "meta_seed_payload_blocked"
    )


def test_db_meta_seed_generated_in_memory(monkeypatch):
    history = FakeHistory(_approved_run())

    monkeypatch.setattr(
        module,
        "AudienceRunHistoryService",
        lambda: history,
    )

    result = module.export_meta_seed_payload(
        "cohort_1",
        module.MetaExportRequest(
            run_id="run_1",
            actor="reviewer",
        ),
        CONTEXT,
    )

    assert result["status"] == "completed"
    assert result["storage_backend"] == (
        "memory_with_postgres_audit"
    )
    assert result["output_path"].startswith("memory://")
    assert result["payload"]["approval_status"] == "approved"
    assert (
        result["payload"]["downstream_export_enabled"]
        is True
    )
    assert (
        result["payload"]["safe_seed_strategy"][
            "raw_maids_included"
        ]
        is False
    )
    assert history.events[-1]["event_type"] == (
        "meta_seed_payload_generated"
    )


def test_production_local_meta_path_is_blocked(monkeypatch):
    monkeypatch.setattr(
        module,
        "local_file_storage_allowed",
        lambda: False,
    )

    with pytest.raises(HTTPException) as exc_info:
        module.export_meta_seed_payload(
            "cohort_1",
            module.MetaExportRequest(
                safe_export_dir="/tmp/export",
            ),
            CONTEXT,
        )

    assert exc_info.value.status_code == 400
