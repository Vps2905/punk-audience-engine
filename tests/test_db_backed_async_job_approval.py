import pytest
from fastapi import HTTPException

from app.api import audience_intelligence_jobs as jobs


class FakeJobStore:
    def __init__(self, record):
        self.record = record
        self.saved = None

    def get(self, job_id):
        assert job_id == self.record["job_id"]
        return self.record

    def save(self, record):
        self.saved = record


class FakeApprovedHistory:
    def approve_run(self, **kwargs):
        assert kwargs["run_id"] == "run_123"
        return {
            "enabled": True,
            "status": "updated",
            "run_id": "run_123",
            "approval_status": "approved",
            "downstream_export_enabled": True,
            "meta_upload_performed": False,
        }

    def get_run(self, run_id):
        return {
            "enabled": True,
            "status": "ok",
            "run": {
                "run_id": run_id,
                "approval_status": "approved",
                "downstream_export_enabled": True,
                "final_summary": {
                    "safe_export": {
                        "approval_status": "approved",
                        "downstream_export_enabled": True,
                        "package": {
                            "cohorts": [
                                {"export_cohort_id": "cohort_1"}
                            ]
                        },
                    }
                },
            },
        }


class FakeBlockedHistory:
    def approve_run(self, **kwargs):
        return {
            "enabled": True,
            "status": "blocked",
            "run_id": kwargs["run_id"],
            "approval_status": "blocked_stale_source",
            "downstream_export_enabled": False,
            "blockers": ["Source data freshness is stale."],
            "meta_upload_performed": False,
        }

    def get_run(self, run_id):
        return {
            "enabled": True,
            "status": "ok",
            "run": {
                "run_id": run_id,
                "approval_status": "blocked_stale_source",
                "downstream_export_enabled": False,
                "final_summary": {
                    "safe_export": {
                        "approval_status": "blocked_stale_source",
                        "downstream_export_enabled": False,
                    }
                },
            },
        }


def _completed_record():
    return {
        "job_id": "job_123",
        "status": "completed",
        "result": {
            "run_id": "run_123",
            "safe_export": {
                "approval_status": "pending_approval",
                "downstream_export_enabled": False,
            },
        },
    }


def test_async_job_approval_delegates_to_run_history(monkeypatch):
    store = FakeJobStore(_completed_record())
    history = FakeApprovedHistory()

    monkeypatch.setattr(jobs, "job_store", store)
    monkeypatch.setattr(
        jobs,
        "AudienceRunHistoryService",
        lambda: history,
    )

    result = jobs.approve_job_export(
        "job_123",
        jobs.ApprovalRequest(
            approver="reviewer",
            note="approved",
        ),
    )

    assert result["approval_status"] == "approved"
    assert result["downstream_export_enabled"] is True
    assert store.saved is not None
    assert (
        store.saved["result"]["safe_export"]["approval_status"]
        == "approved"
    )


def test_async_job_approval_preserves_fail_closed_decision(
    monkeypatch,
):
    store = FakeJobStore(_completed_record())
    history = FakeBlockedHistory()

    monkeypatch.setattr(jobs, "job_store", store)
    monkeypatch.setattr(
        jobs,
        "AudienceRunHistoryService",
        lambda: history,
    )

    with pytest.raises(HTTPException) as exc_info:
        jobs.approve_job_export(
            "job_123",
            jobs.ApprovalRequest(
                approver="reviewer",
                note="attempt",
            ),
        )

    assert exc_info.value.status_code == 400
    assert store.saved is not None
    assert (
        store.saved["result"]["approval_status"]
        == "blocked_stale_source"
    )
    assert (
        store.saved["result"]["downstream_export_enabled"]
        is False
    )
