from __future__ import annotations

from app.api import audience_intelligence_jobs as jobs


def test_safe_result_preserves_supervisor_recovery_metadata():
    result = {
        "status": "completed",
        "run_id": "run_safe_job_metadata",
        "approval_status": "pending_approval",
        "downstream_export_enabled": False,
        "supervisor_decision": {
            "route": "pending_approval",
            "stage": "approval",
            "terminal": True,
            "awaiting_input": True,
            "approval_required": True,
            "downstream_export_enabled": False,
            "reason_codes": ["approval_required"],
            "next_action": "Wait for approval.",
            "unsafe_extra": "secret",
        },
        "supervisor_route": "pending_approval",
        "supervisor_stage": "approval",
        "supervisor_reason_codes": [
            "approval_required",
        ],
        "supervisor_trace": [
            {
                "event": "supervisor_evaluated",
                "route": "pending_approval",
                "secret": "must not persist",
            }
        ],
        "graph_terminal_status": "pending_approval",
        "supervisor_graph_trace": [
            {
                "event": "graph_terminated",
                "terminal_status": "pending_approval",
                "private_prompt": "must not persist",
            }
        ],
        "supervisor_recovery": {
            "attempt_count": 2,
            "max_attempts": 2,
            "retried": True,
            "exhausted": False,
            "last_error_category": "timeout",
            "provider_message": "must not persist",
        },
        "supervisor_graph_error_type": "TimeoutError",
    }

    safe = jobs._safe_result_for_job(result)

    assert safe["supervisor_route"] == "pending_approval"
    assert safe["graph_terminal_status"] == (
        "pending_approval"
    )
    assert safe["supervisor_recovery"] == {
        "attempt_count": 2,
        "max_attempts": 2,
        "retried": True,
        "exhausted": False,
        "last_error_category": "timeout",
    }
    assert "secret" not in str(safe)
    assert "private_prompt" not in str(safe)
    assert "provider_message" not in str(safe)


def test_safe_result_keeps_legacy_contract_when_no_supervisor():
    safe = jobs._safe_result_for_job(
        {
            "status": "completed",
            "run_id": "run_legacy",
        }
    )

    assert "supervisor_decision" not in safe
    assert "supervisor_recovery" not in safe
    assert "graph_terminal_status" not in safe


def test_safe_job_error_never_contains_exception_message():
    error = jobs._safe_job_error(
        ValueError(
            "database password and private prompt"
        )
    )

    assert error == (
        "audience_job_failed:"
        "non_transient:"
        "ValueError"
    )
    assert "password" not in error
    assert "private prompt" not in error


def test_background_failure_persists_sanitized_error(
    monkeypatch,
):
    class FakeJobStore:
        backend = "local"

        def __init__(self):
            self.record = {
                "job_id": "job_failure_test",
                "payload": {
                    "prompt": "Find cafe visitors",
                },
            }
            self.updates = []

        def get(self, job_id):
            return self.record

        def update_status(self, job_id, **kwargs):
            self.updates.append(
                {
                    "job_id": job_id,
                    **kwargs,
                }
            )

    class BrokenAgent:
        def run(self, **kwargs):
            raise ValueError(
                "secret API key and private prompt"
            )

    store = FakeJobStore()

    monkeypatch.setattr(
        jobs,
        "job_store",
        store,
    )
    monkeypatch.setattr(
        jobs,
        "_audience_execution_agent",
        lambda: BrokenAgent(),
    )

    jobs._run_job_background("job_failure_test")

    failed = store.updates[-1]

    assert failed["status"] == "failed"
    assert failed["stage"] == "failed"
    assert failed["error"] == (
        "audience_job_failed:"
        "non_transient:"
        "ValueError"
    )
    assert "secret API key" not in str(failed)
    assert "private prompt" not in str(failed)


def test_run_history_failure_uses_safe_specific_code():
    error = jobs._safe_job_error(
        jobs.AudienceRunHistoryPersistenceError(
            "database URL and private details"
        )
    )

    assert error == (
        "audience_job_failed:"
        "run_history_persistence:"
        "AudienceRunHistoryPersistenceError"
    )
    assert "database URL" not in error
    assert "private details" not in error
