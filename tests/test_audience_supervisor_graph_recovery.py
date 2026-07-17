from __future__ import annotations

from app.agents.audience_supervisor_graph import (
    AudienceSupervisorGraph,
)
from app.services.audience_supervisor_recovery_service import (
    AudienceSupervisorRecoveryService,
)


def completed_result():
    return {
        "status": "completed",
        "run_id": "run_recovery_test",
        "approval_status": "pending_approval",
        "downstream_export_enabled": False,
        "safe_export": {
            "approval_status": "pending_approval",
            "downstream_export_enabled": False,
        },
    }


def test_transient_failure_retries_then_succeeds(
    monkeypatch,
):
    monkeypatch.setenv(
        "AUTONOMOUS_SUPERVISOR_MAX_ORCHESTRATOR_ATTEMPTS",
        "2",
    )
    monkeypatch.setenv(
        "AUTONOMOUS_SUPERVISOR_RETRY_BACKOFF_SECONDS",
        "0.25",
    )

    class FlakyOrchestrator:
        def __init__(self):
            self.calls = 0

        def run(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError(
                    "secret provider detail must not leak"
                )
            return completed_result()

    orchestrator = FlakyOrchestrator()
    sleeps = []
    graph = AudienceSupervisorGraph(
        orchestrator_factory=lambda: orchestrator,
        sleep_fn=sleeps.append,
    )

    result = graph.run(
        prompt="Find cafe visitors in Montreal"
    )

    assert orchestrator.calls == 2
    assert sleeps == [0.25]
    assert result["supervisor_route"] == (
        "pending_approval"
    )
    assert result["supervisor_recovery"] == {
        "attempt_count": 2,
        "max_attempts": 2,
        "retried": True,
        "exhausted": False,
        "last_error_category": "timeout",
    }
    assert "secret provider detail" not in str(result)


def test_non_transient_failure_does_not_retry(
    monkeypatch,
):
    monkeypatch.setenv(
        "AUTONOMOUS_SUPERVISOR_MAX_ORCHESTRATOR_ATTEMPTS",
        "3",
    )

    class InvalidOrchestrator:
        def __init__(self):
            self.calls = 0

        def run(self, **kwargs):
            self.calls += 1
            raise ValueError("invalid request with secret")

    orchestrator = InvalidOrchestrator()
    graph = AudienceSupervisorGraph(
        orchestrator_factory=lambda: orchestrator,
        sleep_fn=lambda _: None,
    )

    result = graph.run(
        prompt="Find cafe visitors in Montreal"
    )

    assert orchestrator.calls == 1
    assert result["supervisor_route"] == "failed"
    assert result["supervisor_recovery"][
        "last_error_category"
    ] == "non_transient"
    assert result["supervisor_recovery"]["exhausted"] is False
    assert "invalid request with secret" not in str(result)


def test_transient_failure_exhausts_bounded_attempts(
    monkeypatch,
):
    monkeypatch.setenv(
        "AUTONOMOUS_SUPERVISOR_MAX_ORCHESTRATOR_ATTEMPTS",
        "3",
    )
    monkeypatch.setenv(
        "AUTONOMOUS_SUPERVISOR_RETRY_BACKOFF_SECONDS",
        "0",
    )

    class BrokenOrchestrator:
        def __init__(self):
            self.calls = 0

        def run(self, **kwargs):
            self.calls += 1
            raise ConnectionError("database credential leak")

    orchestrator = BrokenOrchestrator()
    graph = AudienceSupervisorGraph(
        orchestrator_factory=lambda: orchestrator,
        sleep_fn=lambda _: None,
    )

    result = graph.run(
        prompt="Find cafe visitors in Montreal"
    )

    assert orchestrator.calls == 3
    assert result["supervisor_route"] == "failed"
    assert result["graph_terminal_status"] == "failed"
    assert result["supervisor_recovery"] == {
        "attempt_count": 3,
        "max_attempts": 3,
        "retried": True,
        "exhausted": True,
        "last_error_category": "connection",
    }
    assert "database credential leak" not in str(result)


def test_recovery_configuration_is_bounded(
    monkeypatch,
):
    service = AudienceSupervisorRecoveryService()

    monkeypatch.setenv(
        "AUTONOMOUS_SUPERVISOR_MAX_ORCHESTRATOR_ATTEMPTS",
        "99",
    )
    monkeypatch.setenv(
        "AUTONOMOUS_SUPERVISOR_RETRY_BACKOFF_SECONDS",
        "99",
    )

    assert service.max_attempts() == 3
    assert service.retry_backoff_seconds() == 5.0

    monkeypatch.setenv(
        "AUTONOMOUS_SUPERVISOR_MAX_ORCHESTRATOR_ATTEMPTS",
        "invalid",
    )
    monkeypatch.setenv(
        "AUTONOMOUS_SUPERVISOR_RETRY_BACKOFF_SECONDS",
        "invalid",
    )

    assert service.max_attempts() == 2
    assert service.retry_backoff_seconds() == 0.25


def test_http_status_classification_is_conservative():
    class HttpError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code

    service = AudienceSupervisorRecoveryService()

    assert service.classify(HttpError(429))[
        "error_category"
    ] == "rate_limited"
    assert service.classify(HttpError(503))[
        "error_category"
    ] == "upstream_server"
    assert service.classify(HttpError(400))[
        "retryable"
    ] is False
