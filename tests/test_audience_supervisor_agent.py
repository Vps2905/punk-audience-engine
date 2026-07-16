import pytest

from app.agents.audience_supervisor_agent import AudienceSupervisorAgent


class FakeOrchestrator:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.result)


def factory_for(result):
    return lambda: FakeOrchestrator(result)


def test_wrapper_preserves_pipeline_result_and_attaches_decision():
    agent = AudienceSupervisorAgent(
        orchestrator_factory=factory_for(
            {
                "status": "completed",
                "run_id": "run_123",
                "approval_status": "pending_approval",
                "approval_required": True,
                "downstream_export_enabled": False,
                "prompt_selected_cohorts": 3,
            }
        )
    )

    result = agent.run(prompt="Find cafe visitors near Montreal")

    assert result["run_id"] == "run_123"
    assert result["supervisor_route"] == "pending_approval"
    assert result["supervisor_stage"] == "approval"
    assert result["supervisor_decision"]["downstream_export_enabled"] is False


def test_trace_does_not_copy_prompt_or_secrets():
    prompt = "Find cafe visitors. secret-marker-123"
    agent = AudienceSupervisorAgent(
        orchestrator_factory=factory_for(
            {
                "status": "completed",
                "run_id": "run_456",
                "approval_status": "blocked_stale_source",
                "downstream_export_enabled": False,
            }
        )
    )

    result = agent.run(prompt=prompt)
    serialized = repr(result["supervisor_trace"])

    assert "secret-marker-123" not in serialized
    assert "authorization" not in serialized.lower()
    assert "api_key" not in serialized.lower()


def test_empty_prompt_is_rejected_before_orchestrator_call():
    agent = AudienceSupervisorAgent(
        orchestrator_factory=factory_for({"status": "completed"})
    )

    with pytest.raises(ValueError, match="prompt must not be empty"):
        agent.run(prompt="   ")


def test_non_dictionary_orchestrator_result_is_rejected():
    class InvalidOrchestrator:
        def run(self, **kwargs):
            return []

    agent = AudienceSupervisorAgent(
        orchestrator_factory=InvalidOrchestrator,
    )

    with pytest.raises(TypeError, match="must return a dictionary"):
        agent.run(prompt="Find cafe visitors")
