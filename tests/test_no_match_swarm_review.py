from app.agents.audience_intelligence_orchestrator_agent import (
    AudienceIntelligenceOrchestratorAgent,
)


def test_no_match_swarm_review_is_deterministically_blocked():
    result = (
        AudienceIntelligenceOrchestratorAgent()
        ._build_no_match_swarm_review(
            coverage_warnings=[
                "No exact safe cohort was available."
            ],
        )
    )

    assert result["status"] == "completed"
    assert result["overall_review_status"] == "blocked"
    assert result["review_reason"] == "no_safe_exact_match"
    assert result["coverage_warning_count"] == 1
    assert result["data_gap_count"] == 0
    assert result["approval_required"] is True
    assert result["downstream_export_enabled"] is False
    assert result["recommendations"]
