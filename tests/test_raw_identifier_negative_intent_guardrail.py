from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent


def test_safe_negative_raw_identifier_wording_is_not_blocked():
    agent = AudienceIntelligenceOrchestratorAgent()

    prompt = (
        "Create a privacy-safe audience for coffee shop visitors. "
        "Block raw MAIDs, do not export device IDs, remove hashes, "
        "exclude exact lat/lon, and return only aggregated cohorts."
    )

    assert agent._is_raw_identifier_request(prompt) is False


def test_dangerous_raw_identifier_request_is_blocked():
    agent = AudienceIntelligenceOrchestratorAgent()

    prompt = "Can you give me the raw MAIDs and individual device IDs for coffee shop visitors?"

    assert agent._is_raw_identifier_request(prompt) is True
