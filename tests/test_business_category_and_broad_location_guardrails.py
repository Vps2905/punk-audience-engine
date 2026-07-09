from app.agents.audience_intelligence_orchestrator_agent import AudienceIntelligenceOrchestratorAgent


def test_coffee_shop_request_locks_exports_to_food_related_pois():
    agent = AudienceIntelligenceOrchestratorAgent()

    allowed = agent._allowed_export_poi_terms_for_request(
        ["restaurant", "cafe", "retail", "coffee", "coffee_shop", "shop"]
    )

    assert "cafe" in allowed
    assert "coffee_shop" in allowed
    assert "restaurant" in allowed
    assert "food" in allowed

    assert "barber_shop" not in allowed
    assert "clothing_store" not in allowed
    assert "shopping_mall" not in allowed
    assert "shoe_store" not in allowed


def test_country_only_location_guardrail_blocks_export():
    agent = AudienceIntelligenceOrchestratorAgent()

    report = agent._run_broad_location_guardrail(
        {
            "locations_detected": ["canada"],
            "poi_terms_detected": [],
        }
    )

    assert report["block_export"] is True
    assert report["reason"] == "country_or_broad_location_requires_city_area"
    assert report["downstream_export_enabled"] is False


def test_city_location_does_not_trigger_broad_location_guardrail():
    agent = AudienceIntelligenceOrchestratorAgent()

    report = agent._run_broad_location_guardrail(
        {
            "locations_detected": ["montreal"],
            "poi_terms_detected": ["cafe"],
        }
    )

    assert report["block_export"] is False
    assert report["reason"] == "location_scope_ok"
