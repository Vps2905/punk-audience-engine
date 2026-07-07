from app.agents.semantic_prompt_intelligence_agent import SemanticPromptIntelligenceAgent


def test_semantic_prompt_extracts_restaurant_cafe_evening_locations():
    prompt = "Build me a high-quality restaurant and cafe evening audience for Montreal and San Francisco"

    result = SemanticPromptIntelligenceAgent().analyze_prompt(prompt)

    assert result["status"] == "completed"
    assert result["business_intent"] == "restaurant+cafe"
    assert "restaurant" in result["canonical_categories"]
    assert "cafe" in result["canonical_categories"]
    assert "evening" in result["dayparts"]
    assert "montreal" in result["locations"]
    assert "san francisco" in result["locations"]
    assert result["confidence_score"] >= 0.85


def test_semantic_prompt_handles_cafe_accent_and_coffee_alias():
    prompt = "Find premium café and coffee shop visitors in Montreal Downtown after work"

    result = SemanticPromptIntelligenceAgent().analyze_prompt(prompt)

    assert "cafe" in result["canonical_categories"]
    assert "montreal downtown" in result["locations"]
    assert "evening" in result["dayparts"]
    assert result["quality_intent"] == "high"


def test_semantic_prompt_extracts_gym_fitness_morning():
    prompt = "Build a gym and fitness morning audience for San Francisco"

    result = SemanticPromptIntelligenceAgent().analyze_prompt(prompt)

    assert "gym" in result["canonical_categories"]
    assert "morning" in result["dayparts"]
    assert "san francisco" in result["locations"]


def test_semantic_prompt_extracts_tattoo_body_art_weekend():
    prompt = "Create tattoo and body art weekend audience for Los Angeles"

    result = SemanticPromptIntelligenceAgent().analyze_prompt(prompt)

    assert "tattoo" in result["canonical_categories"]
    assert "weekend" in result["dayparts"]
    assert "los angeles" in result["locations"]


def test_semantic_prompt_detects_strict_exact_preference():
    prompt = "Prioritize exact matches only for restaurant evening audience in Montreal"

    result = SemanticPromptIntelligenceAgent().analyze_prompt(prompt)

    assert result["fallback_tolerance"] == "strict_exact_first"
    assert "restaurant" in result["canonical_categories"]
    assert "evening" in result["dayparts"]
    assert "montreal" in result["locations"]
