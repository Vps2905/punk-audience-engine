import pandas as pd

from app.services.autonomous_audience_intelligence_v2_service import AutonomousAudienceIntelligenceV2Service


def test_v2_service_uses_hybrid_intent_metadata_when_llm_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_LLM_INTENT", "false")
    monkeypatch.setenv("EMBEDDING_MAX_FEATURES", "384")

    safe = pd.DataFrame(
        {
            "location_name": ["montreal", "montreal", "san francisco"],
            "primary_poi_type": ["cafe", "restaurant", "coworking_space"],
            "created_day_part": ["evening", "evening", "evening"],
            "quality_score": [0.5, 0.4, 0.6],
            "total_maid_volume": [10000, 12000, 15000],
            "privacy_status": ["passed", "passed", "passed"],
            "created_at": ["2026-07-06T18:46:16Z"] * 3,
        }
    )

    result = AutonomousAudienceIntelligenceV2Service().run(
        prompt="I need people who grab espresso after work near Montreal.",
        safe_cohorts=safe,
        output_dir=tmp_path,
        freshness_source_df=safe,
    )

    intent = result["prompt_intent"]

    assert intent["llm_used"] is False
    assert intent["resolver_mode"] == "deterministic_fallback_llm_disabled"
    assert intent["rag_context_summary"]["safe_cohort_count"] == 3
    assert "cafe" in intent["canonical_categories"] or "cafe" in intent["poi_terms"]
