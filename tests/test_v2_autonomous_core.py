from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.agents.autonomous_mutation_agent import AutonomousMutationAgent
from app.agents.data_freshness_agent import DataFreshnessAgent
from app.agents.semantic_prompt_intelligence_agent import SemanticPromptIntelligenceAgent
from app.services.all_safe_cohort_embedding_service import AllSafeCohortEmbeddingService
from app.services.audience_match_ranking_service import DynamicAudienceRankingService


def test_v2_freshness_agent_checks_source_timestamp():
    df = pd.DataFrame(
        {
            "created_at": [datetime.now(timezone.utc).isoformat()],
            "location_name": ["montreal"],
            "primary_poi_type": ["restaurant"],
        }
    )

    result = DataFreshnessAgent(stale_after_hours=48).analyze_dataframe(df)

    assert result["status"] == "completed"
    assert result["source_rows_checked"] == 1
    assert result["latest_source_timestamp"] is not None


def test_v2_semantic_prompt_extracts_core_intent():
    prompt = "Build me a high-quality restaurant and cafe evening audience for Montreal and San Francisco"

    result = SemanticPromptIntelligenceAgent().analyze_prompt(prompt)

    assert result["status"] == "completed"
    assert "restaurant" in result["canonical_categories"]
    assert "cafe" in result["canonical_categories"]
    assert "evening" in result["dayparts"]
    assert "montreal" in result["locations"]
    assert "san francisco" in result["locations"]
    assert result["confidence_score"] >= 0.85


def test_v2_embedding_uses_all_safe_cohorts_and_384_dimensions(tmp_path):
    df = pd.DataFrame(
        {
            "location_name": ["montreal", "san francisco", "montreal"],
            "primary_poi_type": ["restaurant", "coworking_space", "cafe"],
            "created_day_part": ["evening", "evening", "afternoon"],
            "quality_score": [0.7, 0.4, 0.5],
            "total_maid_volume": [10000, 20000, 30000],
            "privacy_status": ["passed", "passed", "passed"],
        }
    )

    manifest = AllSafeCohortEmbeddingService(max_features=384).build_index(df, tmp_path)

    assert manifest["vector_count"] == 3
    assert manifest["vector_dimension"] == 384

    vectors = np.load(tmp_path / "all_safe_cohort_vectors.npy")
    assert vectors.shape == (3, 384)


def test_v2_dynamic_ranking_prioritizes_exact_match():
    df = pd.DataFrame(
        {
            "location_name": ["montreal", "san francisco"],
            "primary_poi_type": ["restaurant", "coworking_space"],
            "created_day_part": ["evening", "evening"],
            "quality_score": [0.7, 0.9],
            "total_maid_volume": [10000, 300000],
            "privacy_status": ["passed", "passed"],
        }
    )

    intent = {
        "locations": ["montreal", "san francisco"],
        "canonical_categories": ["restaurant", "cafe"],
        "dayparts": ["evening"],
    }

    ranked = DynamicAudienceRankingService().rank(df, intent, {"freshness_status": "fresh"})

    assert ranked.iloc[0]["location_name"] == "montreal"
    assert ranked.iloc[0]["match_type"] == "exact_match"


def test_v2_mutation_does_not_create_fake_exact_audience():
    ranked = pd.DataFrame(
        {
            "location_name": ["san francisco"],
            "primary_poi_type": ["coworking_space"],
            "created_day_part": ["evening"],
            "final_match_score": [0.52],
            "confidence_score": [0.55],
            "match_type": ["adjacent_category"],
            "match_reason": ["Existing safe cohort is adjacent, not exact restaurant/cafe."],
        }
    )

    intent = {
        "locations": ["san francisco"],
        "canonical_categories": ["restaurant", "cafe"],
        "dayparts": ["evening"],
    }

    result = AutonomousMutationAgent().generate_suggestions(intent, ranked)

    assert result["approval_required"] is True
    assert result["downstream_export_enabled"] is False

    for item in result["mutation_suggestions"]:
        assert not (
            item["location_name"] == "san francisco"
            and item["primary_poi_type"] in {"restaurant", "cafe"}
            and item["mutation_type"] == "exact_match"
        )
