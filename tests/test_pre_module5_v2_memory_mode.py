import pandas as pd

from app.services.autonomous_audience_intelligence_v2_service import (
    AutonomousAudienceIntelligenceV2Service,
)


def test_v2_memory_mode_does_not_write_local_files(
    monkeypatch,
    tmp_path,
):
    safe = pd.DataFrame(
        [
            {
                "location_name": "montreal",
                "primary_poi_type": "cafe",
                "created_day_part": "evening",
                "quality_score": 0.5,
                "privacy_status": "passed",
            }
        ]
    )

    monkeypatch.setattr(
        "app.services.autonomous_audience_intelligence_v2_service."
        "DataFreshnessAgent.analyze_dataframe",
        lambda self, **kwargs: {
            "freshness_status": "fresh",
            "source_rows_checked": 1,
        },
    )
    monkeypatch.setattr(
        "app.services.autonomous_audience_intelligence_v2_service."
        "HybridSemanticIntentAgent.resolve",
        lambda self, **kwargs: {
            "locations": ["montreal"],
            "canonical_categories": ["cafe"],
            "dayparts": ["evening"],
        },
    )
    monkeypatch.setattr(
        "app.services.autonomous_audience_intelligence_v2_service."
        "AllSafeCohortEmbeddingService.build_index",
        lambda self, **kwargs: {
            "vector_count": 1,
            "vector_dimension": 384,
            "embedding_store": "postgres",
        },
    )
    monkeypatch.setattr(
        "app.services.autonomous_audience_intelligence_v2_service."
        "AutonomousMutationAgent.generate_suggestions",
        lambda self, **kwargs: {
            "status": "completed",
            "suggestion_count": 0,
            "mutation_suggestions": [],
            "approval_required": True,
            "downstream_export_enabled": False,
        },
    )

    output = tmp_path / "v2"

    result = AutonomousAudienceIntelligenceV2Service().run(
        prompt="Cafe visitors in Montreal evening",
        safe_cohorts=safe,
        output_dir=output,
        freshness_source_df=safe,
        persist_artifacts=False,
    )

    assert result["ranked_matches"]
    assert result["ranked_matches_path"].startswith(
        "postgres://"
    )
    assert not output.exists()
