from datetime import datetime, timezone

import pandas as pd

from app.services.autonomous_audience_intelligence_v2_service import (
    AutonomousAudienceIntelligenceV2Service,
)


def test_autonomous_audience_intelligence_v2_service_runs_full_pipeline(tmp_path):
    df = pd.DataFrame(
        {
            "created_at": [
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ],
            "location_name": ["montreal", "montreal", "san francisco"],
            "primary_poi_type": ["restaurant", "cafe", "coworking_space"],
            "created_day_part": ["evening", "evening", "evening"],
            "quality_score": [0.8, 0.6, 0.5],
            "total_maid_volume": [10000, 8000, 12000],
            "privacy_status": ["passed", "passed", "passed"],
        }
    )

    result = AutonomousAudienceIntelligenceV2Service().run(
        prompt="Build me a high-quality restaurant and cafe evening audience for Montreal and San Francisco",
        safe_cohorts=df,
        output_dir=tmp_path,
    )

    assert result["status"] == "completed"
    assert result["pipeline_version"] == "v2_autonomous_preview"

    assert result["data_freshness"]["freshness_status"] == "fresh"
    assert result["prompt_intent"]["confidence_score"] >= 0.85

    assert result["embedding_manifest"]["vector_count"] == 3
    assert result["embedding_manifest"]["vector_dimension"] == 384

    assert result["ranked_match_count"] == 3
    assert result["approval_required"] is True
    assert result["downstream_export_enabled"] is False

    assert (tmp_path / "data_freshness_report.json").exists()
    assert (tmp_path / "prompt_intent.json").exists()
    assert (tmp_path / "ranked_audience_matches.csv").exists()
    assert (tmp_path / "mutation_suggestions.json").exists()
    assert (tmp_path / "v2_preview_summary.json").exists()
    assert (tmp_path / "embeddings" / "all_safe_cohort_vectors.npy").exists()
    assert (tmp_path / "embeddings" / "all_safe_cohort_metadata.csv").exists()
    assert (tmp_path / "embeddings" / "all_safe_cohort_embedding_manifest.json").exists()
