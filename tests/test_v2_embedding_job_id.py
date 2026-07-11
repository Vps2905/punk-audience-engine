import pandas as pd

from app.services import autonomous_audience_intelligence_v2_service as v2_module
from app.services.autonomous_audience_intelligence_v2_service import AutonomousAudienceIntelligenceV2Service


def test_v2_passes_unique_job_id_to_all_safe_embedding_service(tmp_path, monkeypatch):
    safe_cohorts = pd.DataFrame(
        [
            {
                "location_name": "Montreal",
                "primary_poi_type": "coffee_shop",
                "created_day_part": "evening",
                "privacy_status": "safe",
                "quality_score": 0.9,
            }
        ]
    )

    captured = {}

    class FakeEmbeddingService:
        def build_index(self, safe_cohorts, output_dir, job_id=None):
            captured["job_id"] = job_id
            return {
                "status": "completed",
                "embedding_store": "postgres",
                "vector_count": 1,
                "vector_dimension": 384,
                "output_vectors": f"postgres://audience_vectors?job_id={job_id}",
            }

    class FakeFreshnessAgent:
        def analyze_dataframe(self, df, run_dir, previous_report_path=None):
            return {"status": "fresh"}

    class FakeIntentAgent:
        def resolve(self, prompt, safe_cohorts, output_dir):
            return {"locations": ["Montreal"], "canonical_categories": ["coffee_shop"], "dayparts": ["evening"]}

    class FakeRankingService:
        def rank(self, safe_cohorts, prompt_intent, freshness_report):
            return safe_cohorts.assign(category_match_score=1.0, daypart_match_score=1.0)

    class FakeMutationAgent:
        def generate_suggestions(self, prompt_intent, ranked_cohorts, coverage_warnings, run_dir):
            return {"suggestions": []}

    monkeypatch.setattr(v2_module, "AllSafeCohortEmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(v2_module, "DataFreshnessAgent", FakeFreshnessAgent)
    monkeypatch.setattr(v2_module, "HybridSemanticIntentAgent", FakeIntentAgent)
    monkeypatch.setattr(v2_module, "DynamicAudienceRankingService", FakeRankingService)
    monkeypatch.setattr(v2_module, "AutonomousMutationAgent", FakeMutationAgent)

    out_dir = tmp_path / "run_abc123" / "06_v2_autonomous_preview"

    AutonomousAudienceIntelligenceV2Service().run(
        prompt="coffee shop montreal evening",
        safe_cohorts=safe_cohorts,
        output_dir=out_dir,
    )

    assert captured["job_id"] == "v2_all_safe_run_abc123_06_v2_autonomous_preview"
