import pandas as pd
import pytest

from app.services import (
    autonomous_audience_intelligence_v2_service as v2_module,
)
from app.services.autonomous_audience_intelligence_v2_service import (
    AutonomousAudienceIntelligenceV2Service,
)


class _Freshness:
    def analyze_dataframe(self, **kwargs):
        return {
            "freshness_status": "fresh",
            "source_rows_checked": 1,
        }


class _Intent:
    def resolve(self, **kwargs):
        return {
            "locations": ["montreal"],
            "canonical_categories": ["cafe"],
            "dayparts": ["evening"],
        }


class _Ranking:
    def rank(self, safe_cohorts, **kwargs):
        return safe_cohorts.assign(
            location_match_score=1.0,
            category_match_score=1.0,
            daypart_match_score=1.0,
            final_match_score=1.0,
            match_type="exact_match",
        )


class _Mutation:
    def generate_suggestions(self, **kwargs):
        return {
            "status": "completed",
            "suggestion_count": 0,
            "mutation_suggestions": [],
        }


def _safe_cohorts():
    return pd.DataFrame(
        [
            {
                "location_name": "montreal",
                "primary_poi_type": "cafe",
                "created_day_part": "evening",
                "privacy_status": "passed",
                "quality_score": 0.8,
            }
        ]
    )


def _patch_common(monkeypatch):
    monkeypatch.setattr(
        v2_module,
        "DataFreshnessAgent",
        _Freshness,
    )
    monkeypatch.setattr(
        v2_module,
        "HybridSemanticIntentAgent",
        _Intent,
    )
    monkeypatch.setattr(
        v2_module,
        "DynamicAudienceRankingService",
        _Ranking,
    )
    monkeypatch.setattr(
        v2_module,
        "AutonomousMutationAgent",
        _Mutation,
    )
    monkeypatch.setattr(
        v2_module.time,
        "sleep",
        lambda seconds: None,
    )


def test_v2_retries_transient_database_disconnect(
    monkeypatch,
    tmp_path,
):
    _patch_common(monkeypatch)
    calls = {"count": 0}

    class _Embedding:
        def build_index(self, **kwargs):
            calls["count"] += 1

            if calls["count"] == 1:
                raise RuntimeError(
                    "server closed the connection unexpectedly"
                )

            return {
                "status": "completed",
                "embedding_store": "postgres",
                "vector_count": 1,
                "vector_dimension": 384,
            }

    monkeypatch.setattr(
        v2_module,
        "AllSafeCohortEmbeddingService",
        _Embedding,
    )

    result = AutonomousAudienceIntelligenceV2Service().run(
        prompt="Cafe visitors in Montreal evening",
        safe_cohorts=_safe_cohorts(),
        output_dir=tmp_path / "v2",
        persist_artifacts=False,
    )

    assert calls["count"] == 2
    assert result["status"] == "completed"
    assert result["embedding_manifest"]["vector_count"] == 1


def test_v2_does_not_retry_non_transient_failure(
    monkeypatch,
    tmp_path,
):
    _patch_common(monkeypatch)
    calls = {"count": 0}

    class _Embedding:
        def build_index(self, **kwargs):
            calls["count"] += 1
            raise ValueError("invalid vector dimension")

    monkeypatch.setattr(
        v2_module,
        "AllSafeCohortEmbeddingService",
        _Embedding,
    )

    with pytest.raises(
        ValueError,
        match="invalid vector dimension",
    ):
        AutonomousAudienceIntelligenceV2Service().run(
            prompt="Cafe visitors in Montreal evening",
            safe_cohorts=_safe_cohorts(),
            output_dir=tmp_path / "v2",
            persist_artifacts=False,
        )

    assert calls["count"] == 1
