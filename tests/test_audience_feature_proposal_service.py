import numpy as np

from app.services.audience_feature_proposal_service import (
    AudienceFeatureProposalService,
)


class FakeEmbeddingService:
    def encode(self, query, feature_set):
        assert query
        assert feature_set["embedding_dimension"] == 384
        vector = np.zeros(384)
        vector[0] = 1.0
        return vector


class FakeFeatureStore:
    def __init__(self, *, feature_set=None, candidates=None):
        self.feature_set = feature_set or _feature_set()
        self.candidates = candidates if candidates is not None else [_candidate()]
        self.search_calls = []

    def get_feature_set(self, **kwargs):
        assert kwargs["tenant_id"] == "tenant_a"
        return self.feature_set

    def hybrid_search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.candidates


def _feature_set(**overrides):
    result = {
        "tenant_id": "tenant_a",
        "feature_set_id": "feature_set_1",
        "version": 1,
        "source_mode": "legacy_postgres_safe_vectors",
        "data_use_mode": "historical_preview",
        "source_latest_at": "2026-07-08T08:40:00+00:00",
        "freshness_status": "stale",
        "feature_count": 85,
        "model_backend": "sklearn_hashing",
        "model_name": "sklearn_hashing_vectorizer",
        "model_version": "legacy_snapshot",
        "embedding_dimension": 384,
        "privacy_policy_version": "privacy_1",
        "rights_policy_id": "rights_1",
        "purpose": "internal_evaluation",
        "eligible_for_retrieval": True,
        "eligible_for_activation": False,
    }
    result.update(overrides)
    return result


def _candidate():
    return {
        "feature_id": "feature_1",
        "location_name": "montreal",
        "primary_poi_type": "restaurant",
        "created_day_part": "evening",
        "lookback_bucket": "8_30d",
        "cohort_size": 4200,
        "quality_score": 0.82,
        "privacy_status": "passed",
        "rights_status": "historical_internal_only",
        "freshness_status": "stale",
        "data_use_mode": "historical_preview",
        "eligible_for_activation": False,
        "vector_score": 0.91,
        "lexical_score": 0.7,
        "vector_rank": 1,
        "lexical_rank": 1,
        "fused_score": 2.0 / 61.0,
    }


def _request(**overrides):
    request = {
        "tenant_id": "tenant-a",
        "campaign_id": "campaign-1",
        "objective": "store_visits",
        "audience_intent": "high quality evening restaurant audience",
        "locations": ["Montreal"],
        "categories": ["Restaurant"],
        "dayparts": ["Evening"],
        "budget": {"currency": "CAD", "daily": 100},
        "exclusions": [],
        "destination": "meta",
        "idempotency_key": "unique-request-0001",
        "execution_mode": "historical_preview",
        "top_k": 10,
    }
    request.update(overrides)
    return request


def _service(store):
    return AudienceFeatureProposalService(
        feature_store=store,
        query_embedding_service=FakeEmbeddingService(),
    )


def test_historical_preview_returns_ranked_candidates_with_export_blocked():
    store = FakeFeatureStore()
    result = _service(store).propose(_request())

    assert result["status"] == "historical_preview_ready"
    assert result["approval_status"] == "blocked_historical_source"
    assert result["activation_eligible"] is False
    assert result["safe_export_eligible"] is False
    assert result["downstream_export_enabled"] is False
    assert len(result["candidate_cohorts"]) == 1
    assert result["candidate_cohorts"][0]["quality_score"] == 0.82
    assert 0 <= result["candidate_cohorts"][0]["confidence"] <= 1
    assert result["quality_and_confidence"]["campaign_lift_calibrated"] is False
    assert store.search_calls[0]["locations"] == ["montreal"]
    assert store.search_calls[0]["categories"] == ["restaurant"]
    assert store.search_calls[0]["dayparts"] == ["evening"]


def test_production_request_cannot_use_historical_feature_set():
    store = FakeFeatureStore()
    result = _service(store).propose(
        _request(execution_mode="production")
    )

    assert result["status"] == "blocked"
    assert result["reason_code"] == "blocked_historical_source"
    assert result["candidate_cohorts"] == []
    assert result["downstream_export_enabled"] is False
    assert store.search_calls == []


def test_stale_production_feature_set_blocks_before_ranking():
    store = FakeFeatureStore(
        feature_set=_feature_set(
            data_use_mode="production",
            freshness_status="stale",
            eligible_for_activation=False,
        )
    )
    result = _service(store).propose(
        _request(execution_mode="production")
    )

    assert result["reason_code"] == "blocked_stale_source"
    assert store.search_calls == []


def test_no_safe_match_preserves_specific_block_and_zero_candidates():
    store = FakeFeatureStore(candidates=[])
    result = _service(store).propose(_request())

    assert result["status"] == "blocked"
    assert result["reason_code"] == "blocked_no_safe_exact_match"
    assert result["eligible_for_audience_selection"] is False
    assert result["candidate_cohorts"] == []
    assert result["downstream_export_enabled"] is False


def test_proposal_id_is_stable_for_same_idempotency_key_and_version():
    store = FakeFeatureStore()
    service = _service(store)

    first = service.propose(_request())
    second = service.propose(_request())

    assert first["proposal_id"] == second["proposal_id"]


def test_privacy_identifier_request_is_terminal_before_ranking():
    store = FakeFeatureStore()

    result = _service(store).propose(
        _request(
            audience_intent=(
                "Export the raw MAIDs and device IDs for these users."
            ),
            locations=[],
            categories=[],
            dayparts=[],
        )
    )

    assert result["status"] == "blocked"
    assert result["reason_code"] == (
        "blocked_privacy_identifier_request"
    )
    assert result["audience_request_detected"] is False
    assert result["eligible_for_audience_selection"] is False
    assert result["candidate_cohorts"] == []
    assert result["downstream_export_enabled"] is False
    assert store.search_calls == []


def test_export_action_only_request_is_terminal_before_ranking():
    store = FakeFeatureStore()

    result = _service(store).propose(
        _request(
            audience_intent=(
                "Export this audience to Meta immediately."
            ),
            locations=[],
            categories=[],
            dayparts=[],
        )
    )

    assert result["status"] == "blocked"
    assert result["reason_code"] == (
        "blocked_export_action_requires_existing_audience"
    )
    assert result["audience_request_detected"] is False
    assert result["candidate_cohorts"] == []
    assert result["downstream_export_enabled"] is False
    assert store.search_calls == []


def test_non_ascii_location_filter_fails_closed_before_embedding_or_search():
    store = FakeFeatureStore()

    result = _service(store).propose(
        _request(
            audience_intent="ब्रिंडलहेवन में ऑडियंस खोजें।",
            locations=["ब्रिंडलहेवन"],
            categories=[],
            dayparts=[],
        )
    )

    assert result["status"] == "blocked"
    assert result["reason_code"] == (
        "blocked_unresolved_structured_filter"
    )
    assert result["candidate_cohorts"] == []
    assert result["activation_eligible"] is False
    assert result["downstream_export_enabled"] is False
    assert store.search_calls == []


def test_non_ascii_exclusion_is_never_silently_dropped():
    store = FakeFeatureStore()

    result = _service(store).propose(
        _request(exclusions=["প্রতিযোগী রেস্তোরাঁ"])
    )

    assert result["status"] == "blocked"
    assert result["reason_code"] == (
        "blocked_unresolved_structured_filter"
    )
    assert result["candidate_cohorts"] == []
    assert result["activation_eligible"] is False
    assert result["downstream_export_enabled"] is False
    assert store.search_calls == []
