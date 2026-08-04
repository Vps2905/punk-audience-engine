from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.models.production_audience_retrieval_contracts import (
    ConstraintTaxonomyEntry,
    GovernedAudienceRetrievalRequest,
    GovernedConstraintTaxonomy,
    ProductionRetrievalModelBinding,
)
from app.models.production_feature_build_contracts import EmbeddingModelSpec
from app.services.production_dual_model_candidate_retrieval_service import (
    ProductionDualModelCandidateRetrievalService,
)
from app.services.production_governed_audience_retrieval_service import (
    ProductionGovernedAudienceRetrievalService,
)
from app.services.production_multilingual_constraint_canonicalization_service import (
    ProductionMultilingualConstraintCanonicalizationService,
    SemanticConstraintScore,
)


E5 = EmbeddingModelSpec(
    backend="sentence_transformers",
    model_name="intfloat/multilingual-e5-small",
    model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    dimension=384,
    normalize_embeddings=True,
    document_prefix="passage: ",
    query_prefix="query: ",
)

MINILM = EmbeddingModelSpec(
    backend="sentence_transformers",
    model_name=(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
    model_revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
    dimension=384,
    normalize_embeddings=True,
)


class FakeRegistry:
    def __init__(self, *, approved: bool = True) -> None:
        self.approved = approved
        self.calls = []

    def require_approved(self, *, tenant_id, model):
        self.calls.append((tenant_id, model.fingerprint))
        if not self.approved:
            raise RuntimeError("Embedding model revision has not passed approval.")
        return {"approved": True}


class FakeSemanticResolver:
    def __init__(self, scores=None) -> None:
        self.scores = scores or {}
        self.calls = []

    def score(self, *, query_text, dimension, entries, model):
        self.calls.append((dimension, model.fingerprint))
        values = self.scores.get(dimension, [])
        return [
            SemanticConstraintScore(canonical_value=label, score=score)
            for label, score in values
        ]


class FakeQueryEncoder:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, *, query_text, model):
        self.calls.append(model.fingerprint)
        return [1.0] + [0.0] * 383


class FakeFeatureStore:
    def __init__(self, *, primary_initial, complementary, primary_expanded):
        self.primary_initial = primary_initial
        self.complementary = complementary
        self.primary_expanded = primary_expanded
        self.search_calls = []

    def get_feature_set(
        self,
        *,
        tenant_id,
        feature_set_id=None,
        version=None,
        data_use_mode=None,
    ):
        model = E5 if feature_set_id == "fs-e5" else MINILM
        return {
            "tenant_id": tenant_id,
            "feature_set_id": feature_set_id,
            "version": version,
            "model_backend": model.backend,
            "model_name": model.model_name,
            "model_version": model.model_revision,
            "embedding_dimension": model.dimension,
            "eligible_for_retrieval": True,
            "data_use_mode": data_use_mode,
            "freshness_status": "fresh",
            "lineage": {"embedding_model_spec": model.to_safe_dict()},
        }

    def hybrid_search(self, **kwargs):
        self.search_calls.append(dict(kwargs))
        feature_set_id = kwargs["feature_set_id"]
        top_k = kwargs["top_k"]
        assert kwargs["locations"] == ()
        assert kwargs["categories"] == ()
        assert kwargs["dayparts"] == ()
        if feature_set_id == "fs-e5" and top_k == 10:
            return list(self.primary_initial)
        if feature_set_id == "fs-e5" and top_k == 30:
            return list(self.primary_expanded)
        if feature_set_id == "fs-minilm" and top_k == 10:
            return list(self.complementary)
        raise AssertionError((feature_set_id, top_k))


def _taxonomy() -> GovernedConstraintTaxonomy:
    return GovernedConstraintTaxonomy(
        taxonomy_id="global-audience-taxonomy",
        version="reviewed-v1",
        reviewed_by="taxonomy-owner",
        locations=(
            ConstraintTaxonomyEntry(
                "montreal",
                aliases=("Montréal", "Montreal", "মন্ট্রিয়ল"),
            ),
            ConstraintTaxonomyEntry(
                "toronto",
                aliases=("Toronto", "টরন্টো"),
            ),
        ),
        categories=(
            ConstraintTaxonomyEntry(
                "car_wash",
                aliases=("car wash", "কার ওয়াশ"),
                descriptions=("audience visiting a car wash",),
            ),
            ConstraintTaxonomyEntry(
                "restaurant",
                aliases=("restaurant", "রেস্তোরাঁ"),
                descriptions=("audience visiting restaurants",),
            ),
        ),
        dayparts=(
            ConstraintTaxonomyEntry(
                "afternoon",
                aliases=("afternoon", "দুপুরে"),
            ),
            ConstraintTaxonomyEntry(
                "evening",
                aliases=("evening", "সন্ধ্যায়"),
            ),
        ),
    )


def _request(**overrides) -> GovernedAudienceRetrievalRequest:
    values = {
        "tenant_id": "tenant-a",
        "query_text": (
            "Montreal-এ দুপুরে কার ওয়াশে যাওয়া বিষয়ে "
            "আগ্রহী অডিয়েন্স খুঁজুন।"
        ),
        "language": "bn",
        "execution_mode": "historical_preview",
        "taxonomy": _taxonomy(),
        "canonicalizer_model": E5,
        "primary_model": ProductionRetrievalModelBinding(
            role="primary",
            feature_set_id="fs-e5",
            feature_set_version=1,
            model=E5,
            initial_depth=10,
            expansion_depth=30,
        ),
        "complementary_model": ProductionRetrievalModelBinding(
            role="complementary",
            feature_set_id="fs-minilm",
            feature_set_version=1,
            model=MINILM,
            initial_depth=10,
            expansion_depth=10,
        ),
        "requested_locations": ("Montreal",),
        "requested_categories": ("কার ওয়াশ",),
        "requested_dayparts": ("দুপুরে",),
        "result_limit": 10,
    }
    values.update(overrides)
    return GovernedAudienceRetrievalRequest(**values)


def _candidate(
    feature_id,
    *,
    location="montreal",
    category="restaurant",
    daypart="afternoon",
    fingerprint=None,
    score=0.8,
):
    return {
        "feature_id": feature_id,
        "location_name": location,
        "primary_poi_type": category,
        "created_day_part": daypart,
        "lookback_bucket": "8_30d",
        "cohort_size": 10000,
        "quality_score": 0.8,
        "privacy_status": "passed",
        "rights_status": "permitted",
        "purpose": "audience_intelligence",
        "source_latest_at": "2026-08-01T00:00:00+00:00",
        "freshness_status": "fresh",
        "data_use_mode": "historical_preview",
        "vector_score": score,
        "lexical_score": 0.1,
        "fused_score": 0.02,
        "metadata": {
            "canonical_feature_fingerprint": fingerprint or feature_id
        },
    }


def test_multilingual_exact_aliases_resolve_without_semantic_guessing():
    registry = FakeRegistry()
    semantic = FakeSemanticResolver()
    service = ProductionMultilingualConstraintCanonicalizationService(
        model_registry=registry,
        semantic_resolver=semantic,
    )

    result = service.canonicalize(_request())

    assert result.ready_for_retrieval is True
    assert result.locations.values == ("montreal",)
    assert result.categories.values == ("car_wash",)
    assert result.dayparts.values == ("afternoon",)
    assert result.language == "bn"
    assert semantic.calls == []
    assert result.to_safe_dict()["raw_query_stored"] is False


def test_unknown_explicit_location_fails_closed_without_nearest_mapping():
    semantic = FakeSemanticResolver()
    service = ProductionMultilingualConstraintCanonicalizationService(
        model_registry=FakeRegistry(),
        semantic_resolver=semantic,
    )

    result = service.canonicalize(
        _request(requested_locations=("Ottawa",))
    )

    assert result.ready_for_retrieval is False
    assert result.locations.status == "unsupported"
    assert result.reason_code == "unsupported_location_requires_clarification"
    assert result.locations.values == ()


def test_semantic_resolution_exposes_per_field_confidence_and_margin():
    semantic = FakeSemanticResolver(
        scores={
            "category": [("car_wash", 0.91), ("_unspecified", 0.20)],
            "daypart": [("afternoon", 0.88), ("_unspecified", 0.22)],
        }
    )
    service = ProductionMultilingualConstraintCanonicalizationService(
        model_registry=FakeRegistry(),
        semantic_resolver=semantic,
        semantic_min_score=0.55,
        semantic_min_margin=0.05,
    )

    result = service.canonicalize(
        _request(
            query_text="Montreal audience with relevant venue and time intent",
            requested_categories=(),
            requested_dayparts=(),
        )
    )

    assert result.ready_for_retrieval is True
    assert result.categories.values == ("car_wash",)
    assert result.categories.confidence == pytest.approx(0.91)
    assert result.categories.margin == pytest.approx(0.71)
    assert result.dayparts.values == ("afternoon",)
    assert result.dayparts.confidence == pytest.approx(0.88)


def test_dual_model_retrieval_expands_primary_and_structured_match_wins():
    shared_wrong = _candidate(
        "wrong-e5",
        category="restaurant",
        fingerprint="wrong-shared",
        score=0.99,
    )
    complementary_wrong = _candidate(
        "wrong-minilm",
        category="restaurant",
        fingerprint="wrong-shared",
        score=0.50,
    )
    correct = _candidate(
        "correct-e5",
        category="car_wash",
        fingerprint="correct",
        score=0.30,
    )
    store = FakeFeatureStore(
        primary_initial=[shared_wrong],
        complementary=[complementary_wrong],
        primary_expanded=[shared_wrong, correct],
    )
    registry = FakeRegistry()
    retrieval = ProductionDualModelCandidateRetrievalService(
        feature_store=store,
        model_registry=registry,
        query_encoder=FakeQueryEncoder(),
    )
    canonicalizer = ProductionMultilingualConstraintCanonicalizationService(
        model_registry=registry,
        semantic_resolver=FakeSemanticResolver(),
    )
    request = _request()
    constraints = canonicalizer.canonicalize(request)

    result = retrieval.retrieve(request=request, constraints=constraints)

    assert result["status"] == "retrieval_ready_for_human_review"
    assert result["retrieval"]["expanded_primary"] is True
    assert result["retrieval"]["raw_cross_model_cosine_combined"] is False
    assert result["selected_candidates"][0]["primary_poi_type"] == "car_wash"
    assert result["selected_candidates"][0]["full_constraint_match"] is True
    assert result["eligible_for_automatic_proposal"] is False
    assert result["production_routing_enabled"] is False
    assert result["activation_or_export_performed"] is False
    assert [(call["feature_set_id"], call["top_k"]) for call in store.search_calls] == [
        ("fs-e5", 10),
        ("fs-minilm", 10),
        ("fs-e5", 30),
    ]


def test_no_full_match_after_expansion_blocks_automatic_proposal():
    wrong = _candidate("wrong", category="restaurant")
    store = FakeFeatureStore(
        primary_initial=[wrong],
        complementary=[wrong],
        primary_expanded=[wrong],
    )
    registry = FakeRegistry()
    request = _request()
    constraints = ProductionMultilingualConstraintCanonicalizationService(
        model_registry=registry,
        semantic_resolver=FakeSemanticResolver(),
    ).canonicalize(request)
    result = ProductionDualModelCandidateRetrievalService(
        feature_store=store,
        model_registry=registry,
        query_encoder=FakeQueryEncoder(),
    ).retrieve(request=request, constraints=constraints)

    assert result["status"] == "blocked"
    assert result["reason_code"] == (
        "no_candidate_satisfies_resolved_constraints"
    )
    assert result["selected_candidates"] == []
    assert result["eligible_for_automatic_proposal"] is False
    assert result["downstream_export_enabled"] is False


def test_unapproved_model_blocks_before_query_encoding_or_search():
    registry = FakeRegistry(approved=False)
    encoder = FakeQueryEncoder()
    store = FakeFeatureStore(
        primary_initial=[],
        complementary=[],
        primary_expanded=[],
    )
    request = _request()
    constraints = ProductionMultilingualConstraintCanonicalizationService(
        model_registry=FakeRegistry(),
        semantic_resolver=FakeSemanticResolver(),
    ).canonicalize(request)
    service = ProductionDualModelCandidateRetrievalService(
        feature_store=store,
        model_registry=registry,
        query_encoder=encoder,
    )

    with pytest.raises(RuntimeError, match="approval"):
        service.retrieve(request=request, constraints=constraints)

    assert encoder.calls == []
    assert store.search_calls == []


def test_orchestrator_remains_disconnected_from_existing_proposal_flow():
    wrong = _candidate("wrong", category="restaurant")
    correct = _candidate("correct", category="car_wash")
    store = FakeFeatureStore(
        primary_initial=[correct],
        complementary=[wrong],
        primary_expanded=[correct],
    )
    registry = FakeRegistry()
    service = ProductionGovernedAudienceRetrievalService(
        canonicalizer=(
            ProductionMultilingualConstraintCanonicalizationService(
                model_registry=registry,
                semantic_resolver=FakeSemanticResolver(),
            )
        ),
        candidate_retrieval=(
            ProductionDualModelCandidateRetrievalService(
                feature_store=store,
                model_registry=registry,
                query_encoder=FakeQueryEncoder(),
            )
        ),
    )

    result = service.retrieve(_request())

    assert result["module"] == "module_2_1_governed_multilingual_retrieval"
    assert result["automatic_proposal_creation_enabled"] is False
    assert result["existing_proposal_flow_modified"] is False
    assert result["model_registration_performed"] is False
    assert result["threshold_changed"] is False
