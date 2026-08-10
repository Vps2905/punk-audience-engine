from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from app.models.production_feature_build_contracts import EmbeddingModelSpec
from app.models.production_historical_semantic_feature_contracts import (
    HistoricalSemanticFeatureBuildRequest,
)
from app.services.production_historical_semantic_feature_service import (
    ProductionHistoricalSemanticFeatureService,
)


MODEL = EmbeddingModelSpec(
    backend="sentence_transformers",
    model_name="intfloat/multilingual-e5-small",
    model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    dimension=384,
    normalize_embeddings=True,
    document_prefix="passage: ",
    query_prefix="query: ",
)


def request() -> HistoricalSemanticFeatureBuildRequest:
    return HistoricalSemanticFeatureBuildRequest(
        tenant_id="punk_internal",
        source_feature_set_id="safe-legacy-set",
        source_feature_set_version=1,
        model=MODEL,
        minimum_cohort_size=1000,
        batch_size=32,
        requested_by="operator-one",
    )


def snapshot(*, cohort_size: int = 1200, activation: bool = False):
    timestamp = datetime(2026, 7, 8, tzinfo=timezone.utc)
    rows = []
    for index, category in enumerate(("restaurant", "fitness"), start=1):
        rows.append(
            {
                "feature_id": f"legacy-{index}",
                "location_name": "montreal",
                "primary_poi_type": category,
                "created_day_part": "evening",
                "lookback_bucket": "0_30d",
                "cohort_size": cohort_size + index,
                "quality_score": 0.8,
                "privacy_status": "privacy_safe",
                "rights_status": "historical_internal_only",
                "purpose": "audience_intelligence",
                "source_latest_at": timestamp,
                "freshness_status": "stale",
                "data_use_mode": "historical_preview",
                "eligible_for_retrieval": True,
                "eligible_for_activation": activation,
                "trait_text": (
                    f"location montreal category {category} daypart evening"
                ),
                "metadata": {"privacy_window": "0_30d"},
            }
        )
    return {
        "feature_set": {
            "tenant_id": "punk_internal",
            "feature_set_id": "safe-legacy-set",
            "version": 1,
            "status": "ready_for_historical_preview",
            "source_mode": "legacy_postgres_safe_vectors",
            "data_use_mode": "historical_preview",
            "source_ref": "postgres://legacy-safe-features",
            "source_version": "snapshot-1",
            "source_fingerprint": "a" * 64,
            "source_latest_at": timestamp,
            "freshness_status": "stale",
            "stale_after_hours": 48,
            "privacy_policy_version": "privacy-v1",
            "rights_policy_id": "historical-internal",
            "purpose": "audience_intelligence",
            "eligible_for_retrieval": True,
            "eligible_for_activation": False,
            "feature_count": len(rows),
            "lineage": {},
        },
        "features": rows,
        "read_only_transaction_verified": True,
        "stored_embeddings_read": False,
        "raw_identifiers_read": False,
    }


class Reader:
    def __init__(self, value):
        self.value = value

    def read(self, **_kwargs):
        return self.value


class Registry:
    def __init__(self, *, approved=True):
        self.approved = approved

    def require_approved(self, **_kwargs):
        return {"approved": self.approved}


class Encoder:
    def encode(self, texts, *, model, batch_size):
        assert len(texts) == 2
        assert model == MODEL
        assert batch_size == 32
        values = np.zeros((2, 384), dtype=np.float32)
        values[0, 0] = 1.0
        values[1, 1] = 1.0
        return values


class Writer:
    def __init__(self):
        self.feature_set = None

    def save_feature_set(self, feature_set):
        self.feature_set = feature_set
        return {
            "status": "completed",
            "feature_set_id": feature_set.feature_set_id,
            "feature_count": feature_set.feature_count,
        }


def service(value=None, *, approved=True):
    writer = Writer()
    instance = ProductionHistoricalSemanticFeatureService(
        source_reader=Reader(value or snapshot()),
        model_registry=Registry(approved=approved),
        feature_writer=writer,
        encoder=Encoder(),
    )
    return instance, writer


def test_builds_model_pinned_nonactivatable_historical_features():
    instance, writer = service()

    result = instance.build(
        request(),
        privacy_safe_historical_reembedding_confirmed=True,
    )

    assert result["status"] == "historical_semantic_features_ready"
    assert result["feature_count"] == 2
    assert result["model_fingerprint"] == MODEL.fingerprint
    assert result["eligible_for_module5_functional_shadow"] is True
    assert result["eligible_for_activation"] is False
    assert result["raw_identifiers_read"] is False
    assert result["activation_or_export_performed"] is False

    feature_set = writer.feature_set
    assert feature_set.model_name == MODEL.model_name
    assert feature_set.model_version == MODEL.model_revision
    assert feature_set.freshness_status == "stale"
    assert feature_set.eligible_for_activation is False
    assert feature_set.feature_count == 2
    assert feature_set.lineage["embedding_model_spec"] == MODEL.to_safe_dict()
    assert all(
        feature.eligible_for_activation is False
        for feature in feature_set.features
    )


def test_requires_explicit_confirmation_before_any_build():
    instance, writer = service()

    with pytest.raises(RuntimeError, match="confirmation"):
        instance.build(
            request(),
            privacy_safe_historical_reembedding_confirmed=False,
        )

    assert writer.feature_set is None


def test_fails_closed_for_unapproved_model():
    instance, writer = service(approved=False)

    with pytest.raises(RuntimeError, match="not approved"):
        instance.build(
            request(),
            privacy_safe_historical_reembedding_confirmed=True,
        )

    assert writer.feature_set is None


@pytest.mark.parametrize(
    "unsafe_snapshot, message",
    [
        (snapshot(cohort_size=10), "k-anonymity"),
        (snapshot(activation=True), "activation-eligible"),
    ],
)
def test_fails_closed_for_unsafe_historical_rows(unsafe_snapshot, message):
    instance, writer = service(unsafe_snapshot)

    with pytest.raises(RuntimeError, match=message):
        instance.build(
            request(),
            privacy_safe_historical_reembedding_confirmed=True,
        )

    assert writer.feature_set is None


def test_contract_rejects_weakened_k_anonymity():
    with pytest.raises(ValueError, match="at least 1000"):
        HistoricalSemanticFeatureBuildRequest(
            tenant_id="punk_internal",
            source_feature_set_id="safe-legacy-set",
            source_feature_set_version=1,
            model=MODEL,
            minimum_cohort_size=999,
        )
