from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pytest

from app.models.production_feature_build_contracts import (
    CanonicalFeatureSourceManifest,
    EmbeddingModelSpec,
    ProductionFeatureBuildRequest,
)
from app.services.production_feature_embedding_service import (
    ProductionCanonicalFeatureEmbeddingService,
)


class FakeEncoder:
    def encode(self, texts, *, model, batch_size):
        assert model.model_revision == "immutable-revision-123"
        assert batch_size == 32
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.zeros(model.dimension, dtype=np.float32)
            vector[0] = 1.0
            vector[1] = digest[0] / 255.0
            vectors.append(vector)
        return np.asarray(vectors)


def _source(**overrides):
    values = {
        "tenant_id": "tenant-a",
        "provider_id": "provider-a",
        "dataset_id": "mobility-signals",
        "schema_version": "v1",
        "source_ref": "s3://canonical/tenant/provider/output/",
        "source_version": "object-version-1",
        "source_fingerprint": "a" * 64,
        "source_latest_at": datetime(
            2026,
            7,
            30,
            10,
            0,
            tzinfo=timezone.utc,
        ),
        "expected_row_count": 2,
        "data_use_mode": "production",
        "privacy_status": "safe",
        "privacy_policy_version": "privacy-v1",
        "privacy_controls": (
            "daily_contribution_bounding",
            "k_anonymity",
            "deterministic_per_release_gaussian_dp",
            "no_identifier_output",
        ),
        "rights_status": "permitted",
        "rights_policy_id": "rights-v1",
        "purpose": "audience-intelligence",
        "location_column": "location_name",
        "category_column": "primary_poi_type",
        "daypart_column": "created_day_part",
        "cohort_size_column": "dp_noisy_count",
        "privacy_window_column": "privacy_window",
    }
    values.update(overrides)
    return CanonicalFeatureSourceManifest(**values)


def _model(**overrides):
    values = {
        "backend": "sentence-transformers",
        "model_name": "organization/multilingual-model",
        "model_revision": "immutable-revision-123",
        "dimension": 384,
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
    }
    values.update(overrides)
    return EmbeddingModelSpec(**values)


def _request(**overrides):
    values = {
        "source": _source(),
        "model": _model(),
        "batch_size": 32,
        "metadata_fields": ("region",),
    }
    values.update(overrides)
    return ProductionFeatureBuildRequest(**values)


def _rows():
    common = {
        "tenant_id": "tenant_a",
        "provider_id": "provider_a",
        "dataset_id": "mobility_signals",
        "schema_version": "v1",
        "source_fingerprint": "a" * 64,
        "rights_policy_id": "rights_v1",
        "processing_purpose": "audience_intelligence",
    }
    return [
        {
            **common,
            "location_name": "Montréal Downtown",
            "primary_poi_type": "Middle Eastern Restaurant",
            "created_day_part": "Evening",
            "dp_noisy_count": 4200,
            "privacy_window": "2026-07-30",
            "region": "Quebec",
        },
        {
            **common,
            "location_name": "Toronto Downtown",
            "primary_poi_type": "Coffee Shop",
            "created_day_part": "Morning",
            "dp_noisy_count": 5100,
            "privacy_window": "2026-07-30",
            "region": "Ontario",
        },
    ]


def _service():
    return ProductionCanonicalFeatureEmbeddingService(
        encoder=FakeEncoder(),
        stale_after_hours=48,
        now_fn=lambda: datetime(
            2026,
            7,
            30,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )


def test_safe_canonical_rows_become_versioned_non_activatable_features():
    feature_set = _service().build(_request(), _rows())

    assert feature_set.status == "ready_for_quality_scoring"
    assert feature_set.source_mode == "canonical_s3_privacy_safe_features"
    assert feature_set.freshness_status == "fresh"
    assert feature_set.eligible_for_retrieval is True
    assert feature_set.eligible_for_activation is False
    assert feature_set.feature_count == 2
    assert feature_set.model_version == "immutable-revision-123"
    assert feature_set.lineage["raw_identifiers_read"] is False
    assert feature_set.lineage["raw_identifiers_stored"] is False
    assert feature_set.lineage["quality_status"] == "not_scored"

    feature = feature_set.features[0]
    assert feature.location_name == "montreal_downtown"
    assert feature.primary_poi_type == "middle_eastern_restaurant"
    assert feature.created_day_part == "evening"
    assert feature.cohort_size == 4200
    assert feature.quality_score == 0.0
    assert feature.eligible_for_activation is False
    assert len(feature.embedding) == 384
    assert "provider_a" not in feature.trait_text


def test_build_identity_is_stable_when_input_order_changes():
    first = _service().build(_request(), _rows())
    second = _service().build(_request(), list(reversed(_rows())))

    assert first.feature_set_id == second.feature_set_id
    assert first.source_fingerprint == second.source_fingerprint
    assert {
        feature.feature_id for feature in first.features
    } == {
        feature.feature_id for feature in second.features
    }


def test_changed_safe_content_creates_new_feature_set_identity():
    first = _service().build(_request(), _rows())
    changed = _rows()
    changed[0]["dp_noisy_count"] = 4300
    second = _service().build(_request(), changed)

    assert first.feature_set_id != second.feature_set_id
    assert first.source_fingerprint != second.source_fingerprint


def test_raw_identifier_column_is_rejected_before_embedding():
    rows = _rows()
    rows[0]["device_id"] = "raw-device"

    with pytest.raises(ValueError, match="blocked sensitive column"):
        _service().build(_request(), rows)


def test_provenance_mismatch_fails_closed():
    rows = _rows()
    rows[0]["tenant_id"] = "other_tenant"

    with pytest.raises(ValueError, match="tenant_id"):
        _service().build(_request(), rows)


def test_duplicate_feature_identity_fails_instead_of_overwriting():
    rows = _rows()
    rows[1] = dict(rows[0])

    with pytest.raises(ValueError, match="duplicate feature identity"):
        _service().build(_request(), rows)


def test_manifest_requires_pinned_model_and_complete_privacy_controls():
    with pytest.raises(ValueError, match="immutable"):
        _model(model_revision="main")

    with pytest.raises(ValueError, match="privacy controls"):
        _source(privacy_controls=("k_anonymity",))


def test_production_mode_requires_permitted_rights_and_never_activation():
    with pytest.raises(ValueError, match="permitted"):
        _source(rights_status="historical_internal_only")

    with pytest.raises(ValueError, match="cannot request"):
        _request(activation_requested=True)


def test_bad_embedding_shape_fails_closed():
    class BadEncoder:
        def encode(self, texts, *, model, batch_size):
            return np.ones((len(texts), 3), dtype=float)

    service = ProductionCanonicalFeatureEmbeddingService(
        encoder=BadEncoder(),
    )
    with pytest.raises(ValueError, match="shape"):
        service.build(_request(), _rows())
