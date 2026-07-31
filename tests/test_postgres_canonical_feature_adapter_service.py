from datetime import datetime, timezone

import pytest

from app.models.audience_feature_contracts import LegacySafeVectorSnapshot
from app.services.postgres_canonical_feature_adapter_service import (
    PostgresCanonicalFeatureAdapterService,
)


def _snapshot(metadata=None):
    vector = [0.0] * 384
    vector[0] = 1.0
    return LegacySafeVectorSnapshot(
        job_id="historical_job_1",
        model_info={
            "backend": "sklearn_hashing",
            "model_name": "sklearn_hashing_vectorizer",
        },
        vector_count=1,
        vector_dimension=384,
        indexed_at=datetime(2026, 7, 9, tzinfo=timezone.utc),
        latest_source_timestamp=datetime(
            2026,
            7,
            8,
            8,
            40,
            tzinfo=timezone.utc,
        ),
        rows=[
            {
                "vector_index": 0,
                "trait_text": "Montreal restaurant evening",
                "location_name": "Montréal Downtown",
                "primary_poi_type": "Middle Eastern Restaurant",
                "created_day_part": "Evening",
                "quality_score": 0.82,
                "embedding": vector,
                "metadata_json": metadata
                or {
                    "location_name": "Montréal Downtown",
                    "primary_poi_type": "Middle Eastern Restaurant",
                    "created_day_part": "Evening",
                    "quality_score": 0.82,
                    "noisy_maid_volume": 4200,
                    "privacy_status": "passed",
                },
            }
        ],
    )


def _adapter():
    return PostgresCanonicalFeatureAdapterService(
        stale_after_hours=48,
        now_fn=lambda: datetime(
            2026,
            7,
            25,
            tzinfo=timezone.utc,
        ),
    )


def test_historical_snapshot_becomes_searchable_but_never_activatable():
    feature_set = _adapter().adapt(
        _snapshot(),
        tenant_id="tenant-a",
        purpose="internal audience evaluation",
        rights_policy_id="rights-2026-01",
        privacy_policy_version="privacy-2026-01",
    )

    assert feature_set.freshness_status == "stale"
    assert feature_set.data_use_mode == "historical_preview"
    assert feature_set.eligible_for_retrieval is True
    assert feature_set.eligible_for_activation is False
    assert feature_set.feature_count == 1

    feature = feature_set.features[0]
    assert feature.location_name == "montreal_downtown"
    assert feature.primary_poi_type == "middle_eastern_restaurant"
    assert feature.created_day_part == "evening"
    assert feature.cohort_size == 4200
    assert feature.privacy_status == "passed"
    assert feature.eligible_for_retrieval is True
    assert feature.eligible_for_activation is False
    assert len(feature.embedding) == 384


def test_feature_identity_is_stable_for_same_versioned_snapshot():
    kwargs = {
        "tenant_id": "tenant-a",
        "purpose": "internal evaluation",
        "rights_policy_id": "rights-policy",
        "privacy_policy_version": "privacy-policy",
    }
    first = _adapter().adapt(_snapshot(), **kwargs)
    second = _adapter().adapt(_snapshot(), **kwargs)

    assert first.feature_set_id == second.feature_set_id
    assert first.source_fingerprint == second.source_fingerprint
    assert first.features[0].feature_id == second.features[0].feature_id


def test_changed_snapshot_content_creates_new_version_identity():
    kwargs = {
        "tenant_id": "tenant-a",
        "purpose": "internal evaluation",
        "rights_policy_id": "rights-policy",
        "privacy_policy_version": "privacy-policy",
    }
    original = _snapshot()
    changed_rows = [dict(original.rows[0])]
    changed_embedding = list(changed_rows[0]["embedding"])
    changed_embedding[0] = 0.5
    changed_embedding[1] = 0.5
    changed_rows[0]["embedding"] = changed_embedding
    changed = LegacySafeVectorSnapshot(
        job_id=original.job_id,
        model_info=original.model_info,
        vector_count=original.vector_count,
        vector_dimension=original.vector_dimension,
        indexed_at=original.indexed_at,
        latest_source_timestamp=original.latest_source_timestamp,
        rows=changed_rows,
    )

    first = _adapter().adapt(original, **kwargs)
    second = _adapter().adapt(changed, **kwargs)

    assert first.source_fingerprint != second.source_fingerprint
    assert first.feature_set_id != second.feature_set_id


def test_raw_identifier_metadata_is_rejected_at_canonical_boundary():
    metadata = {
        "location_name": "montreal",
        "primary_poi_type": "restaurant",
        "created_day_part": "evening",
        "privacy_status": "passed",
        "device_id": "raw-device-value",
    }

    with pytest.raises(ValueError, match="blocked sensitive column"):
        _adapter().adapt(
            _snapshot(metadata),
            tenant_id="tenant-a",
            purpose="internal evaluation",
            rights_policy_id="rights-policy",
            privacy_policy_version="privacy-policy",
        )


def test_legacy_snapshot_cannot_be_relabelled_as_production():
    with pytest.raises(ValueError, match="historical_preview"):
        _adapter().adapt(
            _snapshot(),
            tenant_id="tenant-a",
            purpose="campaign activation",
            rights_policy_id="rights-policy",
            privacy_policy_version="privacy-policy",
            data_use_mode="production",
        )


def test_dimension_mismatch_fails_closed():
    snapshot = _snapshot()
    incompatible = LegacySafeVectorSnapshot(
        job_id=snapshot.job_id,
        model_info=snapshot.model_info,
        vector_count=snapshot.vector_count,
        vector_dimension=3,
        indexed_at=snapshot.indexed_at,
        latest_source_timestamp=snapshot.latest_source_timestamp,
        rows=snapshot.rows,
    )

    with pytest.raises(ValueError, match="Expected 384"):
        _adapter().adapt(
            incompatible,
            tenant_id="tenant-a",
            purpose="internal evaluation",
            rights_policy_id="rights-policy",
            privacy_policy_version="privacy-policy",
        )
