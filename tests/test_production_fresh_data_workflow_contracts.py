from datetime import datetime, timezone

import pytest

from app.models.production_feature_build_contracts import (
    CanonicalFeatureSourceManifest,
    EmbeddingModelSpec,
)
from app.models.production_fresh_data_workflow_contracts import (
    ProductionFreshDataWorkflowRequest,
)


def _source() -> CanonicalFeatureSourceManifest:
    return CanonicalFeatureSourceManifest(
        tenant_id="tenant-a",
        provider_id="provider-a",
        dataset_id="signals-a",
        schema_version="v1",
        source_ref="s3://canonical-safe/tenant-a/object.jsonl",
        source_version="canonical-version-1",
        source_fingerprint="a" * 64,
        source_latest_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
        expected_row_count=2,
        data_use_mode="production",
        privacy_status="privacy_safe",
        privacy_policy_version="provider-privacy-pipeline-v1",
        privacy_controls=(
            "daily_contribution_bounding",
            "k_anonymity",
            "differential_privacy",
            "no_identifier_output",
        ),
        rights_status="permitted",
        rights_policy_id="provider-rights-v1",
        purpose="audience-intelligence",
        location_column="location_name",
        category_column="primary_poi_type",
        daypart_column="created_day_part",
        lookback_bucket_column="lookback_bucket",
        cohort_size_column="dp_noisy_count",
        source_size_bytes=512,
    )


def _model() -> EmbeddingModelSpec:
    return EmbeddingModelSpec(
        backend="sentence_transformers",
        model_name="intfloat/multilingual-e5-small",
        model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    )


def test_workflow_identity_is_stable_and_module2_request_is_safe():
    request = ProductionFreshDataWorkflowRequest(
        ingestion_id="ingestion-1",
        source=_source(),
        model=_model(),
    )

    assert request.workflow_id.startswith("fresh_data_workflow_")
    assert len(request.request_fingerprint) == 64
    assert request.feature_build_request.source == request.source
    assert request.feature_build_request.activation_requested is False
    assert request.to_safe_dict()["approval_required"] is True
    assert request.to_safe_dict()["export_requested"] is False


def test_workflow_rejects_release_affecting_requests():
    with pytest.raises(ValueError, match="lookalikes, activation, or export"):
        ProductionFreshDataWorkflowRequest(
            ingestion_id="ingestion-1",
            source=_source(),
            model=_model(),
            export_requested=True,
        )


def test_workflow_rejects_historical_preview_input():
    source = _source()
    historical = CanonicalFeatureSourceManifest(
        **{
            **source.to_safe_dict(),
            "source_latest_at": source.source_latest_at,
            "privacy_controls": source.privacy_controls,
            "data_use_mode": "historical_preview",
            "rights_status": "historical_internal_only",
        }
    )
    with pytest.raises(ValueError, match="offline_evaluation or production"):
        ProductionFreshDataWorkflowRequest(
            ingestion_id="ingestion-1",
            source=historical,
            model=_model(),
        )
