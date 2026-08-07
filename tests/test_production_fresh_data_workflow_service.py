from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.models.production_feature_build_contracts import (
    CanonicalFeatureSourceManifest,
    EmbeddingModelSpec,
)
from app.models.production_fresh_data_workflow_contracts import (
    ProductionFreshDataWorkflowRequest,
)
from app.services.production_fresh_data_workflow_service import (
    ProductionFreshDataWorkflowService,
)


class FakeState:
    def __init__(self, *, busy=False, terminal_record=None):
        self.busy = busy
        self.record = terminal_record
        self.transitions = []

    def claim(self, request, **_kwargs):
        if self.record is None:
            self.record = {
                "tenant_id": request.source.tenant_id,
                "workflow_id": request.workflow_id,
                "status": "claimed",
                "terminal": False,
                "result_receipt": {},
            }
            return {
                "acquired": True,
                "duplicate": False,
                "busy": False,
                "record": dict(self.record),
            }
        return {
            "acquired": False,
            "duplicate": True,
            "busy": self.busy,
            "record": dict(self.record),
        }

    def transition(self, **kwargs):
        self.transitions.append(dict(kwargs))
        self.record["status"] = kwargs["status"]
        self.record["terminal"] = kwargs["status"] in {
            "awaiting_review",
            "blocked",
            "quarantined",
            "failed",
        }
        for key in (
            "feature_receipt",
            "candidate_report",
            "overlap_report",
            "result_receipt",
            "reason_code",
        ):
            if kwargs.get(key) is not None:
                self.record[key] = kwargs[key]
        return dict(self.record)


class FakeIngestionReader:
    def __init__(self, receipt):
        self.receipt = receipt

    def get(self, ingestion_id):
        assert ingestion_id == self.receipt["ingestion_id"]
        return dict(self.receipt)


class FakeCanonicalReader:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def read(self, manifest):
        self.calls += 1
        assert manifest.source_ref.startswith("s3://")
        return list(self.rows)


class FakeFeatureBuilder:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = 0

    def execute(self, request, rows):
        self.calls += 1
        assert len(rows) == request.source.expected_row_count
        return dict(self.receipt)


class FakeSnapshotReader:
    def __init__(self, feature_set, rows):
        self.feature_set = feature_set
        self.rows = rows

    def read(self, **kwargs):
        assert kwargs["feature_set_id"] == self.feature_set["feature_set_id"]
        return dict(self.feature_set), [dict(row) for row in self.rows]


def _request() -> ProductionFreshDataWorkflowRequest:
    source = CanonicalFeatureSourceManifest(
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
    model = EmbeddingModelSpec(
        backend="sentence_transformers",
        model_name="intfloat/multilingual-e5-small",
        model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    )
    return ProductionFreshDataWorkflowRequest(
        ingestion_id="ingestion-1",
        source=source,
        model=model,
    )


def _ingestion(request):
    return {
        "ingestion_id": request.ingestion_id,
        "tenant_id": request.source.tenant_id,
        "provider_id": request.source.provider_id,
        "dataset_id": request.source.dataset_id,
        "status": "completed",
        "canonical_ref": request.source.source_ref,
        "output_rows": request.source.expected_row_count,
        "privacy_job_id": "privacy-job-1",
        "metadata": {
            "canonical_checksum_sha256": request.source.source_fingerprint,
            "canonical_object_version": request.source.source_version,
            "canonical_object_version_kind": (
                request.source.source_version_kind
            ),
            "canonical_size_bytes": request.source.source_size_bytes,
            "schema_version": request.source.schema_version,
            "purpose": request.source.purpose,
            "rights_policy_id": request.source.rights_policy_id,
            "privacy_policy_version": request.source.privacy_policy_version,
            "rights_status": request.source.rights_status,
            "canonical_source_latest_at": (
                request.source.source_latest_at.isoformat()
            ),
            "canonical_privacy_controls": [
                "daily_contribution_bounding",
                "k_anonymity",
                "differential_privacy",
                "no_identifier_output",
            ],
        },
    }


def _feature_rows():
    return [
        {
            "tenant_id": "tenant_a",
            "feature_set_id": "feature-set-1",
            "feature_set_version": 1,
            "feature_id": "feature-1",
            "location_name": "montreal",
            "primary_poi_type": "cafe",
            "created_day_part": "evening",
            "lookback_bucket": "30d",
            "cohort_size": 1500,
            "quality_score": 0.8,
            "privacy_status": "privacy_safe",
            "rights_status": "permitted",
            "purpose": "audience_intelligence",
            "source_latest_at": "2026-08-06T00:00:00+00:00",
            "freshness_status": "fresh",
            "data_use_mode": "production",
            "eligible_for_retrieval": True,
            "eligible_for_activation": False,
        },
        {
            "tenant_id": "tenant_a",
            "feature_set_id": "feature-set-1",
            "feature_set_version": 1,
            "feature_id": "feature-2",
            "location_name": "toronto",
            "primary_poi_type": "restaurant",
            "created_day_part": "evening",
            "lookback_bucket": "30d",
            "cohort_size": 1800,
            "quality_score": 0.7,
            "privacy_status": "privacy_safe",
            "rights_status": "permitted",
            "purpose": "audience_intelligence",
            "source_latest_at": "2026-08-06T00:00:00+00:00",
            "freshness_status": "fresh",
            "data_use_mode": "production",
            "eligible_for_retrieval": True,
            "eligible_for_activation": False,
        },
    ]


def _service(request, state=None, ingestion=None):
    feature_receipt = {
        "status": "completed",
        "tenant_id": request.source.tenant_id,
        "feature_set_id": "feature-set-1",
        "feature_set_version": 1,
        "feature_count": 2,
        "eligible_for_activation": False,
    }
    feature_set = {
        "tenant_id": request.source.tenant_id,
        "feature_set_id": "feature-set-1",
        "version": 1,
        "data_use_mode": "production",
        "source_fingerprint": request.source.source_fingerprint,
        "privacy_policy_version": request.source.privacy_policy_version,
        "rights_policy_id": request.source.rights_policy_id,
        "eligible_for_retrieval": True,
        "eligible_for_activation": False,
    }
    return ProductionFreshDataWorkflowService(
        state_service=state or FakeState(),
        ingestion_reader=FakeIngestionReader(
            ingestion or _ingestion(request)
        ),
        canonical_source_reader=FakeCanonicalReader(
            [{"safe": 1}, {"safe": 2}]
        ),
        feature_build_executor=FakeFeatureBuilder(feature_receipt),
        feature_snapshot_reader=FakeSnapshotReader(
            feature_set,
            _feature_rows(),
        ),
    )


def test_fresh_data_workflow_reaches_awaiting_review_without_release_side_effects():
    request = _request()
    state = FakeState()
    result = _service(request, state=state).execute(
        request,
        worker_id="worker-1",
    )

    assert result["status"] == "awaiting_human_review"
    assert result["durable_state"] == "awaiting_review"
    assert result["feature_count"] == 2
    assert result["generated_candidate_count"] == 2
    assert result["approval_required"] is True
    assert result["raw_identifiers_read"] is False
    assert result["candidate_database_write_performed"] is False
    assert result["lookalike_generation_performed"] is False
    assert result["activation_or_export_performed"] is False
    assert [item["status"] for item in state.transitions] == [
        "validating_ingestion",
        "reading_canonical",
        "building_features",
        "generating_candidates",
        "analyzing_overlap",
        "awaiting_review",
    ]


def test_ingestion_manifest_mismatch_is_quarantined():
    request = _request()
    state = FakeState()
    ingestion = _ingestion(request)
    ingestion["metadata"]["canonical_checksum_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="canonical_checksum"):
        _service(
            request,
            state=state,
            ingestion=ingestion,
        ).execute(request, worker_id="worker-1")

    assert state.record["status"] == "quarantined"
    assert state.record["reason_code"] == (
        "fresh_data_contract_validation_failed"
    )


def test_terminal_workflow_replays_verified_receipt():
    request = _request()
    stored = {
        "tenant_id": request.source.tenant_id,
        "workflow_id": request.workflow_id,
        "status": "awaiting_review",
        "terminal": True,
        "result_receipt": {
            "status": "awaiting_human_review",
            "workflow_id": request.workflow_id,
            "activation_or_export_performed": False,
        },
    }
    result = _service(
        request,
        state=FakeState(terminal_record=stored),
    ).execute(request, worker_id="worker-2")

    assert result["durable_claim_replayed"] is True
    assert result["workflow_busy"] is False
    assert result["activation_or_export_performed"] is False
