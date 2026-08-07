from datetime import datetime, timezone

import pytest

from app.models.provider_ingestion_contracts import ProviderDatasetContract
from app.services.production_fresh_data_trigger_service import (
    ProductionFreshDataTriggerService,
    ProductionFreshDataWorkflowRequestFactory,
)


def contract():
    return ProviderDatasetContract(
        tenant_id="tenant_a",
        provider_id="provider_a",
        dataset_id="visits",
        schema_version="v1",
        data_format="jsonl",
        allowed_bucket="safe-bucket",
        cohort_columns=("location_name", "primary_poi_type", "created_day_part"),
    )


def ingestion(**overrides):
    payload = {
        "ingestion_id": "ing_1",
        "tenant_id": "tenant_a",
        "provider_id": "provider_a",
        "dataset_id": "visits",
        "status": "completed",
        "canonical_ref": "s3://safe-bucket/canonical/a.jsonl",
        "output_rows": 1200,
        "metadata": {
            "schema_version": "v1",
            "purpose": "audience_intelligence",
            "rights_policy_id": "audience_intelligence_default",
            "rights_status": "permitted",
            "privacy_policy_version": "provider-privacy-pipeline-v1",
            "canonical_checksum_sha256": "a" * 64,
            "canonical_object_version": "version-1",
            "canonical_object_version_kind": "s3_version_id",
            "canonical_source_latest_at": datetime(2026, 8, 7, tzinfo=timezone.utc),
            "canonical_size_bytes": 4096,
            "canonical_privacy_controls": [
                "daily_contribution_bounding",
                "k_anonymity",
                "differential_privacy",
                "no_identifier_output",
                "lineage_logging",
            ],
        },
    }
    payload.update(overrides)
    return payload


def test_factory_builds_safe_pinned_request():
    req = ProductionFreshDataWorkflowRequestFactory().build(
        ingestion=ingestion(), contract=contract()
    )
    assert req.source.expected_row_count == 1200
    assert req.source.location_column == "location_name"
    assert req.source.category_column == "primary_poi_type"
    assert req.source.daypart_column == "created_day_part"
    assert req.source.data_use_mode == "offline_evaluation"
    assert req.model.model_revision == "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    assert req.activation_requested is False
    assert req.export_requested is False


def test_factory_rejects_missing_canonical_evidence():
    row = ingestion()
    row["metadata"] = dict(row["metadata"])
    row["metadata"].pop("canonical_checksum_sha256")
    with pytest.raises(ValueError, match="missing canonical workflow evidence"):
        ProductionFreshDataWorkflowRequestFactory().build(
            ingestion=row, contract=contract()
        )


def test_factory_rejects_contract_mismatch():
    row = ingestion(provider_id="other")
    with pytest.raises(ValueError, match="provider_id"):
        ProductionFreshDataWorkflowRequestFactory().build(
            ingestion=row, contract=contract()
        )


class FakeState:
    def __init__(self, rows): self.rows = rows
    def list_recent(self, **kwargs): return list(self.rows)


class FakeRegistry:
    def get_active(self, **kwargs): return contract()


class FakeExecutor:
    def __init__(self): self.calls = []
    def execute(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return {"durable_state": "awaiting_review", "durable_claim_replayed": False}


def test_trigger_runs_oldest_first_and_never_activates():
    first = ingestion(ingestion_id="old")
    second = ingestion(ingestion_id="new")
    executor = FakeExecutor()
    report = ProductionFreshDataTriggerService(
        ingestion_state=FakeState([second, first]),
        contract_registry=FakeRegistry(),
        workflow_executor=executor,
    ).run_once(worker_id="worker_1")
    assert report["status"] == "completed"
    assert report["triggered_or_replayed"] == 2
    assert [x[0].ingestion_id for x in executor.calls] == ["old", "new"]
    assert report["activation_or_export_performed"] is False


def test_trigger_isolates_bad_receipt_and_continues():
    bad = ingestion(ingestion_id="bad", canonical_ref="")
    good = ingestion(ingestion_id="good")
    executor = FakeExecutor()
    report = ProductionFreshDataTriggerService(
        ingestion_state=FakeState([good, bad]),
        contract_registry=FakeRegistry(),
        workflow_executor=executor,
    ).run_once(worker_id="worker_1")
    assert report["status"] == "completed_with_failures"
    assert report["failed"] == 1
    assert report["triggered_or_replayed"] == 1
    assert executor.calls[0][0].ingestion_id == "good"
