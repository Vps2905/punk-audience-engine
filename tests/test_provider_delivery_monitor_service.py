from datetime import datetime, timezone
from pathlib import Path

from app.models.provider_ingestion_contracts import (
    ProviderDatasetContract,
    ProviderObjectDescriptor,
)
from app.services.provider_contract_registry_service import (
    ProviderContractRegistryService,
)
from app.services.provider_delivery_monitor_service import (
    ProviderDeliveryMonitorService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)


def _contract(dataset_id, *, expected=60, freshness=120):
    return ProviderDatasetContract(
        tenant_id="tenant_a",
        provider_id="provider_a",
        dataset_id=dataset_id,
        schema_version="v1",
        data_format="csv",
        allowed_bucket="provider-landing",
        allowed_prefix=f"{dataset_id}/",
        expected_delivery_interval_minutes=expected,
        freshness_sla_minutes=freshness,
    )


def _completed_delivery(
    state,
    *,
    dataset_id,
    event_time_end,
):
    descriptor = ProviderObjectDescriptor(
        tenant_id="tenant_a",
        provider_id="provider_a",
        dataset_id=dataset_id,
        bucket="provider-landing",
        key=f"{dataset_id}/object.csv",
        version_id="version-1",
        size_bytes=10,
        content_type="text/csv",
        server_side_encryption="AES256",
    )
    claim = state.claim_object(
        descriptor,
        metadata={"event_time_end": event_time_end},
    )
    ingestion_id = claim["record"]["ingestion_id"]
    state.transition(ingestion_id, status="validating")
    state.transition(ingestion_id, status="processing")
    state.transition(
        ingestion_id,
        status="completed",
        input_rows=1000,
        output_rows=1,
        canonical_ref="s3://canonical-safe/output.jsonl",
    )


def test_monitor_reports_fresh_stale_and_never_received_datasets(
    tmp_path: Path,
):
    db_url = f"sqlite:///{tmp_path / 'monitor.db'}"
    registry = ProviderContractRegistryService(database_url=db_url)
    state = ProviderIngestionStateService(database_url=db_url)
    for dataset_id in ("fresh_data", "stale_data", "missing_data"):
        registry.register(_contract(dataset_id), actor="test")
    _completed_delivery(
        state,
        dataset_id="fresh_data",
        event_time_end="2026-07-24T09:30:00+00:00",
    )
    _completed_delivery(
        state,
        dataset_id="stale_data",
        event_time_end="2026-07-24T05:00:00+00:00",
    )

    report = ProviderDeliveryMonitorService(
        registry=registry,
        state_service=state,
        now_provider=lambda: datetime(
            2026,
            7,
            24,
            10,
            0,
            tzinfo=timezone.utc,
        ),
    ).report()

    by_dataset = {
        item["dataset_id"]: item for item in report["datasets"]
    }
    assert by_dataset["fresh_data"]["delivery_status"] == "fresh"
    assert (
        by_dataset["fresh_data"]["eligible_for_audience_intelligence"]
        is True
    )
    assert by_dataset["stale_data"]["delivery_status"] == "stale"
    assert (
        by_dataset["stale_data"]["eligible_for_audience_intelligence"]
        is False
    )
    assert by_dataset["missing_data"]["delivery_status"] == "never_received"
    assert report["status"] == "blocked_source_unavailable"
