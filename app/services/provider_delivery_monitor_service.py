from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from app.models.provider_ingestion_contracts import ProviderDatasetContract
from app.services.provider_contract_registry_service import (
    ProviderContractRegistryService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_queue_service import ProviderQueue
from app.services.provider_privacy_window_service import (
    ProviderPrivacyWindowService,
)


class ProviderDeliveryMonitorService:
    def __init__(
        self,
        *,
        registry: ProviderContractRegistryService,
        state_service: ProviderIngestionStateService,
        queue: Optional[ProviderQueue] = None,
        privacy_window_service: Optional[
            ProviderPrivacyWindowService
        ] = None,
        now_provider: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._registry = registry
        self._state = state_service
        self._queue = queue
        self._privacy_windows = privacy_window_service
        self._now_provider = now_provider or (
            lambda: datetime.now(timezone.utc)
        )

    def report(self, *, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        now = self._now_provider()
        if now.tzinfo is None:
            raise ValueError("now_provider must return a timezone-aware datetime")

        datasets = [
            self._dataset_status(contract, now)
            for contract in self._registry.list_active(tenant_id=tenant_id)
        ]
        status_counts = self._state.status_counts(tenant_id=tenant_id)
        queue_metrics = self._safe_queue_metrics()
        publication = self._publication_health(tenant_id)

        stale_count = sum(
            item["delivery_status"] == "stale" for item in datasets
        )
        never_received_count = sum(
            item["delivery_status"] == "never_received" for item in datasets
        )
        delayed_count = sum(
            item["delivery_status"] == "delayed" for item in datasets
        )
        if not datasets:
            overall = "blocked_no_active_provider_contracts"
        elif stale_count or never_received_count:
            overall = "blocked_source_unavailable"
        elif publication["stale_open_window_count"]:
            overall = "blocked_incomplete_privacy_window"
        elif (
            delayed_count
            or status_counts.get("failed", 0)
            or (queue_metrics.get("dlq") or 0) > 0
            or publication["open_window_count"] > 0
        ):
            overall = "degraded"
        else:
            overall = "healthy"

        return {
            "status": overall,
            "checked_at": now.isoformat(),
            "active_dataset_count": len(datasets),
            "fresh_dataset_count": sum(
                item["delivery_status"] == "fresh" for item in datasets
            ),
            "delayed_dataset_count": delayed_count,
            "stale_dataset_count": stale_count,
            "never_received_dataset_count": never_received_count,
            "ingestion_status_counts": status_counts,
            "queue": queue_metrics,
            "canonical_publication": publication,
            "datasets": datasets,
        }

    def _dataset_status(
        self,
        contract: ProviderDatasetContract,
        now: datetime,
    ) -> Dict[str, Any]:
        records = self._state.list_recent(
            tenant_id=contract.tenant_id,
            provider_id=contract.provider_id,
            dataset_id=contract.dataset_id,
            limit=200,
        )
        completed = [
            record for record in records if record.get("status") == "completed"
        ]
        latest = completed[0] if completed else None
        latest_source_time = self._source_timestamp(latest)

        if latest_source_time is None:
            delivery_status = "never_received"
            age_minutes = None
        else:
            age_minutes = max(
                0.0,
                (now - latest_source_time).total_seconds() / 60.0,
            )
            if age_minutes <= contract.expected_delivery_interval_minutes:
                delivery_status = "fresh"
            elif age_minutes <= contract.freshness_sla_minutes:
                delivery_status = "delayed"
            else:
                delivery_status = "stale"

        recent_counts = {
            status: sum(record.get("status") == status for record in records)
            for status in (
                "completed",
                "blocked",
                "quarantined",
                "failed",
                "processing",
                "validating",
            )
        }
        return {
            "contract_key": contract.contract_key,
            "tenant_id": contract.tenant_id,
            "provider_id": contract.provider_id,
            "dataset_id": contract.dataset_id,
            "schema_version": contract.schema_version,
            "delivery_status": delivery_status,
            "eligible_for_audience_intelligence": delivery_status == "fresh",
            "latest_source_timestamp": (
                latest_source_time.isoformat()
                if latest_source_time is not None
                else None
            ),
            "source_age_minutes": (
                round(age_minutes, 2)
                if age_minutes is not None
                else None
            ),
            "expected_delivery_interval_minutes": (
                contract.expected_delivery_interval_minutes
            ),
            "freshness_sla_minutes": contract.freshness_sla_minutes,
            "latest_ingestion_id": (
                latest.get("ingestion_id") if latest else None
            ),
            "recent_status_counts": recent_counts,
        }

    def _source_timestamp(
        self,
        record: Optional[Dict[str, Any]],
    ) -> Optional[datetime]:
        if not record:
            return None
        metadata = record.get("metadata") or {}
        value = (
            metadata.get("event_time_end")
            or metadata.get("last_modified")
            or record.get("completed_at")
        )
        if not value:
            return None
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _safe_queue_metrics(self) -> Dict[str, Optional[int]]:
        if self._queue is None:
            return {
                "available": None,
                "in_flight": None,
                "delayed": None,
                "dlq": None,
            }
        try:
            return self._queue.attributes()
        except Exception:
            return {
                "available": None,
                "in_flight": None,
                "delayed": None,
                "dlq": None,
            }

    def _publication_health(
        self,
        tenant_id: Optional[str],
    ) -> Dict[str, Any]:
        if self._privacy_windows is None or not tenant_id:
            return {
                "open_window_count": 0,
                "stale_open_window_count": 0,
                "windows": [],
            }
        return self._privacy_windows.publication_health(
            tenant_id=tenant_id
        )
