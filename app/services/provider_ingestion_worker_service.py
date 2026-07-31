from __future__ import annotations

from typing import Any, Dict, Optional

from app.models.provider_ingestion_contracts import (
    CanonicalObjectTarget,
    ProviderObjectDescriptor,
)
from app.models.provider_queue_contracts import (
    ProviderEventValidationError,
    ProviderQueueMessage,
    ProviderWorkerConfig,
)
from app.services.provider_contract_registry_service import (
    ProviderContractRegistryService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_object_ingestion_gateway_service import (
    ProviderObjectIngestionGatewayService,
)
from app.services.provider_object_store_service import ProviderObjectStore
from app.services.provider_queue_service import ProviderQueue
from app.services.provider_s3_event_parser_service import (
    ProviderS3EventParserService,
)


class ProviderIngestionWorkerService:
    TERMINAL_ACK_STATUSES = {
        "completed",
        "blocked",
        "quarantined",
        "dispatched",
    }
    RETRYABLE_GATEWAY_REASONS = {
        "provider_object_read_failed",
        "privacy_pipeline_failed",
        "canonical_write_failed",
        "canonical_receipt_invalid",
        "distributed_dispatch_failed",
        "distributed_dispatch_receipt_invalid",
        "worker_execution_stale",
    }

    def __init__(
        self,
        *,
        queue: ProviderQueue,
        registry: ProviderContractRegistryService,
        state_service: ProviderIngestionStateService,
        object_store: ProviderObjectStore,
        gateway: ProviderObjectIngestionGatewayService,
        canonical_target: CanonicalObjectTarget,
        config: Optional[ProviderWorkerConfig] = None,
        event_parser: Optional[ProviderS3EventParserService] = None,
    ) -> None:
        self._queue = queue
        self._registry = registry
        self._state = state_service
        self._object_store = object_store
        self._gateway = gateway
        self._canonical_target = canonical_target
        self._config = config or ProviderWorkerConfig()
        self._event_parser = event_parser or ProviderS3EventParserService()

    def run_once(self) -> Dict[str, Any]:
        recovered = self._state.recover_stale_in_progress(
            stale_after_seconds=self._config.stale_processing_seconds,
        )
        messages = self._queue.receive_messages(
            max_messages=self._config.max_messages,
            wait_time_seconds=self._config.wait_time_seconds,
            visibility_timeout_seconds=(
                self._config.visibility_timeout_seconds
            ),
        )

        results = [self.process_message(message) for message in messages]
        counts = {
            "acknowledged": 0,
            "retry_scheduled": 0,
            "dead_lettered": 0,
            "queue_action_failed": 0,
        }
        for result in results:
            action = result.get("queue_action")
            if action in counts:
                counts[action] += 1

        return {
            "status": "completed",
            "messages_received": len(messages),
            "stale_executions_recovered": recovered,
            **counts,
            "results": results,
        }

    def process_message(
        self,
        message: ProviderQueueMessage,
    ) -> Dict[str, Any]:
        try:
            event = self._event_parser.parse(message.body)
        except ProviderEventValidationError as exc:
            return self._dead_letter(
                message,
                reason_code=exc.reason_code,
            )

        try:
            head = self._object_store.head_object(event)
        except (TypeError, ValueError):
            return self._dead_letter(
                message,
                reason_code="provider_object_metadata_invalid",
            )
        except Exception as exc:
            return self._retry_or_dead_letter(
                message,
                reason_code="provider_object_head_failed",
                retryable=self._is_transient_exception(exc),
                source_ref=event.source_ref,
            )

        try:
            contract = self._registry.resolve_for_object(
                bucket=event.bucket,
                key=event.key,
                schema_version=head.manifest.schema_version,
            )
        except KeyError:
            return self._retry_or_dead_letter(
                message,
                reason_code="provider_contract_not_available",
                retryable=True,
                source_ref=event.source_ref,
            )
        except (TypeError, ValueError):
            return self._dead_letter(
                message,
                reason_code="provider_contract_resolution_failed",
                source_ref=event.source_ref,
            )
        except Exception as exc:
            return self._retry_or_dead_letter(
                message,
                reason_code="provider_contract_store_unavailable",
                retryable=self._is_transient_exception(exc),
                source_ref=event.source_ref,
            )

        descriptor = ProviderObjectDescriptor(
            tenant_id=contract.tenant_id,
            provider_id=contract.provider_id,
            dataset_id=contract.dataset_id,
            bucket=event.bucket,
            key=event.key,
            size_bytes=head.size_bytes,
            content_type=head.content_type,
            version_id=head.version_id or event.version_id,
            checksum_sha256=head.checksum_sha256,
            server_side_encryption=head.server_side_encryption,
            event_id=event.event_id,
            last_modified=head.last_modified,
        )

        try:
            gateway_result = self._gateway.ingest_object(
                contract=contract,
                descriptor=descriptor,
                manifest=head.manifest,
                canonical_target=self._canonical_target,
                actor="provider_ingestion_worker",
                retry_failed=True,
            )
        except Exception as exc:
            return self._retry_or_dead_letter(
                message,
                reason_code="provider_gateway_unavailable",
                retryable=self._is_transient_exception(exc),
                source_ref=event.source_ref,
            )

        status = str(gateway_result.get("status") or "")
        reason_code = str(gateway_result.get("reason_code") or "") or None
        safe_base = {
            "message_id": message.message_id,
            "receive_count": message.receive_count,
            "source_ref": event.source_ref,
            "ingestion_id": gateway_result.get("ingestion_id"),
            "ingestion_status": status,
            "reason_code": reason_code,
            "duplicate": bool(gateway_result.get("duplicate")),
        }

        if status in self.TERMINAL_ACK_STATUSES:
            return self._acknowledge(message, safe_base)
        if status in {"received", "validating", "dispatching", "processing"}:
            return self._retry_or_dead_letter(
                message,
                reason_code="provider_object_processing_in_progress",
                retryable=True,
                source_ref=event.source_ref,
                ingestion_id=gateway_result.get("ingestion_id"),
            )
        if status == "failed":
            return self._retry_or_dead_letter(
                message,
                reason_code=reason_code or "provider_ingestion_failed",
                retryable=reason_code in self.RETRYABLE_GATEWAY_REASONS,
                source_ref=event.source_ref,
                ingestion_id=gateway_result.get("ingestion_id"),
            )
        return self._retry_or_dead_letter(
            message,
            reason_code="provider_gateway_result_invalid",
            retryable=False,
            source_ref=event.source_ref,
            ingestion_id=gateway_result.get("ingestion_id"),
        )

    def _acknowledge(
        self,
        message: ProviderQueueMessage,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            self._queue.delete_message(message.receipt_handle)
        except Exception:
            return {
                **result,
                "queue_action": "queue_action_failed",
                "queue_reason_code": "queue_ack_failed",
            }
        return {
            **result,
            "queue_action": "acknowledged",
        }

    def _retry_or_dead_letter(
        self,
        message: ProviderQueueMessage,
        *,
        reason_code: str,
        retryable: bool,
        source_ref: Optional[str] = None,
        ingestion_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if retryable and message.receive_count < self._config.max_receive_count:
            delay = self._config.retry_delay_seconds(message.receive_count)
            try:
                self._queue.change_visibility(
                    message.receipt_handle,
                    visibility_timeout_seconds=delay,
                )
            except Exception:
                return {
                    "message_id": message.message_id,
                    "receive_count": message.receive_count,
                    "source_ref": source_ref,
                    "ingestion_id": ingestion_id,
                    "reason_code": reason_code,
                    "queue_action": "queue_action_failed",
                    "queue_reason_code": "queue_visibility_change_failed",
                }
            return {
                "message_id": message.message_id,
                "receive_count": message.receive_count,
                "source_ref": source_ref,
                "ingestion_id": ingestion_id,
                "reason_code": reason_code,
                "retry_delay_seconds": delay,
                "queue_action": "retry_scheduled",
            }
        return self._dead_letter(
            message,
            reason_code=reason_code,
            source_ref=source_ref,
            ingestion_id=ingestion_id,
        )

    def _dead_letter(
        self,
        message: ProviderQueueMessage,
        *,
        reason_code: str,
        source_ref: Optional[str] = None,
        ingestion_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            self._queue.send_to_dlq(
                body=message.body,
                source_message_id=message.message_id,
                reason_code=reason_code,
                receive_count=message.receive_count,
            )
            self._queue.delete_message(message.receipt_handle)
        except Exception:
            return {
                "message_id": message.message_id,
                "receive_count": message.receive_count,
                "source_ref": source_ref,
                "ingestion_id": ingestion_id,
                "reason_code": reason_code,
                "queue_action": "queue_action_failed",
                "queue_reason_code": "dead_letter_transfer_failed",
            }
        return {
            "message_id": message.message_id,
            "receive_count": message.receive_count,
            "source_ref": source_ref,
            "ingestion_id": ingestion_id,
            "reason_code": reason_code,
            "queue_action": "dead_lettered",
        }

    def _is_transient_exception(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429 or (
            isinstance(status_code, int) and status_code >= 500
        ):
            return True
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            metadata = response.get("ResponseMetadata") or {}
            http_status = metadata.get("HTTPStatusCode")
            if http_status == 429 or (
                isinstance(http_status, int) and http_status >= 500
            ):
                return True
            error_code = str((response.get("Error") or {}).get("Code") or "")
            if error_code in {
                "InternalError",
                "RequestTimeout",
                "ServiceUnavailable",
                "SlowDown",
                "Throttling",
                "ThrottlingException",
            }:
                return True
        class_name = type(exc).__name__.lower()
        return any(
            token in class_name
            for token in (
                "connection",
                "operational",
                "throttl",
                "timeout",
                "temporar",
            )
        )
