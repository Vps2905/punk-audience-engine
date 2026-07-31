from __future__ import annotations

import json
from typing import Any, Dict
from urllib.parse import unquote_plus

from app.models.provider_queue_contracts import (
    ProviderEventValidationError,
    ProviderS3ObjectEvent,
)


class ProviderS3EventParserService:
    """Parse native S3, EventBridge, or SNS-wrapped S3 notifications."""

    def parse(self, body: str) -> ProviderS3ObjectEvent:
        payload = self._load_json(body)
        payload = self._unwrap_sns(payload)

        if isinstance(payload.get("Records"), list):
            return self._parse_native_s3(payload)
        if payload.get("source") == "aws.s3" and isinstance(
            payload.get("detail"),
            dict,
        ):
            return self._parse_eventbridge(payload)

        raise ProviderEventValidationError(
            "unsupported_s3_event",
            "The queue message is not a supported S3 object-created event.",
        )

    def _load_json(self, body: str) -> Dict[str, Any]:
        try:
            payload = json.loads(str(body or ""))
        except json.JSONDecodeError as exc:
            raise ProviderEventValidationError(
                "invalid_queue_message_json",
                "The provider queue message is not valid JSON.",
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderEventValidationError(
                "invalid_queue_message_shape",
                "The provider queue message must be a JSON object.",
            )
        return payload

    def _unwrap_sns(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("Type") != "Notification":
            return payload
        message = payload.get("Message")
        if not isinstance(message, str):
            raise ProviderEventValidationError(
                "invalid_sns_envelope",
                "The SNS notification does not contain a message.",
            )
        return self._load_json(message)

    def _parse_native_s3(
        self,
        payload: Dict[str, Any],
    ) -> ProviderS3ObjectEvent:
        records = payload.get("Records") or []
        if len(records) != 1:
            raise ProviderEventValidationError(
                "s3_event_record_count_invalid",
                "Each queue message must contain exactly one S3 event record.",
            )
        record = records[0]
        if not isinstance(record, dict):
            raise ProviderEventValidationError(
                "invalid_s3_event_record",
                "The S3 event record is invalid.",
            )
        event_name = str(record.get("eventName") or "")
        if not event_name.startswith("ObjectCreated:"):
            raise ProviderEventValidationError(
                "s3_event_not_object_created",
                "Only S3 object-created events are accepted.",
            )
        s3 = record.get("s3")
        if not isinstance(s3, dict):
            raise ProviderEventValidationError(
                "invalid_s3_event_record",
                "The S3 event record does not contain object details.",
            )
        bucket = (s3.get("bucket") or {}).get("name")
        object_data = s3.get("object") or {}
        return ProviderS3ObjectEvent(
            bucket=str(bucket or ""),
            key=unquote_plus(str(object_data.get("key") or "")),
            version_id=object_data.get("versionId"),
            event_id=record.get("responseElements", {}).get(
                "x-amz-request-id"
            ),
            event_time=record.get("eventTime"),
            sequencer=object_data.get("sequencer"),
        )

    def _parse_eventbridge(
        self,
        payload: Dict[str, Any],
    ) -> ProviderS3ObjectEvent:
        detail = payload["detail"]
        if detail.get("reason") not in {None, "PutObject", "CompleteMultipartUpload"}:
            raise ProviderEventValidationError(
                "s3_event_not_object_created",
                "Only S3 object-created events are accepted.",
            )
        bucket = (detail.get("bucket") or {}).get("name")
        object_data = detail.get("object") or {}
        return ProviderS3ObjectEvent(
            bucket=str(bucket or ""),
            key=unquote_plus(str(object_data.get("key") or "")),
            version_id=object_data.get("version-id"),
            event_id=payload.get("id"),
            event_time=payload.get("time"),
            sequencer=object_data.get("sequencer"),
        )
