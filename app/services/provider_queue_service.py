from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from app.models.provider_queue_contracts import ProviderQueueMessage


class ProviderQueue(Protocol):
    def receive_messages(
        self,
        *,
        max_messages: int,
        wait_time_seconds: int,
        visibility_timeout_seconds: int,
    ) -> List[ProviderQueueMessage]:
        """Receive a bounded batch without deleting it."""

    def delete_message(self, receipt_handle: str) -> None:
        """Acknowledge a terminally handled message."""

    def change_visibility(
        self,
        receipt_handle: str,
        *,
        visibility_timeout_seconds: int,
    ) -> None:
        """Delay redelivery after a transient failure."""

    def send_to_dlq(
        self,
        *,
        body: str,
        source_message_id: str,
        reason_code: str,
        receive_count: int,
    ) -> None:
        """Move a permanently failed or exhausted message to the DLQ."""

    def send_message(self, *, body: str) -> str:
        """Enqueue a controlled replay message."""

    def attributes(self) -> Dict[str, Optional[int]]:
        """Return safe queue depth metrics."""


class SQSProviderQueue:
    def __init__(
        self,
        *,
        queue_url: str,
        dlq_url: str,
        client: Optional[Any] = None,
        region_name: Optional[str] = None,
    ) -> None:
        if not str(queue_url or "").strip():
            raise ValueError("queue_url is required")
        if not str(dlq_url or "").strip():
            raise ValueError("dlq_url is required")
        self._queue_url = str(queue_url).strip()
        self._dlq_url = str(dlq_url).strip()
        self._client = client or self._create_client(region_name)

    def receive_messages(
        self,
        *,
        max_messages: int,
        wait_time_seconds: int,
        visibility_timeout_seconds: int,
    ) -> List[ProviderQueueMessage]:
        response = self._client.receive_message(
            QueueUrl=self._queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time_seconds,
            VisibilityTimeout=visibility_timeout_seconds,
            AttributeNames=[
                "ApproximateReceiveCount",
                "SentTimestamp",
            ],
        )
        messages: List[ProviderQueueMessage] = []
        for item in response.get("Messages") or []:
            attributes = item.get("Attributes") or {}
            messages.append(
                ProviderQueueMessage(
                    message_id=str(item.get("MessageId") or ""),
                    receipt_handle=str(item.get("ReceiptHandle") or ""),
                    body=str(item.get("Body") or ""),
                    receive_count=int(
                        attributes.get("ApproximateReceiveCount") or 1
                    ),
                    sent_timestamp_ms=self._optional_int(
                        attributes.get("SentTimestamp")
                    ),
                )
            )
        return messages

    def delete_message(self, receipt_handle: str) -> None:
        self._client.delete_message(
            QueueUrl=self._queue_url,
            ReceiptHandle=receipt_handle,
        )

    def change_visibility(
        self,
        receipt_handle: str,
        *,
        visibility_timeout_seconds: int,
    ) -> None:
        self._client.change_message_visibility(
            QueueUrl=self._queue_url,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=visibility_timeout_seconds,
        )

    def send_to_dlq(
        self,
        *,
        body: str,
        source_message_id: str,
        reason_code: str,
        receive_count: int,
    ) -> None:
        self._client.send_message(
            QueueUrl=self._dlq_url,
            MessageBody=body,
            MessageAttributes={
                "source_message_id": {
                    "DataType": "String",
                    "StringValue": source_message_id,
                },
                "reason_code": {
                    "DataType": "String",
                    "StringValue": reason_code,
                },
                "receive_count": {
                    "DataType": "Number",
                    "StringValue": str(receive_count),
                },
            },
        )

    def send_message(self, *, body: str) -> str:
        response = self._client.send_message(
            QueueUrl=self._queue_url,
            MessageBody=body,
        )
        return str(response.get("MessageId") or "")

    def attributes(self) -> Dict[str, Optional[int]]:
        requested = [
            "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesNotVisible",
            "ApproximateNumberOfMessagesDelayed",
        ]
        response = self._client.get_queue_attributes(
            QueueUrl=self._queue_url,
            AttributeNames=requested,
        )
        values = response.get("Attributes") or {}
        dlq_response = self._client.get_queue_attributes(
            QueueUrl=self._dlq_url,
            AttributeNames=["ApproximateNumberOfMessages"],
        )
        dlq_values = dlq_response.get("Attributes") or {}
        return {
            "available": self._optional_int(
                values.get("ApproximateNumberOfMessages")
            ),
            "in_flight": self._optional_int(
                values.get("ApproximateNumberOfMessagesNotVisible")
            ),
            "delayed": self._optional_int(
                values.get("ApproximateNumberOfMessagesDelayed")
            ),
            "dlq": self._optional_int(
                dlq_values.get("ApproximateNumberOfMessages")
            ),
        }

    def _create_client(self, region_name: Optional[str]):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required when an SQS client is not injected."
            ) from exc
        return boto3.client("sqs", region_name=region_name)

    def _optional_int(self, value: Any) -> Optional[int]:
        if value in {None, ""}:
            return None
        return int(value)
