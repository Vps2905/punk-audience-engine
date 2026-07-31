from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


class ProviderEventValidationError(ValueError):
    def __init__(self, reason_code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.reason_code = reason_code
        self.safe_message = safe_message


@dataclass(frozen=True)
class ProviderS3ObjectEvent:
    bucket: str
    key: str
    version_id: Optional[str] = None
    event_id: Optional[str] = None
    event_time: Optional[str] = None
    sequencer: Optional[str] = None

    def __post_init__(self) -> None:
        bucket = str(self.bucket or "").strip()
        key = str(self.key or "").strip().lstrip("/")
        if not bucket:
            raise ProviderEventValidationError(
                "s3_event_bucket_missing",
                "The provider S3 event does not contain a bucket.",
            )
        if not key:
            raise ProviderEventValidationError(
                "s3_event_key_missing",
                "The provider S3 event does not contain an object key.",
            )
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "key", key)

    @property
    def source_ref(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


@dataclass(frozen=True)
class ProviderQueueMessage:
    message_id: str
    receipt_handle: str
    body: str
    receive_count: int
    sent_timestamp_ms: Optional[int] = None

    def __post_init__(self) -> None:
        if not str(self.message_id or "").strip():
            raise ValueError("message_id is required")
        if not str(self.receipt_handle or "").strip():
            raise ValueError("receipt_handle is required")
        if int(self.receive_count) < 1:
            raise ValueError("receive_count must be >= 1")


@dataclass(frozen=True)
class ProviderWorkerConfig:
    max_messages: int = 5
    wait_time_seconds: int = 20
    visibility_timeout_seconds: int = 300
    max_receive_count: int = 5
    retry_base_seconds: int = 30
    retry_max_seconds: int = 900
    stale_processing_seconds: int = 900

    def __post_init__(self) -> None:
        if not 1 <= self.max_messages <= 10:
            raise ValueError("max_messages must be between 1 and 10")
        if not 0 <= self.wait_time_seconds <= 20:
            raise ValueError("wait_time_seconds must be between 0 and 20")
        if not 1 <= self.visibility_timeout_seconds <= 43_200:
            raise ValueError(
                "visibility_timeout_seconds must be between 1 and 43200"
            )
        if self.max_receive_count < 1:
            raise ValueError("max_receive_count must be >= 1")
        if self.retry_base_seconds < 1:
            raise ValueError("retry_base_seconds must be >= 1")
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError(
                "retry_max_seconds must be >= retry_base_seconds"
            )
        if self.stale_processing_seconds < self.visibility_timeout_seconds:
            raise ValueError(
                "stale_processing_seconds must be >= visibility_timeout_seconds"
            )

    def retry_delay_seconds(self, receive_count: int) -> int:
        exponent = max(0, int(receive_count) - 1)
        return min(
            self.retry_max_seconds,
            self.retry_base_seconds * (2**exponent),
        )

    def to_safe_dict(self) -> Dict[str, Any]:
        return {
            "max_messages": self.max_messages,
            "wait_time_seconds": self.wait_time_seconds,
            "visibility_timeout_seconds": self.visibility_timeout_seconds,
            "max_receive_count": self.max_receive_count,
            "retry_base_seconds": self.retry_base_seconds,
            "retry_max_seconds": self.retry_max_seconds,
            "stale_processing_seconds": self.stale_processing_seconds,
        }
