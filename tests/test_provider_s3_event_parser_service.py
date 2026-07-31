import json

import pytest

from app.models.provider_queue_contracts import ProviderEventValidationError
from app.services.provider_s3_event_parser_service import (
    ProviderS3EventParserService,
)


def _native_event():
    return {
        "Records": [
            {
                "eventTime": "2026-07-24T10:00:00Z",
                "eventName": "ObjectCreated:Put",
                "responseElements": {"x-amz-request-id": "request-1"},
                "s3": {
                    "bucket": {"name": "provider-landing"},
                    "object": {
                        "key": "delivery%2Fobject.csv",
                        "versionId": "version-1",
                        "sequencer": "001",
                    },
                },
            }
        ]
    }


def test_parses_native_s3_object_created_event():
    event = ProviderS3EventParserService().parse(json.dumps(_native_event()))

    assert event.bucket == "provider-landing"
    assert event.key == "delivery/object.csv"
    assert event.version_id == "version-1"
    assert event.event_id == "request-1"


def test_parses_sns_wrapped_s3_event():
    body = json.dumps(
        {
            "Type": "Notification",
            "Message": json.dumps(_native_event()),
        }
    )

    event = ProviderS3EventParserService().parse(body)

    assert event.source_ref == "s3://provider-landing/delivery/object.csv"


def test_parses_eventbridge_s3_event():
    body = json.dumps(
        {
            "version": "0",
            "id": "event-1",
            "source": "aws.s3",
            "time": "2026-07-24T10:00:00Z",
            "detail": {
                "reason": "PutObject",
                "bucket": {"name": "provider-landing"},
                "object": {
                    "key": "delivery/object.csv",
                    "version-id": "version-1",
                },
            },
        }
    )

    event = ProviderS3EventParserService().parse(body)

    assert event.event_id == "event-1"
    assert event.key == "delivery/object.csv"


def test_rejects_multi_record_message_to_keep_ack_atomic():
    payload = _native_event()
    payload["Records"].append(payload["Records"][0])

    with pytest.raises(ProviderEventValidationError) as exc_info:
        ProviderS3EventParserService().parse(json.dumps(payload))

    assert exc_info.value.reason_code == "s3_event_record_count_invalid"


def test_rejects_non_creation_event():
    payload = _native_event()
    payload["Records"][0]["eventName"] = "ObjectRemoved:Delete"

    with pytest.raises(ProviderEventValidationError) as exc_info:
        ProviderS3EventParserService().parse(json.dumps(payload))

    assert exc_info.value.reason_code == "s3_event_not_object_created"
