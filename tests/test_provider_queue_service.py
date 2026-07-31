from app.services.provider_queue_service import SQSProviderQueue


class FakeSQSClient:
    def __init__(self):
        self.calls = []

    def receive_message(self, **kwargs):
        self.calls.append(("receive", kwargs))
        return {
            "Messages": [
                {
                    "MessageId": "message-1",
                    "ReceiptHandle": "receipt-1",
                    "Body": "{}",
                    "Attributes": {
                        "ApproximateReceiveCount": "3",
                        "SentTimestamp": "1234",
                    },
                }
            ]
        }

    def delete_message(self, **kwargs):
        self.calls.append(("delete", kwargs))

    def change_message_visibility(self, **kwargs):
        self.calls.append(("visibility", kwargs))

    def send_message(self, **kwargs):
        self.calls.append(("send", kwargs))
        return {"MessageId": "sent-message"}

    def get_queue_attributes(self, **kwargs):
        self.calls.append(("attributes", kwargs))
        if kwargs["QueueUrl"] == "dlq-url":
            return {"Attributes": {"ApproximateNumberOfMessages": "2"}}
        return {
            "Attributes": {
                "ApproximateNumberOfMessages": "4",
                "ApproximateNumberOfMessagesNotVisible": "1",
                "ApproximateNumberOfMessagesDelayed": "3",
            }
        }


def test_sqs_adapter_preserves_receive_count_and_receipt():
    client = FakeSQSClient()
    queue = SQSProviderQueue(
        queue_url="queue-url",
        dlq_url="dlq-url",
        client=client,
    )

    messages = queue.receive_messages(
        max_messages=5,
        wait_time_seconds=20,
        visibility_timeout_seconds=300,
    )

    assert messages[0].message_id == "message-1"
    assert messages[0].receipt_handle == "receipt-1"
    assert messages[0].receive_count == 3


def test_sqs_adapter_dlq_and_depth_operations_are_explicit():
    client = FakeSQSClient()
    queue = SQSProviderQueue(
        queue_url="queue-url",
        dlq_url="dlq-url",
        client=client,
    )

    queue.send_to_dlq(
        body="{}",
        source_message_id="message-1",
        reason_code="invalid_event",
        receive_count=5,
    )
    metrics = queue.attributes()

    assert metrics == {
        "available": 4,
        "in_flight": 1,
        "delayed": 3,
        "dlq": 2,
    }
    send_request = next(
        request for operation, request in client.calls if operation == "send"
    )
    assert send_request["QueueUrl"] == "dlq-url"
    assert send_request["MessageAttributes"]["reason_code"]["StringValue"] == (
        "invalid_event"
    )
