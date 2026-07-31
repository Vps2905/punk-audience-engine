from __future__ import annotations

import json
from urllib.parse import urlparse

from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_queue_service import ProviderQueue


class ProviderReplayService:
    """
    Approval-gated historical replay.

    Completed objects are re-enqueued only to verify idempotency and therefore
    cannot duplicate canonical output. Failed, blocked, or quarantined objects
    are reset to the gateway's controlled retry boundary and retain the same
    immutable object fingerprint.
    """

    def __init__(
        self,
        *,
        state_service: ProviderIngestionStateService,
        queue: ProviderQueue,
    ) -> None:
        self._state = state_service
        self._queue = queue

    def enqueue(
        self,
        ingestion_id: str,
        *,
        requested_by: str,
        reason: str,
        approval_reference: str,
    ) -> dict:
        prepared = self._state.prepare_controlled_replay(
            ingestion_id,
            requested_by=requested_by,
            reason=reason,
            approval_reference=approval_reference,
        )
        ingestion = prepared["ingestion"]
        source_ref = str(ingestion.get("source_ref") or "")
        parsed = urlparse(source_ref)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
            raise ValueError(
                "The stored provider source reference is not replayable."
            )

        event = {
            "version": "0",
            "id": prepared["replay_id"],
            "detail-type": "Object Created",
            "source": "aws.s3",
            "detail": {
                "reason": "PutObject",
                "bucket": {"name": parsed.netloc},
                "object": {
                    "key": parsed.path.lstrip("/"),
                    "version-id": ingestion.get("object_version"),
                },
            },
        }
        message_id = self._queue.send_message(
            body=json.dumps(event, sort_keys=True, separators=(",", ":"))
        )
        replay = self._state.mark_replay_enqueued(
            prepared["replay_id"],
            queue_message_id=message_id,
        )
        return {
            "replay_id": prepared["replay_id"],
            "ingestion_id": ingestion_id,
            "status": replay["status"],
            "replay_mode": prepared["replay_mode"],
            "queue_message_id": message_id,
            "source_ref": source_ref,
        }
