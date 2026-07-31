from __future__ import annotations

import argparse
import json
import os

from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_queue_service import SQSProviderQueue
from app.services.provider_replay_service import ProviderReplayService


def _required_env(name: str) -> str:
    value = str(os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is required.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enqueue an approval-gated provider ingestion replay."
    )
    parser.add_argument("--ingestion-id", required=True)
    parser.add_argument("--requested-by", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--approval-reference", required=True)
    args = parser.parse_args()

    database_url = _required_env("PROVIDER_INGESTION_DATABASE_URL")
    queue = SQSProviderQueue(
        queue_url=_required_env("PROVIDER_SQS_QUEUE_URL"),
        dlq_url=_required_env("PROVIDER_SQS_DLQ_URL"),
        region_name=_required_env("PROVIDER_S3_REGION"),
    )
    result = ProviderReplayService(
        state_service=ProviderIngestionStateService(
            database_url=database_url
        ),
        queue=queue,
    ).enqueue(
        args.ingestion_id,
        requested_by=args.requested_by,
        reason=args.reason,
        approval_reference=args.approval_reference,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
