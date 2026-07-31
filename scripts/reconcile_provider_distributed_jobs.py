from __future__ import annotations

import json
import os

import boto3

from app.services.provider_distributed_privacy_budget_service import (
    ProviderDistributedPrivacyBudgetService,
)
from app.services.provider_distributed_reconciliation_service import (
    ProviderDistributedReconciliationService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)


def main() -> None:
    database_url = str(
        os.getenv("PROVIDER_INGESTION_DATABASE_URL") or ""
    ).strip()
    if not database_url:
        raise SystemExit(
            "STOP: PROVIDER_INGESTION_DATABASE_URL is not configured"
        )
    stale_after_seconds = int(
        os.getenv(
            "PROVIDER_DISTRIBUTED_RECONCILIATION_STALE_SECONDS",
            "900",
        )
    )
    service = ProviderDistributedReconciliationService(
        state_service=ProviderIngestionStateService(
            database_url=database_url
        ),
        step_functions_client=boto3.client(
            "stepfunctions",
            region_name=os.getenv("PROVIDER_S3_REGION") or None,
        ),
        privacy_budget_service=ProviderDistributedPrivacyBudgetService(
            database_url=database_url
        ),
    )
    print(
        json.dumps(
            service.reconcile(
                stale_after_seconds=stale_after_seconds
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
