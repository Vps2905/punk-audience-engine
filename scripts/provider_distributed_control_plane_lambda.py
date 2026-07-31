"""
Lambda entry point used by the provider distributed Step Functions workflow.

Database credentials are resolved from Secrets Manager at invocation time.
The secret value, object payload, and provider identifiers are never logged.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import boto3

from app.models.provider_scale_contracts import (
    ProviderDistributedJobRequest,
)
from app.services.provider_distributed_data_plane_service import (
    ProviderDistributedCompletionService,
    ProviderDistributedControlPlaneService,
    ProviderDistributedJobPlan,
    ProviderDistributedStagingConfig,
    ProviderDistributedStagingService,
)
from app.services.provider_distributed_privacy_budget_service import (
    ProviderDistributedPrivacyBudgetService,
)
from app.services.provider_distributed_reconciliation_service import (
    ProviderDistributedReconciliationService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_privacy_window_service import (
    ProviderPrivacyWindowService,
)


def _required_env(name: str) -> str:
    value = str(os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Required deployment setting is missing: {name}")
    return value


def _database_url() -> str:
    secret_arn = _required_env(
        "PROVIDER_INGESTION_DATABASE_SECRET_ARN"
    )
    response = boto3.client("secretsmanager").get_secret_value(
        SecretId=secret_arn
    )
    secret_string = response.get("SecretString")
    if not secret_string:
        raise RuntimeError(
            "Provider ingestion database secret has no string value"
        )
    try:
        payload = json.loads(secret_string)
    except json.JSONDecodeError:
        payload = {"database_url": secret_string}
    database_url = str(payload.get("database_url") or "").strip()
    if not database_url:
        raise RuntimeError(
            "Provider ingestion database secret lacks database_url"
        )
    return database_url


def _service() -> ProviderDistributedControlPlaneService:
    database_url = _database_url()
    state = ProviderIngestionStateService(database_url=database_url)
    s3 = boto3.client("s3")
    staging = ProviderDistributedStagingService(
        config=ProviderDistributedStagingConfig(
            bucket=_required_env(
                "PROVIDER_DISTRIBUTED_STAGING_BUCKET"
            ),
            prefix=os.getenv(
                "PROVIDER_DISTRIBUTED_STAGING_PREFIX",
                "provider-distributed-staging/",
            ),
            server_side_encryption=os.getenv(
                "PROVIDER_DISTRIBUTED_STAGING_ENCRYPTION",
                "aws:kms",
            ),
            kms_key_id=os.getenv(
                "PROVIDER_DISTRIBUTED_STAGING_KMS_KEY_ID"
            ),
        ),
        client=s3,
    )
    return ProviderDistributedControlPlaneService(
        staging_service=staging,
        completion_service=ProviderDistributedCompletionService(
            state_service=state
        ),
        privacy_budget_service=ProviderDistributedPrivacyBudgetService(
            database_url=database_url
        ),
        result_client=s3,
        privacy_window_service=ProviderPrivacyWindowService(
            database_url=database_url
        ),
    )


def _plan(payload: Dict[str, Any]) -> ProviderDistributedJobPlan:
    try:
        return ProviderDistributedJobPlan(
            ingestion_id=str(payload["ingestion_id"]),
            fingerprint=str(payload["fingerprint"]),
            staged_source_ref=str(payload["staged_source_ref"]),
            canonical_prefix=str(payload["canonical_prefix"]),
            result_ref=str(payload["result_ref"]),
            privacy_budget_scope=str(payload["privacy_budget_scope"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Distributed plan is incomplete") from exc


def lambda_handler(
    event: Dict[str, Any],
    context: Any,
) -> Dict[str, Any]:
    del context
    if not isinstance(event, dict):
        raise ValueError("Control-plane event must be an object")
    action = str(event.get("action") or "").strip().lower()
    if action == "reconcile":
        database_url = _database_url()
        return ProviderDistributedReconciliationService(
            state_service=ProviderIngestionStateService(
                database_url=database_url
            ),
            step_functions_client=boto3.client("stepfunctions"),
            privacy_budget_service=(
                ProviderDistributedPrivacyBudgetService(
                    database_url=database_url
                )
            ),
        ).reconcile(
            stale_after_seconds=int(
                os.getenv(
                    "PROVIDER_DISTRIBUTED_RECONCILIATION_STALE_SECONDS",
                    "900",
                )
            )
        )
    request = ProviderDistributedJobRequest.from_safe_dict(
        event.get("request") or {}
    )
    service = _service()
    if action == "prepare":
        return service.prepare(request)

    plan = _plan(event.get("plan") or {})
    release_id = str(event.get("privacy_release_id") or "").strip()
    if not release_id:
        raise ValueError("privacy_release_id is required")
    if action == "finalize":
        return service.finalize(
            request=request,
            plan=plan,
            privacy_release_id=release_id,
        )
    if action == "fail":
        return service.fail(
            request=request,
            plan=plan,
            privacy_release_id=release_id,
            reason_code="distributed_execution_failed",
        )
    raise ValueError("Unsupported distributed control-plane action")
