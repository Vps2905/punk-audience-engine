from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.core.api_key_auth import require_audience_api_key


router = APIRouter(
    prefix="/api/audience-intelligence/module-1",
    tags=["Audience Intelligence Module 1 Status"],
    dependencies=[Depends(require_audience_api_key)],
)


def _env_present(*names: str) -> Dict[str, bool]:
    return {name: bool(os.getenv(name)) for name in names}


@router.get("/status")
def module_1_status() -> Dict[str, Any]:
    """
    Module 1 readiness/status endpoint.

    Important:
        This endpoint reports configuration presence only.
        It never returns raw secret values.
    """
    database_envs = _env_present(
        "AUDIENCE_INGESTION_DATABASE_URL",
        "AUDIENCE_LINEAGE_DATABASE_URL",
        "AUDIENCE_PRIVACY_BUDGET_DATABASE_URL",
        "AUDIENCE_HISTORY_DATABASE_URL",
        "ECHO_DATABASE_URL",
        "DATABASE_URL",
        "POSTGRES_URL",
        "POSTGRES_DATABASE_URL",
        "SUPABASE_DB_URL",
        "DB_URL",
    )

    api_key_configured = bool(os.getenv("AUDIENCE_API_KEY"))

    has_any_database = any(database_envs.values())
    gateway_enabled = str(
        os.getenv("PROVIDER_GATEWAY_ENABLED") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    distributed_enabled = str(
        os.getenv("PROVIDER_DISTRIBUTED_PROCESSING_ENABLED") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    distributed_settings = _env_present(
        "PROVIDER_DISTRIBUTED_STATE_MACHINE_ARN",
        "PROVIDER_DISTRIBUTED_STAGING_BUCKET",
        "PROVIDER_DISTRIBUTED_STAGING_KMS_KEY_ID",
        "PROVIDER_INGESTION_DATABASE_SECRET_ARN",
        "PROVIDER_DISTRIBUTED_TOKENIZATION_SECRET_ARN",
        "PROVIDER_DISTRIBUTED_DP_SEED_SECRET_ARN",
    )
    distributed_configured = all(distributed_settings.values())
    if gateway_enabled and distributed_enabled and distributed_configured:
        readiness = "configured_pending_scale_certification"
    elif gateway_enabled:
        readiness = "gateway_enabled_distributed_data_plane_incomplete"
    else:
        readiness = "disabled_pending_deployment"

    return {
        "module": "module_1_ingestion_privacy_layer",
        "status": readiness,
        "api_key_configured": api_key_configured,
        "database_configured": has_any_database,
        "database_env_presence": database_envs,
        "gateway_enabled": gateway_enabled,
        "distributed_processing_enabled": distributed_enabled,
        "distributed_processing_configured": distributed_configured,
        "distributed_setting_presence": distributed_settings,
        "production_scale_certified": False,
        "privacy_controls": {
            "hmac_sha256_tokenization": True,
            "contribution_bounding": True,
            "k_anonymity": True,
            "differential_privacy_noise": True,
            "privacy_budget_ledger": True,
            "privacy_budget_approval_enforcement": True,
            "lineage_logging": True,
            "ingestion_job_tracking": True,
            "synthetic_generation_audit": True,
            "legacy_route_auth_protection": True,
            "verified_tenant_request_boundary": True,
            "request_id_propagation": True,
            "production_legacy_route_hiding": True,
            "provider_contract_registry": True,
            "immutable_object_fingerprinting": True,
            "s3_sqs_worker": True,
            "bounded_retry_and_dlq": True,
            "duplicate_delivery_idempotency": True,
            "quarantine": True,
            "controlled_replay": True,
            "provider_delivery_freshness_monitoring": True,
            "canonical_s3_output": True,
            "entity_disjoint_privacy_partitions": True,
            "parallel_dp_composition_accounting": True,
            "privacy_window_sealing": True,
            "correction_supersession": True,
            "deletion_and_opt_out_propagation": True,
            "canonical_partition_revocation": True,
            "billion_event_acceptance_gate": True,
        },
        "protected_endpoints": [
            "POST /api/audience-intelligence/ingest",
            "POST /api/audience-intelligence/ingest/csv",
            "GET /api/audience-intelligence/ingest/{job_id}/status",
            "POST /api/audience-intelligence/synthetic/generate/{job_id}",
            "GET /api/audience-intelligence/synthetic/{synthetic_job_id}/lineage",
            "GET /api/audience-intelligence/module-1/status",
            "GET /api/audience-intelligence/source-health",
            "GET /api/audience-intelligence/provider-ingestion/status",
            "GET /api/audience-intelligence/provider-ingestion/runs/{ingestion_id}",
            "GET /api/audience-intelligence/provider-ingestion/privacy-windows/{window_key}",
            "POST /api/audience-intelligence/provider-ingestion/data-rights",
            "POST /api/audience-intelligence/provider-ingestion/data-rights/{request_id}/apply",
        ],
        "legacy_routes_hardened": [
            "POST /ingest",
            "GET /status/{job_id}",
            "POST /synthetic/generate/{job_id}",
            "POST /audience/generate",
            "POST /cohort/create",
            "GET /cohort/query/{cohort_id}",
            "POST /cohort/lookalike",
            "POST /chat",
            "/agents/*",
        ],
        "remaining_external_gates": [
            "Deploy the AWS data plane into staging with least-privilege roles.",
            "Apply approved migrations through the operator change process.",
            "Run measured billion-event-equivalent scale and recovery tests.",
            "Validate privacy policy, deletion SLAs, cost, and service quotas.",
            "Keep activation blocked until fresh provider data and rights checks pass.",
        ],
    }
