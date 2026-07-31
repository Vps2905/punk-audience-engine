from __future__ import annotations

import json
import os
import signal
from typing import Optional

from app.models.provider_ingestion_contracts import CanonicalObjectTarget
from app.models.provider_queue_contracts import ProviderWorkerConfig
from app.services.privacy_ingestion_pipeline_service import (
    PrivacyIngestionPipelineService,
)
from app.services.provider_contract_registry_service import (
    ProviderContractRegistryService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_data_rights_service import (
    ProviderDataRightsService,
)
from app.services.provider_ingestion_worker_service import (
    ProviderIngestionWorkerService,
)
from app.services.provider_object_ingestion_gateway_service import (
    ProviderObjectIngestionGatewayService,
)
from app.services.provider_object_store_service import (
    S3ClientConfig,
    S3ProviderObjectStore,
)
from app.services.provider_queue_service import SQSProviderQueue
from app.services.provider_scale_execution_service import (
    StepFunctionsLauncherConfig,
    StepFunctionsProviderDistributedJobLauncher,
)


_shutdown_requested = False


def _required_env(name: str) -> str:
    value = str(os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for the provider worker.")
    return value


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _bounded_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = str(os.getenv(name) or default).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc
    return max(minimum, min(maximum, value))


def _handle_shutdown(signum, frame) -> None:
    del signum, frame
    global _shutdown_requested
    _shutdown_requested = True


def build_worker() -> ProviderIngestionWorkerService:
    if not _truthy(os.getenv("PROVIDER_GATEWAY_ENABLED")):
        raise RuntimeError("PROVIDER_GATEWAY_ENABLED must be true.")
    production = (
        _truthy(os.getenv("PRODUCTION_MODE"))
        or os.getenv("APP_ENV", "").strip().lower() == "production"
    )
    if production:
        for setting in (
            "AUDIENCE_DP_SEED_SECRET",
            "AUDIENCE_TOKENIZATION_HMAC_KEY",
        ):
            if not os.getenv(setting):
                raise RuntimeError(f"{setting} is required in production.")
    database_url = _required_env("PROVIDER_INGESTION_DATABASE_URL")
    region_name = _required_env("PROVIDER_S3_REGION")
    queue = SQSProviderQueue(
        queue_url=_required_env("PROVIDER_SQS_QUEUE_URL"),
        dlq_url=_required_env("PROVIDER_SQS_DLQ_URL"),
        region_name=region_name,
    )
    object_store = S3ProviderObjectStore(
        client_config=S3ClientConfig(
            region_name=region_name,
            role_arn=os.getenv("PROVIDER_S3_ROLE_ARN") or None,
            external_id=os.getenv("PROVIDER_S3_EXTERNAL_ID") or None,
        ),
        canonical_client_config=S3ClientConfig(
            region_name=region_name,
            role_arn=os.getenv("CANONICAL_S3_ROLE_ARN") or None,
            role_session_name="punk-audience-canonical-writer",
            external_id=os.getenv("CANONICAL_S3_EXTERNAL_ID") or None,
        ),
    )
    state = ProviderIngestionStateService(database_url=database_url)
    distributed_launcher = None
    if _truthy(os.getenv("PROVIDER_DISTRIBUTED_PROCESSING_ENABLED")):
        distributed_launcher = StepFunctionsProviderDistributedJobLauncher(
            config=StepFunctionsLauncherConfig(
                state_machine_arn=_required_env(
                    "PROVIDER_DISTRIBUTED_STATE_MACHINE_ARN"
                ),
                region_name=region_name,
                execution_name_prefix=os.getenv(
                    "PROVIDER_DISTRIBUTED_EXECUTION_PREFIX",
                    "punk-provider",
                ),
            )
        )
    gateway = ProviderObjectIngestionGatewayService(
        state_service=state,
        object_store=object_store,
        privacy_pipeline=PrivacyIngestionPipelineService(
            database_url=database_url
        ),
        distributed_launcher=distributed_launcher,
        data_rights_service=ProviderDataRightsService(
            database_url=database_url
        ),
    )
    visibility_timeout = _bounded_int(
        "PROVIDER_WORKER_VISIBILITY_TIMEOUT_SECONDS",
        300,
        minimum=1,
        maximum=43_200,
    )
    config = ProviderWorkerConfig(
        max_messages=_bounded_int(
            "PROVIDER_WORKER_MAX_MESSAGES",
            5,
            minimum=1,
            maximum=10,
        ),
        wait_time_seconds=_bounded_int(
            "PROVIDER_WORKER_WAIT_TIME_SECONDS",
            20,
            minimum=0,
            maximum=20,
        ),
        visibility_timeout_seconds=visibility_timeout,
        max_receive_count=_bounded_int(
            "PROVIDER_WORKER_MAX_RECEIVE_COUNT",
            5,
            minimum=1,
            maximum=100,
        ),
        retry_base_seconds=_bounded_int(
            "PROVIDER_WORKER_RETRY_BASE_SECONDS",
            30,
            minimum=1,
            maximum=43_200,
        ),
        retry_max_seconds=_bounded_int(
            "PROVIDER_WORKER_RETRY_MAX_SECONDS",
            900,
            minimum=1,
            maximum=43_200,
        ),
        stale_processing_seconds=_bounded_int(
            "PROVIDER_WORKER_STALE_PROCESSING_SECONDS",
            max(900, visibility_timeout),
            minimum=visibility_timeout,
            maximum=86_400,
        ),
    )
    return ProviderIngestionWorkerService(
        queue=queue,
        registry=ProviderContractRegistryService(
            database_url=database_url
        ),
        state_service=state,
        object_store=object_store,
        gateway=gateway,
        canonical_target=CanonicalObjectTarget(
            bucket=_required_env("CANONICAL_S3_BUCKET"),
            prefix=os.getenv("CANONICAL_S3_PREFIX", "canonical"),
            server_side_encryption=os.getenv(
                "CANONICAL_S3_SERVER_SIDE_ENCRYPTION",
                "AES256",
            ),
            kms_key_id=os.getenv("CANONICAL_S3_KMS_KEY_ID") or None,
        ),
        config=config,
    )


def main(max_cycles: Optional[int] = None) -> None:
    worker = build_worker()
    cycles = 0
    while not _shutdown_requested:
        result = worker.run_once()
        safe_summary = {
            key: value
            for key, value in result.items()
            if key != "results"
        }
        print(json.dumps(safe_summary, sort_keys=True), flush=True)
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)
    main()
