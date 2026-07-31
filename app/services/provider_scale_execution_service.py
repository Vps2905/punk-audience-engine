from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol

from app.models.provider_ingestion_contracts import (
    ProviderDatasetContract,
    ProviderObjectDescriptor,
    ProviderObjectManifest,
    ProviderObjectValidationError,
)
from app.models.provider_scale_contracts import (
    ProviderDistributedDispatchError,
    ProviderDistributedJobReceipt,
    ProviderDistributedJobRequest,
    ProviderExecutionDecision,
)


class ProviderDistributedJobLauncher(Protocol):
    def submit_or_get(
        self,
        request: ProviderDistributedJobRequest,
    ) -> ProviderDistributedJobReceipt:
        """Idempotently submit one immutable provider object for processing."""


class ProviderScaleExecutionPlanner:
    """
    Keep pandas work bounded and route large objects to distributed compute.

    This planner is a deterministic safety boundary. It does not make an
    audience-quality or semantic decision.
    """

    def choose(
        self,
        *,
        contract: ProviderDatasetContract,
        descriptor: ProviderObjectDescriptor,
        manifest: ProviderObjectManifest,
    ) -> ProviderExecutionDecision:
        if contract.data_format == "parquet":
            if contract.execution_mode == "in_process":
                raise ProviderObjectValidationError(
                    "in_process_format_not_supported",
                    "Parquet provider objects require the registered "
                    "distributed processing path.",
                )
            return ProviderExecutionDecision(
                execution_mode="distributed",
                reason_code="parquet_requires_distributed_processing",
            )

        exceeds_bytes = (
            descriptor.size_bytes > contract.max_in_process_object_bytes
        )
        exceeds_rows = (
            manifest.row_count is not None
            and manifest.row_count > contract.max_in_process_rows
        )

        if contract.execution_mode == "distributed":
            return ProviderExecutionDecision(
                execution_mode="distributed",
                reason_code="contract_requires_distributed_processing",
            )

        if contract.execution_mode == "in_process":
            if exceeds_bytes or exceeds_rows:
                raise ProviderObjectValidationError(
                    "in_process_scale_limit_exceeded",
                    "The provider object exceeds the registered in-process "
                    "safety limit.",
                )
            return ProviderExecutionDecision(
                execution_mode="in_process",
                reason_code="within_registered_in_process_limits",
            )

        if exceeds_bytes:
            return ProviderExecutionDecision(
                execution_mode="distributed",
                reason_code="object_bytes_require_distributed_processing",
            )
        if exceeds_rows:
            return ProviderExecutionDecision(
                execution_mode="distributed",
                reason_code="object_rows_require_distributed_processing",
            )
        return ProviderExecutionDecision(
            execution_mode="in_process",
            reason_code="within_registered_in_process_limits",
        )


@dataclass(frozen=True)
class StepFunctionsLauncherConfig:
    state_machine_arn: str
    region_name: Optional[str] = None
    execution_name_prefix: str = "punk-provider"

    def __post_init__(self) -> None:
        if not str(self.state_machine_arn or "").strip():
            raise ValueError("state_machine_arn is required")
        prefix = str(self.execution_name_prefix or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,30}", prefix):
            raise ValueError(
                "execution_name_prefix must contain 1-30 letters, numbers, "
                "underscores, or hyphens"
            )
        object.__setattr__(self, "execution_name_prefix", prefix)


class StepFunctionsProviderDistributedJobLauncher:
    """
    Idempotent boundary from the SQS worker to a distributed processing graph.

    AWS Step Functions Standard executions are keyed by a deterministic name.
    Re-delivery of the same immutable object therefore resolves to the same
    execution instead of starting duplicate privacy transformations.
    """

    RETRYABLE_ERROR_CODES = {
        "InternalError",
        "KmsThrottlingException",
        "ServiceUnavailable",
        "ThrottlingException",
        "TooManyRequestsException",
    }

    def __init__(
        self,
        *,
        config: StepFunctionsLauncherConfig,
        client: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._client = client

    def submit_or_get(
        self,
        request: ProviderDistributedJobRequest,
    ) -> ProviderDistributedJobReceipt:
        client = self._client or self._create_client()
        execution_name = self._execution_name(
            request.fingerprint,
            request.dispatch_attempt,
        )
        payload = json.dumps(
            request.to_safe_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            response = client.start_execution(
                stateMachineArn=self._config.state_machine_arn,
                name=execution_name,
                input=payload,
            )
        except Exception as exc:
            code = self._error_code(exc)
            if code == "ExecutionAlreadyExists":
                return ProviderDistributedJobReceipt(
                    job_id=self._execution_arn(execution_name),
                    backend="aws_step_functions_standard",
                    replayed=True,
                )
            raise ProviderDistributedDispatchError(
                "distributed_dispatch_failed",
                "The distributed provider-processing workflow could not be "
                "started.",
                retryable=code in self.RETRYABLE_ERROR_CODES,
            ) from exc

        execution_arn = str(response.get("executionArn") or "").strip()
        if not execution_arn:
            raise ProviderDistributedDispatchError(
                "distributed_dispatch_receipt_invalid",
                "The distributed provider-processing workflow returned an "
                "invalid receipt.",
                retryable=True,
            )
        started_at = response.get("startDate")
        if isinstance(started_at, datetime):
            submitted_at = started_at.astimezone(timezone.utc).isoformat()
        else:
            submitted_at = str(started_at or "").strip() or None
        return ProviderDistributedJobReceipt(
            job_id=execution_arn,
            backend="aws_step_functions_standard",
            replayed=False,
            submitted_at=submitted_at,
        )

    def _execution_name(
        self,
        fingerprint: str,
        dispatch_attempt: int,
    ) -> str:
        suffix = f"-a{int(dispatch_attempt)}"
        remaining = (
            80
            - len(self._config.execution_name_prefix)
            - 1
            - len(suffix)
        )
        return (
            f"{self._config.execution_name_prefix}-"
            f"{str(fingerprint)[:remaining]}{suffix}"
        )

    def _execution_arn(self, execution_name: str) -> str:
        marker = ":stateMachine:"
        arn = self._config.state_machine_arn
        if marker not in arn:
            return f"{arn}:{execution_name}"
        prefix, state_machine_name = arn.split(marker, 1)
        return (
            f"{prefix}:execution:{state_machine_name}:{execution_name}"
        )

    def _error_code(self, exc: Exception) -> str:
        response = getattr(exc, "response", None)
        if not isinstance(response, dict):
            return type(exc).__name__
        return str((response.get("Error") or {}).get("Code") or "")

    def _create_client(self):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required when a Step Functions client is not injected."
            ) from exc
        return boto3.client(
            "stepfunctions",
            region_name=self._config.region_name,
        )
