from datetime import datetime, timezone

import pytest

from app.models.provider_ingestion_contracts import (
    CanonicalObjectTarget,
    ProviderDatasetContract,
    ProviderObjectDescriptor,
    ProviderObjectManifest,
    ProviderObjectValidationError,
)
from app.models.provider_scale_contracts import ProviderDistributedJobRequest
from app.services.provider_scale_execution_service import (
    ProviderScaleExecutionPlanner,
    StepFunctionsLauncherConfig,
    StepFunctionsProviderDistributedJobLauncher,
)


def _contract(**overrides):
    values = {
        "tenant_id": "tenant_a",
        "provider_id": "provider_a",
        "dataset_id": "signals_a",
        "schema_version": "v1",
        "data_format": "csv",
        "allowed_bucket": "provider-landing",
        "allowed_prefix": "delivery/",
        "max_object_bytes": 10_000,
        "max_rows_per_object": 10_000,
        "max_in_process_object_bytes": 1_000,
        "max_in_process_rows": 100,
    }
    values.update(overrides)
    return ProviderDatasetContract(**values)


def _descriptor(*, size_bytes=500):
    return ProviderObjectDescriptor(
        tenant_id="tenant_a",
        provider_id="provider_a",
        dataset_id="signals_a",
        bucket="provider-landing",
        key="delivery/object.csv",
        size_bytes=size_bytes,
        content_type="text/csv",
        version_id="version-1",
        checksum_sha256="a" * 64,
        server_side_encryption="AES256",
    )


def _manifest(*, row_count=50):
    return ProviderObjectManifest(
        schema_version="v1",
        purpose="audience_intelligence",
        rights_policy_id="audience_intelligence_default",
        checksum_sha256="a" * 64,
        row_count=row_count,
    )


def _request():
    descriptor = _descriptor(size_bytes=2_000)
    return ProviderDistributedJobRequest(
        ingestion_id="provider_ingest_123",
        fingerprint=descriptor.fingerprint,
        contract=_contract(),
        descriptor=descriptor,
        manifest=_manifest(row_count=500),
        canonical_target=CanonicalObjectTarget(
            bucket="canonical-safe",
            prefix="canonical/",
        ),
        actor="provider_ingestion_worker",
    )


def test_auto_execution_routes_only_large_objects_to_distributed_compute():
    planner = ProviderScaleExecutionPlanner()

    small = planner.choose(
        contract=_contract(),
        descriptor=_descriptor(size_bytes=500),
        manifest=_manifest(row_count=50),
    )
    large_bytes = planner.choose(
        contract=_contract(),
        descriptor=_descriptor(size_bytes=2_000),
        manifest=_manifest(row_count=50),
    )
    large_rows = planner.choose(
        contract=_contract(),
        descriptor=_descriptor(size_bytes=500),
        manifest=_manifest(row_count=500),
    )

    assert small.execution_mode == "in_process"
    assert large_bytes.execution_mode == "distributed"
    assert large_rows.execution_mode == "distributed"


def test_explicit_in_process_contract_fails_closed_above_limits():
    with pytest.raises(ProviderObjectValidationError) as exc_info:
        ProviderScaleExecutionPlanner().choose(
            contract=_contract(execution_mode="in_process"),
            descriptor=_descriptor(size_bytes=2_000),
            manifest=_manifest(row_count=50),
        )

    assert exc_info.value.reason_code == "in_process_scale_limit_exceeded"


class FakeStepFunctionsClient:
    def __init__(self):
        self.calls = []

    def start_execution(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "executionArn": (
                "arn:aws:states:ca-central-1:123456789012:"
                "execution:provider-scale:test"
            ),
            "startDate": datetime(2026, 7, 30, tzinfo=timezone.utc),
        }


def test_step_functions_dispatch_contains_only_safe_object_references():
    client = FakeStepFunctionsClient()
    launcher = StepFunctionsProviderDistributedJobLauncher(
        config=StepFunctionsLauncherConfig(
            state_machine_arn=(
                "arn:aws:states:ca-central-1:123456789012:"
                "stateMachine:provider-scale"
            )
        ),
        client=client,
    )

    receipt = launcher.submit_or_get(_request())

    assert receipt.replayed is False
    assert receipt.backend == "aws_step_functions_standard"
    assert len(client.calls) == 1
    submitted = client.calls[0]
    assert len(submitted["name"]) <= 80
    assert "provider-landing" in submitted["input"]
    assert "delivery/object.csv" in submitted["input"]
    lowered = submitted["input"].lower()
    assert "aws_secret_access_key" not in lowered
    assert "password" not in lowered
    assert "maid" not in lowered
    assert "device_id" not in lowered


class ExecutionAlreadyExistsError(RuntimeError):
    response = {"Error": {"Code": "ExecutionAlreadyExists"}}


class DuplicateStepFunctionsClient:
    def start_execution(self, **kwargs):
        del kwargs
        raise ExecutionAlreadyExistsError("already exists")


def test_step_functions_duplicate_resolves_to_same_execution():
    launcher = StepFunctionsProviderDistributedJobLauncher(
        config=StepFunctionsLauncherConfig(
            state_machine_arn=(
                "arn:aws:states:ca-central-1:123456789012:"
                "stateMachine:provider-scale"
            )
        ),
        client=DuplicateStepFunctionsClient(),
    )

    receipt = launcher.submit_or_get(_request())

    assert receipt.replayed is True
    assert ":execution:provider-scale:" in receipt.job_id


def test_retry_attempt_gets_a_new_execution_name():
    client = FakeStepFunctionsClient()
    launcher = StepFunctionsProviderDistributedJobLauncher(
        config=StepFunctionsLauncherConfig(
            state_machine_arn=(
                "arn:aws:states:ca-central-1:123456789012:"
                "stateMachine:provider-scale"
            )
        ),
        client=client,
    )
    first = _request()
    retry = ProviderDistributedJobRequest(
        **{
            **first.__dict__,
            "dispatch_attempt": 2,
        }
    )

    launcher.submit_or_get(first)
    launcher.submit_or_get(retry)

    assert client.calls[0]["name"].endswith("-a1")
    assert client.calls[1]["name"].endswith("-a2")
    assert client.calls[0]["name"] != client.calls[1]["name"]
