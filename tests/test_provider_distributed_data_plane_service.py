import base64
import hashlib
import io
import json

import pytest

from app.models.provider_ingestion_contracts import (
    CanonicalObjectTarget,
    ProviderDatasetContract,
    ProviderObjectDescriptor,
    ProviderObjectManifest,
)
from app.models.provider_scale_contracts import (
    ProviderDistributedJobRequest,
    ProviderDistributedJobResult,
)
from app.services.provider_distributed_data_plane_service import (
    ProviderDistributedCompletionService,
    ProviderDistributedJobPlan,
    ProviderDistributedStagingConfig,
    ProviderDistributedStagingService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_scale_execution_service import (
    ProviderScaleExecutionPlanner,
)


def _request(
    *,
    ingestion_id="provider_ingest_1",
    key="delivery/object.parquet",
    epsilon=1.0,
    max_budget=5.0,
):
    checksum = hashlib.sha256(key.encode("utf-8")).hexdigest()
    contract = ProviderDatasetContract(
        tenant_id="tenant_a",
        provider_id="provider_a",
        dataset_id="dataset_a",
        schema_version="v1",
        data_format="parquet",
        allowed_bucket="provider-bucket",
        allowed_prefix="delivery/",
        entity_id_column="signal_id",
        timestamp_column="event_time",
        cohort_columns=("region", "category", "daypart"),
        epsilon=epsilon,
        max_cumulative_epsilon=max_budget,
        max_object_bytes=10_000,
        max_rows_per_object=10_000,
    )
    descriptor = ProviderObjectDescriptor(
        tenant_id="tenant_a",
        provider_id="provider_a",
        dataset_id="dataset_a",
        bucket="provider-bucket",
        key=key,
        size_bytes=500,
        content_type="application/vnd.apache.parquet",
        version_id="version-1",
        checksum_sha256=checksum,
        server_side_encryption="aws:kms",
    )
    return ProviderDistributedJobRequest(
        ingestion_id=ingestion_id,
        fingerprint=descriptor.fingerprint,
        contract=contract,
        descriptor=descriptor,
        manifest=ProviderObjectManifest(
            schema_version="v1",
            purpose="audience_intelligence",
            rights_policy_id="audience_intelligence_default",
            checksum_sha256=checksum,
            row_count=500,
        ),
        canonical_target=CanonicalObjectTarget(
            bucket="canonical-bucket",
            prefix="canonical/",
            server_side_encryption="aws:kms",
            kms_key_id="alias/canonical-key",
        ),
        actor="provider_worker",
    )


def test_parquet_is_a_distributed_only_provider_format():
    request = _request()

    decision = ProviderScaleExecutionPlanner().choose(
        contract=request.contract,
        descriptor=request.descriptor,
        manifest=request.manifest,
    )

    assert decision.execution_mode == "distributed"
    assert decision.reason_code == "parquet_requires_distributed_processing"

    in_process = ProviderDatasetContract(
        **{
            **request.contract.to_safe_dict(),
            "execution_mode": "in_process",
        }
    )
    with pytest.raises(ValueError):
        ProviderScaleExecutionPlanner().choose(
            contract=in_process,
            descriptor=request.descriptor,
            manifest=request.manifest,
        )


def test_distributed_request_round_trip_revalidates_fingerprint():
    request = _request()
    restored = ProviderDistributedJobRequest.from_safe_dict(
        request.to_safe_dict()
    )

    assert restored == request

    tampered = request.to_safe_dict()
    tampered["descriptor"]["key"] = "delivery/other.parquet"
    with pytest.raises(ValueError, match="fingerprint"):
        ProviderDistributedJobRequest.from_safe_dict(tampered)


def test_distributed_result_contract_blocks_unsafe_terminal_shapes():
    request = _request()
    result = ProviderDistributedJobResult(
        ingestion_id=request.ingestion_id,
        fingerprint=request.fingerprint,
        status="completed",
        input_rows=500,
        output_rows=3,
        privacy_job_id="glue-run-1",
        output_manifest_ref="s3://canonical-bucket/result.json",
        canonical_ref="s3://canonical-bucket/output/",
        canonical_checksum_sha256="a" * 64,
        privacy_controls=("k_anonymity", "no_identifier_output"),
    )

    assert (
        ProviderDistributedJobResult.from_safe_dict(
            result.to_safe_dict()
        )
        == result
    )
    with pytest.raises(ValueError, match="canonical SHA-256"):
        ProviderDistributedJobResult(
            ingestion_id=request.ingestion_id,
            fingerprint=request.fingerprint,
            status="completed",
            input_rows=500,
            output_rows=3,
            privacy_job_id="glue-run-1",
            output_manifest_ref="s3://canonical-bucket/manifest.json",
            canonical_ref="s3://canonical-bucket/output/",
        )
    with pytest.raises(ValueError, match="must not publish output"):
        ProviderDistributedJobResult(
            ingestion_id=request.ingestion_id,
            fingerprint=request.fingerprint,
            status="failed",
            input_rows=500,
            output_rows=1,
            privacy_job_id="glue-run-1",
            reason_code="distributed_privacy_job_failed",
        )


class FakeS3:
    def __init__(self, checksum_hex):
        self.checksum = base64.b64encode(
            bytes.fromhex(checksum_hex)
        ).decode("ascii")
        self.copies = []
        self.deletes = []

    def copy_object(self, **kwargs):
        self.copies.append(kwargs)
        return {
            "CopyObjectResult": {
                "ChecksumSHA256": self.checksum,
            }
        }

    def delete_object(self, **kwargs):
        self.deletes.append(kwargs)
        return {}


def test_exact_version_is_staged_and_checksum_verified():
    request = _request()
    client = FakeS3(request.descriptor.checksum_sha256)
    service = ProviderDistributedStagingService(
        config=ProviderDistributedStagingConfig(
            bucket="staging-bucket",
            kms_key_id="alias/staging-key",
        ),
        client=client,
    )

    plan = service.prepare(request)

    assert client.copies[0]["CopySource"]["VersionId"] == "version-1"
    assert client.copies[0]["ChecksumAlgorithm"] == "SHA256"
    assert request.fingerprint in plan.staged_source_ref
    assert request.fingerprint in plan.canonical_prefix
    assert "password" not in json.dumps(plan.to_safe_dict()).lower()
    service.cleanup(plan)
    assert client.deletes


def test_bad_staging_checksum_is_deleted_and_fails_closed():
    request = _request()
    client = FakeS3("0" * 64)
    service = ProviderDistributedStagingService(
        config=ProviderDistributedStagingConfig(
            bucket="staging-bucket",
            kms_key_id="alias/staging-key",
        ),
        client=client,
    )

    with pytest.raises(ValueError, match="checksum"):
        service.prepare(request)

    assert len(client.deletes) == 1


def test_completion_is_idempotent_and_rejects_replacement(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'state.db'}"
    state = ProviderIngestionStateService(database_url=database_url)
    initial_request = _request()
    claim = state.claim_object(initial_request.descriptor)
    request = _request(
        ingestion_id=claim["record"]["ingestion_id"],
    )
    state.transition(request.ingestion_id, status="validating")
    state.transition(request.ingestion_id, status="dispatching")
    state.transition(request.ingestion_id, status="dispatched")
    completion = ProviderDistributedCompletionService(
        state_service=state
    )
    completion.mark_processing(
        request,
        privacy_job_id="privacy-release-1",
    )
    result = ProviderDistributedJobResult(
        ingestion_id=request.ingestion_id,
        fingerprint=request.fingerprint,
        status="completed",
        input_rows=500,
        output_rows=3,
        privacy_job_id="glue-run-1",
        output_manifest_ref="s3://canonical-bucket/result.json",
        canonical_ref="s3://canonical-bucket/output/",
        canonical_checksum_sha256="a" * 64,
        privacy_controls=("privacy_budget_accounting",),
    )

    first = completion.finalize(result)
    second = completion.finalize(result)

    assert first["status"] == "completed"
    assert first["distributed_result_replayed"] is False
    assert second["distributed_result_replayed"] is True
    replacement = ProviderDistributedJobResult(
        **{
            **result.to_safe_dict(),
            "canonical_ref": "s3://canonical-bucket/replacement/",
        }
    )
    with pytest.raises(ValueError, match="cannot be replaced"):
        completion.finalize(replacement)


def test_control_plane_result_read_is_bounded():
    plan = ProviderDistributedJobPlan(
        ingestion_id="provider_ingest_1",
        fingerprint="a" * 64,
        staged_source_ref="s3://staging/source.parquet",
        canonical_prefix="s3://canonical/output/",
        result_ref="s3://canonical/result.json",
        privacy_budget_scope="provider:tenant:provider:dataset:v1",
    )
    assert plan.glue_arguments(_request())["--result_ref"] == plan.result_ref
    assert io.BytesIO(b"result").read(8) == b"result"
