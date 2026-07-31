from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.models.provider_scale_contracts import (
    ProviderDistributedJobRequest,
    ProviderDistributedJobResult,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_distributed_privacy_budget_service import (
    ProviderDistributedPrivacyBudgetService,
)
from app.models.provider_privacy_window_contracts import (
    ProviderPrivacyPartitionRegistration,
)
from app.services.provider_privacy_window_service import (
    ProviderPrivacyWindowService,
)


_S3_PREFIX_RE = re.compile(r"^[A-Za-z0-9!_.*'()/-]*$")


def _prefix(value: str, field_name: str) -> str:
    normalized = str(value or "").strip().lstrip("/")
    if normalized and not normalized.endswith("/"):
        normalized += "/"
    if (
        ".." in normalized.split("/")
        or not _S3_PREFIX_RE.fullmatch(normalized)
    ):
        raise ValueError(f"{field_name} is not a safe S3 prefix")
    return normalized


@dataclass(frozen=True)
class ProviderDistributedStagingConfig:
    """
    Punk-owned immutable staging area for exact provider object versions.

    The raw staging prefix must have a short lifecycle policy. It is never a
    serving or audience-selection source.
    """

    bucket: str
    prefix: str = "provider-distributed-staging/"
    server_side_encryption: str = "aws:kms"
    kms_key_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.bucket or "").strip():
            raise ValueError("Distributed staging bucket is required")
        object.__setattr__(self, "bucket", str(self.bucket).strip())
        object.__setattr__(
            self,
            "prefix",
            _prefix(self.prefix, "Distributed staging prefix"),
        )
        if self.server_side_encryption not in {
            "AES256",
            "aws:kms",
            "aws:kms:dsse",
        }:
            raise ValueError(
                "Distributed staging requires approved server-side encryption"
            )
        if (
            self.server_side_encryption in {"aws:kms", "aws:kms:dsse"}
            and not str(self.kms_key_id or "").strip()
        ):
            raise ValueError(
                "Distributed staging KMS encryption requires a KMS key reference"
            )


@dataclass(frozen=True)
class ProviderDistributedJobPlan:
    ingestion_id: str
    fingerprint: str
    staged_source_ref: str
    canonical_prefix: str
    result_ref: str
    privacy_budget_scope: str

    def to_safe_dict(self) -> Dict[str, str]:
        return {
            "ingestion_id": self.ingestion_id,
            "fingerprint": self.fingerprint,
            "staged_source_ref": self.staged_source_ref,
            "canonical_prefix": self.canonical_prefix,
            "result_ref": self.result_ref,
            "privacy_budget_scope": self.privacy_budget_scope,
        }

    def glue_arguments(
        self,
        request: ProviderDistributedJobRequest,
    ) -> Dict[str, str]:
        """
        Glue arguments contain only safe references and policy metadata.

        Tokenization and DP secrets are configured on the Glue job as Secrets
        Manager ARNs. Secret values are never Step Functions or Glue arguments.
        """

        return {
            "--request_json": json.dumps(
                request.to_safe_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "--staged_source_ref": self.staged_source_ref,
            "--canonical_prefix": self.canonical_prefix,
            "--result_ref": self.result_ref,
            "--privacy_budget_scope": self.privacy_budget_scope,
        }


class ProviderDistributedStagingService:
    """
    Copies one exact S3 object version into a Punk-owned immutable key.

    Spark reads the staged copy, not a mutable provider key. The copy is
    accepted only when S3 reports the expected full-object SHA-256 checksum.
    """

    def __init__(
        self,
        *,
        config: ProviderDistributedStagingConfig,
        client: Any,
    ) -> None:
        self._config = config
        self._client = client

    def prepare(
        self,
        request: ProviderDistributedJobRequest,
    ) -> ProviderDistributedJobPlan:
        descriptor = request.descriptor
        expected_checksum = (
            request.manifest.checksum_sha256
            or descriptor.checksum_sha256
        )
        if not descriptor.version_id:
            raise ValueError(
                "Distributed processing requires an exact S3 object version"
            )
        if not expected_checksum:
            raise ValueError(
                "Distributed processing requires a full-object SHA-256 checksum"
            )

        extension = {
            "csv": "csv",
            "jsonl": "jsonl",
            "parquet": "parquet",
        }[request.contract.data_format]
        key = (
            f"{self._config.prefix}{request.contract.tenant_id}/"
            f"{request.contract.provider_id}/{request.contract.dataset_id}/"
            f"{request.fingerprint}/source.{extension}"
        )
        copy_source = {
            "Bucket": descriptor.bucket,
            "Key": descriptor.key,
            "VersionId": descriptor.version_id,
        }
        copy_request: Dict[str, Any] = {
            "Bucket": self._config.bucket,
            "Key": key,
            "CopySource": copy_source,
            "ChecksumAlgorithm": "SHA256",
            "MetadataDirective": "COPY",
            "ServerSideEncryption": self._config.server_side_encryption,
        }
        if self._config.server_side_encryption in {"aws:kms", "aws:kms:dsse"}:
            copy_request["SSEKMSKeyId"] = self._config.kms_key_id

        response = self._client.copy_object(**copy_request)
        result = response.get("CopyObjectResult") or {}
        actual_checksum = self._checksum_hex(result.get("ChecksumSHA256"))
        if actual_checksum != expected_checksum:
            self._delete_key(key)
            raise ValueError(
                "Staged object failed full-object SHA-256 checksum verification"
            )

        canonical_prefix = (
            f"s3://{request.canonical_target.bucket}/"
            f"{request.canonical_target.prefix}"
            f"{request.contract.tenant_id}/{request.contract.provider_id}/"
            f"{request.contract.dataset_id}/{request.contract.schema_version}/"
            f"{request.fingerprint}/"
        )
        result_key = (
            f"{request.canonical_target.prefix}_control/"
            f"{request.contract.tenant_id}/{request.fingerprint}/result.json"
        )
        return ProviderDistributedJobPlan(
            ingestion_id=request.ingestion_id,
            fingerprint=request.fingerprint,
            staged_source_ref=f"s3://{self._config.bucket}/{key}",
            canonical_prefix=canonical_prefix,
            result_ref=(
                f"s3://{request.canonical_target.bucket}/{result_key}"
            ),
            privacy_budget_scope=(
                f"provider:{request.contract.tenant_id}:"
                f"{request.contract.provider_id}:"
                f"{request.contract.dataset_id}:"
                f"{request.contract.schema_version}"
            ),
        )

    def cleanup(self, plan: ProviderDistributedJobPlan) -> None:
        marker = f"s3://{self._config.bucket}/"
        if not plan.staged_source_ref.startswith(marker):
            raise ValueError("Staged source reference is outside its bucket")
        self._delete_key(plan.staged_source_ref[len(marker) :])

    def _delete_key(self, key: str) -> None:
        self._client.delete_object(Bucket=self._config.bucket, Key=key)

    def _checksum_hex(self, checksum: Any) -> Optional[str]:
        if not checksum:
            return None
        try:
            decoded = base64.b64decode(str(checksum), validate=True)
        except ValueError as exc:
            raise ValueError(
                "S3 returned an invalid staged-object checksum"
            ) from exc
        if len(decoded) != 32:
            raise ValueError("S3 returned a non-SHA-256 staged checksum")
        return decoded.hex()


class ProviderDistributedCompletionService:
    """
    Idempotent control-plane lifecycle for distributed processing results.

    The result fingerprint must match the durable ingestion claim. Terminal
    results cannot be rewritten with a different result.
    """

    def __init__(
        self,
        *,
        state_service: ProviderIngestionStateService,
    ) -> None:
        self._state = state_service

    def mark_processing(
        self,
        request: ProviderDistributedJobRequest,
        *,
        privacy_job_id: str,
    ) -> Dict[str, Any]:
        record = self._validated_record(
            request.ingestion_id,
            request.fingerprint,
        )
        if record["status"] == "processing":
            return record
        if record["status"] != "dispatched":
            raise ValueError(
                "Distributed processing can start only from dispatched state"
            )
        return self._state.transition(
            request.ingestion_id,
            status="processing",
            privacy_job_id=privacy_job_id,
            increment_attempt=True,
            metadata_update={
                "distributed_processing_started": True,
                "distributed_contract_version": "provider-scale-v1",
            },
        )

    def finalize(
        self,
        result: ProviderDistributedJobResult,
    ) -> Dict[str, Any]:
        record = self._validated_record(
            result.ingestion_id,
            result.fingerprint,
        )
        current_status = str(record["status"])
        terminal = {"completed", "blocked", "quarantined", "failed"}
        if current_status in terminal:
            if (
                current_status == result.status
                and record.get("privacy_job_id") == result.privacy_job_id
                and record.get("canonical_ref") == result.canonical_ref
                and int(record.get("output_rows") or 0)
                == int(result.output_rows)
            ):
                return {**record, "distributed_result_replayed": True}
            raise ValueError(
                "A terminal distributed ingestion result cannot be replaced"
            )
        if current_status not in {"dispatched", "processing"}:
            raise ValueError(
                "Distributed result is not valid for the current ingestion state"
            )

        updated = self._state.transition(
            result.ingestion_id,
            status=result.status,
            reason_code=result.reason_code,
            privacy_job_id=result.privacy_job_id,
            canonical_ref=result.canonical_ref,
            input_rows=result.input_rows,
            output_rows=result.output_rows,
            metadata_update={
                "distributed_result_contract_version": (
                    result.contract_version
                ),
                "distributed_output_manifest_ref": (
                    result.output_manifest_ref
                ),
                "privacy_controls": list(result.privacy_controls),
                "raw_identifiers_returned": False,
            },
        )
        return {**updated, "distributed_result_replayed": False}

    def annotate_terminal_metadata(
        self,
        ingestion_id: str,
        *,
        metadata_update: Dict[str, Any],
    ) -> Dict[str, Any]:
        record = self._state.get(ingestion_id)
        if record["status"] not in {
            "completed",
            "blocked",
            "quarantined",
            "failed",
        }:
            raise ValueError(
                "Only terminal distributed ingestion can be annotated"
            )
        return self._state.transition(
            ingestion_id,
            status=record["status"],
            metadata_update=metadata_update,
        )

    def _validated_record(
        self,
        ingestion_id: str,
        fingerprint: str,
    ) -> Dict[str, Any]:
        record = self._state.get(ingestion_id)
        if str(record.get("fingerprint") or "") != fingerprint:
            raise ValueError(
                "Distributed result fingerprint does not match ingestion state"
            )
        return record


class ProviderDistributedControlPlaneService:
    """
    Coordinates budget reservation, exact-version staging, lifecycle, result
    verification, and raw staging cleanup around one Glue job.
    """

    def __init__(
        self,
        *,
        staging_service: ProviderDistributedStagingService,
        completion_service: ProviderDistributedCompletionService,
        privacy_budget_service: ProviderDistributedPrivacyBudgetService,
        result_client: Any,
        privacy_window_service: Optional[
            ProviderPrivacyWindowService
        ] = None,
    ) -> None:
        self._staging = staging_service
        self._completion = completion_service
        self._budget = privacy_budget_service
        self._result_client = result_client
        self._privacy_windows = privacy_window_service

    def prepare(
        self,
        request: ProviderDistributedJobRequest,
    ) -> Dict[str, Any]:
        reservation = self._budget.reserve(request)
        if reservation["decision"] == "blocked":
            result = ProviderDistributedJobResult(
                ingestion_id=request.ingestion_id,
                fingerprint=request.fingerprint,
                status="blocked",
                input_rows=0,
                output_rows=0,
                privacy_job_id=reservation["release_id"],
                reason_code="privacy_budget_exceeded",
                privacy_controls=("privacy_budget_accounting",),
            )
            ingestion = self._completion.finalize(result)
            return {
                "ready": False,
                "reason_code": "privacy_budget_exceeded",
                "privacy_release": reservation,
                "ingestion": ingestion,
                "privacy_window": None,
            }

        try:
            window_state = self._register_privacy_partition(request)
        except Exception:
            self._budget.mark_terminal(
                reservation["release_id"],
                status="failed",
                reason_code="privacy_window_registration_failed",
            )
            raise

        try:
            plan = self._staging.prepare(request)
        except Exception:
            self._budget.mark_terminal(
                reservation["release_id"],
                status="failed",
                reason_code="distributed_staging_verification_failed",
            )
            result = ProviderDistributedJobResult(
                ingestion_id=request.ingestion_id,
                fingerprint=request.fingerprint,
                status="quarantined",
                input_rows=0,
                output_rows=0,
                privacy_job_id=reservation["release_id"],
                reason_code="distributed_staging_verification_failed",
                privacy_controls=(
                    "exact_version_staging",
                    "privacy_budget_accounting",
                ),
            )
            self._completion.finalize(result)
            raise

        self._completion.mark_processing(
            request,
            privacy_job_id=reservation["release_id"],
        )
        return {
            "ready": True,
            "request": request.to_safe_dict(),
            "plan": plan.to_safe_dict(),
            "privacy_release": reservation,
            "glue_arguments": plan.glue_arguments(request),
            "privacy_window": window_state,
        }

    def finalize(
        self,
        *,
        request: ProviderDistributedJobRequest,
        plan: ProviderDistributedJobPlan,
        privacy_release_id: str,
    ) -> Dict[str, Any]:
        result = self._load_result(plan.result_ref)
        if (
            result.ingestion_id != request.ingestion_id
            or result.fingerprint != request.fingerprint
        ):
            raise ValueError(
                "Distributed result does not match its prepared request"
            )
        if result.status == "completed":
            try:
                self._verify_canonical_manifest(request, result)
            except Exception:
                privacy_release = self._budget.mark_terminal(
                    privacy_release_id,
                    status="failed",
                    reason_code="canonical_manifest_verification_failed",
                )
                failed = ProviderDistributedJobResult(
                    ingestion_id=request.ingestion_id,
                    fingerprint=request.fingerprint,
                    status="failed",
                    input_rows=result.input_rows,
                    output_rows=0,
                    privacy_job_id=result.privacy_job_id,
                    reason_code="canonical_manifest_verification_failed",
                    privacy_controls=(
                        *result.privacy_controls,
                        "canonical_manifest_verification",
                    ),
                )
                ingestion = self._completion.finalize(failed)
                self._staging.cleanup(plan)
                return {
                    "status": "failed",
                    "ingestion": ingestion,
                    "privacy_release": privacy_release,
                    "result": failed.to_safe_dict(),
                    "privacy_window": self._complete_privacy_partition(
                        request=request,
                        result=failed,
                        privacy_release_id=privacy_release_id,
                    ),
                    "raw_staging_deleted": True,
                }
        release_status = {
            "completed": "completed",
            "blocked": "blocked",
            "quarantined": "failed",
            "failed": "failed",
        }[result.status]
        try:
            privacy_release = self._budget.mark_terminal(
                privacy_release_id,
                status=release_status,
                canonical_ref=(
                    result.canonical_ref
                    if release_status == "completed"
                    else None
                ),
                reason_code=(
                    result.reason_code
                    if release_status != "completed"
                    else None
                ),
            )
            ingestion = self._completion.finalize(result)
            window_state = self._complete_privacy_partition(
                request=request,
                result=result,
                privacy_release_id=privacy_release_id,
            )
            if window_state is not None:
                publication_status = (
                    "eligible_after_window_seal"
                    if window_state["status"] == "sealed"
                    else "blocked_pending_window_seal"
                )
                ingestion = self._completion.annotate_terminal_metadata(
                    request.ingestion_id,
                    metadata_update={
                        "delivery_window_id": (
                            request.manifest.delivery_window_id
                        ),
                        "privacy_window_status": window_state["status"],
                        "canonical_publication_status": publication_status,
                    },
                )
            return {
                "status": result.status,
                "ingestion": ingestion,
                "privacy_release": privacy_release,
                "result": result.to_safe_dict(),
                "privacy_window": window_state,
                "raw_staging_deleted": True,
            }
        finally:
            self._staging.cleanup(plan)

    def _register_privacy_partition(
        self,
        request: ProviderDistributedJobRequest,
    ) -> Optional[Dict[str, Any]]:
        manifest = request.manifest
        if manifest.delivery_window_id is None:
            if request.contract.require_complete_privacy_partitions:
                raise ValueError(
                    "A complete privacy partition manifest is required"
                )
            return None
        if self._privacy_windows is None:
            raise RuntimeError(
                "Privacy-window coordination is not configured"
            )
        return self._privacy_windows.register_partition(
            ProviderPrivacyPartitionRegistration(
                tenant_id=request.contract.tenant_id,
                provider_id=request.contract.provider_id,
                dataset_id=request.contract.dataset_id,
                schema_version=request.contract.schema_version,
                delivery_window_id=manifest.delivery_window_id,
                event_time_start=str(manifest.event_time_start),
                event_time_end=str(manifest.event_time_end),
                partition_index=int(manifest.partition_index),
                partition_count=int(manifest.partition_count),
                partition_algorithm=str(manifest.partition_algorithm),
                ingestion_id=request.ingestion_id,
                fingerprint=request.fingerprint,
                row_count=manifest.row_count,
                delivery_type=manifest.delivery_type,
                supersedes_fingerprint=manifest.supersedes_fingerprint,
            )
        )

    def _complete_privacy_partition(
        self,
        *,
        request: ProviderDistributedJobRequest,
        result: ProviderDistributedJobResult,
        privacy_release_id: str,
    ) -> Optional[Dict[str, Any]]:
        if request.manifest.delivery_window_id is None:
            return None
        if self._privacy_windows is None:
            raise RuntimeError(
                "Privacy-window coordination is not configured"
            )
        if result.status != "completed":
            return self._privacy_windows.get_window(
                ":".join(
                    (
                        request.contract.tenant_id,
                        request.contract.provider_id,
                        request.contract.dataset_id,
                        request.contract.schema_version,
                        str(request.manifest.delivery_window_id),
                    )
                ),
                tenant_id=request.contract.tenant_id,
            )
        state = self._privacy_windows.mark_partition_ready(
            tenant_id=request.contract.tenant_id,
            fingerprint=request.fingerprint,
            canonical_ref=str(result.canonical_ref),
            canonical_checksum_sha256=str(
                result.canonical_checksum_sha256
            ),
            privacy_release_id=privacy_release_id,
            output_rows=result.output_rows,
        )
        if (
            state["ready_partition_count"]
            == state["partition_count"]
        ):
            state = self._privacy_windows.seal_window(
                state["window_key"],
                tenant_id=request.contract.tenant_id,
            )
        return state

    def fail(
        self,
        *,
        request: ProviderDistributedJobRequest,
        plan: ProviderDistributedJobPlan,
        privacy_release_id: str,
        reason_code: str = "distributed_execution_failed",
    ) -> Dict[str, Any]:
        try:
            privacy_release = self._budget.mark_terminal(
                privacy_release_id,
                status="failed",
                reason_code=reason_code,
            )
            result = ProviderDistributedJobResult(
                ingestion_id=request.ingestion_id,
                fingerprint=request.fingerprint,
                status="failed",
                input_rows=0,
                output_rows=0,
                privacy_job_id=privacy_release_id,
                reason_code=reason_code,
                privacy_controls=(
                    "exact_version_staging",
                    "privacy_budget_accounting",
                ),
            )
            ingestion = self._completion.finalize(result)
            return {
                "status": "failed",
                "ingestion": ingestion,
                "privacy_release": privacy_release,
                "raw_staging_deleted": True,
            }
        finally:
            self._staging.cleanup(plan)

    def _load_result(
        self,
        result_ref: str,
    ) -> ProviderDistributedJobResult:
        bucket, key = self._parse_s3_ref(result_ref)
        response = self._result_client.get_object(Bucket=bucket, Key=key)
        body = response.get("Body")
        if body is None:
            raise RuntimeError(
                "Distributed result object did not contain a body"
            )
        payload = body.read(65_537)
        if not isinstance(payload, bytes) or len(payload) > 65_536:
            raise ValueError(
                "Distributed result exceeded its bounded control-plane size"
            )
        return ProviderDistributedJobResult.from_safe_dict(
            json.loads(payload.decode("utf-8"))
        )

    def _verify_canonical_manifest(
        self,
        request: ProviderDistributedJobRequest,
        result: ProviderDistributedJobResult,
    ) -> None:
        bucket, key = self._parse_s3_ref(str(result.output_manifest_ref))
        response = self._result_client.get_object(Bucket=bucket, Key=key)
        body = response.get("Body")
        if body is None:
            raise ValueError("Canonical manifest body is missing")
        payload = body.read(8_388_609)
        if not isinstance(payload, bytes) or len(payload) > 8_388_608:
            raise ValueError("Canonical manifest exceeded 8 MiB")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != result.canonical_checksum_sha256:
            raise ValueError("Canonical manifest checksum mismatch")
        manifest = json.loads(payload.decode("utf-8"))
        if manifest.get("contract_version") != "provider-canonical-manifest-v1":
            raise ValueError("Canonical manifest version is invalid")
        expected = {
            "tenant_id": request.contract.tenant_id,
            "provider_id": request.contract.provider_id,
            "dataset_id": request.contract.dataset_id,
            "source_fingerprint": request.fingerprint,
            "delivery_window_id": request.manifest.delivery_window_id,
            "partition_index": request.manifest.partition_index,
            "partition_count": request.manifest.partition_count,
            "output_rows": result.output_rows,
        }
        for field_name, expected_value in expected.items():
            if manifest.get(field_name) != expected_value:
                raise ValueError(
                    f"Canonical manifest {field_name} does not match request"
                )
        objects = manifest.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError("Canonical manifest has no objects")
        canonical_bucket, canonical_prefix = self._parse_s3_ref(
            str(result.canonical_ref)
        )
        if bucket != canonical_bucket:
            raise ValueError("Canonical manifest bucket mismatch")
        for item in objects:
            if not isinstance(item, dict):
                raise ValueError("Canonical manifest object is invalid")
            object_key = str(item.get("key") or "")
            if not object_key.startswith(canonical_prefix):
                raise ValueError("Canonical object escaped attempt prefix")
            if int(item.get("size") or 0) < 0:
                raise ValueError("Canonical object size is invalid")

    def _parse_s3_ref(self, value: str) -> tuple[str, str]:
        normalized = str(value or "").strip()
        if not normalized.startswith("s3://"):
            raise ValueError("Expected an s3:// result reference")
        bucket, separator, key = normalized[5:].partition("/")
        if not separator or not bucket or not key or ".." in key.split("/"):
            raise ValueError("Invalid S3 result reference")
        return bucket, key
