from __future__ import annotations

import base64
import hashlib
import io
import json
import uuid
from typing import Any, Dict, Optional

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
    ProviderDistributedControlPlaneService,
    ProviderDistributedJobPlan,
    ProviderDistributedStagingConfig,
    ProviderDistributedStagingService,
)
from app.services.provider_distributed_privacy_budget_service import (
    ProviderDistributedPrivacyBudgetService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)
from app.services.provider_privacy_window_service import (
    ProviderPrivacyWindowService,
)


class _AcceptanceObjectClient:
    """Bounded in-memory object client for offline control-plane acceptance."""

    def __init__(self) -> None:
        self._source_checksums: Dict[tuple[str, str], str] = {}
        self._objects: Dict[tuple[str, str], bytes] = {}
        self.deleted_keys: list[tuple[str, str]] = []

    def add_source(self, *, bucket: str, key: str, checksum: str) -> None:
        self._source_checksums[(bucket, key)] = checksum

    def put_json(self, reference: str, payload: Dict[str, Any]) -> bytes:
        bucket, key = self._parse(reference)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._objects[(bucket, key)] = encoded
        return encoded

    def copy_object(self, **kwargs: Any) -> Dict[str, Any]:
        source = kwargs["CopySource"]
        checksum = self._source_checksums[(source["Bucket"], source["Key"])]
        return {
            "CopyObjectResult": {
                "ChecksumSHA256": base64.b64encode(
                    bytes.fromhex(checksum)
                ).decode("ascii")
            }
        }

    def delete_object(self, **kwargs: Any) -> Dict[str, Any]:
        self.deleted_keys.append((kwargs["Bucket"], kwargs["Key"]))
        return {}

    def get_object(self, **kwargs: Any) -> Dict[str, Any]:
        return {"Body": io.BytesIO(self._objects[(kwargs["Bucket"], kwargs["Key"])])}

    def _parse(self, reference: str) -> tuple[str, str]:
        bucket, separator, key = reference.removeprefix("s3://").partition("/")
        if not separator or not bucket or not key:
            raise ValueError("Invalid acceptance object reference")
        return bucket, key


class Module1ProviderControlPlaneAcceptanceService:
    """
    Exercise the durable Module 1 control plane without provider identifiers.

    This is an offline acceptance check, not an AWS infrastructure test. It
    uses immutable mock object references while persisting all coordination,
    privacy-budget, idempotency, and publication state in the configured
    provider database.
    """

    def __init__(self, *, database_url: Optional[str] = None) -> None:
        self._database_url = database_url

    def run(self, *, tenant_id: str) -> Dict[str, Any]:
        normalized_tenant = str(tenant_id or "").strip().lower()
        if not normalized_tenant:
            raise ValueError("tenant_id is required")

        state = ProviderIngestionStateService(self._database_url)
        windows = ProviderPrivacyWindowService(self._database_url)
        budget = ProviderDistributedPrivacyBudgetService(self._database_url)
        objects = _AcceptanceObjectClient()
        control = ProviderDistributedControlPlaneService(
            staging_service=ProviderDistributedStagingService(
                config=ProviderDistributedStagingConfig(
                    bucket="punk-module1-acceptance-staging",
                    kms_key_id="alias/punk-module1-acceptance",
                ),
                client=objects,
            ),
            completion_service=ProviderDistributedCompletionService(
                state_service=state
            ),
            privacy_budget_service=budget,
            privacy_window_service=windows,
            result_client=objects,
        )

        run_token = uuid.uuid4().hex
        window_id = f"acceptance_{run_token}"
        contract = self._contract(normalized_tenant)
        requests: list[ProviderDistributedJobRequest] = []
        preparations: list[Dict[str, Any]] = []
        completions: list[Dict[str, Any]] = []

        for partition_index in range(2):
            descriptor = self._descriptor(
                tenant_id=normalized_tenant,
                run_token=run_token,
                partition_index=partition_index,
            )
            objects.add_source(
                bucket=descriptor.bucket,
                key=descriptor.key,
                checksum=str(descriptor.checksum_sha256),
            )
            claim = state.claim_object(
                descriptor,
                metadata={
                    "acceptance_only": True,
                    "raw_identifiers_present": False,
                },
            )
            request = ProviderDistributedJobRequest(
                ingestion_id=claim["record"]["ingestion_id"],
                fingerprint=descriptor.fingerprint,
                contract=contract,
                descriptor=descriptor,
                manifest=self._manifest(
                    checksum=str(descriptor.checksum_sha256),
                    window_id=window_id,
                    partition_index=partition_index,
                ),
                canonical_target=CanonicalObjectTarget(
                    bucket="punk-module1-acceptance-canonical",
                    prefix="candidate/",
                    server_side_encryption="aws:kms",
                    kms_key_id="alias/punk-module1-acceptance",
                ),
                actor="module1_control_plane_acceptance",
            )
            self._mark_dispatched(state, request.ingestion_id)
            prepared = control.prepare(request)
            requests.append(request)
            preparations.append(prepared)

            plan = ProviderDistributedJobPlan(**prepared["plan"])
            release_id = prepared["privacy_release"]["release_id"]
            result = self._store_completed_result(
                objects=objects,
                request=request,
                plan=plan,
                privacy_release_id=release_id,
                partition_index=partition_index,
            )
            completed = control.finalize(
                request=request,
                plan=plan,
                privacy_release_id=release_id,
            )
            if completed["result"] != result.to_safe_dict():
                raise RuntimeError("Acceptance result changed during finalization")
            completions.append(completed)

        duplicate_claim = state.claim_object(requests[0].descriptor)
        replay = control.finalize(
            request=requests[1],
            plan=ProviderDistributedJobPlan(**preparations[1]["plan"]),
            privacy_release_id=preparations[1]["privacy_release"]["release_id"],
        )
        first_window = completions[0]["privacy_window"]
        final_window = completions[1]["privacy_window"]
        charged = sum(
            float(item["privacy_release"]["charged_epsilon"])
            for item in preparations
        )

        if first_window["status"] != "open":
            raise RuntimeError("Incomplete privacy window was published")
        if final_window["status"] != "sealed":
            raise RuntimeError("Complete privacy window was not sealed")
        if final_window["canonical_publication_status"] != "active":
            raise RuntimeError("Sealed canonical candidates were not activated")
        if charged != float(contract.epsilon):
            raise RuntimeError("Entity-disjoint privacy budget was overcharged")
        if not duplicate_claim["duplicate"]:
            raise RuntimeError("Duplicate object notification was not idempotent")
        if not (
            replay["privacy_release"]["replayed"]
            and replay["privacy_window"]["replayed"]
        ):
            raise RuntimeError("Terminal result replay was not idempotent")

        return {
            "status": "module1_control_plane_acceptance_completed",
            "tenant_id": normalized_tenant,
            "partition_count": 2,
            "registered_partition_count": final_window[
                "registered_partition_count"
            ],
            "ready_partition_count": final_window["ready_partition_count"],
            "window_status_after_first_partition": first_window["status"],
            "window_status_after_all_partitions": final_window["status"],
            "canonical_publication_status": final_window[
                "canonical_publication_status"
            ],
            "privacy_epsilon_per_partition": float(contract.epsilon),
            "privacy_epsilon_charged_for_disjoint_window": charged,
            "duplicate_notification_idempotent": True,
            "terminal_result_replay_idempotent": True,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
            "activation_or_export_performed": False,
            "aws_services_called": False,
            "credentials_exposed": False,
        }

    def _contract(self, tenant_id: str) -> ProviderDatasetContract:
        return ProviderDatasetContract(
            tenant_id=tenant_id,
            provider_id="module1_acceptance_provider",
            dataset_id="entity_disjoint_mock_feed",
            schema_version="v1",
            data_format="parquet",
            allowed_bucket="provider-acceptance-source",
            allowed_prefix="entity-hash/",
            entity_id_column="tokenized_entity_id",
            timestamp_column="event_time",
            cohort_columns=("geo_id", "category_id", "daypart"),
            min_cohort_size=1000,
            epsilon=1.0,
            max_cumulative_epsilon=5.0,
            execution_mode="distributed",
            distributed_partition_strategy="entity_hash_v1",
            require_complete_privacy_partitions=True,
            max_object_bytes=10_000_000,
            max_rows_per_object=1_000_000,
        )

    def _descriptor(
        self,
        *,
        tenant_id: str,
        run_token: str,
        partition_index: int,
    ) -> ProviderObjectDescriptor:
        key = f"entity-hash/{run_token}/part-{partition_index:05d}.parquet"
        checksum = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return ProviderObjectDescriptor(
            tenant_id=tenant_id,
            provider_id="module1_acceptance_provider",
            dataset_id="entity_disjoint_mock_feed",
            bucket="provider-acceptance-source",
            key=key,
            size_bytes=4096,
            content_type="application/vnd.apache.parquet",
            version_id=f"acceptance-version-{partition_index}",
            checksum_sha256=checksum,
            server_side_encryption="aws:kms",
        )

    def _manifest(
        self,
        *,
        checksum: str,
        window_id: str,
        partition_index: int,
    ) -> ProviderObjectManifest:
        return ProviderObjectManifest(
            schema_version="v1",
            purpose="audience_intelligence",
            rights_policy_id="audience_intelligence_default",
            checksum_sha256=checksum,
            row_count=250_000,
            event_time_start="2026-07-30T00:00:00Z",
            event_time_end="2026-07-31T00:00:00Z",
            delivery_window_id=window_id,
            partition_index=partition_index,
            partition_count=2,
            partition_algorithm="spark_xxhash64_v1",
            partition_complete=True,
        )

    def _mark_dispatched(
        self,
        state: ProviderIngestionStateService,
        ingestion_id: str,
    ) -> None:
        state.transition(ingestion_id, status="validating")
        state.transition(ingestion_id, status="dispatching")
        state.transition(ingestion_id, status="dispatched")

    def _store_completed_result(
        self,
        *,
        objects: _AcceptanceObjectClient,
        request: ProviderDistributedJobRequest,
        plan: ProviderDistributedJobPlan,
        privacy_release_id: str,
        partition_index: int,
    ) -> ProviderDistributedJobResult:
        canonical_ref = plan.canonical_prefix
        manifest_ref = f"{canonical_ref}manifest.json"
        _, canonical_key = objects._parse(canonical_ref)  # noqa: SLF001
        manifest = {
            "contract_version": "provider-canonical-manifest-v1",
            "tenant_id": request.contract.tenant_id,
            "provider_id": request.contract.provider_id,
            "dataset_id": request.contract.dataset_id,
            "source_fingerprint": request.fingerprint,
            "delivery_window_id": request.manifest.delivery_window_id,
            "partition_index": partition_index,
            "partition_count": 2,
            "output_rows": 3,
            "objects": [
                {
                    "key": f"{canonical_key}part-00000.parquet",
                    "size": 1024,
                }
            ],
        }
        manifest_payload = objects.put_json(manifest_ref, manifest)
        result = ProviderDistributedJobResult(
            ingestion_id=request.ingestion_id,
            fingerprint=request.fingerprint,
            status="completed",
            input_rows=250_000,
            output_rows=3,
            privacy_job_id=privacy_release_id,
            output_manifest_ref=manifest_ref,
            canonical_ref=canonical_ref,
            canonical_checksum_sha256=hashlib.sha256(
                manifest_payload
            ).hexdigest(),
            privacy_controls=(
                "entity_hash_partitioning",
                "cross_partition_contribution_bounding",
                "k_anonymity",
                "differential_privacy",
                "raw_identifier_non_disclosure",
            ),
        )
        objects.put_json(plan.result_ref, result.to_safe_dict())
        return result
