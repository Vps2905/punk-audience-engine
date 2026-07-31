from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

from app.models.provider_ingestion_contracts import (
    CanonicalObjectTarget,
    ProviderDatasetContract,
    ProviderObjectDescriptor,
    ProviderObjectManifest,
)


@dataclass(frozen=True)
class ProviderExecutionDecision:
    execution_mode: str
    reason_code: str

    def __post_init__(self) -> None:
        if self.execution_mode not in {"in_process", "distributed"}:
            raise ValueError(
                "execution_mode must be in_process or distributed"
            )
        if not str(self.reason_code or "").strip():
            raise ValueError("reason_code is required")


@dataclass(frozen=True)
class ProviderDistributedJobRequest:
    ingestion_id: str
    fingerprint: str
    contract: ProviderDatasetContract
    descriptor: ProviderObjectDescriptor
    manifest: ProviderObjectManifest
    canonical_target: CanonicalObjectTarget
    actor: str
    dispatch_attempt: int = 1

    def __post_init__(self) -> None:
        if not str(self.ingestion_id or "").strip():
            raise ValueError("ingestion_id is required")
        if len(str(self.fingerprint or "")) != 64:
            raise ValueError("fingerprint must be a SHA-256 hexadecimal digest")
        if not str(self.actor or "").strip():
            raise ValueError("actor is required")
        if int(self.dispatch_attempt) < 1:
            raise ValueError("dispatch_attempt must be >= 1")

    def to_safe_dict(self) -> Dict[str, Any]:
        """
        Serialize only object references and policy metadata.

        Provider credentials, KMS key material, raw rows, and identifiers are
        intentionally absent. The distributed task obtains short-lived access
        through its workload role.
        """

        target = asdict(self.canonical_target)
        return {
            "contract_version": "provider-scale-v1",
            "ingestion_id": self.ingestion_id,
            "fingerprint": self.fingerprint,
            "actor": str(self.actor).strip(),
            "dispatch_attempt": int(self.dispatch_attempt),
            "contract": self.contract.to_safe_dict(),
            "descriptor": self.descriptor.to_safe_dict(),
            "manifest": self.manifest.to_safe_dict(),
            "canonical_target": target,
        }

    @classmethod
    def from_safe_dict(
        cls,
        payload: Dict[str, Any],
    ) -> "ProviderDistributedJobRequest":
        if not isinstance(payload, dict):
            raise ValueError("Distributed request must be an object.")
        if payload.get("contract_version") != "provider-scale-v1":
            raise ValueError("Unsupported distributed request contract version.")
        try:
            request = cls(
                ingestion_id=str(payload["ingestion_id"]),
                fingerprint=str(payload["fingerprint"]),
                actor=str(payload["actor"]),
                dispatch_attempt=int(payload.get("dispatch_attempt") or 1),
                contract=ProviderDatasetContract(**payload["contract"]),
                descriptor=ProviderObjectDescriptor(**payload["descriptor"]),
                manifest=ProviderObjectManifest(**payload["manifest"]),
                canonical_target=CanonicalObjectTarget(
                    **payload["canonical_target"]
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Distributed request is missing required fields."
            ) from exc
        if request.descriptor.fingerprint != request.fingerprint:
            raise ValueError(
                "Distributed request fingerprint does not match its object."
            )
        return request


@dataclass(frozen=True)
class ProviderDistributedJobReceipt:
    job_id: str
    backend: str
    replayed: bool = False
    submitted_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.job_id or "").strip():
            raise ValueError("job_id is required")
        if not str(self.backend or "").strip():
            raise ValueError("backend is required")

    def to_safe_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderDistributedJobResult:
    """
    Terminal, privacy-safe result written by distributed processing.

    Raw rows, raw identifiers, hashed identifiers, and secret material are
    forbidden by construction: only counts, canonical references, policy
    metadata, and bounded reason codes cross back into the control plane.
    """

    ingestion_id: str
    fingerprint: str
    status: str
    input_rows: int
    output_rows: int
    privacy_job_id: str
    output_manifest_ref: Optional[str] = None
    canonical_ref: Optional[str] = None
    canonical_checksum_sha256: Optional[str] = None
    reason_code: Optional[str] = None
    privacy_controls: Sequence[str] = ()
    contract_version: str = "provider-scale-result-v1"

    TERMINAL_STATUSES = {
        "completed",
        "blocked",
        "quarantined",
        "failed",
    }

    def __post_init__(self) -> None:
        if self.contract_version != "provider-scale-result-v1":
            raise ValueError("Unsupported distributed result contract version.")
        if not str(self.ingestion_id or "").strip():
            raise ValueError("ingestion_id is required")
        fingerprint = str(self.fingerprint or "").strip().lower()
        if len(fingerprint) != 64 or any(
            value not in "0123456789abcdef" for value in fingerprint
        ):
            raise ValueError("fingerprint must be a SHA-256 hexadecimal digest")
        object.__setattr__(self, "fingerprint", fingerprint)
        if self.status not in self.TERMINAL_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(self.TERMINAL_STATUSES)}"
            )
        if int(self.input_rows) < 0 or int(self.output_rows) < 0:
            raise ValueError("input_rows and output_rows must be >= 0")
        if not str(self.privacy_job_id or "").strip():
            raise ValueError("privacy_job_id is required")
        controls = tuple(
            str(value or "").strip()
            for value in self.privacy_controls
            if str(value or "").strip()
        )
        object.__setattr__(
            self,
            "privacy_controls",
            tuple(dict.fromkeys(controls)),
        )
        if self.status == "completed":
            if not self.canonical_ref or not self.output_manifest_ref:
                raise ValueError(
                    "Completed results require canonical and manifest references."
                )
            checksum = str(self.canonical_checksum_sha256 or "").lower()
            if len(checksum) != 64 or any(
                value not in "0123456789abcdef" for value in checksum
            ):
                raise ValueError(
                    "Completed results require a canonical SHA-256 manifest."
                )
            object.__setattr__(
                self, "canonical_checksum_sha256", checksum
            )
            if self.reason_code:
                raise ValueError(
                    "Completed results must not include a failure reason."
                )
        else:
            if not str(self.reason_code or "").strip():
                raise ValueError(
                    "Non-completed results require a bounded reason code."
                )
            if self.output_rows:
                raise ValueError(
                    "Non-completed results must not publish output rows."
                )
            if self.canonical_ref:
                raise ValueError(
                    "Non-completed results must not publish canonical output."
                )
            if self.canonical_checksum_sha256:
                raise ValueError(
                    "Non-completed results must not publish a checksum."
                )

    def to_safe_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["privacy_controls"] = list(self.privacy_controls)
        return payload

    @classmethod
    def from_safe_dict(
        cls,
        payload: Dict[str, Any],
    ) -> "ProviderDistributedJobResult":
        if not isinstance(payload, dict):
            raise ValueError("Distributed result must be an object.")
        try:
            return cls(
                contract_version=str(payload["contract_version"]),
                ingestion_id=str(payload["ingestion_id"]),
                fingerprint=str(payload["fingerprint"]),
                status=str(payload["status"]),
                input_rows=int(payload["input_rows"]),
                output_rows=int(payload["output_rows"]),
                privacy_job_id=str(payload["privacy_job_id"]),
                output_manifest_ref=payload.get("output_manifest_ref"),
                canonical_ref=payload.get("canonical_ref"),
                canonical_checksum_sha256=payload.get(
                    "canonical_checksum_sha256"
                ),
                reason_code=payload.get("reason_code"),
                privacy_controls=tuple(
                    payload.get("privacy_controls") or ()
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and not isinstance(exc, KeyError):
                raise
            raise ValueError(
                "Distributed result is missing required fields."
            ) from exc


class ProviderDistributedDispatchError(RuntimeError):
    def __init__(
        self,
        reason_code: str,
        safe_message: str,
        *,
        retryable: bool,
    ) -> None:
        super().__init__(safe_message)
        self.reason_code = str(reason_code)
        self.safe_message = str(safe_message)
        self.retryable = bool(retryable)
