from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.models.audience_feature_contracts import normalize_taxonomy_value
from app.utils.serialization import make_serializable

FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_RESPONSE_DOCUMENT_BYTES = 1_000_000
BLOCKED_RESPONSE_KEYS = {
    "maid",
    "maids",
    "raw_maid",
    "raw_maids",
    "device_id",
    "device_ids",
    "email",
    "emails",
    "phone",
    "phones",
    "lat",
    "lng",
    "latitude",
    "longitude",
    "raw_observations",
    "embedding",
    "embeddings",
    "lineage",
    "source_ref",
    "database_url",
    "api_key",
    "password",
    "secret",
}
BLOCKED_RESPONSE_KEY_TOKENS = {
    "raw_maid",
    "device_id",
    "raw_observation",
    "database_url",
    "api_key",
    "password",
    "private_key",
}
BLOCKED_RESPONSE_VALUE_PATTERNS = (
    re.compile(r"\bpostgres(?:ql)?://", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_-]?key|password|client[_-]?secret)"
        r"\s*[:=]\s*[^\s,;]{8,}",
        re.IGNORECASE,
    ),
)


class ProposalIdempotencyConflictError(RuntimeError):
    """The same tenant/campaign/key was reused for a different request."""


class UnsafeProposalDocumentError(RuntimeError):
    """A generated proposal response crossed a storage safety boundary."""


class PunkAIAudienceProposalStore:
    """
    Immutable tenant-scoped store for privacy-safe Punk AI proposals.

    The store persists only a request fingerprint and the already-governed
    response document. It never stores the original prompt, embeddings,
    lineage, credentials, raw identifiers, or individual-level records.
    """

    def __init__(
        self,
        *,
        database_url: str | None = None,
        engine_factory: Callable[..., Engine] = create_engine,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._database_url = database_url
        self._engine_factory = engine_factory
        self._environment = environment if environment is not None else os.environ

    def get(
        self,
        *,
        tenant_id: str,
        campaign_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        clean_tenant_id = self._required_slug(tenant_id, "tenant_id")
        clean_campaign_id = self._required_text(campaign_id, "campaign_id")
        clean_idempotency_key = self._required_text(
            idempotency_key,
            "idempotency_key",
        )
        engine = self._engine()
        try:
            with engine.connect() as connection:
                self._set_tenant_context(connection, clean_tenant_id)
                row = connection.execute(
                    text(
                        """
                        SELECT
                            proposal_id,
                            request_fingerprint,
                            response_document
                        FROM public.punk_ai_audience_proposals
                        WHERE
                            tenant_id = :tenant_id
                            AND idempotency_key = :idempotency_key
                        """
                    ),
                    {
                        "tenant_id": clean_tenant_id,
                        "campaign_id": clean_campaign_id,
                        "idempotency_key": clean_idempotency_key,
                    },
                ).mappings().first()
        except SQLAlchemyError:
            raise RuntimeError(
                "The Punk AI proposal ledger is unavailable."
            ) from None
        finally:
            engine.dispose()
        return self._stored_record(row) if row else None

    def record(
        self,
        *,
        tenant_id: str,
        campaign_id: str,
        idempotency_key: str,
        request_fingerprint: str,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        clean_tenant_id = self._required_slug(tenant_id, "tenant_id")
        clean_campaign_id = self._required_text(campaign_id, "campaign_id")
        clean_idempotency_key = self._required_text(
            idempotency_key,
            "idempotency_key",
        )
        clean_fingerprint = str(request_fingerprint or "").strip().lower()
        if not FINGERPRINT_PATTERN.fullmatch(clean_fingerprint):
            raise ValueError("request_fingerprint must be a SHA-256 digest.")

        safe_response = make_serializable(dict(response))
        self._validate_response_document(safe_response)
        response_document = json.dumps(
            safe_response,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(response_document.encode("utf-8")) > MAX_RESPONSE_DOCUMENT_BYTES:
            raise UnsafeProposalDocumentError(
                "The proposal response exceeds the storage limit."
            )

        feature_set = dict(safe_response.get("feature_set") or {})
        params = {
            "tenant_id": clean_tenant_id,
            "proposal_id": self._required_text(
                safe_response.get("proposal_id"),
                "proposal_id",
            ),
            "campaign_id": clean_campaign_id,
            "idempotency_key": clean_idempotency_key,
            "request_fingerprint": clean_fingerprint,
            "contract_version": self._required_text(
                safe_response.get("contract_version"),
                "contract_version",
            ),
            "feature_set_id": self._required_text(
                feature_set.get("feature_set_id"),
                "feature_set.feature_set_id",
            ),
            "feature_set_version": int(feature_set.get("version") or 0),
            "execution_mode": normalize_taxonomy_value(
                safe_response.get("execution_mode")
            ),
            "status": self._required_text(
                safe_response.get("status"),
                "status",
            ),
            "approval_status": self._required_text(
                safe_response.get("approval_status"),
                "approval_status",
            ),
            "activation_eligible": bool(
                safe_response.get("activation_eligible")
            ),
            "safe_export_eligible": bool(
                safe_response.get("safe_export_eligible")
            ),
            "downstream_export_enabled": bool(
                safe_response.get("downstream_export_enabled")
            ),
            "response_document": response_document,
        }
        if params["feature_set_version"] < 1:
            raise ValueError("feature_set.version must be positive.")
        if params["execution_mode"] not in {
            "historical_preview",
            "offline_evaluation",
            "production",
        }:
            raise ValueError("The proposal execution mode is invalid.")
        if params["downstream_export_enabled"]:
            raise UnsafeProposalDocumentError(
                "Proposal ledger records cannot enable downstream export."
            )

        engine = self._engine()
        try:
            with engine.begin() as connection:
                self._set_tenant_context(connection, clean_tenant_id)
                inserted = connection.execute(
                    text(
                        """
                        INSERT INTO public.punk_ai_audience_proposals (
                            tenant_id,
                            proposal_id,
                            campaign_id,
                            idempotency_key,
                            request_fingerprint,
                            contract_version,
                            feature_set_id,
                            feature_set_version,
                            execution_mode,
                            status,
                            approval_status,
                            activation_eligible,
                            safe_export_eligible,
                            downstream_export_enabled,
                            response_document
                        )
                        VALUES (
                            :tenant_id,
                            :proposal_id,
                            :campaign_id,
                            :idempotency_key,
                            :request_fingerprint,
                            :contract_version,
                            :feature_set_id,
                            :feature_set_version,
                            :execution_mode,
                            :status,
                            :approval_status,
                            :activation_eligible,
                            :safe_export_eligible,
                            :downstream_export_enabled,
                            CAST(:response_document AS JSONB)
                        )
                        ON CONFLICT (
                            tenant_id,
                            idempotency_key
                        )
                        DO NOTHING
                        RETURNING proposal_id
                        """
                    ),
                    params,
                ).scalar_one_or_none()
                row = connection.execute(
                    text(
                        """
                        SELECT
                            proposal_id,
                            request_fingerprint,
                            response_document
                        FROM public.punk_ai_audience_proposals
                        WHERE
                            tenant_id = :tenant_id
                            AND idempotency_key = :idempotency_key
                        """
                    ),
                    params,
                ).mappings().one()
        except SQLAlchemyError:
            raise RuntimeError(
                "The Punk AI proposal could not be recorded."
            ) from None
        finally:
            engine.dispose()

        stored = self._stored_record(row)
        if stored["request_fingerprint"] != clean_fingerprint:
            raise ProposalIdempotencyConflictError(
                "The idempotency key is already bound to a different request."
            )
        stored["inserted"] = inserted is not None
        return stored

    def _validate_response_document(self, value: Any, path: str = "response") -> None:
        if isinstance(value, dict):
            for raw_key, child in value.items():
                key = str(raw_key).strip().lower()
                if (
                    key in BLOCKED_RESPONSE_KEYS
                    or any(token in key for token in BLOCKED_RESPONSE_KEY_TOKENS)
                    or key.endswith(("_lat", "_lng"))
                ):
                    raise UnsafeProposalDocumentError(
                        f"{path} contains a blocked field: {raw_key}"
                    )
                self._validate_response_document(
                    child,
                    f"{path}.{raw_key}",
                )
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self._validate_response_document(
                    child,
                    f"{path}[{index}]",
                )
        elif isinstance(value, str) and any(
            pattern.search(value)
            for pattern in BLOCKED_RESPONSE_VALUE_PATTERNS
        ):
            raise UnsafeProposalDocumentError(
                f"{path} contains a credential-like value."
            )

    def _stored_record(self, row: Mapping[str, Any]) -> dict[str, Any]:
        document = row["response_document"]
        if isinstance(document, str):
            document = json.loads(document)
        return {
            "proposal_id": str(row["proposal_id"]),
            "request_fingerprint": str(row["request_fingerprint"]),
            "response": dict(document),
        }

    def _set_tenant_context(self, connection: Any, tenant_id: str) -> None:
        connection.execute(
            text(
                """
                SELECT set_config(
                    'app.tenant_id',
                    :tenant_id,
                    true
                )
                """
            ),
            {"tenant_id": tenant_id},
        )

    def _engine(self) -> Engine:
        database_url = str(
            self._database_url
            or self._environment.get("AUDIENCE_PROPOSAL_DATABASE_URL")
            or ""
        ).strip()
        if not database_url:
            raise RuntimeError(
                "AUDIENCE_PROPOSAL_DATABASE_URL must be configured."
            )
        if database_url.startswith("postgres://"):
            database_url = "postgresql://" + database_url[len("postgres://") :]
        return self._engine_factory(database_url, pool_pre_ping=True)

    def _required_slug(self, value: Any, label: str) -> str:
        clean = normalize_taxonomy_value(value)
        if not clean:
            raise ValueError(f"{label} is required.")
        return clean

    def _required_text(self, value: Any, label: str) -> str:
        clean = " ".join(str(value or "").split())
        if not clean:
            raise ValueError(f"{label} is required.")
        return clean
