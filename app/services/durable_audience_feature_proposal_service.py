from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from app.models.audience_feature_contracts import normalize_taxonomy_value
from app.services.punk_ai_audience_proposal_store_service import (
    ProposalIdempotencyConflictError,
)


class AudienceProposalDelegate(Protocol):
    def propose(self, request: dict[str, Any]) -> dict[str, Any]: ...


class AudienceProposalStore(Protocol):
    def get(
        self,
        *,
        tenant_id: str,
        campaign_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None: ...

    def record(
        self,
        *,
        tenant_id: str,
        campaign_id: str,
        idempotency_key: str,
        request_fingerprint: str,
        response: Mapping[str, Any],
    ) -> dict[str, Any]: ...


class DurableAudienceFeatureProposalService:
    """
    Add immutable replay and collision detection to audience proposals.

    A repeated identical request returns the recorded governed response. The
    same tenant/idempotency tuple cannot be rebound to changed input.
    """

    def __init__(
        self,
        *,
        delegate: AudienceProposalDelegate,
        proposal_store: AudienceProposalStore,
    ) -> None:
        self._delegate = delegate
        self._proposal_store = proposal_store

    def propose(self, request: dict[str, Any]) -> dict[str, Any]:
        canonical = self._canonical_request(request)
        tenant_id = canonical["tenant_id"]
        campaign_id = canonical["campaign_id"]
        idempotency_key = canonical["idempotency_key"]
        fingerprint = hashlib.sha256(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        existing = self._proposal_store.get(
            tenant_id=tenant_id,
            campaign_id=campaign_id,
            idempotency_key=idempotency_key,
        )
        if existing:
            return self._replay(existing, fingerprint)

        response = self._delegate.propose(request)
        if (
            normalize_taxonomy_value(response.get("tenant_id")) != tenant_id
            or self._required_text(
                response.get("campaign_id"),
                "response.campaign_id",
            )
            != campaign_id
            or self._required_text(
                response.get("idempotency_key"),
                "response.idempotency_key",
            )
            != idempotency_key
        ):
            raise RuntimeError(
                "The proposal response does not match its request identity."
            )
        recorded = self._proposal_store.record(
            tenant_id=tenant_id,
            campaign_id=campaign_id,
            idempotency_key=idempotency_key,
            request_fingerprint=fingerprint,
            response=response,
        )
        if not recorded.get("inserted"):
            return self._replay(recorded, fingerprint)
        output = dict(recorded["response"])
        output["idempotency_replayed"] = False
        output["proposal_persisted"] = True
        return output

    def _replay(
        self,
        existing: Mapping[str, Any],
        fingerprint: str,
    ) -> dict[str, Any]:
        if existing.get("request_fingerprint") != fingerprint:
            raise ProposalIdempotencyConflictError(
                "The idempotency key is already bound to a different request."
            )
        output = dict(existing["response"])
        output["idempotency_replayed"] = True
        output["proposal_persisted"] = True
        return output

    def _canonical_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        execution_mode = normalize_taxonomy_value(
            request.get("execution_mode") or "historical_preview"
        )
        if execution_mode not in {"historical_preview", "production"}:
            raise ValueError(
                "execution_mode must be historical_preview or production."
            )
        return {
            "tenant_id": self._required_slug(
                request.get("tenant_id"),
                "tenant_id",
            ),
            "campaign_id": self._required_text(
                request.get("campaign_id"),
                "campaign_id",
            ),
            "idempotency_key": self._required_text(
                request.get("idempotency_key"),
                "idempotency_key",
            ),
            "objective": normalize_taxonomy_value(request.get("objective")),
            "audience_intent": self._required_text(
                request.get("audience_intent"),
                "audience_intent",
            ).casefold(),
            "locations": self._canonical_list(request.get("locations")),
            "categories": self._canonical_list(request.get("categories")),
            "dayparts": self._canonical_list(request.get("dayparts")),
            "exclusions": self._canonical_list(request.get("exclusions")),
            "destination": normalize_taxonomy_value(
                request.get("destination")
            ),
            "budget": self._canonical_budget(request.get("budget")),
            "execution_mode": execution_mode,
            "feature_set_id": self._optional_text(
                request.get("feature_set_id")
            ),
            "feature_set_version": (
                int(request["feature_set_version"])
                if request.get("feature_set_version") is not None
                else None
            ),
            "top_k": max(1, min(int(request.get("top_k") or 10), 50)),
        }

    def _canonical_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        values: Sequence[Any] = [value] if isinstance(value, str) else list(value)
        return sorted(
            {
                normalized
                for item in values
                if (normalized := normalize_taxonomy_value(item))
            }
        )

    def _canonical_budget(self, value: Any) -> dict[str, Any]:
        budget = dict(value or {})
        return {
            str(key): budget[key]
            for key in sorted(budget)
            if budget[key] is not None
        }

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

    def _optional_text(self, value: Any) -> str | None:
        clean = " ".join(str(value or "").split())
        return clean or None
