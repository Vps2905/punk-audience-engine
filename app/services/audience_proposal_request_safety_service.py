from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AudienceProposalSafetyDecision:
    terminal: bool
    reason_code: str | None = None
    explanation: str | None = None


class AudienceProposalRequestSafetyService:
    """Deterministic privacy and action-eligibility boundary."""

    _IDENTIFIER_TERMS = (
        r"raw\s+ma ids?",
        r"raw\s+maids?",
        r"ma ids?",
        r"maid\s+ids?",
        r"device\s+ids?",
        r"mobile\s+advertising\s+ids?",
        r"hashed\s+identifiers?",
        r"raw\s+hash(?:es)?",
        r"exact\s+(?:lat|lon|lng|latitude|longitude)",
        r"lat(?:itude)?\s*[/, ]\s*(?:lon|lng|longitude)",
        r"individual(?:-level|\s+level)?\s+(?:users?|data|records?)",
        r"user(?:-level|\s+level)\s+(?:data|records?)",
    )
    _DISCLOSURE_ACTIONS = (
        r"give",
        r"show",
        r"list",
        r"display",
        r"return",
        r"export",
        r"download",
        r"send",
        r"provide",
        r"share",
        r"reveal",
        r"extract",
        r"upload",
    )
    _SAFE_NEGATIONS = (
        r"do\s+not",
        r"don['’]?t",
        r"never",
        r"block",
        r"remove",
        r"exclude",
        r"without",
        r"no",
    )
    _EXPORT_ACTION = re.compile(
        r"\b(?:export|upload|push|activate)\b"
        r".{0,80}\b(?:audience|cohort|meta|destination|platform|campaign)\b"
        r"|"
        r"\b(?:audience|cohort)\b"
        r".{0,80}\b(?:export|upload|push|activate)\b"
        r"|"
        r"\b(?:send|deliver)\b"
        r".{0,80}\b(?:meta|destination|platform|ads?\s+manager)\b",
        re.IGNORECASE,
    )

    def evaluate(
        self,
        request: Mapping[str, Any],
    ) -> AudienceProposalSafetyDecision:
        intent = re.sub(
            r"\s+",
            " ",
            str(request.get("audience_intent") or ""),
        ).strip()
        if self._is_raw_identifier_request(intent):
            return AudienceProposalSafetyDecision(
                terminal=True,
                reason_code="blocked_privacy_identifier_request",
                explanation=(
                    "Raw identifiers and individual-level user data cannot "
                    "be retrieved or exported. Only privacy-safe aggregated "
                    "cohorts are allowed."
                ),
            )
        if (
            self._EXPORT_ACTION.search(intent)
            and not self._has_structured_targeting(request)
        ):
            return AudienceProposalSafetyDecision(
                terminal=True,
                reason_code=(
                    "blocked_export_action_requires_existing_audience"
                ),
                explanation=(
                    "An export action requires an existing approved audience. "
                    "No new audience was ranked, prepared or exported."
                ),
            )
        return AudienceProposalSafetyDecision(terminal=False)

    def _is_raw_identifier_request(self, intent: str) -> bool:
        identifiers = "(?:" + "|".join(self._IDENTIFIER_TERMS) + ")"
        actions = "(?:" + "|".join(self._DISCLOSURE_ACTIONS) + ")"
        negations = "(?:" + "|".join(self._SAFE_NEGATIONS) + ")"
        remaining_intent = re.sub(
            rf"\b{negations}\b.{{0,80}}?{identifiers}",
            " ",
            intent,
            flags=re.IGNORECASE,
        )
        return bool(
            re.search(
                rf"\b{actions}\b.{{0,80}}{identifiers}"
                rf"|{identifiers}.{{0,80}}\b{actions}\b",
                remaining_intent,
                re.IGNORECASE,
            )
        )

    def _has_structured_targeting(
        self,
        request: Mapping[str, Any],
    ) -> bool:
        return any(
            self._has_values(request.get(field))
            for field in (
                "locations",
                "categories",
                "dayparts",
            )
        )

    def _has_values(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        try:
            return any(str(item or "").strip() for item in value)
        except TypeError:
            return bool(str(value).strip())
