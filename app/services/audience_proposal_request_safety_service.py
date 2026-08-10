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
        r"\b(?:export|upload|push|activate|publish|launch)\b"
        r".{0,80}\b(?:audience|cohort|recommendation|selection|meta|"
        r"destination|platform|campaign)\b"
        r"|"
        r"\b(?:audience|cohort|recommendation|selection)\b"
        r".{0,80}\b(?:export|upload|push|activate|publish|launch)\b"
        r"|"
        r"\b(?:send|deliver)\b"
        r".{0,80}\b(?:meta|destination|platform|ads?\s+manager)\b"
        r"|"
        r"\b(?:deliver|publish|launch)\b"
        r".{0,80}\b(?:it|audience|cohort|recommendation|campaign)\b",
        re.IGNORECASE,
    )
    _ACTION_PLANNING_OR_NEGATION = (
        re.compile(
            r"\b(?:do\s+not|don['’]?t|never|without)\b"
            r".{0,80}?\b(?:export|upload|push|activate|publish|launch|"
            r"send|deliver)\b"
            r".{0,80}?(?=$|[,.;]|\b(?:but|however|yet|then)\b)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bkeep\b.{0,80}\b(?:activation|export|delivery)\b"
            r".{0,40}\b(?:blocked|disabled|off)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:prepare|design|build|create|plan|recommend|evaluate|"
            r"review|explain)\b.{0,120}\b(?:for\s+export|export-ready|"
            r"activation-ready|delivery-ready)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:explain|describe|document|review)\b.{0,80}"
            r"\bhow\s+to\s+(?:export|activate|deliver|publish|launch)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:export|activation|delivery)\s+"
            r"(?:plan|strategy|workflow|readiness|documentation)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bplan\b.{0,40}\b(?:an?\s+)?"
            r"(?:export|activation|delivery)\b",
            re.IGNORECASE,
        ),
    )
    _POLICY_BYPASS = re.compile(
        r"\b(?:ignore|skip|bypass|override|disable|turn\s+off|"
        r"circumvent|evade)\b.{0,100}\b(?:safety|safeguards?|"
        r"guardrails?|governance|freshness|approval|checks?|"
        r"restrictions?|polic(?:y|ies))\b"
        r"|"
        r"\b(?:safety|safeguards?|guardrails?|governance|freshness|"
        r"approval|checks?|restrictions?|polic(?:y|ies))\b.{0,100}"
        r"\b(?:ignore|skip|bypass|override|disable|circumvent|evade)\b",
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
        executable_intent = self._remove_safe_action_language(intent)
        if self._EXPORT_ACTION.search(executable_intent):
            if self._POLICY_BYPASS.search(intent):
                return AudienceProposalSafetyDecision(
                    terminal=True,
                    reason_code="blocked_approval_bypass_attempt",
                    explanation=(
                        "Safety, freshness, governance, and manual approval "
                        "controls cannot be bypassed. No audience data was "
                        "read, ranked, activated, or exported."
                    ),
                )
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

    def _remove_safe_action_language(self, intent: str) -> str:
        remaining = intent
        for pattern in self._ACTION_PLANNING_OR_NEGATION:
            remaining = pattern.sub(" ", remaining)
        return re.sub(r"\s+", " ", remaining).strip()

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
