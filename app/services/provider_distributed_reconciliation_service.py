from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from app.services.provider_distributed_privacy_budget_service import (
    ProviderDistributedPrivacyBudgetService,
)
from app.services.provider_ingestion_state_service import (
    ProviderIngestionStateService,
)


class ProviderDistributedReconciliationService:
    """
    Repairs control-plane records when Step Functions terminal callbacks fail.

    This is intentionally conservative: a terminal execution that left an
    ingestion non-terminal becomes failed, never completed. Canonical output
    can be published only through a validated distributed result contract.
    """

    ACTIVE_EXECUTION_STATUSES = {"RUNNING", "PENDING_REDRIVE"}
    TERMINAL_EXECUTION_STATUSES = {
        "SUCCEEDED",
        "FAILED",
        "TIMED_OUT",
        "ABORTED",
    }

    def __init__(
        self,
        *,
        state_service: ProviderIngestionStateService,
        step_functions_client: Any,
        privacy_budget_service: Optional[
            ProviderDistributedPrivacyBudgetService
        ] = None,
    ) -> None:
        self._state = state_service
        self._step_functions = step_functions_client
        self._privacy_budget = privacy_budget_service

    def reconcile(
        self,
        *,
        stale_after_seconds: int = 900,
        limit: int = 500,
    ) -> Dict[str, Any]:
        if stale_after_seconds < 1:
            raise ValueError("stale_after_seconds must be >= 1")
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=stale_after_seconds
        )
        records = []
        for status in ("dispatched", "processing"):
            records.extend(
                self._state.list_recent(status=status, limit=limit)
            )

        checked = 0
        still_running = 0
        failed_closed = 0
        unavailable = 0
        for record in records:
            metadata = record.get("metadata") or {}
            if metadata.get("execution_mode") != "distributed":
                continue
            updated_at = self._timestamp(record.get("updated_at"))
            if updated_at and updated_at > cutoff:
                continue
            execution_arn = str(
                metadata.get("distributed_job_id") or ""
            ).strip()
            if not execution_arn:
                self._fail(record, "distributed_execution_receipt_missing")
                failed_closed += 1
                continue
            try:
                response = self._step_functions.describe_execution(
                    executionArn=execution_arn
                )
            except Exception:
                unavailable += 1
                continue
            checked += 1
            execution_status = str(
                response.get("status") or ""
            ).upper()
            if execution_status in self.ACTIVE_EXECUTION_STATUSES:
                still_running += 1
                continue
            if execution_status in self.TERMINAL_EXECUTION_STATUSES:
                reason = (
                    "distributed_completion_callback_missing"
                    if execution_status == "SUCCEEDED"
                    else "distributed_execution_failed"
                )
                self._fail(record, reason)
                failed_closed += 1
            else:
                unavailable += 1

        return {
            "status": "completed",
            "records_considered": len(records),
            "executions_checked": checked,
            "still_running": still_running,
            "failed_closed": failed_closed,
            "execution_status_unavailable": unavailable,
        }

    def _fail(self, record: Dict[str, Any], reason_code: str) -> None:
        release_id = str(record.get("privacy_job_id") or "").strip()
        if self._privacy_budget is not None and release_id:
            try:
                self._privacy_budget.mark_terminal(
                    release_id,
                    status="failed",
                    reason_code=reason_code,
                )
            except (KeyError, ValueError):
                pass
        self._state.transition(
            str(record["ingestion_id"]),
            status="failed",
            reason_code=reason_code,
            metadata_update={
                "distributed_reconciled": True,
                "distributed_reconciliation_reason": reason_code,
            },
        )

    def _timestamp(self, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            parsed = value
        elif value:
            parsed = datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )
        else:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
