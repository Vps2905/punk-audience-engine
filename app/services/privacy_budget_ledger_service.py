from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text

from app.core.tenant_request_auth import TENANT_PATTERN


@dataclass(frozen=True)
class PrivacyBudgetRequest:
    """
    One privacy budget spend request.

    epsilon:
        Privacy budget consumed by this DP release.

    delta:
        Failure probability used for approximate DP, usually Gaussian DP.

    budget_scope:
        The scope we are protecting, for example:
            global_audience_export
            customer_123
            campaign_456
            cohort_abc
    """

    tenant_id: str
    run_id: str
    budget_scope: str
    epsilon: float
    delta: float = 1e-5
    mechanism: str = "gaussian"
    sensitivity: float = 1.0
    max_budget: float = 5.0
    cohort_id: Optional[str] = None
    query_type: str = "audience_count_release"
    actor: str = "system"
    note: Optional[str] = None


class PrivacyBudgetLedgerService:
    """
    DB-backed privacy budget ledger.

    Why this exists:
        DP is not only adding noise. We must track total privacy budget usage.

    Production rule:
        If used_epsilon + requested_epsilon > max_budget,
        block the release/export.
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._explicit_database_url = database_url

    def check_and_record(self, request: PrivacyBudgetRequest) -> Dict[str, Any]:
        self._validate_request(request)
        tenant_id = self._tenant_id(request.tenant_id)

        db_url = self._db_url()
        if not db_url:
            return {
                "enabled": False,
                "status": "skipped",
                "reason": "No database URL configured for privacy budget ledger.",
            }

        engine = create_engine(self._connection_url(db_url))

        with engine.begin() as conn:
            self._set_tenant_context(
                conn,
                tenant_id,
                engine.dialect.name,
            )
            self._ensure_table(conn, engine.dialect.name)

            used_before = self._used_budget(
                conn,
                tenant_id=tenant_id,
                budget_scope=request.budget_scope,
            )
            requested_epsilon = float(request.epsilon)
            budget_after_if_allowed = used_before + requested_epsilon

            if budget_after_if_allowed > request.max_budget:
                decision = "blocked"
                reason = "Privacy budget exceeded."
                budget_after = used_before
                spent = False
            else:
                decision = "allowed"
                reason = "Privacy budget available."
                budget_after = budget_after_if_allowed
                spent = True

            conn.execute(
                text(
                    """
                    INSERT INTO audience_privacy_budget_ledger (
                        tenant_id,
                        run_id,
                        cohort_id,
                        budget_scope,
                        query_type,
                        mechanism,
                        epsilon,
                        delta,
                        sensitivity,
                        budget_before,
                        budget_after,
                        max_budget,
                        decision,
                        reason,
                        actor,
                        note,
                        created_at
                    )
                    VALUES (
                        :tenant_id,
                        :run_id,
                        :cohort_id,
                        :budget_scope,
                        :query_type,
                        :mechanism,
                        :epsilon,
                        :delta,
                        :sensitivity,
                        :budget_before,
                        :budget_after,
                        :max_budget,
                        :decision,
                        :reason,
                        :actor,
                        :note,
                        :created_at
                    )
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "run_id": request.run_id,
                    "cohort_id": request.cohort_id,
                    "budget_scope": request.budget_scope,
                    "query_type": request.query_type,
                    "mechanism": request.mechanism,
                    "epsilon": request.epsilon,
                    "delta": request.delta,
                    "sensitivity": request.sensitivity,
                    "budget_before": used_before,
                    "budget_after": budget_after,
                    "max_budget": request.max_budget,
                    "decision": decision,
                    "reason": reason,
                    "actor": request.actor,
                    "note": request.note,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        return {
            "enabled": True,
            "status": decision,
            "spent": spent,
            "tenant_id": tenant_id,
            "run_id": request.run_id,
            "cohort_id": request.cohort_id,
            "budget_scope": request.budget_scope,
            "mechanism": request.mechanism,
            "epsilon": request.epsilon,
            "delta": request.delta,
            "sensitivity": request.sensitivity,
            "budget_before": used_before,
            "budget_after": budget_after,
            "max_budget": request.max_budget,
            "remaining_budget": max(0.0, request.max_budget - budget_after),
            "reason": reason,
            "dp_probability_ratio_bound": round(math.exp(request.epsilon), 6),
            "privacy_note": (
                "Only allowed records spend privacy budget. Blocked records are audited but do not spend epsilon."
            ),
        }

    def get_budget_status(
        self,
        budget_scope: str,
        *,
        tenant_id: str,
        max_budget: float = 5.0,
    ) -> Dict[str, Any]:
        normalized_tenant_id = self._tenant_id(tenant_id)
        db_url = self._db_url()
        if not db_url:
            return {
                "enabled": False,
                "status": "skipped",
                "reason": "No database URL configured for privacy budget ledger.",
            }

        engine = create_engine(self._connection_url(db_url))

        with engine.begin() as conn:
            self._set_tenant_context(
                conn,
                normalized_tenant_id,
                engine.dialect.name,
            )
            self._ensure_table(conn, engine.dialect.name)
            used = self._used_budget(
                conn,
                tenant_id=normalized_tenant_id,
                budget_scope=budget_scope,
            )

        return {
            "enabled": True,
            "status": "ok",
            "tenant_id": normalized_tenant_id,
            "budget_scope": budget_scope,
            "used_budget": used,
            "max_budget": max_budget,
            "remaining_budget": max(0.0, max_budget - used),
            "budget_exhausted": used >= max_budget,
        }

    def _used_budget(
        self,
        conn,
        *,
        tenant_id: str,
        budget_scope: str,
    ) -> float:
        value = conn.execute(
            text(
                """
                SELECT COALESCE(SUM(epsilon), 0)
                FROM audience_privacy_budget_ledger
                WHERE tenant_id = :tenant_id
                  AND budget_scope = :budget_scope
                  AND decision = 'allowed'
                """
            ),
            {
                "tenant_id": self._tenant_id(tenant_id),
                "budget_scope": budget_scope,
            },
        ).scalar()

        return float(value or 0.0)

    def _ensure_table(self, conn, dialect_name: str) -> None:
        if dialect_name == "postgresql":
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS audience_privacy_budget_ledger (
                        id BIGSERIAL PRIMARY KEY,
                        tenant_id TEXT NOT NULL CHECK (
                            tenant_id ~ '^[a-z0-9][a-z0-9_.-]{0,127}$'
                        ),
                        run_id TEXT NOT NULL,
                        cohort_id TEXT,
                        budget_scope TEXT NOT NULL,
                        query_type TEXT NOT NULL,
                        mechanism TEXT NOT NULL,
                        epsilon DOUBLE PRECISION NOT NULL,
                        delta DOUBLE PRECISION,
                        sensitivity DOUBLE PRECISION NOT NULL,
                        budget_before DOUBLE PRECISION NOT NULL,
                        budget_after DOUBLE PRECISION NOT NULL,
                        max_budget DOUBLE PRECISION NOT NULL,
                        decision TEXT NOT NULL,
                        reason TEXT,
                        actor TEXT NOT NULL DEFAULT 'system',
                        note TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
            )
        else:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS audience_privacy_budget_ledger (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tenant_id TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        cohort_id TEXT,
                        budget_scope TEXT NOT NULL,
                        query_type TEXT NOT NULL,
                        mechanism TEXT NOT NULL,
                        epsilon REAL NOT NULL,
                        delta REAL,
                        sensitivity REAL NOT NULL,
                        budget_before REAL NOT NULL,
                        budget_after REAL NOT NULL,
                        max_budget REAL NOT NULL,
                        decision TEXT NOT NULL,
                        reason TEXT,
                        actor TEXT NOT NULL DEFAULT 'system',
                        note TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
            )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_audience_privacy_budget_scope
                ON audience_privacy_budget_ledger(
                    tenant_id,
                    budget_scope
                )
                """
            )
        )

        if dialect_name == "postgresql":
            conn.execute(
                text(
                    "ALTER TABLE public."
                    "audience_privacy_budget_ledger "
                    "ENABLE ROW LEVEL SECURITY"
                )
            )
            conn.execute(
                text(
                    "ALTER TABLE public."
                    "audience_privacy_budget_ledger "
                    "FORCE ROW LEVEL SECURITY"
                )
            )
            conn.execute(
                text(
                    "DROP POLICY IF EXISTS "
                    "audience_privacy_budget_ledger_tenant_policy "
                    "ON public.audience_privacy_budget_ledger"
                )
            )
            conn.execute(
                text(
                    "CREATE POLICY "
                    "audience_privacy_budget_ledger_tenant_policy "
                    "ON public.audience_privacy_budget_ledger "
                    "USING (tenant_id = current_setting("
                    "'app.tenant_id', true)) "
                    "WITH CHECK (tenant_id = current_setting("
                    "'app.tenant_id', true))"
                )
            )
            conn.execute(
                text(
                    "REVOKE ALL ON public."
                    "audience_privacy_budget_ledger FROM PUBLIC"
                )
            )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_audience_privacy_budget_run
                ON audience_privacy_budget_ledger(
                    tenant_id,
                    run_id
                )
                """
            )
        )

    def _db_url(self) -> Optional[str]:
        return (
            self._explicit_database_url
            or os.getenv("AUDIENCE_PRIVACY_BUDGET_DATABASE_URL")
            or os.getenv("AUDIENCE_HISTORY_DATABASE_URL")
            or os.getenv("ECHO_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or os.getenv("POSTGRES_URL")
            or os.getenv("POSTGRES_DATABASE_URL")
            or os.getenv("SUPABASE_DB_URL")
            or os.getenv("DB_URL")
        )

    def _connection_url(self, db_url: str) -> str:
        if db_url.startswith("postgres://"):
            return "postgresql://" + db_url[len("postgres://") :]
        return db_url

    def _tenant_id(self, tenant_id: str) -> str:
        normalized = str(tenant_id or "").strip().lower()
        if not TENANT_PATTERN.fullmatch(normalized):
            raise ValueError("Invalid tenant identifier.")
        return normalized

    def _set_tenant_context(
        self,
        conn,
        tenant_id: str,
        dialect_name: str,
    ) -> None:
        if dialect_name != "postgresql":
            return
        conn.execute(
            text(
                "SELECT set_config("
                "'app.tenant_id', :tenant_id, true"
                ")"
            ),
            {"tenant_id": self._tenant_id(tenant_id)},
        )

    def _validate_request(self, request: PrivacyBudgetRequest) -> None:
        self._tenant_id(request.tenant_id)

        if not request.run_id:
            raise ValueError("run_id is required")

        if not request.budget_scope:
            raise ValueError("budget_scope is required")

        if request.epsilon <= 0:
            raise ValueError("epsilon must be > 0")

        if request.delta <= 0 or request.delta >= 1:
            raise ValueError("delta must be between 0 and 1")

        if request.sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")

        if request.max_budget <= 0:
            raise ValueError("max_budget must be > 0")
