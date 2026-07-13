from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from app.services.privacy_budget_ledger_service import PrivacyBudgetLedgerService, PrivacyBudgetRequest


class AudienceRunHistoryService:
    """
    Production DB-backed audience run history and approval audit service.

    Safety rules:
    - Store only privacy-safe summaries, cohort metadata, warnings, and artifact paths.
    - Do not store raw MAIDs, raw observations, raw lat/lng, email, phone, or individual-level data.
    - Approval/rejection is audit logged in Postgres.
    """

    def __init__(self, db_url: Optional[str] = None):
        self._explicit_db_url = db_url

    def persist_run(
        self,
        *,
        final_summary: Dict[str, Any],
        selected_cohorts: pd.DataFrame | List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        db_url = self._db_url()
        if not db_url:
            return {"enabled": False, "status": "skipped", "reason": "no_database_url"}

        run_id = str(final_summary.get("run_id") or "").strip()
        if not run_id:
            return {"enabled": False, "status": "skipped", "reason": "missing_run_id"}

        clean_summary = self._clean_json(final_summary)
        cohort_rows = self._build_cohort_rows(
            run_id=run_id,
            final_summary=clean_summary,
            selected_cohorts=selected_cohorts,
        )
        artifact_rows = self._build_artifact_rows(run_id=run_id, final_summary=clean_summary)
        warning_rows = self._build_warning_rows(run_id=run_id, final_summary=clean_summary)

        try:
            engine = create_engine(self._connection_url(db_url))
            with engine.begin() as conn:
                self._ensure_tables(conn)
                self._upsert_run(conn, clean_summary)
                self._replace_cohorts(conn, run_id, cohort_rows)
                self._replace_artifacts(conn, run_id, artifact_rows)
                self._replace_warnings(conn, run_id, warning_rows)
                self._record_event_conn(
                    conn,
                    run_id=run_id,
                    event_type="run_persisted",
                    actor="system",
                    details={
                        "cohort_rows": len(cohort_rows),
                        "artifact_rows": len(artifact_rows),
                        "warning_rows": len(warning_rows),
                        "approval_status": (clean_summary.get("safe_export") or {}).get("approval_status"),
                        "downstream_export_enabled": (clean_summary.get("safe_export") or {}).get(
                            "downstream_export_enabled"
                        ),
                    },
                )
        except Exception as exc:
            return {
                "enabled": True,
                "status": "failed",
                "run_id": run_id,
                "error": str(exc),
            }

        return {
            "enabled": True,
            "status": "persisted",
            "run_id": run_id,
            "tables": [
                "audience_run_history",
                "audience_run_cohorts",
                "audience_run_artifacts",
                "audience_run_warnings",
                "audience_run_events",
                "audience_run_approvals",
            ],
            "cohort_rows": len(cohort_rows),
            "artifact_rows": len(artifact_rows),
            "warning_rows": len(warning_rows),
        }

    def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        approval_status: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        db_url = self._db_url()
        if not db_url:
            return {"enabled": False, "status": "skipped", "reason": "no_database_url", "runs": []}

        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))

        where = []
        params: Dict[str, Any] = {"limit": limit, "offset": offset}

        if approval_status:
            where.append("approval_status = :approval_status")
            params["approval_status"] = approval_status
        if status:
            where.append("status = :status")
            params["status"] = status

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        engine = create_engine(self._connection_url(db_url))
        with engine.begin() as conn:
            self._ensure_tables(conn)
            rows = conn.execute(
                text(
                    f"""
                    SELECT
                        run_id,
                        prompt,
                        status,
                        source_mode,
                        source_rows,
                        privacy_cohorts,
                        prompt_selected_cohorts,
                        exported_cohorts,
                        approval_status,
                        downstream_export_enabled,
                        filter_mode,
                        locations_detected,
                        poi_terms_detected,
                        dayparts_detected,
                        run_dir,
                        created_at,
                        updated_at
                    FROM audience_run_history
                    {where_sql}
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                    """
                ),
                params,
            ).fetchall()

        return {
            "enabled": True,
            "status": "ok",
            "limit": limit,
            "offset": offset,
            "runs": [self._row_to_dict(row) for row in rows],
        }

    def get_run(self, run_id: str) -> Dict[str, Any]:
        db_url = self._db_url()
        if not db_url:
            return {"enabled": False, "status": "skipped", "reason": "no_database_url"}

        engine = create_engine(self._connection_url(db_url))
        with engine.begin() as conn:
            self._ensure_tables(conn)

            row = conn.execute(
                text("SELECT * FROM audience_run_history WHERE run_id = :run_id"),
                {"run_id": run_id},
            ).fetchone()

            if not row:
                return {"enabled": True, "status": "not_found", "run_id": run_id}

            cohorts = conn.execute(
                text(
                    """
                    SELECT
                        export_cohort_id,
                        audience_name,
                        location_name,
                        primary_poi_type,
                        created_day_part,
                        lookback_bucket,
                        quality_score,
                        management_quality_score,
                        approval_status,
                        privacy_mode,
                        data_safety_status,
                        risk_decision,
                        risk_level,
                        metadata,
                        created_at
                    FROM audience_run_cohorts
                    WHERE run_id = :run_id
                    ORDER BY quality_score DESC NULLS LAST, created_at ASC
                    """
                ),
                {"run_id": run_id},
            ).fetchall()

            artifacts = conn.execute(
                text(
                    """
                    SELECT artifact_type, artifact_path, metadata, created_at
                    FROM audience_run_artifacts
                    WHERE run_id = :run_id
                    ORDER BY created_at ASC
                    """
                ),
                {"run_id": run_id},
            ).fetchall()

            warnings = conn.execute(
                text(
                    """
                    SELECT warning_type, warning_message, metadata, created_at
                    FROM audience_run_warnings
                    WHERE run_id = :run_id
                    ORDER BY created_at ASC
                    """
                ),
                {"run_id": run_id},
            ).fetchall()

        return {
            "enabled": True,
            "status": "ok",
            "run": self._row_to_dict(row),
            "cohorts": [self._row_to_dict(r) for r in cohorts],
            "artifacts": [self._row_to_dict(r) for r in artifacts],
            "warnings": [self._row_to_dict(r) for r in warnings],
        }

    def list_cohorts(
        self,
        *,
        run_id: str,
        location: Optional[str] = None,
        poi_type: Optional[str] = None,
        daypart: Optional[str] = None,
        approval_status: Optional[str] = None,
        min_quality: float = 0.0,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        db_url = self._db_url()

        if not db_url:
            return {
                "enabled": False,
                "status": "skipped",
                "reason": "no_database_url",
                "cohorts": [],
            }

        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        min_quality = max(
            0.0,
            min(float(min_quality), 1.0),
        )

        where = [
            "run_id = :run_id",
            (
                "COALESCE("
                "management_quality_score, "
                "quality_score, 0"
                ") >= :min_quality"
            ),
        ]

        params: Dict[str, Any] = {
            "run_id": run_id,
            "min_quality": min_quality,
            "limit": limit,
            "offset": offset,
        }

        if location:
            where.append(
                "LOWER(location_name) = LOWER(:location)"
            )
            params["location"] = location

        if poi_type:
            where.append(
                "LOWER(primary_poi_type) = "
                "LOWER(:poi_type)"
            )
            params["poi_type"] = poi_type

        if daypart:
            where.append(
                "LOWER(created_day_part) = "
                "LOWER(:daypart)"
            )
            params["daypart"] = daypart

        if approval_status:
            where.append(
                "LOWER(approval_status) = "
                "LOWER(:approval_status)"
            )
            params[
                "approval_status"
            ] = approval_status

        where_sql = " AND ".join(where)

        engine = create_engine(
            self._connection_url(db_url)
        )

        with engine.begin() as conn:
            self._ensure_tables(conn)

            run_exists = conn.execute(
                text(
                    """
                    SELECT 1
                    FROM audience_run_history
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id},
            ).scalar()

            if not run_exists:
                return {
                    "enabled": True,
                    "status": "not_found",
                    "run_id": run_id,
                    "cohorts": [],
                }

            total_count = int(
                conn.execute(
                    text(
                        f"""
                        SELECT COUNT(*)
                        FROM audience_run_cohorts
                        WHERE {where_sql}
                        """
                    ),
                    params,
                ).scalar_one()
            )

            rows = conn.execute(
                text(
                    f"""
                    SELECT
                        export_cohort_id,
                        audience_name,
                        location_name,
                        primary_poi_type,
                        created_day_part,
                        lookback_bucket,
                        quality_score,
                        management_quality_score,
                        approval_status,
                        privacy_mode,
                        data_safety_status,
                        risk_decision,
                        risk_level,
                        metadata,
                        created_at
                    FROM audience_run_cohorts
                    WHERE {where_sql}
                    ORDER BY
                        COALESCE(
                            management_quality_score,
                            quality_score,
                            0
                        ) DESC,
                        export_cohort_id ASC
                    LIMIT :limit OFFSET :offset
                    """
                ),
                params,
            ).fetchall()

        returned_count = len(rows)
        next_offset = offset + returned_count
        has_more = next_offset < total_count

        return {
            "enabled": True,
            "status": "ok",
            "run_id": run_id,
            # Backward-compatible field.
            "count": returned_count,
            "returned_count": returned_count,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "next_offset": (
                next_offset if has_more else None
            ),
            "cohorts": [
                self._row_to_dict(row)
                for row in rows
            ],
        }

    def get_audit(self, run_id: str) -> Dict[str, Any]:
        db_url = self._db_url()
        if not db_url:
            return {"enabled": False, "status": "skipped", "reason": "no_database_url"}

        engine = create_engine(self._connection_url(db_url))
        with engine.begin() as conn:
            self._ensure_tables(conn)

            events = conn.execute(
                text(
                    """
                    SELECT event_type, actor, details, created_at
                    FROM audience_run_events
                    WHERE run_id = :run_id
                    ORDER BY created_at ASC, id ASC
                    """
                ),
                {"run_id": run_id},
            ).fetchall()

            approvals = conn.execute(
                text(
                    """
                    SELECT action, actor, note, previous_status, new_status,
                           downstream_export_enabled, privacy_snapshot,
                           artifacts_snapshot, created_at
                    FROM audience_run_approvals
                    WHERE run_id = :run_id
                    ORDER BY created_at ASC, id ASC
                    """
                ),
                {"run_id": run_id},
            ).fetchall()

        return {
            "enabled": True,
            "status": "ok",
            "run_id": run_id,
            "events": [self._row_to_dict(r) for r in events],
            "approvals": [self._row_to_dict(r) for r in approvals],
        }


    def _approval_blockers(self, final_summary: Dict[str, Any]) -> List[str]:
        """
        Return hard blockers that must prevent approval and downstream export.

        Only a clean pending_approval package may move to approved.
        Blocked, rejected, failed, stale, already-approved, or unsafe packages
        must fail closed.
        """
        blockers: List[str] = []

        final_summary = final_summary or {}
        safe_export = final_summary.get("safe_export") or {}
        v2 = final_summary.get("v2_autonomous") or {}
        swarm = final_summary.get("v2_swarm_review") or {}
        v2_freshness = v2.get("data_freshness") or {}
        source_freshness = final_summary.get("source_freshness") or {}
        prompt_filter_report = final_summary.get("prompt_filter_report") or {}
        privacy = final_summary.get("privacy_guarantees") or {}

        def normalize(value: Any) -> str:
            return str(value or "").strip().lower()

        # Approval-state enforcement.
        approval_sources = {
            "run": final_summary.get("approval_status"),
            "safe_export": safe_export.get("approval_status"),
            "privacy_guarantees": privacy.get("approval_status"),
        }

        terminal_blocked_statuses = {
            "blocked",
            "blocked_approval",
            "blocked_stale_source",
            "blocked_no_safe_exact_match",
            "blocked_privacy_budget",
            "rejected",
            "failed",
            "error",
            "not_ready",
        }

        for source, value in approval_sources.items():
            status = normalize(value)

            if not status:
                continue

            if status.startswith("blocked_") or status in terminal_blocked_statuses:
                blockers.append(f"{source} approval_status is {status}.")

            # Prevent repeat approval and repeat privacy-budget spending.
            if status == "approved":
                blockers.append(
                    f"{source} approval_status is already approved; "
                    "repeat approval is not allowed."
                )

            if status == "not_required":
                blockers.append(
                    f"{source} approval_status is not_required; "
                    "manual approval is not applicable."
                )

        # Freshness can be surfaced at several levels.
        freshness_values = [
            final_summary.get("freshness_status"),
            source_freshness.get("freshness_status"),
            source_freshness.get("status"),
            v2_freshness.get("freshness_status"),
            v2_freshness.get("status"),
            swarm.get("freshness_status"),
        ]

        normalized_freshness = {
            normalize(value)
            for value in freshness_values
            if normalize(value)
        }

        if normalized_freshness.intersection({"stale", "expired", "outdated"}):
            blockers.append(
                "Source freshness is stale; refresh or verify Echo/Postgres "
                "source before approval."
            )

        if bool(v2_freshness.get("stale_data_warning")):
            blockers.append("Freshness agent raised stale_data_warning.")

        # Fail closed on all export-block representations.
        export_is_blocked = bool(
            final_summary.get("block_export")
            or safe_export.get("block_export")
            or safe_export.get("export_blocked")
            or safe_export.get("export_blocked_until_source_refresh")
        )

        if export_is_blocked:
            blockers.append("Safe export is explicitly blocked.")

        filter_mode = normalize(prompt_filter_report.get("filter_mode"))
        blocked_filter_modes = {
            "location_category_gap_no_export",
            "broad_location_no_export",
            "raw_identifier_request_blocked",
        }

        if filter_mode in blocked_filter_modes:
            blockers.append(f"Prompt filter mode blocks approval: {filter_mode}.")

        overall_review_status = normalize(
            swarm.get("overall_review_status")
            or swarm.get("status")
        )

        if overall_review_status == "blocked":
            blockers.append("Swarm review status is blocked.")

        raw_flags = [
            "raw_maids_exported",
            "hashed_identifiers_exported",
            "raw_observations_exported",
            "raw_lat_lng_exported",
            "raw_email_exported",
            "raw_phone_exported",
            "individual_user_data_exported",
        ]

        leaked = [flag for flag in raw_flags if bool(privacy.get(flag))]
        if leaked:
            blockers.append(
                "Privacy guarantee failed; unsafe export flags detected: "
                + ", ".join(leaked)
            )

        if export_is_blocked and bool(
            final_summary.get("downstream_export_enabled")
            or safe_export.get("downstream_export_enabled")
        ):
            blockers.append(
                "Unsafe state: downstream export is enabled while export is blocked."
            )

        # Keep blocker output deterministic and duplicate-free.
        return list(dict.fromkeys(blockers))

    def approve_run(
        self,
        *,
        run_id: str,
        actor: str = "internal_reviewer",
        note: Optional[str] = None,
        downstream_export_enabled: bool = True,
    ) -> Dict[str, Any]:
        return self._decision(
            run_id=run_id,
            action="approved",
            actor=actor,
            note=note,
            downstream_export_enabled=downstream_export_enabled,
        )

    def reject_run(
        self,
        *,
        run_id: str,
        actor: str = "internal_reviewer",
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._decision(
            run_id=run_id,
            action="rejected",
            actor=actor,
            note=note,
            downstream_export_enabled=False,
        )

    def record_event(
        self,
        *,
        run_id: str,
        event_type: str,
        actor: str = "system",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        db_url = self._db_url()
        if not db_url:
            return {"enabled": False, "status": "skipped", "reason": "no_database_url"}

        engine = create_engine(self._connection_url(db_url))
        with engine.begin() as conn:
            self._ensure_tables(conn)
            self._record_event_conn(
                conn,
                run_id=run_id,
                event_type=event_type,
                actor=actor,
                details=details or {},
            )

        return {"enabled": True, "status": "recorded", "run_id": run_id, "event_type": event_type}

    def _decision(
        self,
        *,
        run_id: str,
        action: str,
        actor: str,
        note: Optional[str],
        downstream_export_enabled: bool,
    ) -> Dict[str, Any]:
        db_url = self._db_url()
        if not db_url:
            return {"enabled": False, "status": "skipped", "reason": "no_database_url"}

        if action not in {"approved", "rejected"}:
            return {"enabled": True, "status": "failed", "error": f"Unsupported action: {action}"}

        engine = create_engine(self._connection_url(db_url))

        with engine.begin() as conn:
            self._ensure_tables(conn)

            row = conn.execute(
                text("SELECT * FROM audience_run_history WHERE run_id = :run_id FOR UPDATE"),
                {"run_id": run_id},
            ).fetchone()

            if not row:
                return {"enabled": True, "status": "not_found", "run_id": run_id}

            run = self._row_to_dict(row)
            previous_status = run.get("approval_status")
            exported_cohorts = max(
                self._safe_int(run.get("exported_cohorts")) or 0,
                self._exportable_cohort_count_conn(conn, run_id) or 0,
            )

            if action == "approved":
                final_summary = run.get("final_summary") or {}
                blockers = self._approval_blockers(final_summary)
                if blockers:
                    blocked_status = "blocked_approval"

                    self._record_event_conn(
                        conn,
                        run_id=run_id,
                        event_type="approval_blocked",
                        actor=actor,
                        details={
                            "note": note,
                            "blockers": blockers,
                            "previous_status": previous_status,
                            "new_status": blocked_status,
                            "downstream_export_enabled": False,
                        },
                    )

                    conn.execute(
                        text(
                            """
                            UPDATE audience_run_history
                            SET approval_status = :approval_status,
                                downstream_export_enabled = false,
                                updated_at = now()
                            WHERE run_id = :run_id
                            """
                        ),
                        {
                            "run_id": run_id,
                            "approval_status": blocked_status,
                        },
                    )

                    conn.execute(
                        text(
                            """
                            UPDATE audience_run_cohorts
                            SET approval_status = :approval_status
                            WHERE run_id = :run_id
                            """
                        ),
                        {
                            "run_id": run_id,
                            "approval_status": blocked_status,
                        },
                    )

                    return {
                        "enabled": True,
                        "status": "blocked",
                        "run_id": run_id,
                        "previous_status": previous_status,
                        "approval_status": blocked_status,
                        "downstream_export_enabled": False,
                        "actor": actor,
                        "note": note,
                        "blockers": blockers,
                        "meta_upload_performed": False,
                    }

                if action == "approved":
                    if previous_status and str(previous_status).startswith("blocked"):
                        self._record_event_conn(
                            conn,
                            run_id=run_id,
                            event_type="approval_blocked",
                            actor=actor,
                            details={
                                "reason": "blocked_runs_cannot_be_approved",
                                "previous_status": previous_status,
                                "note": note,
                            },
                        )
                        return {
                            "enabled": True,
                            "status": "blocked",
                            "run_id": run_id,
                            "reason": "blocked_runs_cannot_be_approved",
                            "previous_status": previous_status,
                        }

                    if exported_cohorts <= 0:
                        self._record_event_conn(
                            conn,
                            run_id=run_id,
                            event_type="approval_blocked",
                            actor=actor,
                            details={
                                "reason": "no_exported_cohorts_to_approve",
                                "previous_status": previous_status,
                                "note": note,
                            },
                        )
                        return {
                            "enabled": True,
                            "status": "blocked",
                            "run_id": run_id,
                            "reason": "no_exported_cohorts_to_approve",
                            "previous_status": previous_status,
                        }

                privacy_budget_result = PrivacyBudgetLedgerService(
                    database_url=db_url
                ).check_and_record(
                    self._privacy_budget_request_for_run(
                        run_id=run_id,
                        run=run,
                        actor=actor,
                        note=note,
                    )
                )

                if privacy_budget_result.get("status") == "blocked":
                    blocked_status = "blocked_privacy_budget"

                    self._record_event_conn(
                        conn,
                        run_id=run_id,
                        event_type="privacy_budget_blocked",
                        actor=actor,
                        details={
                            "note": note,
                            "previous_status": previous_status,
                            "new_status": blocked_status,
                            "downstream_export_enabled": False,
                            "privacy_budget": privacy_budget_result,
                        },
                    )

                    conn.execute(
                        text(
                            """
                            UPDATE audience_run_history
                            SET approval_status = :approval_status,
                                downstream_export_enabled = false,
                                updated_at = now()
                            WHERE run_id = :run_id
                            """
                        ),
                        {
                            "run_id": run_id,
                            "approval_status": blocked_status,
                        },
                    )

                    conn.execute(
                        text(
                            """
                            UPDATE audience_run_cohorts
                            SET approval_status = :approval_status
                            WHERE run_id = :run_id
                            """
                        ),
                        {
                            "run_id": run_id,
                            "approval_status": blocked_status,
                        },
                    )

                    return {
                        "enabled": True,
                        "status": "blocked",
                        "run_id": run_id,
                        "previous_status": previous_status,
                        "approval_status": blocked_status,
                        "downstream_export_enabled": False,
                        "actor": actor,
                        "note": note,
                        "blockers": ["Privacy budget exceeded."],
                        "privacy_budget": privacy_budget_result,
                        "meta_upload_performed": False,
                    }

                self._record_event_conn(
                    conn,
                    run_id=run_id,
                    event_type="privacy_budget_spent",
                    actor=actor,
                    details={
                        "note": note,
                        "privacy_budget": privacy_budget_result,
                    },
                )

            if action == "approved":
                if previous_status and str(previous_status).startswith("blocked"):
                    self._record_event_conn(
                        conn,
                        run_id=run_id,
                        event_type="approval_blocked",
                        actor=actor,
                        details={
                            "reason": "blocked_runs_cannot_be_approved",
                            "previous_status": previous_status,
                            "note": note,
                        },
                    )
                    return {
                        "enabled": True,
                        "status": "blocked",
                        "run_id": run_id,
                        "reason": "blocked_runs_cannot_be_approved",
                        "previous_status": previous_status,
                    }

                if exported_cohorts <= 0:
                    self._record_event_conn(
                        conn,
                        run_id=run_id,
                        event_type="approval_blocked",
                        actor=actor,
                        details={
                            "reason": "no_exported_cohorts_to_approve",
                            "previous_status": previous_status,
                            "note": note,
                        },
                    )
                    return {
                        "enabled": True,
                        "status": "blocked",
                        "run_id": run_id,
                        "reason": "no_exported_cohorts_to_approve",
                        "previous_status": previous_status,
                    }

            new_status = "approved" if action == "approved" else "rejected"
            new_downstream = bool(downstream_export_enabled) if action == "approved" else False

            final_summary = run.get("final_summary") or {}
            if isinstance(final_summary, str):
                final_summary = json.loads(final_summary)

            safe_export = final_summary.get("safe_export") or {}
            safe_export["approval_status"] = new_status
            safe_export["downstream_export_enabled"] = new_downstream
            safe_export["reviewed_at"] = self._now()
            safe_export["reviewed_by"] = actor
            safe_export["review_note"] = note
            final_summary["safe_export"] = safe_export

            privacy_snapshot = run.get("privacy_guarantees") or final_summary.get("privacy_guarantees") or {}
            artifacts_snapshot = safe_export.get("outputs") or {}

            conn.execute(
                text(
                    """
                    UPDATE audience_run_history
                    SET approval_status = :approval_status,
                        downstream_export_enabled = :downstream_export_enabled,
                        safe_export = CAST(:safe_export AS jsonb),
                        final_summary = CAST(:final_summary AS jsonb),
                        updated_at = now()
                    WHERE run_id = :run_id
                    """
                ),
                {
                    "run_id": run_id,
                    "approval_status": new_status,
                    "downstream_export_enabled": new_downstream,
                    "safe_export": json.dumps(self._clean_json(safe_export)),
                    "final_summary": json.dumps(self._clean_json(final_summary)),
                },
            )

            conn.execute(
                text(
                    """
                    UPDATE audience_run_cohorts
                    SET approval_status = :approval_status
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id, "approval_status": new_status},
            )

            conn.execute(
                text(
                    """
                    INSERT INTO audience_run_approvals (
                        run_id,
                        action,
                        actor,
                        note,
                        previous_status,
                        new_status,
                        downstream_export_enabled,
                        privacy_snapshot,
                        artifacts_snapshot
                    )
                    VALUES (
                        :run_id,
                        :action,
                        :actor,
                        :note,
                        :previous_status,
                        :new_status,
                        :downstream_export_enabled,
                        CAST(:privacy_snapshot AS jsonb),
                        CAST(:artifacts_snapshot AS jsonb)
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "action": action,
                    "actor": actor,
                    "note": note,
                    "previous_status": previous_status,
                    "new_status": new_status,
                    "downstream_export_enabled": new_downstream,
                    "privacy_snapshot": json.dumps(self._clean_json(privacy_snapshot)),
                    "artifacts_snapshot": json.dumps(self._clean_json(artifacts_snapshot)),
                },
            )

            self._record_event_conn(
                conn,
                run_id=run_id,
                event_type=f"run_{action}",
                actor=actor,
                details={
                    "previous_status": previous_status,
                    "new_status": new_status,
                    "downstream_export_enabled": new_downstream,
                    "note": note,
                },
            )

        return {
            "enabled": True,
            "status": new_status,
            "run_id": run_id,
            "previous_status": previous_status,
            "approval_status": new_status,
            "downstream_export_enabled": new_downstream,
            "actor": actor,
            "note": note,
            "meta_upload_performed": False,
        }


    def _exportable_cohort_count_conn(self, conn, run_id: str) -> int:
        """
        Count persisted cohorts for approval eligibility.

        Production reason:
        Some runs persist selected/exportable cohorts in audience_run_cohorts
        even when final_summary.exported_cohorts is missing or zero.
        Approval should use actual persisted cohort rows as the source of truth.
        """
        value = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM audience_run_cohorts
                WHERE run_id = :run_id
                """
            ),
            {"run_id": run_id},
        ).scalar()

        return int(value or 0)

    def _privacy_budget_request_for_run(
        self,
        *,
        run_id: str,
        run: Dict[str, Any],
        actor: str,
        note: Optional[str],
    ) -> PrivacyBudgetRequest:
        """
        Build privacy budget request for an approved export.

        Production rule:
        Every approved export spends epsilon. If the budget is exhausted,
        approval/export must be blocked.

        Defaults are intentionally conservative and can be overridden later
        from final_summary["privacy_budget"].
        """
        final_summary = run.get("final_summary") or {}
        privacy_budget = final_summary.get("privacy_budget") or {}

        safe_export = final_summary.get("safe_export") or {}
        prompt_filter_report = final_summary.get("prompt_filter_report") or {}

        locations = prompt_filter_report.get("locations_detected") or run.get("locations_detected") or []
        poi_terms = prompt_filter_report.get("poi_terms_detected") or run.get("poi_terms_detected") or []
        dayparts = prompt_filter_report.get("dayparts_detected") or run.get("dayparts_detected") or []

        scope_parts = [
            "audience_export",
            ",".join(str(item).lower().strip() for item in locations) or "unknown_location",
            ",".join(str(item).lower().strip() for item in poi_terms) or "unknown_poi",
            ",".join(str(item).lower().strip() for item in dayparts) or "unknown_daypart",
        ]

        budget_scope = str(
            privacy_budget.get("budget_scope")
            or safe_export.get("budget_scope")
            or "|".join(scope_parts)
        )

        epsilon = float(privacy_budget.get("epsilon", safe_export.get("epsilon", 1.0)))
        delta = float(privacy_budget.get("delta", safe_export.get("delta", 1e-5)))
        sensitivity = float(privacy_budget.get("sensitivity", safe_export.get("sensitivity", 1.0)))
        max_budget = float(privacy_budget.get("max_budget", safe_export.get("max_budget", 5.0)))

        return PrivacyBudgetRequest(
            run_id=run_id,
            cohort_id=None,
            budget_scope=budget_scope,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            max_budget=max_budget,
            mechanism=str(privacy_budget.get("mechanism") or safe_export.get("mechanism") or "gaussian"),
            query_type="audience_export_approval",
            actor=actor,
            note=note,
        )

    def _db_url(self) -> str | None:
        return (
            self._explicit_db_url
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

    def _ensure_tables(self, conn) -> None:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS audience_run_history (
                    id BIGSERIAL PRIMARY KEY,
                    run_id TEXT UNIQUE NOT NULL,
                    prompt TEXT,
                    status TEXT,
                    source_mode TEXT,
                    source_rows INTEGER,
                    privacy_cohorts INTEGER,
                    prompt_selected_cohorts INTEGER,
                    exported_cohorts INTEGER,
                    approval_status TEXT,
                    downstream_export_enabled BOOLEAN,
                    filter_mode TEXT,
                    locations_detected JSONB NOT NULL DEFAULT '[]'::jsonb,
                    poi_terms_detected JSONB NOT NULL DEFAULT '[]'::jsonb,
                    dayparts_detected JSONB NOT NULL DEFAULT '[]'::jsonb,
                    coverage_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                    prompt_filter_report JSONB NOT NULL DEFAULT '{}'::jsonb,
                    sensitive_poi_privacy_risk JSONB NOT NULL DEFAULT '{}'::jsonb,
                    hybrid_retrieval_intelligence JSONB NOT NULL DEFAULT '{}'::jsonb,
                    privacy_guarantees JSONB NOT NULL DEFAULT '{}'::jsonb,
                    v2_autonomous JSONB NOT NULL DEFAULT '{}'::jsonb,
                    v2_swarm_review JSONB NOT NULL DEFAULT '{}'::jsonb,
                    safe_export JSONB NOT NULL DEFAULT '{}'::jsonb,
                    final_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
                    run_dir TEXT,
                    final_summary_path TEXT,
                    business_summary_path TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
        )

        for ddl in [
            "ALTER TABLE audience_run_history ADD COLUMN IF NOT EXISTS filter_mode TEXT;",
            "ALTER TABLE audience_run_history ADD COLUMN IF NOT EXISTS locations_detected JSONB NOT NULL DEFAULT '[]'::jsonb;",
            "ALTER TABLE audience_run_history ADD COLUMN IF NOT EXISTS poi_terms_detected JSONB NOT NULL DEFAULT '[]'::jsonb;",
            "ALTER TABLE audience_run_history ADD COLUMN IF NOT EXISTS dayparts_detected JSONB NOT NULL DEFAULT '[]'::jsonb;",
            "ALTER TABLE audience_run_history ADD COLUMN IF NOT EXISTS privacy_guarantees JSONB NOT NULL DEFAULT '{}'::jsonb;",
            "ALTER TABLE audience_run_history ADD COLUMN IF NOT EXISTS v2_autonomous JSONB NOT NULL DEFAULT '{}'::jsonb;",
            "ALTER TABLE audience_run_history ADD COLUMN IF NOT EXISTS v2_swarm_review JSONB NOT NULL DEFAULT '{}'::jsonb;",
            "ALTER TABLE audience_run_history ADD COLUMN IF NOT EXISTS run_dir TEXT;",
            "ALTER TABLE audience_run_history ADD COLUMN IF NOT EXISTS final_summary_path TEXT;",
            "ALTER TABLE audience_run_history ADD COLUMN IF NOT EXISTS business_summary_path TEXT;",
        ]:
            conn.execute(text(ddl))

        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audience_run_history_run_id ON audience_run_history(run_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audience_run_history_created_at ON audience_run_history(created_at DESC);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audience_run_history_approval ON audience_run_history(approval_status);"))

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS audience_run_cohorts (
                    id BIGSERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    export_cohort_id TEXT NOT NULL,
                    audience_name TEXT,
                    location_name TEXT,
                    primary_poi_type TEXT,
                    created_day_part TEXT,
                    lookback_bucket TEXT,
                    quality_score DOUBLE PRECISION,
                    management_quality_score DOUBLE PRECISION,
                    approval_status TEXT,
                    privacy_mode TEXT,
                    data_safety_status TEXT,
                    risk_decision TEXT,
                    risk_level TEXT,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (run_id, export_cohort_id)
                );
                """
            )
        )
        for ddl in [
            "ALTER TABLE audience_run_cohorts "
            "ADD COLUMN IF NOT EXISTS export_cohort_id TEXT;",
            "ALTER TABLE audience_run_cohorts "
            "ADD COLUMN IF NOT EXISTS lookback_bucket TEXT;",
            "ALTER TABLE audience_run_cohorts "
            "ADD COLUMN IF NOT EXISTS "
            "management_quality_score DOUBLE PRECISION;",
            "ALTER TABLE audience_run_cohorts "
            "ADD COLUMN IF NOT EXISTS privacy_mode TEXT;",
            "ALTER TABLE audience_run_cohorts "
            "ADD COLUMN IF NOT EXISTS data_safety_status TEXT;",
        ]:
            conn.execute(text(ddl))

        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS "
                "idx_audience_run_cohorts_run_id "
                "ON audience_run_cohorts(run_id);"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS "
                "idx_audience_run_cohorts_export_id "
                "ON audience_run_cohorts(export_cohort_id);"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS "
                "idx_audience_run_cohorts_location "
                "ON audience_run_cohorts(location_name);"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS "
                "idx_audience_run_cohorts_poi "
                "ON audience_run_cohorts(primary_poi_type);"
            )
        )

        # Backfill historical cohort rows that predate
        # normalized export cohort IDs.
        conn.execute(
            text(
                """
                UPDATE audience_run_cohorts
                SET export_cohort_id =
                    'legacy_' || md5(
                        run_id || ':' || id::text
                    )
                WHERE export_cohort_id IS NULL
                   OR BTRIM(export_cohort_id) = ''
                """
            )
        )

        # Keep the newest row if an older database already
        # contains a duplicate normalized cohort.
        conn.execute(
            text(
                """
                DELETE FROM audience_run_cohorts older
                USING audience_run_cohorts newer
                WHERE older.run_id = newer.run_id
                  AND older.export_cohort_id =
                      newer.export_cohort_id
                  AND older.id < newer.id
                """
            )
        )

        conn.execute(
            text(
                """
                ALTER TABLE audience_run_cohorts
                ALTER COLUMN export_cohort_id
                SET NOT NULL
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                uq_audience_run_cohorts_run_export
                ON audience_run_cohorts(
                    run_id,
                    export_cohort_id
                )
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                idx_audience_run_cohorts_filter
                ON audience_run_cohorts(
                    run_id,
                    approval_status,
                    created_day_part,
                    location_name,
                    primary_poi_type
                )
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS audience_run_artifacts (
                    id BIGSERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    artifact_type TEXT,
                    artifact_path TEXT,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audience_run_artifacts_run_id ON audience_run_artifacts(run_id);"))

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS audience_run_warnings (
                    id BIGSERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    warning_type TEXT,
                    warning_message TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audience_run_warnings_run_id ON audience_run_warnings(run_id);"))

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS audience_run_events (
                    id BIGSERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL DEFAULT 'system',
                    details JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audience_run_events_run_id ON audience_run_events(run_id);"))

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS audience_run_approvals (
                    id BIGSERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    note TEXT,
                    previous_status TEXT,
                    new_status TEXT,
                    downstream_export_enabled BOOLEAN NOT NULL DEFAULT false,
                    privacy_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
                    artifacts_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audience_run_approvals_run_id ON audience_run_approvals(run_id);"))

    def _upsert_run(self, conn, final_summary: Dict[str, Any]) -> None:
        safe_export = final_summary.get("safe_export") or {}
        prompt_filter_report = final_summary.get("prompt_filter_report") or {}
        hybrid = prompt_filter_report.get("hybrid_retrieval_intelligence") or {}
        sensitive = (
            final_summary.get("sensitive_poi_privacy_risk")
            or prompt_filter_report.get("sensitive_poi_privacy_risk")
            or {}
        )

        conn.execute(
            text(
                """
                INSERT INTO audience_run_history (
                    run_id,
                    prompt,
                    status,
                    source_mode,
                    source_rows,
                    privacy_cohorts,
                    prompt_selected_cohorts,
                    exported_cohorts,
                    approval_status,
                    downstream_export_enabled,
                    filter_mode,
                    locations_detected,
                    poi_terms_detected,
                    dayparts_detected,
                    coverage_warnings,
                    prompt_filter_report,
                    sensitive_poi_privacy_risk,
                    hybrid_retrieval_intelligence,
                    privacy_guarantees,
                    v2_autonomous,
                    v2_swarm_review,
                    safe_export,
                    final_summary,
                    run_dir,
                    final_summary_path,
                    business_summary_path,
                    updated_at
                )
                VALUES (
                    :run_id,
                    :prompt,
                    :status,
                    :source_mode,
                    :source_rows,
                    :privacy_cohorts,
                    :prompt_selected_cohorts,
                    :exported_cohorts,
                    :approval_status,
                    :downstream_export_enabled,
                    :filter_mode,
                    CAST(:locations_detected AS jsonb),
                    CAST(:poi_terms_detected AS jsonb),
                    CAST(:dayparts_detected AS jsonb),
                    CAST(:coverage_warnings AS jsonb),
                    CAST(:prompt_filter_report AS jsonb),
                    CAST(:sensitive_poi_privacy_risk AS jsonb),
                    CAST(:hybrid_retrieval_intelligence AS jsonb),
                    CAST(:privacy_guarantees AS jsonb),
                    CAST(:v2_autonomous AS jsonb),
                    CAST(:v2_swarm_review AS jsonb),
                    CAST(:safe_export AS jsonb),
                    CAST(:final_summary AS jsonb),
                    :run_dir,
                    :final_summary_path,
                    :business_summary_path,
                    now()
                )
                ON CONFLICT (run_id)
                DO UPDATE SET
                    prompt = EXCLUDED.prompt,
                    status = EXCLUDED.status,
                    source_mode = EXCLUDED.source_mode,
                    source_rows = EXCLUDED.source_rows,
                    privacy_cohorts = EXCLUDED.privacy_cohorts,
                    prompt_selected_cohorts = EXCLUDED.prompt_selected_cohorts,
                    exported_cohorts = EXCLUDED.exported_cohorts,
                    approval_status = EXCLUDED.approval_status,
                    downstream_export_enabled = EXCLUDED.downstream_export_enabled,
                    filter_mode = EXCLUDED.filter_mode,
                    locations_detected = EXCLUDED.locations_detected,
                    poi_terms_detected = EXCLUDED.poi_terms_detected,
                    dayparts_detected = EXCLUDED.dayparts_detected,
                    coverage_warnings = EXCLUDED.coverage_warnings,
                    prompt_filter_report = EXCLUDED.prompt_filter_report,
                    sensitive_poi_privacy_risk = EXCLUDED.sensitive_poi_privacy_risk,
                    hybrid_retrieval_intelligence = EXCLUDED.hybrid_retrieval_intelligence,
                    privacy_guarantees = EXCLUDED.privacy_guarantees,
                    v2_autonomous = EXCLUDED.v2_autonomous,
                    v2_swarm_review = EXCLUDED.v2_swarm_review,
                    safe_export = EXCLUDED.safe_export,
                    final_summary = EXCLUDED.final_summary,
                    run_dir = EXCLUDED.run_dir,
                    final_summary_path = EXCLUDED.final_summary_path,
                    business_summary_path = EXCLUDED.business_summary_path,
                    updated_at = now();
                """
            ),
            {
                "run_id": final_summary.get("run_id"),
                "prompt": final_summary.get("prompt"),
                "status": final_summary.get("status"),
                "source_mode": final_summary.get("source_mode"),
                "source_rows": self._safe_int(
                    final_summary.get("source_rows")
                    or ((final_summary.get("v2_autonomous") or {}).get("data_freshness") or {}).get(
                        "source_rows_checked"
                    )
                ),
                "privacy_cohorts": self._safe_int(final_summary.get("privacy_cohorts")),
                "prompt_selected_cohorts": self._safe_int(final_summary.get("prompt_selected_cohorts")),
                "exported_cohorts": self._safe_int(safe_export.get("exported_cohorts")),
                "approval_status": safe_export.get("approval_status"),
                "downstream_export_enabled": bool(safe_export.get("downstream_export_enabled", False)),
                "filter_mode": prompt_filter_report.get("filter_mode"),
                "locations_detected": json.dumps(prompt_filter_report.get("locations_detected") or []),
                "poi_terms_detected": json.dumps(prompt_filter_report.get("poi_terms_detected") or []),
                "dayparts_detected": json.dumps(prompt_filter_report.get("dayparts_detected") or []),
                "coverage_warnings": json.dumps(final_summary.get("coverage_warnings") or []),
                "prompt_filter_report": json.dumps(self._clean_json(prompt_filter_report)),
                "sensitive_poi_privacy_risk": json.dumps(self._clean_json(sensitive)),
                "hybrid_retrieval_intelligence": json.dumps(self._clean_json(hybrid)),
                "privacy_guarantees": json.dumps(self._clean_json(final_summary.get("privacy_guarantees") or {})),
                "v2_autonomous": json.dumps(self._clean_json(final_summary.get("v2_autonomous") or {})),
                "v2_swarm_review": json.dumps(self._clean_json(final_summary.get("v2_swarm_review") or {})),
                "safe_export": json.dumps(self._clean_json(safe_export)),
                "final_summary": json.dumps(self._clean_json(final_summary)),
                "run_dir": final_summary.get("run_dir"),
                "final_summary_path": final_summary.get("final_summary_path"),
                "business_summary_path": final_summary.get("business_summary_path"),
            },
        )

    def _replace_cohorts(
        self,
        conn,
        run_id: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        # Serialize concurrent persistence attempts for
        # the same run.
        conn.execute(
            text(
                """
                SELECT pg_advisory_xact_lock(
                    hashtext(:run_id)
                )
                """
            ),
            {"run_id": run_id},
        )

        conn.execute(
            text(
                """
                DELETE FROM audience_run_cohorts
                WHERE run_id = :run_id
                """
            ),
            {"run_id": run_id},
        )

        for row in rows:
            export_cohort_id = str(
                row.get("export_cohort_id") or ""
            ).strip()

            if not export_cohort_id:
                raise ValueError(
                    "Normalized cohort is missing "
                    "export_cohort_id."
                )

            quality_value = row.get("quality_score")

            if quality_value is None:
                quality_value = row.get("score")

            if quality_value is None:
                quality_value = row.get(
                    "management_quality_score"
                )

            management_quality = row.get(
                "management_quality_score"
            )

            approval_status = (
                row.get(
                    "_normalized_approval_status"
                )
                or row.get("export_status")
                or row.get("approval_status")
                or "pending_approval"
            )

            metadata = {
                key: value
                for key, value in row.items()
                if not str(key).startswith("_")
            }

            normalized_source = row.get(
                "_normalized_source"
            )

            if normalized_source:
                metadata[
                    "normalized_source"
                ] = normalized_source

            params = {
                "run_id": run_id,
                "export_cohort_id": export_cohort_id,
                "audience_name": row.get(
                    "audience_name"
                ),
                "location_name": row.get(
                    "location_name"
                ),
                "primary_poi_type": row.get(
                    "primary_poi_type"
                ),
                "created_day_part": row.get(
                    "created_day_part"
                ),
                "lookback_bucket": row.get(
                    "lookback_bucket"
                ),
                "quality_score": self._safe_float(
                    quality_value
                ),
                "management_quality_score": (
                    self._safe_float(
                        management_quality
                    )
                ),
                "approval_status": approval_status,
                "privacy_mode": row.get(
                    "privacy_mode"
                ),
                "data_safety_status": row.get(
                    "data_safety_status"
                ),
                "risk_decision": row.get(
                    "risk_decision"
                ),
                "risk_level": row.get(
                    "risk_level"
                ),
                "metadata": json.dumps(
                    self._clean_json(metadata)
                ),
            }

            conn.execute(
                text(
                    """
                    INSERT INTO audience_run_cohorts (
                        run_id,
                        export_cohort_id,
                        audience_name,
                        location_name,
                        primary_poi_type,
                        created_day_part,
                        lookback_bucket,
                        quality_score,
                        management_quality_score,
                        approval_status,
                        privacy_mode,
                        data_safety_status,
                        risk_decision,
                        risk_level,
                        metadata
                    )
                    VALUES (
                        :run_id,
                        :export_cohort_id,
                        :audience_name,
                        :location_name,
                        :primary_poi_type,
                        :created_day_part,
                        :lookback_bucket,
                        :quality_score,
                        :management_quality_score,
                        :approval_status,
                        :privacy_mode,
                        :data_safety_status,
                        :risk_decision,
                        :risk_level,
                        CAST(:metadata AS jsonb)
                    )
                    ON CONFLICT (
                        run_id,
                        export_cohort_id
                    )
                    DO UPDATE SET
                        audience_name = EXCLUDED.audience_name,
                        location_name = EXCLUDED.location_name,
                        primary_poi_type = EXCLUDED.primary_poi_type,
                        created_day_part = EXCLUDED.created_day_part,
                        lookback_bucket = EXCLUDED.lookback_bucket,
                        quality_score = EXCLUDED.quality_score,
                        management_quality_score =
                            EXCLUDED.management_quality_score,
                        approval_status =
                            EXCLUDED.approval_status,
                        privacy_mode = EXCLUDED.privacy_mode,
                        data_safety_status =
                            EXCLUDED.data_safety_status,
                        risk_decision = EXCLUDED.risk_decision,
                        risk_level = EXCLUDED.risk_level,
                        metadata = EXCLUDED.metadata,
                        created_at = now()
                    """
                ),
                params,
            )

    def _replace_artifacts(self, conn, run_id: str, rows: List[Dict[str, Any]]) -> None:
        conn.execute(text("DELETE FROM audience_run_artifacts WHERE run_id = :run_id"), {"run_id": run_id})

        for row in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO audience_run_artifacts (
                        run_id,
                        artifact_type,
                        artifact_path,
                        metadata
                    )
                    VALUES (
                        :run_id,
                        :artifact_type,
                        :artifact_path,
                        CAST(:metadata AS jsonb)
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "artifact_type": row.get("artifact_type"),
                    "artifact_path": row.get("artifact_path"),
                    "metadata": json.dumps(self._clean_json(row.get("metadata") or {})),
                },
            )

    def _replace_warnings(self, conn, run_id: str, rows: List[Dict[str, Any]]) -> None:
        conn.execute(text("DELETE FROM audience_run_warnings WHERE run_id = :run_id"), {"run_id": run_id})

        for row in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO audience_run_warnings (
                        run_id,
                        warning_type,
                        warning_message,
                        metadata
                    )
                    VALUES (
                        :run_id,
                        :warning_type,
                        :warning_message,
                        CAST(:metadata AS jsonb)
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "warning_type": row.get("warning_type") or "coverage_warning",
                    "warning_message": row.get("warning_message"),
                    "metadata": json.dumps(self._clean_json(row.get("metadata") or {})),
                },
            )

    def _record_event_conn(
        self,
        conn,
        *,
        run_id: str,
        event_type: str,
        actor: str,
        details: Dict[str, Any],
    ) -> None:
        conn.execute(
            text(
                """
                INSERT INTO audience_run_events (
                    run_id,
                    event_type,
                    actor,
                    details
                )
                VALUES (
                    :run_id,
                    :event_type,
                    :actor,
                    CAST(:details AS jsonb)
                )
                """
            ),
            {
                "run_id": run_id,
                "event_type": event_type,
                "actor": actor,
                "details": json.dumps(self._clean_json(details)),
            },
        )

    def _build_cohort_rows(
        self,
        *,
        run_id: str,
        final_summary: Dict[str, Any],
        selected_cohorts: (
            pd.DataFrame
            | List[Dict[str, Any]]
            | None
        ) = None,
    ) -> List[Dict[str, Any]]:
        """
        Build deterministic, normalized cohort records.

        Source priority:
        1. safe_export.package.cohorts
        2. selected_cohorts
        3. legacy final_summary safe-export rows
        4. legacy local CSV artifact

        Duplicate export cohort IDs are collapsed before
        persistence. The highest-quality record wins.
        """
        safe_export = (
            final_summary.get("safe_export") or {}
        )
        package = safe_export.get("package") or {}

        rows: List[Dict[str, Any]] = []
        source = "none"

        package_cohorts = package.get("cohorts") or []

        if isinstance(package_cohorts, list):
            rows = [
                dict(item)
                for item in package_cohorts
                if isinstance(item, dict)
            ]

            if rows:
                source = "safe_export.package.cohorts"

        if not rows and selected_cohorts is not None:
            if isinstance(selected_cohorts, pd.DataFrame):
                rows = selected_cohorts.to_dict(
                    orient="records"
                )
            elif isinstance(selected_cohorts, list):
                rows = [
                    dict(item)
                    for item in selected_cohorts
                    if isinstance(item, dict)
                ]

            if rows:
                source = "selected_cohorts"

        if not rows:
            legacy_rows = (
                final_summary.get(
                    "safe_export_cohorts"
                )
                or []
            )

            if isinstance(legacy_rows, list):
                rows = [
                    dict(item)
                    for item in legacy_rows
                    if isinstance(item, dict)
                ]

                if rows:
                    source = (
                        "final_summary."
                        "safe_export_cohorts"
                    )

        if not rows:
            outputs = safe_export.get("outputs") or {}
            path_value = str(
                outputs.get("safe_export_cohorts")
                or ""
            ).strip()

            if path_value and "://" not in path_value:
                local_path = Path(path_value)

                if local_path.exists():
                    rows = pd.read_csv(
                        local_path
                    ).to_dict(orient="records")
                    source = (
                        "safe_export.outputs."
                        "safe_export_cohorts"
                    )

        authoritative_approval = str(
            safe_export.get("approval_status")
            or final_summary.get("approval_status")
            or "pending_approval"
        ).strip()

        deduplicated: Dict[str, Dict[str, Any]] = {}
        insertion_order: List[str] = []

        for item in rows:
            clean_item = self._clean_json(item)

            if not isinstance(clean_item, dict):
                continue

            export_cohort_id = str(
                clean_item.get("export_cohort_id")
                or clean_item.get("cohort_id")
                or ""
            ).strip()

            if not export_cohort_id:
                identity = {
                    "audience_name": clean_item.get(
                        "audience_name"
                    ),
                    "location_name": clean_item.get(
                        "location_name"
                    ),
                    "primary_poi_type": clean_item.get(
                        "primary_poi_type"
                    ),
                    "created_day_part": clean_item.get(
                        "created_day_part"
                    ),
                    "lookback_bucket": clean_item.get(
                        "lookback_bucket"
                    ),
                    "cluster_id": clean_item.get(
                        "cluster_id"
                    ),
                }

                canonical = json.dumps(
                    self._clean_json(identity),
                    sort_keys=True,
                    separators=(",", ":"),
                )

                digest = hashlib.sha256(
                    f"{run_id}|{canonical}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:16]

                export_cohort_id = (
                    f"punk_audience_{digest}"
                )

            clean_item[
                "export_cohort_id"
            ] = export_cohort_id
            clean_item[
                "_normalized_source"
            ] = source
            clean_item[
                "_normalized_approval_status"
            ] = authoritative_approval

            existing = deduplicated.get(
                export_cohort_id
            )

            if existing is None:
                deduplicated[
                    export_cohort_id
                ] = clean_item
                insertion_order.append(
                    export_cohort_id
                )
                continue

            existing_quality = self._safe_float(
                existing.get(
                    "management_quality_score"
                )
            )

            if existing_quality is None:
                existing_quality = self._safe_float(
                    existing.get("quality_score")
                )

            incoming_quality = self._safe_float(
                clean_item.get(
                    "management_quality_score"
                )
            )

            if incoming_quality is None:
                incoming_quality = self._safe_float(
                    clean_item.get("quality_score")
                )

            if (
                incoming_quality is not None
                and (
                    existing_quality is None
                    or incoming_quality
                    > existing_quality
                )
            ):
                deduplicated[
                    export_cohort_id
                ] = clean_item

        normalized = [
            deduplicated[export_id]
            for export_id in insertion_order
        ]

        return self._apply_sensitive_risk_to_cohort_rows(
            rows=normalized,
            final_summary=final_summary,
        )

    def _apply_sensitive_risk_to_cohort_rows(
        self,
        *,
        rows: List[Dict[str, Any]],
        final_summary: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        sensitive = (
            final_summary.get("sensitive_poi_privacy_risk")
            or (final_summary.get("prompt_filter_report") or {}).get("sensitive_poi_privacy_risk")
            or {}
        )

        assessed = sensitive.get("assessed_audiences") or []
        if not rows or not assessed:
            return rows

        def norm(value: Any) -> str:
            return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

        def key(item: Dict[str, Any]) -> tuple[str, str, str]:
            return (
                norm(item.get("location_name")),
                norm(item.get("primary_poi_type")),
                norm(item.get("created_day_part")),
            )

        risk_by_key: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        risk_by_loose_key: Dict[tuple[str, str], Dict[str, Any]] = {}

        for risk in assessed:
            if not isinstance(risk, dict):
                continue

            full_key = key(risk)
            loose_key = (full_key[0], full_key[1])

            risk_by_key[full_key] = risk
            risk_by_loose_key[loose_key] = risk

        enriched: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                enriched.append(row)
                continue

            clean_row = dict(row)
            row_key = key(clean_row)
            loose_key = (row_key[0], row_key[1])
            risk = risk_by_key.get(row_key) or risk_by_loose_key.get(loose_key)

            if risk:
                clean_row.setdefault(
                    "risk_decision",
                    risk.get("decision") or risk.get("risk_decision"),
                )
                clean_row.setdefault("risk_level", risk.get("risk_level"))
                clean_row.setdefault("sensitive_poi_reason", risk.get("reason"))

            enriched.append(clean_row)

        return enriched

    def _build_artifact_rows(self, *, run_id: str, final_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        safe_export = final_summary.get("safe_export") or {}
        outputs = safe_export.get("outputs") or {}

        for artifact_type, artifact_path in outputs.items():
            if artifact_path:
                rows.append(
                    {
                        "run_id": run_id,
                        "artifact_type": str(artifact_type),
                        "artifact_path": str(artifact_path),
                        "metadata": {"source": "safe_export.outputs"},
                    }
                )

        for artifact_type in ["run_dir", "final_summary_path", "business_summary_path"]:
            artifact_path = final_summary.get(artifact_type)
            if artifact_path:
                rows.append(
                    {
                        "run_id": run_id,
                        "artifact_type": artifact_type,
                        "artifact_path": str(artifact_path),
                        "metadata": {"source": "final_summary"},
                    }
                )

        return rows

    def _build_warning_rows(self, *, run_id: str, final_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for warning in final_summary.get("coverage_warnings") or []:
            rows.append(
                {
                    "run_id": run_id,
                    "warning_type": "coverage_warning",
                    "warning_message": str(warning),
                    "metadata": {},
                }
            )

        prompt_filter_report = final_summary.get("prompt_filter_report") or {}
        for warning in prompt_filter_report.get("coverage_warnings") or []:
            message = str(warning)
            if not any(row["warning_message"] == message for row in rows):
                rows.append(
                    {
                        "run_id": run_id,
                        "warning_type": "prompt_filter_warning",
                        "warning_message": message,
                        "metadata": {"source": "prompt_filter_report"},
                    }
                )

        return rows

    def _row_to_dict(self, row) -> Dict[str, Any]:
        raw = dict(row._mapping)
        return self._clean_json(raw)

    def _safe_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return int(value)
        except Exception:
            return None

    def _safe_float(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            value = float(value)
            if not math.isfinite(value):
                return None
            return value
        except Exception:
            return None

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _clean_json(self, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, dict):
            return {str(k): self._clean_json(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._clean_json(v) for v in value]

        if isinstance(value, tuple):
            return [self._clean_json(v) for v in value]

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return value

        if hasattr(value, "item"):
            try:
                return self._clean_json(value.item())
            except Exception:
                pass

        return value
