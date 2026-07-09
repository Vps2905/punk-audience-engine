from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from sqlalchemy import create_engine, text


class AudienceRunHistoryService:
    """
    Persists privacy-safe audience run history into Postgres.

    Stores:
    - run-level summary JSONB
    - selected cohort metadata
    - artifact paths

    No raw MAIDs, hashed IDs, raw observations, raw lat/lng, email, or phone.
    """

    def persist_run(
        self,
        *,
        final_summary: Dict[str, Any],
        selected_cohorts: pd.DataFrame | List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        db_url = self._db_url()

        if not db_url:
            return {
                "enabled": False,
                "status": "skipped",
                "reason": "no_database_url",
            }

        run_id = str(final_summary.get("run_id") or "").strip()
        if not run_id:
            return {
                "enabled": False,
                "status": "skipped",
                "reason": "missing_run_id",
            }

        clean_summary = self._clean_json(final_summary)
        cohort_rows = self._build_cohort_rows(
            run_id=run_id,
            final_summary=clean_summary,
            selected_cohorts=selected_cohorts,
        )
        artifact_rows = self._build_artifact_rows(
            run_id=run_id,
            final_summary=clean_summary,
        )

        engine = create_engine(db_url)

        with engine.begin() as conn:
            self._ensure_tables(conn)
            self._upsert_run(conn, clean_summary)
            self._replace_cohorts(conn, run_id, cohort_rows)
            self._replace_artifacts(conn, run_id, artifact_rows)

        return {
            "enabled": True,
            "status": "persisted",
            "run_id": run_id,
            "tables": [
                "audience_run_history",
                "audience_run_cohorts",
                "audience_run_artifacts",
            ],
            "cohort_rows": len(cohort_rows),
            "artifact_rows": len(artifact_rows),
        }

    def _db_url(self) -> str | None:
        return (
            os.getenv("AUDIENCE_HISTORY_DATABASE_URL")
            or os.getenv("ECHO_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or os.getenv("POSTGRES_URL")
            or os.getenv("POSTGRES_DATABASE_URL")
            or os.getenv("SUPABASE_DB_URL")
            or os.getenv("DB_URL")
        )

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
                    coverage_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                    prompt_filter_report JSONB NOT NULL DEFAULT '{}'::jsonb,
                    sensitive_poi_privacy_risk JSONB NOT NULL DEFAULT '{}'::jsonb,
                    hybrid_retrieval_intelligence JSONB NOT NULL DEFAULT '{}'::jsonb,
                    safe_export JSONB NOT NULL DEFAULT '{}'::jsonb,
                    final_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS audience_run_cohorts (
                    id BIGSERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    audience_name TEXT,
                    location_name TEXT,
                    primary_poi_type TEXT,
                    created_day_part TEXT,
                    quality_score DOUBLE PRECISION,
                    approval_status TEXT,
                    risk_decision TEXT,
                    risk_level TEXT,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_audience_run_cohorts_run_id
                ON audience_run_cohorts(run_id);
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

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_audience_run_artifacts_run_id
                ON audience_run_artifacts(run_id);
                """
            )
        )

    def _upsert_run(self, conn, final_summary: Dict[str, Any]) -> None:
        safe_export = final_summary.get("safe_export") or {}
        prompt_filter_report = final_summary.get("prompt_filter_report") or {}
        hybrid = prompt_filter_report.get("hybrid_retrieval_intelligence") or {}
        sensitive = final_summary.get("sensitive_poi_privacy_risk") or prompt_filter_report.get("sensitive_poi_privacy_risk") or {}

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
                    coverage_warnings,
                    prompt_filter_report,
                    sensitive_poi_privacy_risk,
                    hybrid_retrieval_intelligence,
                    safe_export,
                    final_summary,
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
                    CAST(:coverage_warnings AS jsonb),
                    CAST(:prompt_filter_report AS jsonb),
                    CAST(:sensitive_poi_privacy_risk AS jsonb),
                    CAST(:hybrid_retrieval_intelligence AS jsonb),
                    CAST(:safe_export AS jsonb),
                    CAST(:final_summary AS jsonb),
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
                    coverage_warnings = EXCLUDED.coverage_warnings,
                    prompt_filter_report = EXCLUDED.prompt_filter_report,
                    sensitive_poi_privacy_risk = EXCLUDED.sensitive_poi_privacy_risk,
                    hybrid_retrieval_intelligence = EXCLUDED.hybrid_retrieval_intelligence,
                    safe_export = EXCLUDED.safe_export,
                    final_summary = EXCLUDED.final_summary,
                    updated_at = now();
                """
            ),
            {
                "run_id": final_summary.get("run_id"),
                "prompt": final_summary.get("prompt"),
                "status": final_summary.get("status"),
                "source_mode": final_summary.get("source_mode"),
                "source_rows": self._safe_int(final_summary.get("source_rows")),
                "privacy_cohorts": self._safe_int(final_summary.get("privacy_cohorts")),
                "prompt_selected_cohorts": self._safe_int(final_summary.get("prompt_selected_cohorts")),
                "exported_cohorts": self._safe_int(safe_export.get("exported_cohorts")),
                "approval_status": safe_export.get("approval_status"),
                "downstream_export_enabled": bool(safe_export.get("downstream_export_enabled", False)),
                "coverage_warnings": json.dumps(final_summary.get("coverage_warnings") or []),
                "prompt_filter_report": json.dumps(prompt_filter_report),
                "sensitive_poi_privacy_risk": json.dumps(sensitive),
                "hybrid_retrieval_intelligence": json.dumps(hybrid),
                "safe_export": json.dumps(safe_export),
                "final_summary": json.dumps(final_summary),
            },
        )

    def _replace_cohorts(self, conn, run_id: str, rows: List[Dict[str, Any]]) -> None:
        conn.execute(text("DELETE FROM audience_run_cohorts WHERE run_id = :run_id"), {"run_id": run_id})

        for row in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO audience_run_cohorts (
                        run_id,
                        audience_name,
                        location_name,
                        primary_poi_type,
                        created_day_part,
                        quality_score,
                        approval_status,
                        risk_decision,
                        risk_level,
                        metadata
                    )
                    VALUES (
                        :run_id,
                        :audience_name,
                        :location_name,
                        :primary_poi_type,
                        :created_day_part,
                        :quality_score,
                        :approval_status,
                        :risk_decision,
                        :risk_level,
                        CAST(:metadata AS jsonb)
                    );
                    """
                ),
                {
                    "run_id": run_id,
                    "audience_name": row.get("audience_name"),
                    "location_name": row.get("location_name"),
                    "primary_poi_type": row.get("primary_poi_type"),
                    "created_day_part": row.get("created_day_part"),
                    "quality_score": self._safe_float(row.get("quality_score")),
                    "approval_status": row.get("approval_status"),
                    "risk_decision": row.get("risk_decision"),
                    "risk_level": row.get("risk_level"),
                    "metadata": json.dumps(row.get("metadata") or {}),
                },
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
                    );
                    """
                ),
                {
                    "run_id": run_id,
                    "artifact_type": row.get("artifact_type"),
                    "artifact_path": row.get("artifact_path"),
                    "metadata": json.dumps(row.get("metadata") or {}),
                },
            )

    def _build_cohort_rows(
        self,
        *,
        run_id: str,
        final_summary: Dict[str, Any],
        selected_cohorts: pd.DataFrame | List[Dict[str, Any]] | None,
    ) -> List[Dict[str, Any]]:
        rows = self._records(selected_cohorts)
        risk_by_key = self._risk_index(final_summary)

        output = []

        for row in rows:
            audience_name = str(row.get("audience_name") or self._build_audience_name(row)).strip()
            location_name = str(row.get("location_name") or "").strip()
            poi = str(row.get("primary_poi_type") or "").strip()
            daypart = str(row.get("created_day_part") or "").strip()

            risk = risk_by_key.get(self._risk_key(audience_name, location_name, poi, daypart), {})

            output.append(
                {
                    "run_id": run_id,
                    "audience_name": audience_name,
                    "location_name": location_name,
                    "primary_poi_type": poi,
                    "created_day_part": daypart,
                    "quality_score": self._first_float(
                        row,
                        ["quality_score", "management_quality", "cohort_quality_component"],
                    ),
                    "approval_status": str(row.get("approval_status") or "").strip() or None,
                    "risk_decision": risk.get("decision"),
                    "risk_level": risk.get("risk_level"),
                    "metadata": self._clean_json(row),
                }
            )

        return output

    def _build_artifact_rows(self, *, run_id: str, final_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        def walk(value: Any, path: str = "") -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    key_path = f"{path}.{key}" if path else str(key)
                    if isinstance(item, str) and self._looks_like_artifact_path(key, item):
                        rows.append(
                            {
                                "run_id": run_id,
                                "artifact_type": key,
                                "artifact_path": item,
                                "metadata": {"json_path": key_path},
                            }
                        )
                    else:
                        walk(item, key_path)

            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    walk(item, f"{path}[{idx}]")

        walk(final_summary)
        deduped = {}
        for row in rows:
            key = (row["artifact_type"], row["artifact_path"])
            deduped[key] = row

        return list(deduped.values())

    def _risk_index(self, final_summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        sensitive = final_summary.get("sensitive_poi_privacy_risk") or {}
        assessed = sensitive.get("assessed_audiences") or []

        output = {}

        for row in assessed:
            audience_name = str(row.get("audience_name") or "").strip()
            location_name = str(row.get("location_name") or "").strip()
            poi = str(row.get("primary_poi_type") or "").strip()
            daypart = str(row.get("created_day_part") or "").strip()

            output[self._risk_key(audience_name, location_name, poi, daypart)] = row

        return output

    def _risk_key(self, audience_name: str, location_name: str, poi: str, daypart: str) -> str:
        return "|".join(
            [
                self._norm(audience_name),
                self._norm(location_name),
                self._norm(poi),
                self._norm(daypart),
            ]
        )

    def _records(self, value: pd.DataFrame | List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
        if value is None:
            return []

        if isinstance(value, pd.DataFrame):
            return self._clean_json(value.to_dict(orient="records"))

        if isinstance(value, list):
            return self._clean_json(value)

        return []

    def _build_audience_name(self, row: Dict[str, Any]) -> str:
        parts = [
            row.get("primary_poi_type"),
            row.get("created_day_part"),
            row.get("location_name"),
        ]
        return " - ".join([str(item).strip() for item in parts if str(item or "").strip()])

    def _looks_like_artifact_path(self, key: str, value: str) -> bool:
        key = str(key or "").lower()
        value = str(value or "")

        return (
            key.endswith("_path")
            or key.endswith("_manifest")
            or key.endswith("_cohorts")
            or key.endswith("_lookalikes")
            or key.endswith("_payload")
            or value.startswith("data/")
            or value.endswith((".json", ".csv", ".md", ".npy"))
        )

    def _first_float(self, row: Dict[str, Any], keys: Iterable[str]) -> float | None:
        for key in keys:
            value = self._safe_float(row.get(key))
            if value is not None:
                return value
        return None

    def _safe_int(self, value: Any) -> int:
        try:
            if value is None:
                return 0
            if isinstance(value, float) and math.isnan(value):
                return 0
            return int(value)
        except Exception:
            return 0

    def _safe_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            result = float(value)
            if math.isnan(result) or math.isinf(result):
                return None
            return result
        except Exception:
            return None

    def _clean_json(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._clean_json(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._clean_json(item) for item in value]

        if isinstance(value, tuple):
            return [self._clean_json(item) for item in value]

        if isinstance(value, Path):
            return str(value)

        if hasattr(value, "item"):
            try:
                return self._clean_json(value.item())
            except Exception:
                pass

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None

        if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
            return None

        return value

    def _norm(self, value: Any) -> str:
        return str(value or "").strip().lower().replace(" ", "_")
