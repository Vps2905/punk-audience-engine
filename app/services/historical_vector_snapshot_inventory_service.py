from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


class HistoricalVectorSnapshotInventoryService:
    """
    Inspect legacy privacy-safe vector snapshots without returning row data.

    Only model metadata and aggregate validation statistics are selected.
    Embeddings, traits, cohort metadata and raw source records never leave the
    database. The transaction is explicitly read-only and always rolled back.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: Engine | None = None,
        expected_dimension: int | None = None,
    ) -> None:
        self._database_url = database_url
        self._engine_override = engine
        self._expected_dimension = int(
            expected_dimension
            or os.getenv("PGVECTOR_DIMENSION", "384")
        )

    def inventory(self, *, limit: int = 100) -> dict[str, Any]:
        bounded_limit = max(1, min(int(limit), 1000))
        database_url = self._resolved_database_url()
        if self._engine_override is None and not database_url:
            return self._blocked_report(
                "historical_source_database_not_configured"
            )

        engine: Engine | None = None
        owns_engine = self._engine_override is None
        try:
            engine = self._engine_override or create_engine(
                self._normalize_url(database_url),
                pool_pre_ping=True,
            )
            if engine.dialect.name != "postgresql":
                return self._blocked_report("postgresql_required")

            with engine.connect() as connection:
                transaction = connection.begin()
                try:
                    connection.exec_driver_sql(
                        "SET TRANSACTION READ ONLY"
                    )
                    read_only = connection.execute(
                        text(
                            """
                            SELECT current_setting(
                                'transaction_read_only'
                            ) AS transaction_read_only
                            """
                        )
                    ).mappings().one()
                    if str(
                        read_only["transaction_read_only"]
                    ).lower() != "on":
                        return self._blocked_report(
                            "read_only_transaction_not_verified"
                        )

                    tables = connection.execute(
                        text(
                            """
                            SELECT
                                (
                                    to_regclass(
                                        'public.audience_vector_models'
                                    ) IS NOT NULL
                                ) AS models_present,
                                (
                                    to_regclass(
                                        'public.audience_vectors'
                                    ) IS NOT NULL
                                ) AS vectors_present,
                                (
                                    to_regclass(
                                        'public.maid_extractions'
                                    ) IS NOT NULL
                                ) AS source_events_present
                            """
                        )
                    ).mappings().one()
                    if not (
                        tables["models_present"]
                        and tables["vectors_present"]
                    ):
                        return self._blocked_report(
                            "legacy_vector_tables_not_found",
                            read_only_verified=True,
                        )

                    source_latest_at = None
                    if tables["source_events_present"]:
                        source_row = connection.execute(
                            text(
                                """
                                SELECT MAX(created_at) AS source_latest_at
                                FROM public.maid_extractions
                                """
                            )
                        ).mappings().one()
                        source_latest_at = source_row["source_latest_at"]

                    rows = connection.execute(
                        text(
                            """
                            WITH vector_stats AS (
                                SELECT
                                    job_id,
                                    COUNT(*) AS stored_vector_count,
                                    COUNT(
                                        DISTINCT vector_index
                                    ) AS distinct_vector_indices,
                                    MIN(vector_index) AS min_vector_index,
                                    MAX(vector_index) AS max_vector_index,
                                    COUNT(*) FILTER (
                                        WHERE cardinality(embedding)
                                            <> :expected_dimension
                                    ) AS incompatible_embedding_count
                                FROM public.audience_vectors
                                GROUP BY job_id
                            )
                            SELECT
                                model.job_id,
                                model.vector_count
                                    AS declared_vector_count,
                                model.vector_dimension,
                                model.updated_at,
                                COALESCE(
                                    stats.stored_vector_count,
                                    0
                                ) AS stored_vector_count,
                                COALESCE(
                                    stats.distinct_vector_indices,
                                    0
                                ) AS distinct_vector_indices,
                                stats.min_vector_index,
                                stats.max_vector_index,
                                COALESCE(
                                    stats.incompatible_embedding_count,
                                    0
                                ) AS incompatible_embedding_count
                            FROM public.audience_vector_models model
                            LEFT JOIN vector_stats stats
                                ON stats.job_id = model.job_id
                            ORDER BY
                                model.updated_at DESC,
                                model.job_id ASC
                            LIMIT :limit
                            """
                        ),
                        {
                            "expected_dimension": (
                                self._expected_dimension
                            ),
                            "limit": bounded_limit,
                        },
                    ).mappings().all()
                finally:
                    transaction.rollback()
        except (SQLAlchemyError, TypeError, ValueError):
            return self._blocked_report(
                "historical_snapshot_inventory_failed"
            )
        finally:
            if owns_engine and engine is not None:
                engine.dispose()

        return self._build_report(
            rows,
            source_latest_at=source_latest_at,
            limit=bounded_limit,
        )

    def _build_report(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        source_latest_at: datetime | None,
        limit: int,
    ) -> dict[str, Any]:
        snapshots = [
            self._snapshot_report(row, position=index)
            for index, row in enumerate(rows)
        ]
        compatible = [
            snapshot
            for snapshot in snapshots
            if snapshot["compatible_complete_snapshot"]
        ]
        largest = sorted(
            compatible,
            key=lambda item: (
                -int(item["stored_vector_count"]),
                int(item["source_order"]),
                str(item["job_id"]),
            ),
        )

        return {
            "status": (
                "inventory_ready" if snapshots else "no_snapshots_found"
            ),
            "read_only": True,
            "read_only_transaction_verified": True,
            "credentials_exposed": False,
            "raw_identifiers_read": False,
            "embeddings_returned": False,
            "trait_or_metadata_rows_returned": False,
            "expected_vector_dimension": self._expected_dimension,
            "source_latest_timestamp": (
                source_latest_at.isoformat()
                if source_latest_at is not None
                else None
            ),
            "snapshot_count_returned": len(snapshots),
            "query_limit": limit,
            "latest_snapshot_job_id": (
                snapshots[0]["job_id"] if snapshots else None
            ),
            "latest_compatible_snapshot_job_id": (
                compatible[0]["job_id"] if compatible else None
            ),
            "largest_compatible_snapshot_job_id": (
                largest[0]["job_id"] if largest else None
            ),
            "compatible_complete_snapshot_count": len(compatible),
            "selection_requires_operator_review": bool(compatible),
            "automatic_selection_performed": False,
            "snapshots": snapshots,
        }

    def _snapshot_report(
        self,
        row: Mapping[str, Any],
        *,
        position: int,
    ) -> dict[str, Any]:
        declared_count = int(row.get("declared_vector_count") or 0)
        stored_count = int(row.get("stored_vector_count") or 0)
        distinct_indices = int(row.get("distinct_vector_indices") or 0)
        dimension = int(row.get("vector_dimension") or 0)
        incompatible_embeddings = int(
            row.get("incompatible_embedding_count") or 0
        )
        min_index = row.get("min_vector_index")
        max_index = row.get("max_vector_index")

        issues: list[str] = []
        if stored_count < 1:
            issues.append("empty_snapshot")
        if declared_count != stored_count:
            issues.append("declared_stored_count_mismatch")
        if dimension != self._expected_dimension:
            issues.append("model_dimension_incompatible")
        if incompatible_embeddings:
            issues.append("embedding_dimension_incompatible")
        indices_contiguous = bool(
            stored_count > 0
            and distinct_indices == stored_count
            and int(min_index) == 0
            and int(max_index) == stored_count - 1
        )
        if stored_count and not indices_contiguous:
            issues.append("vector_indices_not_contiguous")

        updated_at = row.get("updated_at")
        return {
            "job_id": str(row.get("job_id") or ""),
            "source_order": position,
            "updated_at": (
                updated_at.isoformat()
                if isinstance(updated_at, datetime)
                else str(updated_at or "") or None
            ),
            "declared_vector_count": declared_count,
            "stored_vector_count": stored_count,
            "distinct_vector_indices": distinct_indices,
            "min_vector_index": (
                int(min_index) if min_index is not None else None
            ),
            "max_vector_index": (
                int(max_index) if max_index is not None else None
            ),
            "vector_dimension": dimension,
            "incompatible_embedding_count": incompatible_embeddings,
            "vector_indices_contiguous": indices_contiguous,
            "compatible_complete_snapshot": not issues,
            "issues": issues,
        }

    def _blocked_report(
        self,
        reason_code: str,
        *,
        read_only_verified: bool = False,
    ) -> dict[str, Any]:
        return {
            "status": "blocked",
            "reason_code": reason_code,
            "read_only": True,
            "read_only_transaction_verified": read_only_verified,
            "credentials_exposed": False,
            "raw_identifiers_read": False,
            "embeddings_returned": False,
            "trait_or_metadata_rows_returned": False,
            "automatic_selection_performed": False,
            "snapshots": [],
        }

    def _resolved_database_url(self) -> str:
        return str(
            self._database_url
            or os.getenv("ECHO_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or ""
        ).strip()

    def _normalize_url(self, value: str) -> str:
        if value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://") :]
        return value
