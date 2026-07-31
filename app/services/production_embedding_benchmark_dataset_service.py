from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.models.audience_feature_contracts import (
    normalize_taxonomy_value,
    stable_digest,
)
from app.models.embedding_benchmark_authoring_contracts import (
    EmbeddingBenchmarkCaseCatalog,
    EmbeddingBenchmarkDocumentCatalog,
)
from app.models.embedding_benchmark_contracts import (
    EmbeddingBenchmarkDataset,
    EmbeddingBenchmarkDocument,
    ProductionEmbeddingBenchmarkPolicy,
)
from app.services.production_embedding_benchmark_service import (
    REQUIRED_BENCHMARK_COVERAGE,
    benchmark_dataset_coverage,
)


class ProductionEmbeddingBenchmarkDatasetBuilderService:
    """
    Compile reviewed gold labels and safe documents into one immutable dataset.

    The builder validates labels against structured document attributes. It
    never generates relevance truth from the candidate embedding model.
    """

    def __init__(
        self,
        *,
        policy: ProductionEmbeddingBenchmarkPolicy | None = None,
    ) -> None:
        self._policy = policy or ProductionEmbeddingBenchmarkPolicy()

    def build(
        self,
        *,
        benchmark_id: str,
        dataset_version: str,
        document_catalogs: Sequence[
            EmbeddingBenchmarkDocumentCatalog
        ],
        case_catalog: EmbeddingBenchmarkCaseCatalog,
    ) -> tuple[EmbeddingBenchmarkDataset, dict[str, Any]]:
        if not document_catalogs:
            raise ValueError(
                "At least one benchmark document catalog is required."
            )
        documents = self._merge_documents(document_catalogs)
        document_by_id = {
            value.document_id: value
            for value in documents
        }
        self._validate_gold_labels(
            cases=case_catalog.cases,
            document_by_id=document_by_id,
        )
        rights_status = self._combined_rights_status(
            document_catalogs
        )
        authoring_manifest = {
            "schema_version": "punk-embedding-benchmark-authoring-v1",
            "builder_version": "production-dataset-builder-v1",
            "review_status": case_catalog.review_status,
            "reviewed_by": case_catalog.reviewed_by,
            "reviewed_at": case_catalog.reviewed_at.isoformat(),
            "case_catalog_fingerprint": case_catalog.fingerprint,
            "document_catalog_fingerprints": sorted(
                value.fingerprint
                for value in document_catalogs
            ),
        }
        dataset = EmbeddingBenchmarkDataset(
            benchmark_id=benchmark_id,
            dataset_version=dataset_version,
            privacy_status="safe",
            rights_status=rights_status,
            contains_raw_identifiers=False,
            authoring_manifest=authoring_manifest,
            documents=documents,
            cases=case_catalog.cases,
        )
        coverage = benchmark_dataset_coverage(dataset)
        coverage_thresholds = dict(
            self._policy.minimum_coverage_thresholds
        )
        coverage_results = {
            key: {
                "observed": int(coverage[key]),
                "threshold": int(coverage_thresholds[key]),
                "operator": ">=",
                "passed": int(coverage[key])
                >= int(coverage_thresholds[key]),
            }
            for key in sorted(REQUIRED_BENCHMARK_COVERAGE)
        }
        ready = all(
            value["passed"]
            for value in coverage_results.values()
        )
        report = {
            "status": (
                "ready_for_model_evaluation"
                if ready
                else "blocked_insufficient_benchmark_coverage"
            ),
            "benchmark_id": dataset.benchmark_id,
            "dataset_version": dataset.dataset_version,
            "dataset_fingerprint": dataset.fingerprint,
            "policy_id": self._policy.policy_id,
            "coverage": coverage,
            "coverage_results": coverage_results,
            "gold_label_integrity": "passed",
            "review_status": case_catalog.review_status,
            "reviewed_by": case_catalog.reviewed_by,
            "reviewed_at": case_catalog.reviewed_at.isoformat(),
            "document_catalog_count": len(document_catalogs),
            "document_catalog_fingerprints": sorted(
                value.fingerprint
                for value in document_catalogs
            ),
            "case_catalog_fingerprint": case_catalog.fingerprint,
            "ready_for_model_evaluation": ready,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
            "activation_or_export_performed": False,
        }
        report["report_fingerprint"] = stable_digest(report)
        return dataset, report

    def _merge_documents(
        self,
        catalogs: Sequence[EmbeddingBenchmarkDocumentCatalog],
    ) -> tuple[EmbeddingBenchmarkDocument, ...]:
        merged: dict[str, EmbeddingBenchmarkDocument] = {}
        for catalog in catalogs:
            for document in catalog.documents:
                if document.document_id in merged:
                    raise ValueError(
                        "Benchmark document IDs must be unique across catalogs."
                    )
                merged[document.document_id] = document
        return tuple(
            merged[key]
            for key in sorted(merged)
        )

    def _validate_gold_labels(
        self,
        *,
        cases: Sequence[Any],
        document_by_id: dict[str, EmbeddingBenchmarkDocument],
    ) -> None:
        seen_queries: set[tuple[str, str]] = set()
        semantic_groups: dict[str, list[Any]] = defaultdict(list)
        available_locations = {
            value.location
            for value in document_by_id.values()
        }
        for case in cases:
            query_identity = (
                case.language,
                " ".join(case.query.lower().split()),
            )
            if query_identity in seen_queries:
                raise ValueError(
                    "Benchmark queries must be unique within each language."
                )
            seen_queries.add(query_identity)
            if case.semantic_group_id:
                semantic_groups[case.semantic_group_id].append(case)

            if case.unsupported_location:
                if not case.expected_locations:
                    raise ValueError(
                        "Unsupported-location cases require expected_locations."
                    )
                if set(case.expected_locations).intersection(
                    available_locations
                ):
                    raise ValueError(
                        "Unsupported-location labels conflict with "
                        "available documents."
                    )
                continue

            for document_id in case.relevant_document_ids:
                document = document_by_id[document_id]
                self._assert_relevant_constraints(case, document)

            if case.hard_negative_document_ids and not any(
                (
                    case.expected_locations,
                    case.expected_categories,
                    case.expected_dayparts,
                )
            ):
                raise ValueError(
                    "Hard-negative cases require explicit expected constraints."
                )
            for document_id in case.hard_negative_document_ids:
                document = document_by_id[document_id]
                if self._matches_all_constraints(case, document):
                    raise ValueError(
                        "Hard-negative labels cannot satisfy every "
                        "explicit expected constraint."
                    )

        for group_id, grouped_cases in semantic_groups.items():
            if len(grouped_cases) < 2:
                continue
            languages = {
                value.language
                for value in grouped_cases
            }
            if len(languages) != len(grouped_cases):
                raise ValueError(
                    f"Semantic group {group_id} repeats a language."
                )
            reference = self._semantic_group_label(grouped_cases[0])
            if any(
                self._semantic_group_label(value) != reference
                for value in grouped_cases[1:]
            ):
                raise ValueError(
                    f"Semantic group {group_id} has inconsistent gold labels."
                )

    def _assert_relevant_constraints(
        self,
        case: Any,
        document: EmbeddingBenchmarkDocument,
    ) -> None:
        if not self._matches_all_constraints(case, document):
            raise ValueError(
                f"Relevant document {document.document_id} conflicts with "
                f"the gold constraints for case {case.case_id}."
            )

    def _matches_all_constraints(
        self,
        case: Any,
        document: EmbeddingBenchmarkDocument,
    ) -> bool:
        checks = []
        if case.expected_locations:
            checks.append(document.location in case.expected_locations)
        if case.expected_categories:
            checks.append(document.category in case.expected_categories)
        if case.expected_dayparts:
            checks.append(document.daypart in case.expected_dayparts)
        return all(checks)

    def _semantic_group_label(self, case: Any) -> tuple[Any, ...]:
        return (
            case.relevant_document_ids,
            case.hard_negative_document_ids,
            case.expected_locations,
            case.expected_categories,
            case.expected_dayparts,
            case.unsupported_location,
        )

    def _combined_rights_status(
        self,
        catalogs: Sequence[EmbeddingBenchmarkDocumentCatalog],
    ) -> str:
        values = {
            value.rights_status
            for value in catalogs
        }
        if values == {"synthetic_evaluation"}:
            return "synthetic_evaluation"
        if values == {"permitted"}:
            return "permitted"
        return "offline_evaluation_only"


class PgvectorEmbeddingBenchmarkCatalogSourceService:
    """Read only safe trait documents from one tenant-scoped feature set."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: Engine | None = None,
    ) -> None:
        self._database_url = database_url
        self._engine_override = engine

    def export_catalog(
        self,
        *,
        tenant_id: str,
        feature_set_id: str,
        feature_set_version: int,
        catalog_id: str,
        catalog_version: str,
    ) -> EmbeddingBenchmarkDocumentCatalog:
        clean_tenant_id = normalize_taxonomy_value(tenant_id)
        if not clean_tenant_id:
            raise ValueError("tenant_id is required.")
        engine = self._engine()
        with engine.connect() as connection:
            self._assert_schema_ready(connection)
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
                {"tenant_id": clean_tenant_id},
            )
            feature_set = connection.execute(
                text(
                    """
                    SELECT
                        tenant_id,
                        feature_set_id,
                        version,
                        source_fingerprint,
                        data_use_mode,
                        freshness_status,
                        privacy_policy_version,
                        rights_policy_id,
                        purpose,
                        eligible_for_retrieval,
                        feature_count
                    FROM audience_feature_sets
                    WHERE tenant_id = :tenant_id
                      AND feature_set_id = :feature_set_id
                      AND version = :feature_set_version
                    """
                ),
                {
                    "tenant_id": clean_tenant_id,
                    "feature_set_id": feature_set_id,
                    "feature_set_version": int(feature_set_version),
                },
            ).mappings().first()
            if not feature_set:
                raise FileNotFoundError(
                    "The requested benchmark source feature set was not found."
                )
            if not feature_set["eligible_for_retrieval"]:
                raise RuntimeError(
                    "The requested feature set is not retrieval-eligible."
                )
            rows = connection.execute(
                text(
                    """
                    SELECT
                        feature_id,
                        location_name,
                        primary_poi_type,
                        created_day_part,
                        trait_text,
                        privacy_status,
                        rights_status
                    FROM audience_feature_vectors
                    WHERE tenant_id = :tenant_id
                      AND feature_set_id = :feature_set_id
                      AND feature_set_version = :feature_set_version
                      AND eligible_for_retrieval = TRUE
                      AND privacy_status IN (
                          'safe',
                          'passed',
                          'privacy_safe'
                      )
                    ORDER BY feature_id
                    """
                ),
                {
                    "tenant_id": clean_tenant_id,
                    "feature_set_id": feature_set_id,
                    "feature_set_version": int(feature_set_version),
                },
            ).mappings().all()
        if len(rows) != int(feature_set["feature_count"]):
            raise RuntimeError(
                "Safe benchmark catalog count does not match feature lineage."
            )
        documents = tuple(
            EmbeddingBenchmarkDocument(
                document_id=f"feature:{row['feature_id']}",
                text=row["trait_text"],
                location=row["location_name"],
                category=row["primary_poi_type"],
                daypart=row["created_day_part"],
            )
            for row in rows
        )
        unsafe_rights = {
            normalize_taxonomy_value(row["rights_status"])
            for row in rows
        }.difference(
            {
                "permitted",
                "historical_internal_only",
                "offline_evaluation_only",
            }
        )
        if unsafe_rights:
            raise RuntimeError(
                "Feature rows contain unsupported evaluation rights."
            )
        lineage = {
            "tenant_id": clean_tenant_id,
            "feature_set_id": feature_set_id,
            "feature_set_version": int(feature_set_version),
            "feature_set_source_fingerprint": feature_set[
                "source_fingerprint"
            ],
            "data_use_mode": feature_set["data_use_mode"],
            "freshness_status": feature_set["freshness_status"],
            "privacy_policy_version": feature_set[
                "privacy_policy_version"
            ],
            "rights_policy_id": feature_set["rights_policy_id"],
            "purpose": feature_set["purpose"],
            "feature_count": len(documents),
            "embeddings_exported": False,
            "cohort_sizes_exported": False,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
        }
        source_fingerprint = stable_digest(
            {
                "lineage": lineage,
                "documents": [
                    value.to_dict()
                    for value in documents
                ],
            }
        )
        return EmbeddingBenchmarkDocumentCatalog(
            catalog_id=catalog_id,
            catalog_version=catalog_version,
            source_type="pgvector_privacy_safe_features",
            source_fingerprint=source_fingerprint,
            privacy_status="safe",
            rights_status="offline_evaluation_only",
            contains_raw_identifiers=False,
            documents=documents,
            lineage=lineage,
        )

    def _assert_schema_ready(self, connection: Any) -> None:
        ready = connection.execute(
            text(
                """
                SELECT
                    to_regclass(
                        'public.audience_feature_sets'
                    ) IS NOT NULL
                    AND
                    to_regclass(
                        'public.audience_feature_vectors'
                    ) IS NOT NULL
                    AS ready
                """
            )
        ).scalar()
        if not ready:
            raise RuntimeError(
                "The versioned pgvector feature schema is not ready."
            )

    def _engine(self) -> Engine:
        if self._engine_override is not None:
            return self._engine_override
        database_url = (
            self._database_url
            or os.getenv("AUDIENCE_FEATURE_DATABASE_URL")
            or ""
        ).strip()
        if not database_url:
            raise RuntimeError(
                "AUDIENCE_FEATURE_DATABASE_URL is not configured."
            )
        if database_url.startswith("postgres://"):
            database_url = (
                "postgresql://" + database_url[len("postgres://") :]
            )
        return create_engine(database_url, pool_pre_ping=True)
