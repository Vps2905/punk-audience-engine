from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from app.models.audience_feature_contracts import stable_digest
from app.models.embedding_benchmark_authoring_contracts import (
    EmbeddingBenchmarkDocumentCatalog,
)
from app.models.embedding_benchmark_contracts import (
    EmbeddingBenchmarkDocument,
    ProductionEmbeddingBenchmarkPolicy,
)

CURATION_PLAN_SCHEMA_VERSION = (
    "punk-embedding-benchmark-curation-plan-v1"
)
STRUCTURED_CONSTRAINTS = ("location", "category", "daypart")


class ProductionEmbeddingBenchmarkCurationService:
    """
    Plan reviewed benchmark authoring from safe structured documents.

    The planner never writes queries, generates source documents, approves
    labels, evaluates a model, or claims audience volume. It identifies exact
    coverage deficits and grounded candidates for human gold-label review.
    """

    def __init__(
        self,
        *,
        policy: ProductionEmbeddingBenchmarkPolicy | None = None,
    ) -> None:
        self._policy = policy or ProductionEmbeddingBenchmarkPolicy()

    def plan(
        self,
        *,
        document_catalogs: Sequence[
            EmbeddingBenchmarkDocumentCatalog
        ],
    ) -> dict[str, Any]:
        documents = self._merge_documents(document_catalogs)
        thresholds = self._policy.minimum_coverage_thresholds
        observed = {
            "document_count": len(documents),
            "location_value_count": len(
                {value.location for value in documents}
            ),
            "category_value_count": len(
                {value.category for value in documents}
            ),
            "daypart_value_count": len(
                {value.daypart for value in documents}
            ),
        }
        coverage_results = {
            key: self._minimum_result(
                observed=observed[key],
                threshold=int(thresholds[key]),
            )
            for key in sorted(observed)
        }
        hard_negative_candidates = (
            self._hard_negative_candidates(documents)
        )
        hard_negative_result = self._minimum_result(
            observed=len(hard_negative_candidates),
            threshold=int(thresholds["hard_negative_case_count"]),
        )
        blockers = [
            key
            for key, result in coverage_results.items()
            if not result["passed"]
        ]
        if not hard_negative_result["passed"]:
            blockers.append("grounded_hard_negative_candidate_count")

        document_deficit = max(
            0,
            int(thresholds["document_count"])
            - len(documents),
        )
        counts = {
            constraint: Counter(
                getattr(value, constraint)
                for value in documents
            )
            for constraint in STRUCTURED_CONSTRAINTS
        }
        signatures: dict[
            tuple[str, str, str],
            list[str],
        ] = defaultdict(list)
        for document in documents:
            signatures[
                (
                    document.location,
                    document.category,
                    document.daypart,
                )
            ].append(document.document_id)

        plan: dict[str, Any] = {
            "schema_version": CURATION_PLAN_SCHEMA_VERSION,
            "status": (
                "ready_for_human_case_authoring"
                if not blockers
                else "blocked_curation_coverage"
            ),
            "policy_id": self._policy.policy_id,
            "document_catalog_fingerprints": sorted(
                value.fingerprint
                for value in document_catalogs
            ),
            "document_coverage": observed,
            "document_coverage_results": coverage_results,
            "document_deficit": document_deficit,
            "supplemental_document_slots": [
                {
                    "slot": value + 1,
                    "required_source": (
                        "reviewed_privacy_safe_aggregate_or_synthetic"
                    ),
                    "must_have_unique_document_id": True,
                    "must_have_unique_source_lineage": True,
                    "audience_volume_claim_allowed": False,
                }
                for value in range(document_deficit)
            ],
            "taxonomy_distribution": {
                key: self._distribution(values, len(documents))
                for key, values in counts.items()
            },
            "constraint_signature_count": len(signatures),
            "duplicate_constraint_signature_count": sum(
                len(document_ids) > 1
                for document_ids in signatures.values()
            ),
            "grounded_hard_negative_candidate_count": len(
                hard_negative_candidates
            ),
            "grounded_hard_negative_result": hard_negative_result,
            "grounded_hard_negative_candidates": (
                hard_negative_candidates
            ),
            "human_case_requirements": {
                key: int(value)
                for key, value in sorted(thresholds.items())
                if key not in observed
            },
            "blockers": sorted(blockers),
            "requires_human_gold_label_review": True,
            "gold_labels_auto_approved": False,
            "queries_auto_generated": False,
            "documents_auto_generated": False,
            "synthetic_audience_volume_claimed": False,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
            "activation_or_export_performed": False,
        }
        plan["plan_fingerprint"] = stable_digest(plan)
        return plan

    def _merge_documents(
        self,
        catalogs: Sequence[EmbeddingBenchmarkDocumentCatalog],
    ) -> tuple[EmbeddingBenchmarkDocument, ...]:
        if not catalogs:
            raise ValueError(
                "At least one safe document catalog is required."
            )
        documents: dict[str, EmbeddingBenchmarkDocument] = {}
        for catalog in catalogs:
            for document in catalog.documents:
                if document.document_id in documents:
                    raise ValueError(
                        "Benchmark document IDs must be unique "
                        "across catalogs."
                    )
                documents[document.document_id] = document
        return tuple(
            documents[key]
            for key in sorted(documents)
        )

    def _hard_negative_candidates(
        self,
        documents: Sequence[EmbeddingBenchmarkDocument],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for anchor in documents:
            for violated in STRUCTURED_CONSTRAINTS:
                matches = [
                    value
                    for value in documents
                    if value.document_id != anchor.document_id
                    and getattr(value, violated)
                    != getattr(anchor, violated)
                    and all(
                        getattr(value, constraint)
                        == getattr(anchor, constraint)
                        for constraint in STRUCTURED_CONSTRAINTS
                        if constraint != violated
                    )
                ]
                if not matches:
                    continue
                negative = min(
                    matches,
                    key=lambda value: value.document_id,
                )
                candidates.append(
                    {
                        "anchor_document_id": anchor.document_id,
                        "hard_negative_document_id": (
                            negative.document_id
                        ),
                        "violated_constraint": violated,
                        "shared_constraints": sorted(
                            constraint
                            for constraint in STRUCTURED_CONSTRAINTS
                            if constraint != violated
                        ),
                        "requires_human_review": True,
                    }
                )
        return candidates

    def _distribution(
        self,
        counts: Counter[str],
        total: int,
    ) -> list[dict[str, Any]]:
        return [
            {
                "value": value,
                "count": count,
                "share": round(count / total, 6),
            }
            for value, count in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

    def _minimum_result(
        self,
        *,
        observed: int,
        threshold: int,
    ) -> dict[str, Any]:
        return {
            "operator": ">=",
            "threshold": threshold,
            "observed": observed,
            "passed": observed >= threshold,
        }
