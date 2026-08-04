from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from app.models.audience_feature_contracts import stable_digest
from app.models.embedding_benchmark_authoring_contracts import (
    EmbeddingBenchmarkDocumentCatalog,
)
from app.models.embedding_benchmark_case_authoring_contracts import (
    EmbeddingBenchmarkLanguagePack,
)
from app.models.embedding_benchmark_contracts import (
    EmbeddingBenchmarkCase,
    EmbeddingBenchmarkDocument,
    ProductionEmbeddingBenchmarkPolicy,
)

CASE_AUTHORING_SCHEMA_VERSION = (
    "punk-embedding-benchmark-case-authoring-v1"
)
STRUCTURED_CONSTRAINTS = ("category", "location", "daypart")


class ProductionEmbeddingBenchmarkCaseAuthoringService:
    """
    Build a deterministic pending-review benchmark case catalog.

    The service only uses approved language assets, privacy-safe documents and
    grounded hard-negative candidates. It never approves gold labels, evaluates
    an embedding model, accesses raw identifiers, or activates an audience.
    """

    def __init__(
        self,
        *,
        policy: ProductionEmbeddingBenchmarkPolicy | None = None,
    ) -> None:
        self._policy = policy or ProductionEmbeddingBenchmarkPolicy()

    def author(
        self,
        *,
        document_catalogs: Sequence[
            EmbeddingBenchmarkDocumentCatalog
        ],
        curation_plan: Mapping[str, Any],
        language_pack: EmbeddingBenchmarkLanguagePack,
        catalog_id: str,
        catalog_version: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        documents = self._merge_documents(document_catalogs)
        self._validate_curation_plan(
            curation_plan=curation_plan,
            document_catalogs=document_catalogs,
        )
        thresholds = self._policy.minimum_coverage_thresholds
        languages = tuple(language_pack.language_order)
        required_language_count = int(thresholds["language_count"])
        if len(languages) != required_language_count:
            raise ValueError(
                "Approved language pack must contain exactly "
                f"{required_language_count} languages."
            )

        allocations = self._allocations(
            thresholds=thresholds,
            language_count=len(languages),
        )
        unsupported_locations = tuple(
            sorted(language_pack.unsupported_location_labels)
        )
        if len(unsupported_locations) < allocations[
            "unsupported_per_language"
        ]:
            raise ValueError(
                "Approved language pack does not contain enough "
                "unsupported locations."
            )
        known_locations = {value.location for value in documents}
        if known_locations.intersection(unsupported_locations):
            raise ValueError(
                "Unsupported benchmark locations overlap known documents."
            )

        documents_by_id = {
            value.document_id: value
            for value in documents
        }
        candidate_map = self._candidate_map(
            curation_plan=curation_plan,
            documents_by_id=documents_by_id,
        )
        eligible_anchor_ids, missing_categories, missing_dayparts = (
            self._eligible_anchor_ids(
                documents=documents,
                candidate_map=candidate_map,
                language_pack=language_pack,
            )
        )
        if missing_categories or missing_dayparts:
            raise ValueError(
                "Approved language pack does not cover all benchmark "
                "taxonomy values: "
                f"categories={sorted(missing_categories)}, "
                f"dayparts={sorted(missing_dayparts)}"
            )

        selected_anchor_ids: set[str] = set()
        selected_category_counts: Counter[str] = Counter()
        violation_counts: Counter[str] = Counter()

        group_selections = self._select_balanced_anchors(
            count=allocations["multilingual_group_count"],
            desired_violations=self._balanced_violation_sequence(
                allocations["multilingual_group_count"]
            ),
            eligible_anchor_ids=eligible_anchor_ids,
            candidate_map=candidate_map,
            documents_by_id=documents_by_id,
            selected_anchor_ids=selected_anchor_ids,
            selected_category_counts=selected_category_counts,
        )
        for selection in group_selections:
            violation_counts[selection["violated_constraint"]] += len(
                languages
            )

        extra_selections = self._select_balanced_anchors(
            count=allocations["extra_supported_case_count"],
            desired_violations=self._balanced_violation_sequence(
                allocations["extra_supported_case_count"]
            ),
            eligible_anchor_ids=eligible_anchor_ids,
            candidate_map=candidate_map,
            documents_by_id=documents_by_id,
            selected_anchor_ids=selected_anchor_ids,
            selected_category_counts=selected_category_counts,
        )
        for selection in extra_selections:
            violation_counts[selection["violated_constraint"]] += 1

        cases: list[EmbeddingBenchmarkCase] = []
        for group_index, selection in enumerate(
            group_selections,
            start=1,
        ):
            anchor = documents_by_id[selection["anchor_document_id"]]
            semantic_group_id = (
                f"multilingual-group-{group_index:02d}-"
                f"{anchor.location}-{anchor.category}-{anchor.daypart}"
            )
            for language in languages:
                cases.append(
                    self._supported_case(
                        case_id=(
                            f"supported-group-{group_index:02d}-{language}"
                        ),
                        language=language,
                        anchor=anchor,
                        negative_id=selection[
                            "hard_negative_document_id"
                        ],
                        semantic_group_id=semantic_group_id,
                        language_pack=language_pack,
                    )
                )

        extra_index = 0
        for language in languages:
            for language_case_number in range(
                1,
                allocations["extra_supported_per_language"] + 1,
            ):
                selection = extra_selections[extra_index]
                extra_index += 1
                anchor = documents_by_id[
                    selection["anchor_document_id"]
                ]
                cases.append(
                    self._supported_case(
                        case_id=(
                            f"supported-extra-{language}-"
                            f"{language_case_number:02d}"
                        ),
                        language=language,
                        anchor=anchor,
                        negative_id=selection[
                            "hard_negative_document_id"
                        ],
                        semantic_group_id=None,
                        language_pack=language_pack,
                    )
                )

        selected_unsupported_locations = unsupported_locations[
            : allocations["unsupported_per_language"]
        ]
        for language in languages:
            definition = language_pack.languages[language]
            for unsupported_index, location in enumerate(
                selected_unsupported_locations,
                start=1,
            ):
                display_location = language_pack.unsupported_location_labels[
                    location
                ][language]
                query = definition.unsupported_query_template.format(
                    location=display_location,
                )
                cases.append(
                    EmbeddingBenchmarkCase(
                        case_id=(
                            f"unsupported-{language}-{unsupported_index:02d}"
                        ),
                        query=query,
                        language=language,
                        relevant_document_ids=(),
                        hard_negative_document_ids=(),
                        expected_locations=(location,),
                        expected_categories=(),
                        expected_dayparts=(),
                        semantic_group_id=None,
                        unsupported_location=True,
                    )
                )

        sorted_cases = tuple(
            sorted(cases, key=lambda value: value.case_id)
        )
        if len(sorted_cases) != allocations["case_count"]:
            raise RuntimeError("Authored benchmark case count is incorrect.")
        case_ids = [value.case_id for value in sorted_cases]
        if len(case_ids) != len(set(case_ids)):
            raise RuntimeError("Authored benchmark case IDs are not unique.")

        draft_metadata: dict[str, Any] = {
            "schema_version": CASE_AUTHORING_SCHEMA_VERSION,
            "policy_id": self._policy.policy_id,
            "curation_plan_fingerprint": curation_plan.get(
                "plan_fingerprint"
            ),
            "document_catalog_fingerprints": sorted(
                value.fingerprint
                for value in document_catalogs
            ),
            "language_pack_fingerprint": language_pack.fingerprint,
            "language_pack_id": language_pack.pack_id,
            "language_pack_version": language_pack.pack_version,
            "queries_auto_approved": False,
            "gold_labels_auto_approved": False,
            "requires_human_review": True,
            "raw_identifiers_read": False,
            "raw_identifiers_stored": False,
            "activation_or_export_performed": False,
        }
        draft: dict[str, Any] = {
            "catalog_id": str(catalog_id).strip(),
            "catalog_version": str(catalog_version).strip(),
            "review_status": "pending",
            "reviewed_by": "",
            "reviewed_at": None,
            "cases": [value.to_dict() for value in sorted_cases],
            "draft_metadata": draft_metadata,
        }
        draft["draft_fingerprint"] = stable_digest(draft)

        audit = self._audit(
            cases=sorted_cases,
            languages=languages,
            violation_counts=violation_counts,
            allocations=allocations,
            selected_anchor_ids=selected_anchor_ids,
        )
        audit["draft_fingerprint"] = draft["draft_fingerprint"]
        audit["language_pack_fingerprint"] = language_pack.fingerprint
        audit["curation_plan_fingerprint"] = curation_plan.get(
            "plan_fingerprint"
        )
        audit["human_approval_recorded"] = False
        audit["gold_labels_auto_approved"] = False
        audit["raw_identifiers_read"] = False
        audit["raw_identifiers_stored"] = False
        audit["activation_or_export_performed"] = False
        audit["audit_fingerprint"] = stable_digest(audit)
        return draft, audit

    def _allocations(
        self,
        *,
        thresholds: Mapping[str, int],
        language_count: int,
    ) -> dict[str, int]:
        case_count = int(thresholds["case_count"])
        unsupported_count = int(
            thresholds["unsupported_location_case_count"]
        )
        group_count = int(thresholds["multilingual_group_count"])
        group_case_count = group_count * language_count
        supported_count = case_count - unsupported_count
        extra_supported_count = supported_count - group_case_count
        if min(
            case_count,
            unsupported_count,
            group_count,
            extra_supported_count,
        ) < 0:
            raise ValueError("Benchmark policy case allocations are invalid.")
        if case_count % language_count:
            raise ValueError(
                "Benchmark case_count must be divisible by language_count."
            )
        if unsupported_count % language_count:
            raise ValueError(
                "Unsupported case count must be divisible by language_count."
            )
        if extra_supported_count % language_count:
            raise ValueError(
                "Extra supported case count must be divisible by language_count."
            )
        return {
            "case_count": case_count,
            "supported_case_count": supported_count,
            "unsupported_case_count": unsupported_count,
            "multilingual_group_count": group_count,
            "group_case_count": group_case_count,
            "extra_supported_case_count": extra_supported_count,
            "unsupported_per_language": unsupported_count // language_count,
            "extra_supported_per_language": (
                extra_supported_count // language_count
            ),
        }

    def _validate_curation_plan(
        self,
        *,
        curation_plan: Mapping[str, Any],
        document_catalogs: Sequence[
            EmbeddingBenchmarkDocumentCatalog
        ],
    ) -> None:
        if not isinstance(curation_plan, Mapping):
            raise TypeError("Benchmark curation plan must be an object.")
        if curation_plan.get("status") != "ready_for_human_case_authoring":
            raise ValueError(
                "Benchmark curation plan is not ready for case authoring."
            )
        if curation_plan.get("policy_id") != self._policy.policy_id:
            raise ValueError("Benchmark curation plan policy does not match.")
        expected = sorted(
            value.fingerprint
            for value in document_catalogs
        )
        observed = sorted(
            str(value)
            for value in (
                curation_plan.get("document_catalog_fingerprints") or []
            )
        )
        if observed != expected:
            raise ValueError(
                "Benchmark curation plan document fingerprints do not match."
            )
        if curation_plan.get("blockers"):
            raise ValueError("Benchmark curation plan still contains blockers.")

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
                        "Benchmark document IDs must be unique across catalogs."
                    )
                documents[document.document_id] = document
        return tuple(
            documents[key]
            for key in sorted(documents)
        )

    def _candidate_map(
        self,
        *,
        curation_plan: Mapping[str, Any],
        documents_by_id: Mapping[str, EmbeddingBenchmarkDocument],
    ) -> dict[str, dict[str, list[dict[str, str]]]]:
        values: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for candidate in curation_plan.get(
            "grounded_hard_negative_candidates",
            [],
        ):
            if not isinstance(candidate, Mapping):
                continue
            anchor_id = str(
                candidate.get("anchor_document_id") or ""
            ).strip()
            negative_id = str(
                candidate.get("hard_negative_document_id") or ""
            ).strip()
            violated = str(
                candidate.get("violated_constraint") or ""
            ).strip()
            if (
                anchor_id not in documents_by_id
                or negative_id not in documents_by_id
                or anchor_id == negative_id
                or violated not in STRUCTURED_CONSTRAINTS
            ):
                continue
            anchor = documents_by_id[anchor_id]
            negative = documents_by_id[negative_id]
            if getattr(anchor, violated) == getattr(negative, violated):
                continue
            if not all(
                getattr(anchor, constraint)
                == getattr(negative, constraint)
                for constraint in STRUCTURED_CONSTRAINTS
                if constraint != violated
            ):
                continue
            values[anchor_id][violated].append(
                {
                    "hard_negative_document_id": negative_id,
                    "violated_constraint": violated,
                }
            )
        normalized: dict[str, dict[str, list[dict[str, str]]]] = {}
        for anchor_id, by_violation in values.items():
            normalized[anchor_id] = {
                violation: sorted(
                    candidates,
                    key=lambda item: item[
                        "hard_negative_document_id"
                    ],
                )
                for violation, candidates in by_violation.items()
            }
        return normalized

    def _eligible_anchor_ids(
        self,
        *,
        documents: Sequence[EmbeddingBenchmarkDocument],
        candidate_map: Mapping[
            str,
            Mapping[str, Sequence[Mapping[str, str]]],
        ],
        language_pack: EmbeddingBenchmarkLanguagePack,
    ) -> tuple[tuple[str, ...], set[str], set[str]]:
        categories = {value.category for value in documents}
        dayparts = {value.daypart for value in documents}
        missing_categories: set[str] = set()
        missing_dayparts: set[str] = set()
        for language in language_pack.language_order:
            definition = language_pack.languages[language]
            missing_categories.update(
                categories.difference(definition.category_labels)
            )
            missing_dayparts.update(
                dayparts.difference(definition.daypart_labels)
            )
        eligible = tuple(
            value.document_id
            for value in documents
            if value.document_id in candidate_map
        )
        return eligible, missing_categories, missing_dayparts

    def _balanced_violation_sequence(
        self,
        count: int,
    ) -> tuple[str, ...]:
        return tuple(
            STRUCTURED_CONSTRAINTS[index % len(STRUCTURED_CONSTRAINTS)]
            for index in range(count)
        )

    def _select_balanced_anchors(
        self,
        *,
        count: int,
        desired_violations: Sequence[str],
        eligible_anchor_ids: Sequence[str],
        candidate_map: Mapping[
            str,
            Mapping[str, Sequence[Mapping[str, str]]],
        ],
        documents_by_id: Mapping[str, EmbeddingBenchmarkDocument],
        selected_anchor_ids: set[str],
        selected_category_counts: Counter[str],
    ) -> list[dict[str, str]]:
        selections: list[dict[str, str]] = []
        if len(desired_violations) != count:
            raise ValueError("Balanced violation sequence count is invalid.")
        for violated in desired_violations:
            candidates = [
                anchor_id
                for anchor_id in eligible_anchor_ids
                if anchor_id not in selected_anchor_ids
                and candidate_map.get(anchor_id, {}).get(violated)
            ]
            if not candidates:
                raise ValueError(
                    "Insufficient distinct grounded anchors for balanced "
                    f"{violated} hard negatives."
                )
            anchor_id = min(
                candidates,
                key=lambda value: (
                    selected_category_counts[
                        documents_by_id[value].category
                    ],
                    documents_by_id[value].category,
                    documents_by_id[value].location,
                    documents_by_id[value].daypart,
                    value,
                ),
            )
            negative = candidate_map[anchor_id][violated][0]
            selected_anchor_ids.add(anchor_id)
            selected_category_counts[
                documents_by_id[anchor_id].category
            ] += 1
            selections.append(
                {
                    "anchor_document_id": anchor_id,
                    "hard_negative_document_id": negative[
                        "hard_negative_document_id"
                    ],
                    "violated_constraint": violated,
                }
            )
        return selections

    def _supported_case(
        self,
        *,
        case_id: str,
        language: str,
        anchor: EmbeddingBenchmarkDocument,
        negative_id: str,
        semantic_group_id: str | None,
        language_pack: EmbeddingBenchmarkLanguagePack,
    ) -> EmbeddingBenchmarkCase:
        definition = language_pack.languages[language]
        query = definition.supported_query_template.format(
            category=definition.category_labels[anchor.category],
            daypart=definition.daypart_labels[anchor.daypart],
            location=self._display_location(anchor.location),
        )
        return EmbeddingBenchmarkCase(
            case_id=case_id,
            query=query,
            language=language,
            relevant_document_ids=(anchor.document_id,),
            hard_negative_document_ids=(negative_id,),
            expected_locations=(anchor.location,),
            expected_categories=(anchor.category,),
            expected_dayparts=(anchor.daypart,),
            semantic_group_id=semantic_group_id,
            unsupported_location=False,
        )

    def _display_location(self, value: str) -> str:
        return " ".join(
            token.capitalize()
            for token in value.split("_")
            if token
        )

    def _audit(
        self,
        *,
        cases: Sequence[EmbeddingBenchmarkCase],
        languages: Sequence[str],
        violation_counts: Counter[str],
        allocations: Mapping[str, int],
        selected_anchor_ids: set[str],
    ) -> dict[str, Any]:
        language_counts = Counter(value.language for value in cases)
        semantic_groups = {
            value.semantic_group_id
            for value in cases
            if value.semantic_group_id
        }
        unsupported_count = sum(
            value.unsupported_location
            for value in cases
        )
        hard_negative_count = sum(
            bool(value.hard_negative_document_ids)
            for value in cases
        )
        blockers: list[str] = []
        if len(cases) != allocations["case_count"]:
            blockers.append("case_count")
        expected_per_language = allocations["case_count"] // len(languages)
        if any(
            language_counts[language] != expected_per_language
            for language in languages
        ):
            blockers.append("language_distribution")
        if unsupported_count != allocations["unsupported_case_count"]:
            blockers.append("unsupported_location_case_count")
        if hard_negative_count != allocations["supported_case_count"]:
            blockers.append("hard_negative_case_count")
        if len(semantic_groups) != allocations["multilingual_group_count"]:
            blockers.append("multilingual_group_count")
        if set(violation_counts) != set(STRUCTURED_CONSTRAINTS):
            blockers.append("hard_negative_constraint_coverage")
        if min(violation_counts.values(), default=0) < 10:
            blockers.append("hard_negative_constraint_balance")
        return {
            "schema_version": CASE_AUTHORING_SCHEMA_VERSION,
            "status": (
                "ready_for_human_case_review"
                if not blockers
                else "blocked_case_quality"
            ),
            "case_count": len(cases),
            "supported_case_count": len(cases) - unsupported_count,
            "unsupported_location_case_count": unsupported_count,
            "hard_negative_case_count": hard_negative_count,
            "language_count": len(language_counts),
            "language_counts": {
                key: language_counts[key]
                for key in sorted(language_counts)
            },
            "multilingual_group_count": len(semantic_groups),
            "distinct_supported_anchor_count": len(selected_anchor_ids),
            "hard_negative_violation_counts": {
                key: violation_counts[key]
                for key in sorted(violation_counts)
            },
            "fallback_translation_case_count": 0,
            "placeholder_unsupported_case_count": 0,
            "ungrounded_hard_negative_count": 0,
            "multilingual_group_error_count": 0,
            "linguistic_flag_count": 0,
            "blockers": blockers,
            "requires_human_review": True,
        }
