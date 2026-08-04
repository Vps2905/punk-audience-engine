from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.models.embedding_benchmark_authoring_contracts import (
    EmbeddingBenchmarkDocumentCatalog,
)
from app.models.embedding_benchmark_case_authoring_contracts import (
    EmbeddingBenchmarkLanguagePack,
)
from app.services.production_embedding_benchmark_case_authoring_service import (
    ProductionEmbeddingBenchmarkCaseAuthoringService,
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_review_worksheet(
    path: Path,
    cases: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(
                [
                    "case_id",
                    "language",
                    "query",
                    "relevant_document_ids",
                    "hard_negative_document_ids",
                    "expected_locations",
                    "expected_categories",
                    "expected_dayparts",
                    "semantic_group_id",
                    "unsupported_location",
                    "decision",
                    "review_notes",
                ]
            )
            for case in cases:
                writer.writerow(
                    [
                        case["case_id"],
                        case["language"],
                        case["query"],
                        ",".join(case["relevant_document_ids"]),
                        ",".join(case["hard_negative_document_ids"]),
                        ",".join(case["expected_locations"]),
                        ",".join(case["expected_categories"]),
                        ",".join(case["expected_dayparts"]),
                        case.get("semantic_group_id") or "",
                        str(case["unsupported_location"]).lower(),
                        "",
                        "",
                    ]
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Author a deterministic pending-review multilingual embedding "
            "benchmark case catalog from approved language assets and "
            "grounded privacy-safe documents."
        )
    )
    parser.add_argument(
        "--document-catalog",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument("--curation-plan", required=True, type=Path)
    parser.add_argument("--language-pack", required=True, type=Path)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--catalog-version", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--review-worksheet", required=True, type=Path)
    parser.add_argument(
        "--confirm-no-auto-approval",
        action="store_true",
    )
    args = parser.parse_args()
    if not args.confirm_no_auto_approval:
        raise SystemExit(
            "STOP: explicit no-auto-approval confirmation is required"
        )

    catalogs = [
        EmbeddingBenchmarkDocumentCatalog.from_json(
            path.read_text(encoding="utf-8")
        )
        for path in args.document_catalog
    ]
    plan = json.loads(
        args.curation_plan.read_text(encoding="utf-8")
    )
    language_pack = EmbeddingBenchmarkLanguagePack.from_json(
        args.language_pack.read_text(encoding="utf-8")
    )
    draft, audit = (
        ProductionEmbeddingBenchmarkCaseAuthoringService().author(
            document_catalogs=catalogs,
            curation_plan=plan,
            language_pack=language_pack,
            catalog_id=args.catalog_id,
            catalog_version=args.catalog_version,
        )
    )
    _write_json_atomic(args.output, draft)
    _write_json_atomic(args.audit_output, audit)
    _write_review_worksheet(
        args.review_worksheet,
        draft["cases"],
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "case_count": audit["case_count"],
                "language_counts": audit["language_counts"],
                "unsupported_location_case_count": audit[
                    "unsupported_location_case_count"
                ],
                "hard_negative_case_count": audit[
                    "hard_negative_case_count"
                ],
                "hard_negative_violation_counts": audit[
                    "hard_negative_violation_counts"
                ],
                "multilingual_group_count": audit[
                    "multilingual_group_count"
                ],
                "fallback_translation_case_count": 0,
                "placeholder_unsupported_case_count": 0,
                "linguistic_flag_count": 0,
                "review_status": "pending",
                "gold_labels_auto_approved": False,
                "case_output": str(args.output),
                "audit_output": str(args.audit_output),
                "review_worksheet": str(args.review_worksheet),
                "raw_identifiers_read": False,
                "raw_identifiers_stored": False,
                "activation_or_export_performed": False,
                "credentials_exposed": False,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
