from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from app.models.embedding_benchmark_authoring_contracts import (
    EmbeddingBenchmarkCaseCatalog,
    EmbeddingBenchmarkDocumentCatalog,
)
from app.services.production_embedding_benchmark_dataset_service import (
    ProductionEmbeddingBenchmarkDatasetBuilderService,
)


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile reviewed gold labels and privacy-safe feature catalogs "
            "into an immutable production embedding benchmark dataset."
        )
    )
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument(
        "--document-catalog",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument("--case-catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--readiness-report", required=True, type=Path)
    parser.add_argument(
        "--confirm-gold-label-review-complete",
        action="store_true",
    )
    args = parser.parse_args()
    if not args.confirm_gold_label_review_complete:
        raise SystemExit(
            "STOP: explicit gold-label review confirmation is required"
        )

    document_catalogs = [
        EmbeddingBenchmarkDocumentCatalog.from_json(
            path.read_text(encoding="utf-8")
        )
        for path in args.document_catalog
    ]
    case_catalog = EmbeddingBenchmarkCaseCatalog.from_json(
        args.case_catalog.read_text(encoding="utf-8")
    )
    dataset, readiness = (
        ProductionEmbeddingBenchmarkDatasetBuilderService().build(
            benchmark_id=args.benchmark_id,
            dataset_version=args.dataset_version,
            document_catalogs=document_catalogs,
            case_catalog=case_catalog,
        )
    )
    _write_atomic(args.output, dataset.to_dict())
    _write_atomic(args.readiness_report, readiness)
    print(
        json.dumps(
            {
                "status": readiness["status"],
                "benchmark_id": dataset.benchmark_id,
                "dataset_version": dataset.dataset_version,
                "dataset_fingerprint": dataset.fingerprint,
                "document_count": len(dataset.documents),
                "case_count": len(dataset.cases),
                "ready_for_model_evaluation": readiness[
                    "ready_for_model_evaluation"
                ],
                "dataset_output": str(args.output),
                "readiness_output": str(args.readiness_report),
                "raw_identifiers_read": False,
                "raw_identifiers_stored": False,
                "activation_or_export_performed": False,
                "credentials_exposed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
