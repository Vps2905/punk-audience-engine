from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from app.models.embedding_benchmark_authoring_contracts import (
    EmbeddingBenchmarkDocumentCatalog,
)
from app.services.production_embedding_benchmark_curation_service import (
    ProductionEmbeddingBenchmarkCurationService,
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
            "Plan human-reviewed production embedding benchmark "
            "curation from privacy-safe document catalogs."
        )
    )
    parser.add_argument(
        "--document-catalog",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
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
    plan = ProductionEmbeddingBenchmarkCurationService().plan(
        document_catalogs=catalogs
    )
    _write_atomic(args.output, plan)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "policy_id": plan["policy_id"],
                "document_count": plan["document_coverage"][
                    "document_count"
                ],
                "document_deficit": plan["document_deficit"],
                "location_value_count": plan["document_coverage"][
                    "location_value_count"
                ],
                "category_value_count": plan["document_coverage"][
                    "category_value_count"
                ],
                "daypart_value_count": plan["document_coverage"][
                    "daypart_value_count"
                ],
                "grounded_hard_negative_candidate_count": plan[
                    "grounded_hard_negative_candidate_count"
                ],
                "blockers": plan["blockers"],
                "requires_human_gold_label_review": True,
                "gold_labels_auto_approved": False,
                "documents_auto_generated": False,
                "output": str(args.output),
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
