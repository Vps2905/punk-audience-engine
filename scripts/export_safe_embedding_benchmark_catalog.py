from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from app.services.production_embedding_benchmark_dataset_service import (
    PgvectorEmbeddingBenchmarkCatalogSourceService,
)

ROOT = Path(__file__).resolve().parents[1]


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
            "Export privacy-safe feature trait documents for offline "
            "embedding benchmark authoring."
        )
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--feature-set-version", type=int, required=True)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--catalog-version", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--confirm-offline-evaluation-only",
        action="store_true",
    )
    args = parser.parse_args()
    if not args.confirm_offline_evaluation_only:
        raise SystemExit(
            "STOP: explicit offline-evaluation confirmation is required"
        )

    load_dotenv(ROOT / ".env", override=False)
    catalog = (
        PgvectorEmbeddingBenchmarkCatalogSourceService().export_catalog(
            tenant_id=args.tenant_id,
            feature_set_id=args.feature_set_id,
            feature_set_version=args.feature_set_version,
            catalog_id=args.catalog_id,
            catalog_version=args.catalog_version,
        )
    )
    _write_atomic(args.output, catalog.to_dict())
    print(
        json.dumps(
            {
                "status": "safe_benchmark_catalog_exported",
                "catalog_id": catalog.catalog_id,
                "catalog_version": catalog.catalog_version,
                "catalog_fingerprint": catalog.fingerprint,
                "document_count": len(catalog.documents),
                "output": str(args.output),
                "embeddings_exported": False,
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
