from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from app.models.embedding_benchmark_contracts import (
    EmbeddingBenchmarkDataset,
)
from app.models.production_feature_build_contracts import EmbeddingModelSpec
from app.services.production_embedding_model_registration_service import (
    ProductionEmbeddingModelRegistrationService,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Register one benchmark-approved immutable embedding model "
            "revision for a tenant."
        )
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--embedding-dimension", type=int, default=384)
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--approved-by", required=True)
    parser.add_argument(
        "--benchmark-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--benchmark-dataset",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
    )
    parser.add_argument(
        "--confirm-punk-owned-target",
        action="store_true",
    )
    parser.add_argument(
        "--confirm-benchmark-accepted",
        action="store_true",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=False)
    report = json.loads(
        args.benchmark_report.read_text(encoding="utf-8")
    )
    dataset = EmbeddingBenchmarkDataset.from_json(
        args.benchmark_dataset.read_text(encoding="utf-8")
    )
    result = ProductionEmbeddingModelRegistrationService().register_approved(
        tenant_id=args.tenant_id,
        model=EmbeddingModelSpec(
            backend=args.backend,
            model_name=args.model_name,
            model_revision=args.model_revision,
            dimension=args.embedding_dimension,
            normalize_embeddings=not args.no_normalize_embeddings,
            document_prefix=args.document_prefix,
            query_prefix=args.query_prefix,
        ),
        benchmark_report=report,
        benchmark_dataset=dataset,
        approved_by=args.approved_by,
        punk_owned_target_confirmed=args.confirm_punk_owned_target,
        benchmark_accepted=args.confirm_benchmark_accepted,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
