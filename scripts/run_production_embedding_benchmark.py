from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.models.embedding_benchmark_contracts import (
    EmbeddingBenchmarkDataset,
)
from app.models.production_feature_build_contracts import EmbeddingModelSpec
from app.services.production_embedding_benchmark_service import (
    ProductionEmbeddingBenchmarkService,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise SystemExit(f"STOP: {path} is not valid JSON") from None
    if not isinstance(payload, dict):
        raise SystemExit(f"STOP: {path} must contain a JSON object")
    return payload


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
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
            "Run the versioned Punk production embedding benchmark for one "
            "immutable model revision."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--embedding-dimension", type=int, default=384)
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--rejection-similarity-threshold",
        type=float,
        default=0.78,
    )
    parser.add_argument(
        "--cost-per-1000-queries-usd",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--threshold-overrides",
        type=Path,
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=False)
    dataset = EmbeddingBenchmarkDataset.from_mapping(
        _load_json_object(args.dataset)
    )
    threshold_overrides = (
        _load_json_object(args.threshold_overrides)
        if args.threshold_overrides
        else None
    )
    model = EmbeddingModelSpec(
        backend="sentence_transformers",
        model_name=args.model_name,
        model_revision=args.model_revision,
        dimension=args.embedding_dimension,
        normalize_embeddings=not args.no_normalize_embeddings,
        document_prefix=args.document_prefix,
        query_prefix=args.query_prefix,
    )
    report = ProductionEmbeddingBenchmarkService().evaluate(
        dataset=dataset,
        model=model,
        top_k=args.top_k,
        batch_size=args.batch_size,
        rejection_similarity_threshold=(
            args.rejection_similarity_threshold
        ),
        cost_per_1000_queries_usd=(
            args.cost_per_1000_queries_usd
        ),
        threshold_overrides=threshold_overrides,
    )
    _write_atomic(args.output, report)
    print(
        json.dumps(
            {
                "status": (
                    "benchmark_passed"
                    if report["passed"]
                    else "benchmark_failed"
                ),
                "benchmark_id": report["benchmark_id"],
                "dataset_version": report["dataset_version"],
                "dataset_fingerprint": report[
                    "dataset_fingerprint"
                ],
                "model_fingerprint": report["model"][
                    "model_fingerprint"
                ],
                "policy_id": report["policy"]["policy_id"],
                "passed": report["passed"],
                "report_fingerprint": report[
                    "report_fingerprint"
                ],
                "output": str(args.output),
                "activation_or_export_performed": False,
                "credentials_exposed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
