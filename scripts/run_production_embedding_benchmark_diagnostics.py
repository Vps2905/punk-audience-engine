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
from app.services.production_embedding_benchmark_diagnostics_service import (
    ProductionEmbeddingBenchmarkDiagnosticsService,
    ProductionEmbeddingBenchmarkDiagnosticsValidator,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate privacy-safe case-level diagnostics and a review-only "
            "model-specific rejection-threshold candidate for one immutable "
            "embedding model revision."
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
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--diagnostic-rank-depth", type=int, default=10)
    parser.add_argument(
        "--rejection-similarity-threshold",
        type=float,
        default=0.78,
    )
    parser.add_argument(
        "--target-unsupported-false-match-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
    )
    parser.add_argument(
        "--confirm-offline-diagnostics-only",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.confirm_offline_diagnostics_only:
        raise SystemExit(
            "STOP: pass --confirm-offline-diagnostics-only to acknowledge "
            "that diagnostics cannot register or activate a model"
        )

    load_dotenv(ROOT / ".env", override=False)
    dataset = EmbeddingBenchmarkDataset.from_mapping(
        _load_json_object(args.dataset)
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
    diagnostics = ProductionEmbeddingBenchmarkDiagnosticsService().evaluate(
        dataset=dataset,
        model=model,
        top_k=args.top_k,
        diagnostic_rank_depth=args.diagnostic_rank_depth,
        batch_size=args.batch_size,
        rejection_similarity_threshold=(
            args.rejection_similarity_threshold
        ),
        target_unsupported_false_match_rate=(
            args.target_unsupported_false_match_rate
        ),
        confidence_level=args.confidence_level,
    )
    validated = ProductionEmbeddingBenchmarkDiagnosticsValidator().validate(
        diagnostics,
        model=model,
        dataset=dataset,
    )
    _write_atomic(args.output, validated)

    calibration = validated["calibration"]
    print(
        json.dumps(
            {
                "status": "diagnostics_ready_for_human_review",
                "benchmark_id": validated["benchmark_id"],
                "dataset_fingerprint": validated[
                    "dataset_fingerprint"
                ],
                "model_fingerprint": validated["model"][
                    "model_fingerprint"
                ],
                "case_count": validated["evaluation"]["case_count"],
                "exact_top1_accuracy": validated["aggregate"][
                    "exact_top1_accuracy"
                ],
                "semantic_signature_top1_accuracy": validated[
                    "aggregate"
                ]["semantic_signature_top1_accuracy"],
                "ambiguous_supported_case_count": validated[
                    "ambiguity"
                ]["ambiguous_supported_case_count"],
                "recommended_threshold_candidate": (
                    calibration["recommended_candidate"]
                ),
                "production_calibration_ready": calibration[
                    "production_calibration_ready"
                ],
                "registration_allowed": False,
                "activation_or_export_performed": False,
                "credentials_exposed": False,
                "diagnostics_fingerprint": validated[
                    "diagnostics_fingerprint"
                ],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
