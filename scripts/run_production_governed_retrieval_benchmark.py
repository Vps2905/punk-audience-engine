from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.production_governed_retrieval_benchmark_service import (
    ProductionGovernedRetrievalBenchmarkService,
    validate_governed_retrieval_benchmark_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline, side-effect-free Module 2.1 governed retrieval "
            "benchmark using the real canonicalizer and retrieval pipeline."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--semantic-min-score", type=float, default=0.55)
    parser.add_argument("--semantic-min-margin", type=float, default=0.05)
    parser.add_argument(
        "--confirm-offline-benchmark-only",
        action="store_true",
        help="Required acknowledgement that no registration, routing, or export occurs.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.confirm_offline_benchmark_only:
        raise SystemExit("STOP: --confirm-offline-benchmark-only is required.")
    if args.dataset.resolve() == args.output.resolve():
        raise SystemExit("STOP: output cannot overwrite the immutable dataset.")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    report = ProductionGovernedRetrievalBenchmarkService(
        device=args.device,
        batch_size=args.batch_size,
        semantic_min_score=args.semantic_min_score,
        semantic_min_margin=args.semantic_min_margin,
    ).evaluate(dataset)
    validate_governed_retrieval_benchmark_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "benchmark_id": report["benchmark"]["benchmark_id"],
                "dataset_fingerprint": report["benchmark"]["dataset_fingerprint"],
                "report_fingerprint": report["report_fingerprint"],
                "engineering_passed": report["engineering_passed"],
                "benchmark_evidence_ready": report["benchmark_evidence_ready"],
                "production_certification_ready": report[
                    "production_certification"
                ]["ready"],
                "full_canonicalization_accuracy": summary[
                    "full_canonicalization_accuracy"
                ],
                "base_candidate_recall": summary["base_candidate_recall"],
                "final_candidate_recall": summary["final_candidate_recall"],
                "structured_semantic_top1_accuracy": summary[
                    "structured_semantic_top1_accuracy"
                ],
                "unsupported_rejection_rate": summary[
                    "unsupported_rejection_rate"
                ],
                "p95_latency_ms": summary["latency_ms"]["p95"],
                "output": str(args.output),
                "registration_allowed": report["registration_allowed"],
                "production_routing_enabled": report[
                    "production_routing_enabled"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
