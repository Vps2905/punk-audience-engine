from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from app.services.production_governed_constraint_taxonomy_service import (
    attach_governed_taxonomy_to_dataset,
)
from app.services.production_governed_retrieval_benchmark_service import (
    ProductionGovernedRetrievalBenchmarkService,
    validate_governed_retrieval_benchmark_report,
)


def _load_json_object(path: Path, label: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"STOP: {label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"STOP: invalid {label} JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"STOP: {label} must be a JSON object: {path}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline, side-effect-free Module 2.1 governed retrieval "
            "benchmark using the real canonicalizer and retrieval pipeline."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        help=(
            "Optional separately reviewed governed taxonomy JSON. The "
            "canonical benchmark dataset identity is verified after attachment."
        ),
    )
    parser.add_argument(
        "--expected-taxonomy-sha256",
        help="Required SHA-256 when --taxonomy is supplied.",
    )
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
    dataset_path = args.dataset.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    taxonomy_path = (
        args.taxonomy.expanduser().resolve() if args.taxonomy else None
    )
    if dataset_path == output_path:
        raise SystemExit("STOP: output cannot overwrite the immutable dataset.")
    if taxonomy_path and taxonomy_path == output_path:
        raise SystemExit("STOP: output cannot overwrite the taxonomy input.")
    if taxonomy_path and taxonomy_path == dataset_path:
        raise SystemExit("STOP: taxonomy and dataset must be separate files.")

    dataset = _load_json_object(dataset_path, "benchmark dataset")
    if taxonomy_path:
        expected_taxonomy_sha = str(
            args.expected_taxonomy_sha256 or ""
        ).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_taxonomy_sha):
            raise SystemExit(
                "STOP: --expected-taxonomy-sha256 is required with --taxonomy."
            )
        actual_taxonomy_sha = hashlib.sha256(
            taxonomy_path.read_bytes()
        ).hexdigest()
        if actual_taxonomy_sha != expected_taxonomy_sha:
            raise SystemExit(
                "STOP: taxonomy checksum mismatch: "
                f"expected={expected_taxonomy_sha} "
                f"actual={actual_taxonomy_sha}"
            )
        taxonomy_payload = _load_json_object(
            taxonomy_path,
            "governed taxonomy",
        )
        dataset = attach_governed_taxonomy_to_dataset(
            dataset_payload=dataset,
            taxonomy_payload=taxonomy_payload,
        )
    elif args.expected_taxonomy_sha256:
        raise SystemExit(
            "STOP: --expected-taxonomy-sha256 requires --taxonomy."
        )
    report = ProductionGovernedRetrievalBenchmarkService(
        device=args.device,
        batch_size=args.batch_size,
        semantic_min_score=args.semantic_min_score,
        semantic_min_margin=args.semantic_min_margin,
    ).evaluate(dataset)
    validate_governed_retrieval_benchmark_report(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
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
                "output": str(output_path),
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
