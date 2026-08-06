#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.models.production_module3_overlap_contracts import (
    Module3OverlapDeduplicationPolicy,
    Module3OverlapDeduplicationRequest,
)
from app.services.production_module3_overlap_deduplication_service import (
    ProductionModule3OverlapDeduplicationService,
)


ROOT = Path(__file__).resolve().parents[1]


def load_environment() -> None:
    load_dotenv(ROOT / ".env", override=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a no-write Module 3.3 exact-duplicate and potential-overlap "
            "engineering evidence report from a Module 3.1-3.2 candidate report."
        )
    )
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-input-candidates", type=int, default=1000)
    parser.add_argument("--max-group-members", type=int, default=250)
    parser.add_argument(
        "--min-potential-overlap-group-size",
        type=int,
        default=2,
    )
    parser.add_argument("--confirm-engineering-preview-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_environment()
    args = parse_args()
    if not args.confirm_engineering_preview_only:
        raise SystemExit("--confirm-engineering-preview-only is required")

    candidate_path = Path(args.candidate_report).resolve()
    output = Path(args.output).resolve()
    if candidate_path == output:
        raise SystemExit("Candidate report and output paths must be different.")
    if not candidate_path.is_file():
        raise SystemExit(f"Candidate report not found: {candidate_path}")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")

    candidate_report = _read_mapping(candidate_path)
    batch_request = candidate_report.get("request")
    if not isinstance(batch_request, dict):
        raise SystemExit("Candidate report request metadata is missing.")

    request = Module3OverlapDeduplicationRequest(
        tenant_id=str(batch_request.get("tenant_id") or ""),
        batch_fingerprint=str(
            candidate_report.get("batch_fingerprint") or ""
        ),
        execution_mode=str(batch_request.get("execution_mode") or ""),
        purpose=str(batch_request.get("purpose") or ""),
    )
    policy = Module3OverlapDeduplicationPolicy(
        max_input_candidates=args.max_input_candidates,
        max_group_members=args.max_group_members,
        min_potential_overlap_group_size=(
            args.min_potential_overlap_group_size
        ),
    )
    report = ProductionModule3OverlapDeduplicationService(
        policy=policy
    ).analyze(
        request=request,
        candidate_batch=candidate_report,
    ).to_record()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("STATUS=engineering_preview_ready")
    print("SOURCE_CANDIDATE_COUNT=", report["source_candidate_count"])
    print("RETAINED_CANDIDATE_COUNT=", report["retained_candidate_count"])
    print(
        "SUPPRESSED_EXACT_DUPLICATE_OCCURRENCE_COUNT=",
        report["suppressed_exact_duplicate_occurrence_count"],
    )
    print(
        "POTENTIAL_OVERLAP_GROUP_COUNT=",
        report["potential_overlap_group_count"],
    )
    print(
        "CANDIDATES_REQUIRING_OVERLAP_REVIEW_COUNT=",
        report["candidates_requiring_overlap_review_count"],
    )
    print("MEMBERSHIP_INTERSECTION_READ=False")
    print("OVERLAP_RATE_COMPUTED=False")
    print("UNIQUE_REACH_CLAIMED=False")
    print("COHORT_SIZES_SUMMED=False")
    print("DATABASE_WRITE_PERFORMED=False")
    print("LOOKALIKE_GENERATION_PERFORMED=False")
    print("ACTIVATION_OR_EXPORT_PERFORMED=False")
    print("OUTPUT=", output)
    return 0


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Candidate report is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("Candidate report must be a JSON object.")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
