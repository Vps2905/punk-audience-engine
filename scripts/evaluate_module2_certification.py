#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.models.production_module2_completion_contracts import (
    NativeLanguageReviewManifest,
    UnsupportedCalibrationDataset,
)
from app.services.production_module2_certification_service import (
    ProductionModule2CertificationService,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Module 2 evidence without registering or routing."
    )
    parser.add_argument("--benchmark-report", required=True, type=Path)
    parser.add_argument("--native-review-manifest", required=True, type=Path)
    parser.add_argument("--unsupported-calibration", required=True, type=Path)
    parser.add_argument("--external-gates", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-registration-approved", action="store_true")
    parser.add_argument("--index-release-ready", action="store_true")
    parser.add_argument("--shadow-release-ready", action="store_true")
    parser.add_argument("--confirm-evaluation-only", action="store_true")
    args = parser.parse_args()
    if not args.confirm_evaluation_only:
        raise SystemExit("--confirm-evaluation-only is required.")
    external = _load(args.external_gates) if args.external_gates else {}
    report = ProductionModule2CertificationService().evaluate(
        benchmark_report=_load(args.benchmark_report),
        native_review_manifest=NativeLanguageReviewManifest.from_mapping(
            _load(args.native_review_manifest)
        ),
        calibration_dataset=UnsupportedCalibrationDataset.from_mapping(
            _load(args.unsupported_calibration)
        ),
        external_gates=external,
        model_registration_approved=args.model_registration_approved,
        index_release_ready=args.index_release_ready,
        shadow_release_ready=args.shadow_release_ready,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "module2_evidence_ready": report["module2_evidence_ready"],
                "production_certification_ready": report[
                    "production_certification_ready"
                ],
                "report_fingerprint": report["report_fingerprint"],
                "output": str(args.output),
                "output_sha256": hashlib.sha256(
                    args.output.read_bytes()
                ).hexdigest(),
                "model_registration_performed": False,
                "production_routing_enabled": False,
                "activation_or_export_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
