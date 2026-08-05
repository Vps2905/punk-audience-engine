#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.models.production_module2_completion_contracts import (
    Module2ShadowObservation,
    Module2ShadowPolicy,
)
from app.services.production_module2_shadow_serving_service import (
    evaluate_module2_shadow_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate privacy-safe Module 2 shadow observations."
    )
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-samples", type=int, default=1000)
    parser.add_argument("--minimum-agreement-rate", type=float, default=0.95)
    parser.add_argument("--maximum-safety-divergence-rate", type=float, default=0.0)
    parser.add_argument("--maximum-candidate-error-rate", type=float, default=0.005)
    parser.add_argument("--maximum-p95-latency-overhead-ms", type=float, default=250.0)
    parser.add_argument("--confirm-shadow-only", action="store_true")
    args = parser.parse_args()
    if not args.confirm_shadow_only:
        raise SystemExit("--confirm-shadow-only is required.")
    payload = json.loads(args.observations.read_text(encoding="utf-8"))
    observations = tuple(
        Module2ShadowObservation(**dict(item))
        for item in payload.get("observations") or ()
    )
    report = evaluate_module2_shadow_readiness(
        observations,
        policy=Module2ShadowPolicy(
            minimum_samples=args.minimum_samples,
            minimum_agreement_rate=args.minimum_agreement_rate,
            maximum_safety_divergence_rate=args.maximum_safety_divergence_rate,
            maximum_candidate_error_rate=args.maximum_candidate_error_rate,
            maximum_p95_latency_overhead_ms=(
                args.maximum_p95_latency_overhead_ms
            ),
        ),
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
                "shadow_release_ready": report["shadow_release_ready"],
                "output": str(args.output),
                "output_sha256": hashlib.sha256(
                    args.output.read_bytes()
                ).hexdigest(),
                "candidate_output_routed_to_user": False,
                "production_routing_enabled": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
