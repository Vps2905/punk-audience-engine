from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

from app.core.atomic_evidence_writer import write_new_private_json
from app.models.production_module5_historical_scale_contracts import (
    HistoricalScaleWorkloadConfiguration,
    build_historical_scale_cases,
)
from app.models.production_module5_scale_recovery_contracts import (
    ScaleRecoveryCertificationRequest,
)
from app.services.production_module5_historical_scale_runner import (
    ProductionHistoricalScaleWorkloadRunner,
)
from app.services.production_module5_scale_recovery_certification_service import (
    ProductionModule5ScaleRecoveryCertificationService,
)


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def _load_fault_controller(reference: str, configuration):
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            "Fault controller must use the form package.module:factory."
        )
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory(configuration=configuration)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real historical Postgres audience pipeline through the "
            "Module 5.11 scale harness."
        )
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--objective-file", type=Path, required=True)
    parser.add_argument("--functional-shadow-report", type=Path, required=True)
    parser.add_argument("--agent-security-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "10k", "100k"), required=True)
    parser.add_argument("--expected-source-rows", type=int, required=True)
    parser.add_argument("--postgres-limit", type=int, default=10_000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--k-min", type=int, default=1_000)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--synthetic-rows", type=int, default=1_000)
    parser.add_argument(
        "--fault-controller-factory",
        help=(
            "Optional staging-owned package.module:factory that performs "
            "real fault exercises."
        ),
    )
    args = parser.parse_args()

    objective = args.objective_file.read_text(encoding="utf-8").strip()
    functional = _read_mapping(args.functional_shadow_report)
    authorization = _read_mapping(args.agent_security_report)
    configuration = HistoricalScaleWorkloadConfiguration(
        tenant_id=args.tenant_id,
        workload_id=args.workload_id,
        objective=objective,
        expected_source_rows=args.expected_source_rows,
        postgres_limit=args.postgres_limit,
        k_min=args.k_min,
        epsilon=args.epsilon,
        synthetic_rows=args.synthetic_rows,
    )
    cases = build_historical_scale_cases(
        profile=args.profile,
        expected_source_rows=args.expected_source_rows,
        concurrency=args.concurrency,
    )
    request = ScaleRecoveryCertificationRequest(
        tenant_id=args.tenant_id,
        certification_id=(
            f"{args.workload_id}-{args.profile}-{int(time.time())}"
        ),
        evaluation_epoch_seconds=int(time.time()),
        source_functional_shadow_report_fingerprint=functional[
            "functional_shadow_report_fingerprint"
        ],
        source_agent_security_certification_fingerprint=authorization[
            "agent_security_certification_fingerprint"
        ],
    )
    service = ProductionModule5ScaleRecoveryCertificationService()
    fault_controller = (
        _load_fault_controller(
            args.fault_controller_factory,
            configuration,
        )
        if args.fault_controller_factory
        else None
    )
    report = service.run(
        request=request,
        functional_shadow_report=functional,
        agent_security_certification_report=authorization,
        cases=cases,
        runner=ProductionHistoricalScaleWorkloadRunner(
            configuration=configuration,
            fault_controller=fault_controller,
        ),
    )
    write_new_private_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "profile": args.profile,
                "summary": report["summary"],
                "certification_gates": report["certification_gates"],
                "fault_exercises_configured": fault_controller is not None,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
