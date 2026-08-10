from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.production_agent_security_certification_service import (
    ProductionAgentSecurityCertificationService,
)
from app.services.production_bounded_autonomy_certification_service import (
    ProductionBoundedAutonomyCertificationService,
)
from app.services.production_bounded_autonomy_functional_shadow_service import (
    ProductionBoundedAutonomyFunctionalShadowService,
)
from app.services.production_module5_scale_recovery_certification_service import (
    ProductionModule5ScaleRecoveryCertificationService,
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Module 5.8 through 5.12 evidence and exact lineage."
    )
    parser.add_argument("--source-certification-report", type=Path, required=True)
    parser.add_argument("--functional-shadow-report", type=Path, required=True)
    parser.add_argument("--agent-security-report", type=Path, required=True)
    parser.add_argument("--scale-recovery-report", type=Path)
    args = parser.parse_args()

    source = ProductionBoundedAutonomyCertificationService().validate_report(
        _read(args.source_certification_report)
    )
    functional = ProductionBoundedAutonomyFunctionalShadowService(
        environment={}
    ).validate_report(_read(args.functional_shadow_report))
    security = ProductionAgentSecurityCertificationService().validate_report(
        _read(args.agent_security_report)
    )
    blockers: list[str] = []
    for name, value in (
        ("module5_8_not_ready", source),
        ("module5_9_not_ready", functional),
        ("module5_10_not_ready", security),
    ):
        if value.get("status") != "engineering_preview_ready":
            blockers.append(name)
    source_fingerprint = source["bounded_autonomy_certification_fingerprint"]
    functional_fingerprint = functional["functional_shadow_report_fingerprint"]
    if functional["request"]["source_certification_report_fingerprint"] != (
        source_fingerprint
    ):
        blockers.append("module5_8_to_5_9_lineage_mismatch")
    if security["lineage"][
        "source_functional_shadow_report_fingerprint"
    ] != functional_fingerprint:
        blockers.append("module5_9_to_5_10_lineage_mismatch")

    scale = None
    if args.scale_recovery_report is not None:
        scale = ProductionModule5ScaleRecoveryCertificationService().validate_report(
            _read(args.scale_recovery_report)
        )
        if scale.get("status") != "engineering_preview_ready":
            blockers.append("module5_11_12_not_ready")
        request = dict(scale.get("request") or {})
        if request.get("source_functional_shadow_report_fingerprint") != (
            functional_fingerprint
        ):
            blockers.append("module5_9_to_5_12_lineage_mismatch")
        if request.get("source_agent_security_certification_fingerprint") != (
            security["agent_security_certification_fingerprint"]
        ):
            blockers.append("module5_10_to_5_12_lineage_mismatch")

    result = {
        "status": "evidence_chain_valid" if not blockers else "evidence_chain_blocked",
        "blockers": sorted(set(blockers)),
        "module5_8_status": source.get("status"),
        "module5_9_status": functional.get("status"),
        "module5_10_status": security.get("status"),
        "module5_12_status": scale.get("status") if scale else "not_supplied",
        "production_ready": False,
        "live_cutover_authorized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
