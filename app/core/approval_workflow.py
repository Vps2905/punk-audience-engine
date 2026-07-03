from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class ApprovalWorkflow:
    """
    Creates approval requests before any downstream/export use.
    """

    def __init__(self, approval_dir: str | Path):
        self.approval_dir = Path(approval_dir)

    def create_request(
        self,
        *,
        run_id: str,
        module: str,
        output_dir: str | Path,
        artifacts: List[str],
        summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        self.approval_dir.mkdir(parents=True, exist_ok=True)

        request = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "module": module,
            "status": "pending_approval",
            "artifacts": artifacts,
            "summary": summary,
            "required_checks": [
                "privacy_review",
                "schema_review",
                "business_review",
                "export_approval",
            ],
        }

        approval_path = output_dir / "approval_request.json"
        approval_path.write_text(json.dumps(request, indent=2, default=str, allow_nan=False))

        global_approval_path = self.approval_dir / f"{run_id}_{module}_approval_request.json"
        global_approval_path.write_text(json.dumps(request, indent=2, default=str, allow_nan=False))

        return request
