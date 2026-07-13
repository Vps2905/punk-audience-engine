import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from app.core.production_guardrails import (
    require_local_file_storage_allowed,
)


LINEAGE_DIR = Path("data/lineage")


def write_lineage(job_id: str, lineage: Dict[str, Any]) -> str:
    require_local_file_storage_allowed(
        "legacy ingestion lineage"
    )

    LINEAGE_DIR.mkdir(parents=True, exist_ok=True)

    lineage["job_id"] = job_id
    lineage["created_at"] = datetime.utcnow().isoformat() + "Z"

    output_path = LINEAGE_DIR / f"{job_id}_lineage.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2)

    return str(output_path)
