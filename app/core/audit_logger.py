from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class AuditLogger:
    """
    Append-only audit event logger.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def log(self, event: str, details: Dict[str, Any] | None = None) -> Dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "details": details or {},
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, allow_nan=False) + "\n")

        return record
