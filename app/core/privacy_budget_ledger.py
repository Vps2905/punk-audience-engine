from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class PrivacyBudgetLedger:
    """
    Append-only JSONL privacy budget ledger.
    """

    def __init__(self, path: str | Path, max_epsilon_per_run: float = 10.0):
        self.path = Path(path)
        self.max_epsilon_per_run = float(max_epsilon_per_run)

    def record_spend(
        self,
        *,
        run_id: str,
        module: str,
        epsilon: float,
        engine: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        epsilon = float(epsilon)

        if epsilon <= 0:
            raise ValueError("epsilon must be greater than 0.")

        if epsilon > self.max_epsilon_per_run:
            raise ValueError(
                f"epsilon {epsilon} exceeds max_epsilon_per_run {self.max_epsilon_per_run}."
            )

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "module": module,
            "engine": engine,
            "epsilon_spent": epsilon,
            "max_epsilon_per_run": self.max_epsilon_per_run,
            "metadata": metadata or {},
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")

        return record
