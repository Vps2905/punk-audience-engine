import os
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone
from app.agents.autonomous_decision_core_agent import AutonomousDecisionCoreAgent
from app.models.autonomous_decision_state import (
    AutonomousDecisionState,
    ConstraintLedger,
    ExplicitConstraint,
    SemanticDecision,
    CoverageState,
    QualityState,
    REPAIR_IMPACT_REEXECUTE_PIPELINE,
)
from app.services.decision_repair_service import DecisionRepairService
from app.services.decision_verification_service import DecisionVerificationService

os.environ["ENABLE_AUTONOMOUS_SUPERVISOR"] = "true"
os.environ["ENABLE_AUTONOMOUS_SUPERVISOR_GRAPH"] = "true"
os.environ["ALLOW_LOCAL_FILE_STORAGE"] = "true"

agent = AutonomousDecisionCoreAgent()

print("\n==== Real Local Prompt Run ====")
res = agent.run(prompt="Build a restaurant evening audience for Montreal.", output_root="/tmp/punk-test")

print("Status:", res.get("status"))
print("Exact Candidates:", res.get("prompt_selected_cohorts", "N/A"))
print("Pipeline Execution Count:", res.get("pipeline_execution_count", "N/A"))

print("\n==== System Info ====")
print("Checkpoint Backend:", type(agent.checkpointer).__name__)
print("Thread ID:", res.get("run_id"))

print("\n==== Adversarial Mocked Contradiction Graph Test ====")

# Use the ContradictionOrchestrator to prove pipeline re-execution
class ContradictionOrchestrator:
    call_count = 0

    def run(self, **kwargs):
        ContradictionOrchestrator.call_count += 1
        if ContradictionOrchestrator.call_count == 1:
            return {
                "status": "completed",
                "run_id": "contradiction_test",
                "prompt": kwargs.get("prompt"),
                "prompt_selected_cohorts": 2,
                "prompt_filter_report": {
                    "locations_detected": ["locationb"],
                    "matched_requested_locations": ["locationb"],
                    "missing_requested_locations": [],
                    "coverage_status": "complete",
                    "v2_guided_selection": {
                        "rows": 2,
                        "coverage_status": "complete",
                        "matched_requested_locations": ["locationb"],
                        "missing_requested_locations": [],
                    },
                },
                "safe_export": {
                    "approval_status": "pending_approval",
                    "downstream_export_enabled": False,
                    "exported_cohorts": 2,
                },
                "privacy_guarantees": {},
            }
        else:
            return {
                "status": "completed",
                "run_id": "contradiction_test",
                "prompt": kwargs.get("prompt"),
                "prompt_selected_cohorts": 0,
                "prompt_filter_report": {
                    "locations_detected": [],
                    "matched_requested_locations": [],
                    "missing_requested_locations": ["locationa"],
                    "coverage_status": "blocked",
                    "v2_guided_selection": {
                        "rows": 0,
                        "coverage_status": "blocked",
                        "matched_requested_locations": [],
                        "missing_requested_locations": ["locationa"],
                    },
                },
                "safe_export": {
                    "approval_status": "blocked_no_safe_exact_match",
                    "downstream_export_enabled": False,
                    "exported_cohorts": 0,
                },
                "privacy_guarantees": {},
            }

ContradictionOrchestrator.call_count = 0
contradiction_agent = AutonomousDecisionCoreAgent(
    orchestrator_factory=ContradictionOrchestrator,
)
contradiction_res = contradiction_agent.run(
    prompt="Build audience for LocationA",
    output_root="/tmp/punk-test-contradiction"
)

print("Original Ledger Locations: LocationA (from prompt)")
print("Original Semantic Interpretation: locationa (from constraint ledger)")
print("First Pipeline Result Locations: locationb (incompatible)")
print("Pipeline Execution Count:", contradiction_res.get("pipeline_execution_count"))
print("Orchestrator Call Count:", ContradictionOrchestrator.call_count)
print("Final Status:", contradiction_res.get("graph_terminal_status"))
print("Downstream Export:", contradiction_res.get("safe_export", {}).get("downstream_export_enabled", False)
      if "safe_export" in contradiction_res else False)

# Verify locationb is absent from final result
tool_results = contradiction_res
pfr = tool_results.get("prompt_filter_report") or {}
v2gs = pfr.get("v2_guided_selection") or {}
final_matched = v2gs.get("matched_requested_locations") or pfr.get("matched_requested_locations") or []
print("Final Matched Locations:", final_matched)
print("LocationB Absent:", "locationb" not in final_matched)

print("\n==== File Verification ====")
decision_dir = Path("/tmp/punk-test") / res.get("run_id") / "07_autonomous_decision"

artifacts = [
    "constraint_ledger.json",
    "semantic_decision.json",
    "execution_plan.json",
    "verification_report.json",
    "repair_history.json",
    "final_decision.json",
    "world_state.json",
    "final_explanation.md"
]

for artifact in artifacts:
    p = decision_dir / artifact
    print(f"{artifact} exists:", p.exists())
