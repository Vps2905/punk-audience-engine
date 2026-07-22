import uuid
from typing import Any, Dict, List
from app.models.autonomous_decision_state import ExecutionPlan, PlanStep, Capability

class AutonomousPlanService:
    def __init__(self):
        self.registry: Dict[str, Capability] = {
            "privacy_safe_cohort_retrieval": Capability(
                capability_id="privacy_safe_cohort_retrieval",
                name="Privacy-Safe Cohort Retrieval",
                description="Retrieves cohorts securely.",
                input_schema={},
                output_schema={"privacy_cohorts": "DataFrame"},
                dependencies=[],
                mandatory_safety_checks=["k-anonymity", "freshness_check"],
                service_reference="SafeRawInputAgent"
            ),
            "cohort_management": Capability(
                capability_id="cohort_management",
                name="Cohort Management",
                description="Scores and clusters cohorts.",
                input_schema={"privacy_cohorts_ref": "DataFrame"},
                output_schema={"managed_cohorts": "DataFrame"},
                dependencies=["privacy_safe_cohort_retrieval"],
                mandatory_safety_checks=[],
                service_reference="CohortManagementAgent"
            ),
            "final_quality_evaluation": Capability(
                capability_id="final_quality_evaluation",
                name="Final Quality Evaluation",
                description="Enforces strict quality policy.",
                input_schema={"managed_cohorts_ref": "DataFrame"},
                output_schema={"quality_cohorts": "DataFrame"},
                dependencies=["cohort_management"],
                mandatory_safety_checks=["strict_threshold"],
                service_reference="AudienceQualityPolicyService"
            ),
            "safe_export": Capability(
                capability_id="safe_export",
                name="Safe Export",
                description="Exports cohorts safely.",
                input_schema={"quality_cohorts_ref": "DataFrame"},
                output_schema={"export_result": "dict"},
                dependencies=["final_quality_evaluation"],
                mandatory_safety_checks=["approval_gated", "no_raw_ids"],
                service_reference="SafeExportAgent"
            )
        }

    def build_plan(self, request_id: str, prompt: str) -> ExecutionPlan:
        plan = ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            request_id=request_id,
            steps=[]
        )

        # Build steps from registry through topological sort of dependencies
        visited = set()
        temp_mark = set()
        sorted_caps = []

        def visit(cap_id):
            if cap_id in temp_mark:
                raise ValueError(f"Cyclic dependency detected at {cap_id}")
            if cap_id not in visited:
                temp_mark.add(cap_id)
                cap = self.registry[cap_id]
                for dep in cap.dependencies:
                    visit(dep)
                temp_mark.remove(cap_id)
                visited.add(cap_id)
                sorted_caps.append(cap)

        for cap_id in self.registry:
            if cap_id not in visited:
                visit(cap_id)

        for cap in sorted_caps:
            step = PlanStep(
                step_id=f"step_{cap.capability_id}",
                capability_id=cap.capability_id,
                inputs=cap.input_schema,
                output_reference=list(cap.output_schema.keys())[0] if cap.output_schema else "output",
                dependencies=[f"step_{d}" for d in cap.dependencies],
                mandatory=True,
                safety_requirements=cap.mandatory_safety_checks
            )
            plan.steps.append(step)

        self.validate_plan(plan)

        plan.mandatory_steps = [s.step_id for s in plan.steps if s.mandatory]
        return plan

    def validate_plan(self, plan: ExecutionPlan):
        capabilities = [s.capability_id for s in plan.steps]

        if "privacy_safe_cohort_retrieval" not in capabilities:
            raise ValueError("Plan omits mandatory privacy step")

        if "final_quality_evaluation" not in capabilities:
            raise ValueError("Plan omits mandatory verification/quality step")

        if "safe_export" not in capabilities:
            raise ValueError("Plan omits mandatory safe export step")

        # Check that safety requirements exist in the plan
        safety_reqs = set()
        for s in plan.steps:
            safety_reqs.update(s.safety_requirements)

        if "freshness_check" not in safety_reqs:
            raise ValueError("Plan omits mandatory freshness check")

        if "approval_gated" not in safety_reqs:
            raise ValueError("Plan omits mandatory approval gate")

        # Dependency check
        step_ids = [s.step_id for s in plan.steps]
        for step in plan.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise ValueError(f"Missing dependency {dep} for step {step.step_id}")
