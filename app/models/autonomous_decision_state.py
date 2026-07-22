from typing import Any, Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field


class ExplicitConstraint(BaseModel):
    constraint_id: str
    constraint_type: Literal["location", "category", "daypart", "date", "time", "quality", "reach", "exclusion", "budget", "currency", "destination", "other"]
    raw_text: str
    normalized_value: str
    source_span_start: Optional[int] = None
    source_span_end: Optional[int] = None
    source: str = "user_prompt"
    confidence: float = 1.0
    resolution_status: Literal["resolved", "unresolved", "ambiguous", "unsupported", "contradicted"] = "unresolved"
    resolved_value: Optional[str] = None
    must_preserve: bool = True
    alternatives: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class ConstraintLedger(BaseModel):
    original_prompt: str
    locations: List[ExplicitConstraint] = Field(default_factory=list)
    categories: List[ExplicitConstraint] = Field(default_factory=list)
    dayparts: List[ExplicitConstraint] = Field(default_factory=list)
    date_constraints: List[ExplicitConstraint] = Field(default_factory=list)
    clock_time_constraints: List[ExplicitConstraint] = Field(default_factory=list)
    quality_objective: Optional[ExplicitConstraint] = None
    reach_objective: Optional[ExplicitConstraint] = None
    budget: Optional[ExplicitConstraint] = None
    currency: Optional[ExplicitConstraint] = None
    destination: Optional[ExplicitConstraint] = None
    exclusions: List[ExplicitConstraint] = Field(default_factory=list)
    unresolved_constraints: List[ExplicitConstraint] = Field(default_factory=list)
    ambiguous_constraints: List[ExplicitConstraint] = Field(default_factory=list)
    confirmation_required: bool = False
    ledger_version: str = "1.0"


class SemanticDecision(BaseModel):
    business_intent: str = ""
    locations: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    canonical_categories: List[str] = Field(default_factory=list)
    dayparts: List[str] = Field(default_factory=list)
    date_constraints: List[str] = Field(default_factory=list)
    time_constraints: List[str] = Field(default_factory=list)
    quality_intent: Literal["high", "balanced", "broad", "unknown"] = "unknown"
    quality_preference_score: float = 0.5
    reach_preference_score: float = 0.5
    objective_relationship: Literal["quality_dominant", "co_optimize", "reach_dominant", "ambiguous"] = "ambiguous"
    budget: Optional[float] = None
    currency: Optional[str] = None
    destination: Optional[str] = None
    exclusions: List[str] = Field(default_factory=list)
    confidence_by_field: Dict[str, float] = Field(default_factory=dict)
    evidence_by_field: Dict[str, str] = Field(default_factory=dict)
    alternatives_considered: List[str] = Field(default_factory=list)
    unresolved_fields: List[str] = Field(default_factory=list)
    quality_reasoning: str = ""
    evidence_source: str = ""
    confidence: float = 0.0
    resolver_mode: str = "hybrid"
    model_provider: str = ""
    model_used: str = ""
    semantic_version: str = "1.0"


class Capability(BaseModel):
    capability_id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    dependencies: List[str] = Field(default_factory=list)
    mandatory_safety_checks: List[str] = Field(default_factory=list)
    idempotent: bool = True
    retry_policy: str = "exponential_backoff"
    failure_mode: str = "fail_closed"
    service_reference: str


class PlanStep(BaseModel):
    step_id: str
    capability_id: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    output_reference: str
    dependencies: List[str] = Field(default_factory=list)
    mandatory: bool = True
    safety_requirements: List[str] = Field(default_factory=list)
    retry_policy: str = "none"
    failure_behaviour: str = "abort_plan"
    status: Literal["pending", "executing", "completed", "failed", "skipped"] = "pending"


class ExecutionPlan(BaseModel):
    plan_id: str
    request_id: str
    steps: List[PlanStep] = Field(default_factory=list)
    mandatory_steps: List[str] = Field(default_factory=list)
    optional_steps: List[str] = Field(default_factory=list)
    plan_version: str = "1.0"


class CoverageState(BaseModel):
    requested_locations: List[str] = Field(default_factory=list)
    evaluated_locations: List[str] = Field(default_factory=list)
    matched_locations: List[str] = Field(default_factory=list)
    missing_locations: List[str] = Field(default_factory=list)
    requested_categories: List[str] = Field(default_factory=list)
    category_matches: List[str] = Field(default_factory=list)
    requested_dayparts: List[str] = Field(default_factory=list)
    daypart_matches: List[str] = Field(default_factory=list)
    exact_candidates_by_location: Dict[str, int] = Field(default_factory=dict)
    exact_candidate_count: int = 0
    coverage_status: Literal["complete", "partial", "blocked", "unresolved"] = "unresolved"


class QualityState(BaseModel):
    quality_intent: str = "unknown"
    objective_relationship: str = "ambiguous"
    quality_reasoning: str = ""
    evidence_source: str = ""
    confidence: float = 0.0
    evaluated_locations: List[str] = Field(default_factory=list)
    locations_meeting_quality: List[str] = Field(default_factory=list)
    locations_missing_quality: List[str] = Field(default_factory=list)
    candidates_before: int = 0
    candidates_after: int = 0
    excluded_count: int = 0
    quality_score_field: str = ""
    quality_score_sources: List[str] = Field(default_factory=list)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    quality_policy_status: Literal["not_requested", "satisfied", "partial", "unmet", "pending_final_quality_evaluation"] = "pending_final_quality_evaluation"


class VerificationIssue(BaseModel):
    issue_code: str
    severity: Literal["info", "warning", "error", "critical"]
    description: str
    expected: str
    observed: str
    evidence: str
    repairable: bool = False
    recommended_action: str = ""


REPAIR_IMPACT_REEXECUTE_PIPELINE = "reexecute_pipeline"
REPAIR_IMPACT_REASSEMBLE_STATE = "reassemble_state"
REPAIR_IMPACT_EXPLANATION_ONLY = "explanation_only"
REPAIR_IMPACT_NON_REPAIRABLE = "non_repairable"


class RepairRecord(BaseModel):
    attempt: int
    issue_codes: List[str]
    repair_strategy: str
    repair_impact: str = REPAIR_IMPACT_REASSEMBLE_STATE
    fields_changed: List[str]
    before_state_reference: str
    after_state_reference: str
    verification_result: str
    timestamp: str


class AutonomousDecisionState(BaseModel):
    request_id: str = ""
    run_id: str = ""
    original_prompt: str = ""
    constraint_ledger: Optional[ConstraintLedger] = None
    semantic_decision: Optional[SemanticDecision] = None
    execution_plan: Optional[ExecutionPlan] = None
    completed_steps: List[str] = Field(default_factory=list)
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    pipeline_execution_count: int = 0
    coverage_state: Optional[CoverageState] = None
    quality_state: Optional[QualityState] = None
    privacy_state: Dict[str, Any] = Field(default_factory=dict)
    freshness_state: Dict[str, Any] = Field(default_factory=dict)
    approval_state: Dict[str, Any] = Field(default_factory=dict)
    export_state: Dict[str, Any] = Field(default_factory=dict)
    contradictions: List[VerificationIssue] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    repair_history: List[RepairRecord] = Field(default_factory=list)
    repair_attempts: int = 0
    last_repair_impact: str = REPAIR_IMPACT_REASSEMBLE_STATE
    final_decision: str = ""
    explanation: str = ""
    status: Literal[
        "interpreting",
        "planning",
        "executing",
        "verifying",
        "repairing",
        "completed",
        "needs_clarification",
        "blocked_unresolved_constraint",
        "blocked_verification_failed",
        "blocked_no_safe_exact_match",
        "blocked_requested_quality_unmet",
        "blocked_stale_source",
        "pending_approval",
        "approved"
    ] = "interpreting"
    created_at: str = ""
    updated_at: str = ""
    state_version: str = "1.1"
