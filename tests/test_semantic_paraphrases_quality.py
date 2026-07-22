import pytest
from app.agents.autonomous_decision_core_agent import AutonomousDecisionCoreAgent
from app.models.autonomous_decision_state import AutonomousDecisionState, ConstraintLedger, ExplicitConstraint

def _run_semantic_interpretation(prompt: str) -> str:
    agent = AutonomousDecisionCoreAgent()
    state = AutonomousDecisionState(request_id="test_req", original_prompt=prompt)
    # mock constraints for fallback
    state.constraint_ledger = ConstraintLedger(
        original_prompt=prompt,
        locations=[],
        categories=[],
        dayparts=[],
    )
    # run interpret semantics
    state = agent._interpret_semantics(state)
    return state.semantic_decision.quality_intent

def test_semantic_paraphrases_balanced():
    assert _run_semantic_interpretation("sensible balance between reliability and scale") == "balanced"
    assert _run_semantic_interpretation("quality and reach equally important") == "balanced"
    assert _run_semantic_interpretation("scale mentioned without sacrificing quality") == "balanced"
    assert _run_semantic_interpretation("reliability mentioned without sacrificing reach") == "balanced"

def test_semantic_paraphrases_broad():
    assert _run_semantic_interpretation("maximize reach and accept varying quality") == "broad"
    assert _run_semantic_interpretation("we want the widest volume of users") == "broad"

def test_semantic_paraphrases_high():
    assert _run_semantic_interpretation("prioritize dependable cohorts even with smaller scale") == "high"
    assert _run_semantic_interpretation("we need the best premium users") == "high"

def test_propagation_and_evaluated_locations():
    # Test that matched location enters quality evaluated_locations; unsupported location does not
    # and completed coverage is not described as unresolved.
    agent = AutonomousDecisionCoreAgent()
    state = AutonomousDecisionState(request_id="test_req", original_prompt="Build audience for Montreal")
    
    # Mock constraints
    state.constraint_ledger = ConstraintLedger(
        original_prompt="Build audience for Montreal",
        locations=[
            ExplicitConstraint(constraint_id="1", constraint_type="location", raw_text="Montreal", normalized_value="montreal", resolution_status="unresolved")
        ],
        categories=[],
        dayparts=[],
    )
    
    # Mock semantic decision
    state = agent._interpret_semantics(state)
    
    # Mock tool results for assemble_world_state
    state.tool_results = {
        "prompt_filter_report": {
            "locations_detected": ["montreal"],
            "matched_requested_locations": ["montreal"],
            "missing_requested_locations": [],
            "coverage_status": "complete",
            "quality_policy_report": {
                "quality_intent": "balanced",
                "quality_policy_status": "satisfied",
                "quality_thresholds_by_location": {"montreal": 0.5},
                "quality_candidates_before": 4,
                "quality_candidates_after": 4,
                "locations_meeting_quality": ["montreal"],
            },
            "v2_guided_selection": {
                "rows": 4,
                "coverage_status": "complete",
                "matched_requested_locations": ["montreal"],
                "missing_requested_locations": [],
                "block_export": False,
                "downstream_export_enabled": False,
            },
        },
        "safe_export": {
            "approval_status": "blocked_no_safe_exact_match",
            "downstream_export_enabled": False,
            "exported_cohorts": 0,
        },
    }
    
    state = agent._assemble_world_state(state)
    
    # Check evaluated_locations
    assert "montreal" in state.quality_state.evaluated_locations
    
    # Check that unsupported location does not enter
    state.tool_results["prompt_filter_report"]["locations_detected"] = []
    state.tool_results["prompt_filter_report"]["matched_requested_locations"] = []
    state.tool_results["prompt_filter_report"]["v2_guided_selection"]["matched_requested_locations"] = []
    state.tool_results["prompt_filter_report"]["missing_requested_locations"] = ["montreal"]
    state.tool_results["prompt_filter_report"]["v2_guided_selection"]["missing_requested_locations"] = ["montreal"]
    state.tool_results["prompt_filter_report"]["coverage_status"] = "blocked"
    state.tool_results["prompt_filter_report"]["v2_guided_selection"]["coverage_status"] = "blocked"
    
    state2 = agent._assemble_world_state(state)
    assert "montreal" not in state2.quality_state.evaluated_locations
    
def test_explanation_formatting():
    from app.services.autonomous_explanation_service import AutonomousExplanationService
    from pathlib import Path
    
    svc = AutonomousExplanationService()
    
    agent = AutonomousDecisionCoreAgent()
    state = AutonomousDecisionState(request_id="test_req", original_prompt="Build audience for Montreal", status="completed")
    
    # Mock constraints
    state.constraint_ledger = ConstraintLedger(
        original_prompt="Build audience for Montreal",
        locations=[
            ExplicitConstraint(constraint_id="1", constraint_type="location", raw_text="Montreal", normalized_value="montreal", resolution_status="unresolved")
        ],
        categories=[],
        dayparts=[],
    )
    
    # Mock coverage complete
    from app.models.autonomous_decision_state import CoverageState
    state.coverage_state = CoverageState(
        matched_locations=["montreal"],
        missing_locations=[],
        coverage_status="complete"
    )
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        expl = svc.build_explanation(state, Path(tmpdir))
        
        # Verify it does not say unresolved
        assert "coverage result: matched" in expl
        assert "(unresolved)" not in expl
