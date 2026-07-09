from app.services.data_coverage_intelligence_service import DataCoverageIntelligenceService
from app.services.guardrail_decision_report_service import GuardrailDecisionReportService
from app.services.hybrid_audience_retrieval_service import HybridAudienceRetrievalService


SAFE_COHORTS = [
    {
        "audience_name": "Restaurant - Evening - Montreal",
        "location_name": "montreal",
        "primary_poi_type": "restaurant",
        "created_day_part": "evening",
        "quality_score": 0.6,
        "privacy_status": "passed",
    },
    {
        "audience_name": "Coworking Space - Evening - San Francisco",
        "location_name": "san francisco",
        "primary_poi_type": "coworking_space",
        "created_day_part": "evening",
        "quality_score": 0.7,
        "privacy_status": "passed",
    },
    {
        "audience_name": "Health - Afternoon - Montreal",
        "location_name": "montreal",
        "primary_poi_type": "health",
        "created_day_part": "afternoon",
        "quality_score": 0.5,
        "privacy_status": "passed",
    },
]


def test_multi_location_partial_coverage_reports_missing_city_but_keeps_safe_matches():
    intent = {
        "business_intent": "restaurant",
        "locations": ["montreal", "san francisco"],
        "requested_categories": ["restaurant"],
        "dayparts": ["evening"],
    }

    coverage = DataCoverageIntelligenceService().build_report(intent, SAFE_COHORTS)
    retrieval = HybridAudienceRetrievalService().retrieve(
        prompt="Build restaurant evening audience for Montreal and San Francisco",
        intent=intent,
        safe_cohorts=SAFE_COHORTS,
        top_k=10,
    )
    guardrail = GuardrailDecisionReportService().build_report(coverage, retrieval)

    assert coverage["coverage_decision"] == "partial_safe_data_available"
    assert coverage["partial_location_coverage"] is True
    assert coverage["missing_requested_categories_by_location"] == {
        "san_francisco": ["restaurant"]
    }
    assert retrieval["selection_count"] == 1
    assert retrieval["selected_candidates"][0]["location_name"] == "montreal"
    assert guardrail["decision"] == "review_required"
    assert guardrail["safe_to_export"] is False


def test_healthcare_wrong_city_wrong_category_blocks_export():
    intent = {
        "business_intent": "healthcare",
        "locations": ["san francisco"],
        "requested_categories": ["healthcare", "clinic", "pharmacy", "hospital"],
        "dayparts": ["afternoon"],
    }

    coverage = DataCoverageIntelligenceService().build_report(intent, SAFE_COHORTS)
    retrieval = HybridAudienceRetrievalService().retrieve(
        prompt="health checkup campaign in San Francisco during afternoon",
        intent=intent,
        safe_cohorts=SAFE_COHORTS,
        top_k=10,
    )
    guardrail = GuardrailDecisionReportService().build_report(coverage, retrieval)

    assert coverage["coverage_decision"] in {
        "requested_category_not_available",
        "partial_safe_data_available",
    }
    assert retrieval["selection_count"] == 0
    assert guardrail["decision"] == "blocked"
    assert guardrail["safe_to_export"] is False
