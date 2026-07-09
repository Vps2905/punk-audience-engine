from app.services.data_coverage_intelligence_service import DataCoverageIntelligenceService
from app.services.guardrail_decision_report_service import GuardrailDecisionReportService
from app.services.hybrid_audience_retrieval_service import HybridAudienceRetrievalService


SAFE_COHORTS = [
    {
        "location_name": "quebec",
        "primary_poi_type": "casino",
        "created_day_part": "afternoon",
        "quality_score": 0.678,
        "privacy_status": "safe",
        "trait_text": "casino entertainment gaming visitors quebec afternoon",
    },
    {
        "location_name": "quebec",
        "primary_poi_type": "casino",
        "created_day_part": "evening",
        "quality_score": 0.572,
        "privacy_status": "safe",
        "trait_text": "casino gaming evening entertainment quebec",
    },
    {
        "location_name": "montreal",
        "primary_poi_type": "clothing_store",
        "created_day_part": "evening",
        "quality_score": 0.8,
        "privacy_status": "safe",
        "trait_text": "fashion retail clothing lifestyle shopping",
    },
    {
        "location_name": "san francisco",
        "primary_poi_type": "coworking_space",
        "created_day_part": "evening",
        "quality_score": 0.7,
        "privacy_status": "safe",
        "trait_text": "office coworking business work",
    },
    {
        "location_name": "montreal",
        "primary_poi_type": "store",
        "created_day_part": "evening",
        "quality_score": 0.5,
        "privacy_status": "safe",
        "trait_text": "generic store",
    },
]


def test_quebec_casino_selects_exact_candidates():
    prompt = "Casino entertainment offer in Quebec during afternoon or evening."
    intent = {
        "business_intent": "casino",
        "locations": ["quebec"],
        "requested_categories": ["casino", "gaming"],
        "dayparts": ["afternoon", "evening"],
    }

    report = HybridAudienceRetrievalService().retrieve(prompt, intent, SAFE_COHORTS, top_k=10)
    selected = report["selected_candidates"]

    assert selected
    assert all(c["location_name"] == "quebec" for c in selected)
    assert all(c["primary_poi_type"] == "casino" for c in selected)
    assert {c["created_day_part"] for c in selected} == {"afternoon", "evening"}


def test_london_retail_blocks_wrong_city_fallback():
    prompt = "Premium lifestyle and fashion store in London during weekend evenings."
    intent = {
        "business_intent": "retail",
        "locations": ["london"],
        "requested_categories": ["retail", "fashion", "clothing_store"],
        "dayparts": ["evening"],
    }

    coverage = DataCoverageIntelligenceService().build_report(intent, SAFE_COHORTS)
    retrieval = HybridAudienceRetrievalService().retrieve(prompt, intent, SAFE_COHORTS, top_k=10)
    guardrail = GuardrailDecisionReportService().build_report(coverage, retrieval)

    assert coverage["coverage_decision"] == "requested_location_not_available"
    assert retrieval["selection_count"] == 0
    assert guardrail["decision"] == "blocked"
    assert guardrail["safe_to_export"] is False


def test_caffeine_after_office_does_not_select_coworking_when_cafe_missing():
    prompt = "Find people taking a caffeine break after office near San Francisco."
    intent = {
        "business_intent": "cafe",
        "locations": ["san francisco"],
        "requested_categories": ["cafe", "coffee", "caffeine"],
        "dayparts": ["evening"],
    }

    report = HybridAudienceRetrievalService().retrieve(prompt, intent, SAFE_COHORTS, top_k=10)

    assert report["selection_count"] == 0
    assert any(c["primary_poi_type"] == "coworking_space" for c in report["ranked_candidates"])
    assert all(c["decision"] != "selected_for_review" for c in report["ranked_candidates"])


def test_generic_store_blocked_for_specific_retail_if_not_direct_supported():
    prompt = "Find premium fashion shoppers in Montreal during evening."
    intent = {
        "business_intent": "retail",
        "locations": ["montreal"],
        "requested_categories": ["fashion", "clothing_store"],
        "dayparts": ["evening"],
    }

    report = HybridAudienceRetrievalService().retrieve(prompt, intent, SAFE_COHORTS, top_k=10)

    selected_pois = {c["primary_poi_type"] for c in report["selected_candidates"]}
    assert "clothing_store" in selected_pois
    assert "store" not in selected_pois
