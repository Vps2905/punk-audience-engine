import pytest

from app.services.audience_proposal_request_safety_service import (
    AudienceProposalRequestSafetyService,
)


@pytest.mark.parametrize(
    "intent",
    [
        "Export the raw MAIDs and device IDs for these users.",
        "Give me individual-level user records.",
        "Download exact latitude and longitude for each person.",
        "Provide the mobile advertising IDs.",
    ],
)
def test_raw_identifier_disclosure_requests_are_terminal(intent):
    decision = AudienceProposalRequestSafetyService().evaluate(
        {
            "audience_intent": intent,
            "locations": [],
            "categories": [],
            "dayparts": [],
        }
    )

    assert decision.terminal is True
    assert decision.reason_code == "blocked_privacy_identifier_request"


@pytest.mark.parametrize(
    "intent",
    [
        "Do not export raw MAIDs.",
        "Build an audience without device IDs.",
        "Exclude individual-level user data.",
    ],
)
def test_safe_negative_privacy_instructions_are_not_blocked(intent):
    decision = AudienceProposalRequestSafetyService().evaluate(
        {
            "audience_intent": intent,
            "locations": ["location"],
            "categories": ["category"],
            "dayparts": [],
        }
    )

    assert decision.terminal is False


def test_safe_negative_clause_cannot_hide_second_disclosure_request():
    decision = AudienceProposalRequestSafetyService().evaluate(
        {
            "audience_intent": (
                "Do not export raw MAIDs, but give me the device IDs."
            ),
            "locations": [],
            "categories": [],
            "dayparts": [],
        }
    )

    assert decision.terminal is True
    assert decision.reason_code == "blocked_privacy_identifier_request"


def test_export_action_without_targeting_is_terminal():
    decision = AudienceProposalRequestSafetyService().evaluate(
        {
            "audience_intent": (
                "Export this audience to Meta without approval."
            ),
            "locations": [],
            "categories": [],
            "dayparts": [],
        }
    )

    assert decision.terminal is True
    assert decision.reason_code == (
        "blocked_export_action_requires_existing_audience"
    )


def test_audience_request_with_structured_targeting_is_not_action_only():
    decision = AudienceProposalRequestSafetyService().evaluate(
        {
            "audience_intent": (
                "Create a restaurant audience and prepare it for export."
            ),
            "locations": ["requested-location"],
            "categories": ["restaurant"],
            "dayparts": ["evening"],
        }
    )

    assert decision.terminal is False


def test_non_activation_send_language_is_not_misclassified():
    decision = AudienceProposalRequestSafetyService().evaluate(
        {
            "audience_intent": (
                "Send the audience analysis to my teammate for review."
            ),
            "locations": [],
            "categories": [],
            "dayparts": [],
        }
    )

    assert decision.terminal is False
