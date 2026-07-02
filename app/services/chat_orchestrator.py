from typing import Dict, Any, Optional
import re

from app.services.cohort_service import create_cohort, get_cohort
from app.services.lookalike_service import create_lookalike
from app.services.meta_export_service import generate_meta_safe_export


DEFAULT_JOB_ID_HINT = "Pass job_id when creating a cohort from chat."


def extract_cohort_id(message: str) -> Optional[str]:
    """
    Finds cohort_id from message if user provides one.
    Example: export cohort_abc123 to meta
    """
    match = re.search(r"(cohort_[a-zA-Z0-9]+)", message)
    if match:
        return match.group(1)
    return None


def detect_intent(message: str) -> str:
    """
    Simple deterministic intent parser.

    This is v1. Later this can be replaced by LLM/LangGraph.
    """
    msg = message.lower()

    if "lookalike" in msg:
        return "create_lookalike"

    if "export" in msg or "meta" in msg:
        return "export_meta"

    if "create" in msg and ("cohort" in msg or "audience" in msg):
        return "create_cohort"

    if "query" in msg or "show cohort" in msg or "get cohort" in msg:
        return "query_cohort"

    return "unknown"


def clean_query_text(message: str) -> str:
    """
    Converts chat message into a search query.

    Removes command words but keeps audience intent words.
    """
    msg = message.lower()

    removable = [
        "create",
        "cohort",
        "audience",
        "prepare",
        "export",
        "meta",
        "for",
        "to",
        "please",
        "make",
        "build"
    ]

    for word in removable:
        msg = msg.replace(word, " ")

    msg = " ".join(msg.split())

    if not msg:
        return message

    return msg


def handle_chat(
    message: str,
    job_id: Optional[str] = None,
    cohort_name: Optional[str] = None,
    top_k: int = 20,
    seed_limit: int = 1000
) -> Dict[str, Any]:
    """
    Main chat orchestration function.

    It does not directly process data.
    It calls existing services.
    """
    intent = detect_intent(message)
    cohort_id = extract_cohort_id(message)

    if intent == "create_cohort":
        if not job_id:
            return {
                "status": "needs_input",
                "intent": intent,
                "message": DEFAULT_JOB_ID_HINT
            }

        query = clean_query_text(message)
        name = cohort_name or f"Chat Cohort - {query.title()}"

        cohort_result = create_cohort(
            job_id=job_id,
            name=name,
            query=query,
            filters=None,
            top_k=top_k
        )

        return {
            "status": "completed",
            "intent": intent,
            "message": "Cohort created from chat request",
            "input_message": message,
            "query_used": query,
            "result": cohort_result
        }

    if intent == "export_meta":
        if not cohort_id:
            return {
                "status": "needs_input",
                "intent": intent,
                "message": "Please provide a cohort_id like cohort_xxxxx to export."
            }

        export_result = generate_meta_safe_export(
            cohort_id=cohort_id,
            seed_limit=seed_limit,
            approval_status="pending_approval"
        )

        return {
            "status": "completed",
            "intent": intent,
            "message": "Meta-safe export prepared from chat request",
            "input_message": message,
            "result": export_result
        }

    if intent == "create_lookalike":
        if not cohort_id:
            return {
                "status": "needs_input",
                "intent": intent,
                "message": "Please provide a source cohort_id like cohort_xxxxx for lookalike."
            }

        lookalike_result = create_lookalike(
            cohort_id=cohort_id,
            top_k=top_k
        )

        return {
            "status": "completed",
            "intent": intent,
            "message": "Lookalike created from chat request",
            "input_message": message,
            "result": lookalike_result
        }

    if intent == "query_cohort":
        if not cohort_id:
            return {
                "status": "needs_input",
                "intent": intent,
                "message": "Please provide a cohort_id like cohort_xxxxx to query."
            }

        cohort = get_cohort(cohort_id)

        return {
            "status": "completed",
            "intent": intent,
            "message": "Cohort fetched from chat request",
            "input_message": message,
            "result": cohort
        }

    return {
        "status": "unknown_intent",
        "intent": intent,
        "message": "I can create cohorts, create lookalikes, query cohorts, or prepare Meta exports.",
        "examples": [
            "Create fitness evening cohort",
            "Export cohort_xxxxx to Meta",
            "Create lookalike for cohort_xxxxx",
            "Show cohort cohort_xxxxx"
        ]
    }
