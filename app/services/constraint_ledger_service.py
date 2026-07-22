import json
import os
import uuid
from typing import Any, Dict, List, Optional
from app.models.autonomous_decision_state import ConstraintLedger, ExplicitConstraint
from app.services.llm_model_router_service import LLMModelRouterService, LLMResponseValidationError
from app.agents.semantic_prompt_intelligence_agent import SemanticPromptIntelligenceAgent

class ConstraintLedgerService:
    def capture(self, prompt: str) -> ConstraintLedger:
        fallback = SemanticPromptIntelligenceAgent().analyze_prompt(prompt)

        # Try LLM
        enabled = os.getenv("ENABLE_LLM_INTENT", "false").strip().lower() in {"1", "true", "yes", "on"}

        if enabled:
            provider = os.getenv("LLM_INTENT_PROVIDER", "openrouter").strip().lower()
            model = os.getenv("LLM_INTENT_MODEL", "").strip()
            api_key = os.getenv("LLM_INTENT_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            if provider == "openrouter":
                base_url = os.getenv("LLM_INTENT_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
            elif provider == "openai":
                base_url = os.getenv("LLM_INTENT_BASE_URL", "https://api.openai.com/v1/chat/completions")
            else:
                base_url = os.getenv("LLM_INTENT_BASE_URL", "").strip()

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Extract EXPLICIT constraints from the user prompt into a strict JSON schema. "
                        "Do not infer, guess, or map to internal categories. Just record what the user literally asked for. "
                        "Preserve explicitly supplied entities exactly, including entities not present in known data. "
                        "Return JSON only with keys: locations (list of str), categories (list of str), dayparts (list of str), "
                        "quality_objective (str or null), reach_objective (str or null), exclusions (list of str)."
                    )
                },
                {"role": "user", "content": prompt}
            ]

            def validator(content: str) -> dict[str, Any]:
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()
                return json.loads(content)

            try:
                routed = LLMModelRouterService().route(
                    messages=messages,
                    primary_provider=provider,
                    primary_model=model,
                    primary_api_key=api_key,
                    primary_base_url=base_url,
                    timeout_seconds=20,
                    max_tokens=1000,
                    validator=validator,
                )
                parsed = routed["validated"]
                return self._build_ledger(prompt, parsed)
            except Exception as e:
                pass

        import re
        locs = []
        # Extract multiple locations using a robust pattern
        # Matches 'in Montreal' or 'in Montreal and Reykjavik'
        match = re.search(r"(?:for|in|near|at|around)\s+((?:[A-Z][a-zA-Z]+\s*(?:and\s+|,\s*)?)+)", prompt)
        if match:
            loc_str = match.group(1).strip()
            # Split by ' and ', ',', etc.
            raw_locs = re.split(r'\s+and\s+|,\s*', loc_str)
            locs = [l.strip() for l in raw_locs if l.strip() and l.strip()[0].isupper()]

        if not locs:
            locs = fallback.get("locations_detected", [])

        cats = fallback.get("poi_terms_detected", [])
        if not cats:
            import re
            m = re.search(r"(?:a|an)\s+([a-zA-Z]+)\s+(?:evening|morning|afternoon|audience)", prompt)
            if m:
                cats = [m.group(1).strip()]

        return self._build_ledger(prompt, {
            "locations": locs,
            "categories": cats,
            "dayparts": fallback.get("dayparts", []),
            "quality_objective": fallback.get("quality_intent") if fallback.get("quality_intent") != "unknown" else None,
            "reach_objective": None,
            "exclusions": [],
        })

    def _build_ledger(self, prompt: str, parsed: Dict[str, Any]) -> ConstraintLedger:
        ledger = ConstraintLedger(original_prompt=prompt)

        for loc in parsed.get("locations", []):
            if isinstance(loc, str) and loc.strip():
                ledger.locations.append(ExplicitConstraint(
                    constraint_id=str(uuid.uuid4()),
                    constraint_type="location",
                    raw_text=loc,
                    normalized_value=loc.lower().strip()
                ))

        for cat in parsed.get("categories", []):
            if isinstance(cat, str) and cat.strip():
                ledger.categories.append(ExplicitConstraint(
                    constraint_id=str(uuid.uuid4()),
                    constraint_type="category",
                    raw_text=cat,
                    normalized_value=cat.lower().strip()
                ))

        for dp in parsed.get("dayparts", []):
            if isinstance(dp, str) and dp.strip():
                ledger.dayparts.append(ExplicitConstraint(
                    constraint_id=str(uuid.uuid4()),
                    constraint_type="daypart",
                    raw_text=dp,
                    normalized_value=dp.lower().strip()
                ))

        if parsed.get("quality_objective") and isinstance(parsed["quality_objective"], str):
            ledger.quality_objective = ExplicitConstraint(
                constraint_id=str(uuid.uuid4()),
                constraint_type="quality",
                raw_text=parsed["quality_objective"],
                normalized_value=parsed["quality_objective"].lower().strip()
            )

        if parsed.get("reach_objective") and isinstance(parsed["reach_objective"], str):
            ledger.reach_objective = ExplicitConstraint(
                constraint_id=str(uuid.uuid4()),
                constraint_type="reach",
                raw_text=parsed["reach_objective"],
                normalized_value=parsed["reach_objective"].lower().strip()
            )

        for exc in parsed.get("exclusions", []):
            if isinstance(exc, str) and exc.strip():
                ledger.exclusions.append(ExplicitConstraint(
                    constraint_id=str(uuid.uuid4()),
                    constraint_type="exclusion",
                    raw_text=exc,
                    normalized_value=exc.lower().strip()
                ))

        return ledger
