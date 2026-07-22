import pandas as pd
from typing import Any

def norm_text(text: Any) -> str:
    if pd.isna(text) or not text:
        return ""
    return str(text).strip().lower()

def location_matches_request(cohort_location: Any, requested_location: str) -> bool:
    cohort = norm_text(cohort_location)
    requested = norm_text(requested_location)

    if not requested:
        return True
    if not cohort:
        return False
    if cohort == requested:
        return True

    import re
    # Extract alphanumeric components for granular matching
    def tokenize(text: str) -> set:
        return set(re.findall(r'\b\w+\b', text, flags=re.UNICODE))

    cohort_components = tokenize(cohort)
    req_components = tokenize(requested)

    if req_components and req_components.issubset(cohort_components):
        return True

    return False
