# Privacy and Safety Policy

This document outlines the strict privacy and safety guarantees implemented in the Audience Intelligence Engine.

## Export Restrictions
- **No raw identifiers exported:** The system strictly blocks the export of any raw identifiers.
- **No raw MAIDs exported:** Mobile Advertising IDs are never included in exports.
- **No hashed IDs exported:** Even hashed identifiers are scrubbed before export.
- **No raw lat/lng exported:** Granular geolocation data is removed.
- **No emails or phones exported:** PII (Personally Identifiable Information) such as emails and phone numbers are completely blocked.
- **No individual rows exported:** Data is always aggregated; individual user records are never exposed.

## Aggregation and Anonymization
- **k-anonymity minimum:** A strict k-anonymity minimum (default: 1000) is enforced on all cohorts. Any cohort falling below this threshold is dropped or merged.
- **DP aggregation:** Differential Privacy (DP) aggregation techniques are used to inject noise and prevent linkage attacks.

## Workflow Safety
- **Approval-gated export:** All exports require explicit manual approval via the API before any data payload is generated.
- **Fail-closed behavior:** Any error or anomaly in the privacy pipeline results in an immediate halt (fail-closed), preventing accidental data leakage.
- **No silent fallback in production:** In production mode, the synthetic engine does not silently fall back to unsafe methods if generation fails.
- **Real Meta upload disabled by default:** The final step of uploading to Meta is disabled by default and requires explicit configuration and review to enable.
