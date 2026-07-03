# Audience Intelligence Integration Contract

This document outlines the API contract for integrating Punk AI with the Audience Intelligence microservice.

## Authentication
All endpoints require authentication via an API key.

Headers:
`X-Audience-API-Key: <key>`
or
`Authorization: Bearer <key>`

## Endpoints

### 1. Prompt Run API
Executes an end-to-end audience intelligence prompt.

`POST /api/audience-intelligence/prompt/run`

**Request Body:**
```json
{
  "prompt": "Build me a high-quality restaurant and cafe evening audience for Montreal and San Francisco",
  "source": "postgres",
  "approval_required": true,
  "postgres_limit": 10000,
  "k_min": 1000,
  "epsilon": 1.0,
  "synthetic_rows": 1000,
  "max_export_cohorts": 25,
  "min_export_quality": 0.25
}
```

**Response Notes:**
- Returns a `run_id` and the status.
- If `approval_required` is true, the `safe_export.approval_status` will be `pending_approval`.

### 2. Async Job APIs
For running and monitoring long-running processes.

`POST /api/audience-intelligence/jobs/run`
`GET /api/audience-intelligence/jobs/status/{job_id}`
`GET /api/audience-intelligence/jobs/result/{job_id}`

### 3. Approval APIs
To approve an export after review.

`POST /api/audience-intelligence/jobs/approve/{job_id}`

### 4. Safe Meta Payload API
To generate the final payload for Meta (after approval).

`POST /api/audience-intelligence/modules/export/meta/{cohort_id}`

## Error and Warning Handling

### Coverage Warnings
The orchestrator may return coverage warnings if the exact requested criteria (e.g., specific city, category, or daypart) are not sufficiently represented in the safe feature table.
Example Warning: "Insufficient exact matches for 'San Francisco'. Expanded search to broader region while maintaining privacy limits."

### Privacy Rules
- All responses are aggregated.
- Raw identifiers (MAIDs, emails, etc.) are never returned.
- If k-anonymity constraints are violated, the API will fail closed and return an appropriate error message rather than exposing unsafe data.
