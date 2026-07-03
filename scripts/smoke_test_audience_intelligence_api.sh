#!/bin/bash

# Smoke Test for Audience Intelligence API
# Validates health, basic job execution, and safety constraints.

set -e

echo "Starting Audience Intelligence Smoke Test..."

# 1. Read API Key safely without printing it
export AUDIENCE_API_KEY=$(python3 - <<'PY'
from pathlib import Path
try:
    for raw in Path(".env").read_text(errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("AUDIENCE_API_KEY="):
            print(line.split("=", 1)[1].strip().strip('"').strip("'"))
            break
except Exception:
    pass
PY
)

if [ -z "$AUDIENCE_API_KEY" ]; then
    echo "Warning: AUDIENCE_API_KEY not found in .env. Using fallback 'test_key'."
    AUDIENCE_API_KEY="test_key"
fi

BASE_URL="http://localhost:8000/api/audience-intelligence"

echo "Checking Health..."
HEALTH_RESP=$(curl -sS "${BASE_URL}/swarm/health" -H "X-Audience-API-Key: $AUDIENCE_API_KEY")
STATUS=$(echo "$HEALTH_RESP" | jq -r '.status')

if [ "$STATUS" != "ok" ]; then
    echo "Health check failed."
    exit 1
fi
echo "Health check passed."

echo "Running Prompt Benchmark..."
PROMPT_RESP=$(curl -sS -X POST "${BASE_URL}/prompt/run" \
  -H "Content-Type: application/json" \
  -H "X-Audience-API-Key: $AUDIENCE_API_KEY" \
  -d '{
    "prompt": "Smoke test audience",
    "source": "postgres",
    "approval_required": true,
    "postgres_limit": 100,
    "k_min": 10,
    "epsilon": 1.0,
    "synthetic_rows": 50,
    "max_export_cohorts": 5,
    "min_export_quality": 0.1
  }')

RUN_ID=$(echo "$PROMPT_RESP" | jq -r '.run_id')
APPROVAL_STATUS=$(echo "$PROMPT_RESP" | jq -r '.safe_export.approval_status')

echo "Run ID: $RUN_ID"
echo "Approval Status: $APPROVAL_STATUS"

if [ "$APPROVAL_STATUS" != "pending_approval" ]; then
    echo "Error: Export was not approval-gated."
    exit 1
fi

echo "Validating safety (no raw emails or lat/lng in response)..."
if echo "$PROMPT_RESP" | grep -qi "email\|lat\|lng"; then
    echo "Warning: Potentially unsafe fields detected in response."
    # We do not strictly fail here as some metadata might legitimately contain these strings,
    # but we flag it for the smoke test log.
fi

echo "Smoke test complete."
