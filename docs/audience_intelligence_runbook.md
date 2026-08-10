# Audience Intelligence Runbook

This guide covers setup, execution, and troubleshooting for the Audience Intelligence microservice.

## Setup and Execution

1. **Local Setup:**
   Ensure Python 3.11+ is installed.
   ```bash
   cd ~/Downloads/punk-audience-engine
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Backend:**
   Run the loopback-only operator workspace with production guardrail
   defaults preserved:
   ```bash
   PYTHONPATH=. AUDIENCE_LOCAL_UI_ENABLED=true AUDIENCE_LOCAL_UI_TENANT_ID=punk_internal PRODUCTION_MODE=false APP_ENV=local REQUIRE_AUDIENCE_API_KEY=true SYNTHETIC_ENGINE=dp_aggregate ALLOW_SYNTHETIC_FALLBACK=false K_ANONYMITY_MIN=1000 uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

3. **Open Prompt UI:**
   Navigate in your browser to:
   `http://127.0.0.1:8000/audience-workspace`

   The local workspace uses an HttpOnly, loopback-only session. It never asks
   the operator to paste an API key, tenant signature, or server data path into
   the browser. The production `/run` API continues to require the shared
   authenticated tenant boundary.

4. **Run Health Check:**
   ```bash
   export AUDIENCE_API_KEY="your_api_key_here" # Or extract from .env
   curl -sS http://localhost:8000/api/audience-intelligence/swarm/health \
     -H "X-Audience-API-Key: $AUDIENCE_API_KEY" | jq '.status, .health_status'
   ```

5. **Run Latency Benchmark:**
   ```bash
   time curl -sS -X POST http://localhost:8000/api/audience-intelligence/prompt/run \
     -H "Content-Type: application/json" \
     -H "X-Audience-API-Key: $AUDIENCE_API_KEY" \
     -d '{
       "prompt": "Build me a high-quality restaurant and cafe evening audience for Montreal and San Francisco",
       "source": "postgres",
       "approval_required": true,
       "postgres_limit": 10000,
       "k_min": 1000,
       "epsilon": 1.0,
       "synthetic_rows": 1000,
       "max_export_cohorts": 25,
       "min_export_quality": 0.25
     }' > /tmp/audience_latency_test.json
   cat /tmp/audience_latency_test.json | jq '{
     status,
     run_id,
     prompt_selected_cohorts,
     exported_cohorts: .safe_export.exported_cohorts,
     approval_status: .safe_export.approval_status
   }'
   ```

6. **Run Tests:**
   ```bash
   PYTHONPATH=. pytest -q \
     tests/test_synthetic_engine_agent.py \
     tests/test_synthetic_engine_agent_production_paths.py \
     tests/test_privacy_layer_agent.py \
     tests/test_embedding_feature_store_agent.py \
     tests/test_cohort_management_agent.py \
     tests/test_safe_export_agent.py \
     tests/test_audience_intelligence_orchestrator_agent.py \
     tests/test_audience_job_store.py \
     tests/test_audience_swarm_monitor_agent.py \
     tests/test_audience_api_key_auth.py
   ```

7. **Run Smoke Test:**
   ```bash
   ./scripts/smoke_test_audience_intelligence_api.sh
   ```

## Troubleshooting

- **`localhost:8000` connection refused:** The `uvicorn` server is not running. Check the backend run command.
- **`401 Unauthorized`:** The `AUDIENCE_API_KEY` is missing or incorrect in the headers.
- **UI stuck on Waiting:** The async job might have failed silently or the backend is not responding. Check the backend logs.
- **`favicon.ico 404` is harmless:** This error in the browser console can be safely ignored.
- **Coverage warning explanation:** If you receive a warning that exact city/category/daypart data is unavailable, it means the required sample size in the underlying data did not meet the strict `k_min` privacy threshold. The system will safely expand the search or provide a reduced cohort rather than returning unsafe sparse data.

## Terminal safety and action boundary

The prompt endpoint is an analysis and audience-preparation interface. It is
not an activation or delivery interface. Explicit raw-identifier disclosure,
immediate export/activation, and policy-bypass language is evaluated before
source access, privacy processing, semantic retrieval, cohort generation, or
export agents are invoked.

Terminal decisions use these contracts:

- `blocked_privacy_identifier_request`
- `blocked_export_action_requires_existing_audience`
- `blocked_approval_bypass_attempt`

For these decisions, `source_mode` and `freshness_status` are
`not_evaluated`, `source_rows` is null, and the backend returns authoritative
blocked/skipped pipeline stages. No location or category mentioned in the
prompt can turn an explicit action command into an audience-analysis request.

Production approval remains a separate authenticated workflow bound to an
explicit `run_id`. Prompt text such as "approved", "skip approval", or
"ignore safety" is never treated as authorization.
