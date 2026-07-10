# Module 2 Review — Synthetic Data Layer

## Validation

Default local validation:
180 passed, 3 skipped, 3 warnings

Postgres integration validation:
183 passed, 3 warnings

## Module 2 Hardening

- Legacy synthetic service defaults to production-safe aggregate sampling
- Production mode blocks SDV Gaussian generation
- Production mode blocks silent fallback engines
- Unsafe source columns such as device_id, raw IDs, email, phone, lat/lon are blocked
- Synthetic output is aggregate seed-profile only
- Synthetic manifests include production_mode and allow_fallback
- Legacy synthetic route exposes production_mode and allow_fallback controls
- MaidSwarm synthetic generation routes through SyntheticEngineAgent

## Real Artifact Validation

Real UI prompt generated:
- engine_used: DPAggregateCohortSynthesizer
- production_mode: true
- allow_fallback: false
- rows_generated: 1000
- privacy_budget_recorded: true
- unsafe synthetic columns: none
