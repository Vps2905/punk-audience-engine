# Production aggregate quality monitoring

This control plane monitors quality across ingestion, privacy, feature,
embedding, retrieval, cohort and agent domains without reading row-level data.
Every observation must reference existing evidence by SHA-256 fingerprint and
must include a sample count, healthy target range, outer critical range and
bounded drift thresholds.

## Evidence chain

1. A quality snapshot classifies each aggregate metric as healthy, warning or
   critical using its declared ranges.
2. A drift report compares two immutable snapshots from the same tenant. New,
   missing, changed and stable metrics remain explicitly distinguishable.
3. An alert plan translates drift severity into review-only actions. It never
   dispatches an external alert, quarantines a source, performs remediation,
   changes routing, activates a campaign or exports data.

The design is metric-agnostic. It does not hardcode cities, business categories,
providers, models or customer use cases. Latency, rates, scores, counts and other
finite aggregate values can use the same range contract.

## Persistence and tenant isolation

Migration `0023_production_aggregate_quality_monitoring.sql` creates immutable
snapshot, drift and alert-plan tables. All tables force tenant row-level
security using `app.tenant_id`, revoke public privileges, and reject updates and
deletes.

## Runtime gates

All monitoring and alert-planning flags default to false. Automatic remediation
and production routing must remain disabled until separate operator, security,
observability, load and recovery certification is complete.

The authenticated tenant-bound endpoint is:

`GET /api/audience-intelligence/production-quality/status`

Engineering evidence readiness does not claim live-production certification.
