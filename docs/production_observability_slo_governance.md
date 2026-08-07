# Production observability and SLO governance

This control plane evaluates privacy-minimized telemetry coverage and
service-level objectives across arbitrary Punk AI components. It operates only
on aggregate counters, rates and finite SLI values. Prompt text, request and
response payloads, raw identifiers and sensitive telemetry attributes are not
accepted as evidence.

## Telemetry coverage

Each component declares metrics, logs or traces coverage with emitted and
accepted event counts, sensitive-attribute rejection counts, correlation
coverage, export success and minimum required rates. Missing signals are
critical; coverage below a declared minimum is a warning.

## Service-level objectives

SLO evidence supports both `at_least` and `at_most` objectives. Each observation
declares a target, warning boundary, critical boundary, evidence window and
minimum sample count. Results are classified as met, at risk, warning, critical
or insufficient data without hardcoding a provider, service, metric or region.

## Incident review planning

Every telemetry and SLO result receives a deterministic review action. Plans do
not dispatch alerts, declare incidents, remediate services, change production
routing, activate campaigns or export data. Release-affecting actions require a
separate human-controlled incident process.

Migration `0024_production_observability_slo_governance.sql` stores immutable
tenant-scoped evidence with forced row-level security, revoked public access and
update/delete prevention.

The authenticated tenant-bound endpoint is:

`GET /api/audience-intelligence/observability/status`

All runtime flags default to false. Engineering evidence readiness is not
live-production certification.
