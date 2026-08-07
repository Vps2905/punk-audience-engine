# Authenticated tenant API boundary

## Scope

Every business API mounted by `app.main` must pass the shared Audience
Intelligence request boundary. The dependency validates the Audience API key,
normalizes and verifies the tenant identity, assigns a log-safe request ID, and
stores the resulting context on `request.state.audience_request_context`.

The only public application endpoints are service metadata, liveness, readiness,
and local/demo UI surfaces. UI actions still call protected APIs.

## Required headers

All business requests require:

- `X-Audience-API-Key` (or an accepted existing API-key alias);
- `X-Audience-Tenant-Id`;
- `X-Audience-Tenant-Signature` in production;
- optional `X-Request-Id`, restricted to log-safe characters and 128 bytes.

Production tenant signatures are HMAC-SHA256 values produced from the normalized
tenant ID and `PUNK_AI_TENANT_AUTH_SECRET`. The secret belongs in the deployment
secret manager and must never be logged, committed, or returned by an API.

## Production route policy

Local-file pipelines and the legacy cohort/chat/export APIs are hidden with 404
responses when local file storage is disabled. Adaptive `/agents/*` routes are
also hidden unless an explicitly approved internal environment enables demo
routes. Health and readiness endpoints remain available to infrastructure.

This perimeter does not replace service-level tenant authorization. Every
durable store must continue to filter by the verified tenant, set the database
tenant context before queries, and enforce row-level security. New routes must
be registered through `_include_audience_router` or declare
`require_authenticated_audience_request` directly when a router also contains a
public UI endpoint.

## Validation

The route inventory test fails whenever a new business API is mounted without
the shared boundary. Regression tests also verify missing credentials, tenant
signature failure, request-ID propagation, production hiding of legacy/demo
routes, and the public health exception.
