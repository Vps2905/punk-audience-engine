# Production Application and Deployment Security Hardening

This layer provides fail-closed application and container controls plus a
non-secret evidence chain for preproduction security review. It does not scan
for or return credential values, mutate infrastructure, remediate findings, or
authorize a production release.

## Enforced application boundaries

- Production API documentation is hidden unless `EXPOSE_API_DOCS=true` is
  explicitly approved.
- API responses receive anti-sniffing, anti-framing, referrer, permissions and
  cross-origin resource policy headers. Production responses also receive HSTS.
- API responses are non-cacheable and use a restrictive content security policy.
- `PRODUCTION_ALLOWED_HOSTS` optionally rejects untrusted Host headers.
- `MAX_REQUEST_BODY_BYTES` rejects declared oversized request bodies before
  routing or authentication work begins.
- Existing API-key, signed-tenant, tenant-isolation, demo-route and local-file
  production guardrails remain authoritative.

## Container boundary

The image runs as UID/GID 10001, removes build tooling after dependency
installation, and the Compose services drop all Linux capabilities. Compose
also enables `no-new-privileges`, a read-only root filesystem and a bounded
temporary filesystem. Deployment values should come from the platform secret
manager; the checked-in example intentionally does not attest that this is
already configured.

## Evidence and approval flow

1. An operator constructs a `ProductionSecurityAssessmentRequest` and supplies
   the effective deployment environment to `ProductionSecurityPostureService`.
2. The service emits control names, boolean outcomes and evidence codes only.
   Secret values and environment values are never copied into the report.
3. Failed controls produce `fail_closed`. A failed posture cannot receive an
   `approved_for_preproduction_review` decision.
4. An approved preproduction review remains immutable, tenant-scoped and still
   carries `production_release_authorized=false`.
5. `GET /api/audience-intelligence/security/status` validates the full evidence
   chain. It reports engineering readiness, never live-production certification.

## Required deployment attestations

Set these only when they accurately describe the deployed environment:

- `REQUIRE_AUDIENCE_API_KEY=true`
- `REQUIRE_PUNK_AI_TENANT_SIGNATURE=true`
- `SECURITY_HEADERS_ENABLED=true`
- `PRODUCTION_ALLOWED_HOSTS` to an explicit comma-separated allowlist
- `PRODUCTION_CORS_ORIGINS` to HTTPS origins only
- `REQUIRE_DATABASE_TLS=true` and `REQUIRE_OUTBOUND_TLS=true`
- `CONTAINER_RUNTIME_NON_ROOT=true`
- `CONTAINER_ROOT_FILESYSTEM_READ_ONLY=true`
- `CONTAINER_NO_NEW_PRIVILEGES=true`
- `SECRETS_INJECTED_BY_SECRET_MANAGER=true` only after platform verification

All routing, autonomous mutation, automatic remediation and production-release
flags must remain disabled during evidence generation and preproduction review.
