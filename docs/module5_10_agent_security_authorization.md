# Module 5.10 — Agent Security and Authorization Certification

## Purpose

Module 5.10 adds a mandatory authorization decision immediately before each
real Module 5.9 capability handler. Planning an allowlisted capability is not
enough to execute it: the caller must also present a short-lived,
tenant-bound principal with the required scope and an explicit capability
allowlist entry.

This stage certifies the application authorization boundary and its minimized
evidence. It does not claim external identity-provider, penetration-test,
fresh-data, activation-connector, infrastructure-scale, or live-production
certification.

## Enforcement model

Every non-terminal functional-shadow invocation is checked for:

- request and tenant lineage;
- principal issue and expiry time, with a maximum one-hour lifetime;
- the exact capability ID in the principal allowlist;
- a scope matching the capability action;
- consistency between risk class and action, so a production effect cannot be
  disguised as a read;
- execution-mode boundaries;
- the principal role; and
- an approval chain with three distinct actors before delivery authorization.

The bounded agent and system worker roles can read and propose. A human
reviewer can review and approve, but cannot act as the bounded agent. A
delivery service can receive delivery authorization only after a separate
proposer and approver have produced valid, tenant-matched, unexpired approval
evidence. A bounded agent can never approve, deliver, or execute a production
effect.

Authorization denial occurs before the real handler is invoked. Consequently,
an expired, cross-tenant, under-scoped, or non-allowlisted caller cannot reach
the feature reader, semantic retrieval, cohort, evolution, or delivery-review
handler.

## Terminal safety ordering

Prohibited identifier requests and attempts to bypass safety or approval are
still resolved by the terminal safety preflight before source configuration,
authorization, retrieval, or cohort processing. Those reports explicitly mark
source and authorization evaluation as not performed.

## Adversarial certification

The certification suite runs positive and negative authorization cases. It
covers valid read and proposal access, tenant isolation, missing scope,
capability allowlisting, expired identity, agent self-approval and delivery,
non-production effect boundaries, missing approval, separation of duties, and
role confusion.

One positive delivery case evaluates authorization logic only. It deliberately
does not call a delivery connector or perform an activation/export. Every
certification report requires:

- all scenarios to match their expected decision;
- at least three positive and eight negative cases;
- all required control families;
- a lineage-matched, ready Module 5.9 functional report;
- a passing production security-posture report; and
- all production effects and live cutover fields to remain false.

Migration `0032_module5_agent_security_authorization_certification.sql`
provides immutable, tenant-scoped evidence with forced PostgreSQL row-level
security, public privilege revocation, and SQL checks that prevent the report
from claiming credentials, tokens, cross-tenant access, approval, delivery, or
production effects.

## Configuration

```dotenv
MODULE5_AGENT_SECURITY_CERTIFICATION_EVIDENCE_PATH=
MODULE5_AGENT_SECURITY_CERTIFICATION_ENABLED=false
MODULE5_AGENT_CAPABILITY_AUTHORIZATION_REQUIRED=true
MODULE5_AGENT_PRODUCTION_EFFECT_AUTHORIZATION_ENABLED=false
```

The certification flag defaults to false and fails closed when enabled without
valid evidence. Production-effect authorization remains a release-affecting
flag and must remain false in this stage.

## Authentication boundary

The existing API-key and signed-tenant request boundary remains useful for
local and application-level isolation, but it is not a substitute for a
production identity provider. Before a canary, staging must bind the principal
contract to a real OIDC or workload-identity issuer, validate audience and
issuer claims, rotate signing keys, and persist replay protection outside the
application process. Those controls then require external penetration and IAM
review evidence.

## Acceptance boundary

`module5_agent_security_authorization_ready=true` means the application-level
authorization scenarios, functional enforcement, lineage, and security posture
are ready for staging review. It always coexists with:

- `production_identity_provider_certified=false`;
- `external_penetration_test_completed=false`;
- `fresh_data_certified=false`;
- `live_cutover_authorized=false`;
- `agent_approval_authority=false`;
- `agent_delivery_authority=false`;
- `production_effect_performed=false`; and
- `activation_or_export_performed=false`.

The next stages are external identity-provider integration, replay-store and
key-rotation certification, staging penetration testing, then load, soak,
failure-recovery, and fresh-data shadow certification.
