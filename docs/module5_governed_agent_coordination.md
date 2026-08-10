# Module 5 governed agent coordination

Module 5 converts the existing LLM router, supervisor wrapper, LangGraph
decision path and bounded recovery foundation into a tenant-scoped engineering
control plane. It does not replace the Module 1-4 services and it does not grant
an agent permission to mutate audiences, approve a release, route production
traffic, activate a campaign or export data.

## Evidence stages

1. **5.1 minimized execution evidence** records only supervisor status, route,
   reason codes, trace event counts and bounded recovery counters. Prompt text,
   tool arguments, tool results and identifiers are excluded.
2. **5.2 supervisor policy evaluation** maps the execution route to a review-only
   recommendation. Every decision keeps production execution, mutation,
   activation and export unauthorized.
3. **5.3 circuit-breaker governance** evaluates a tenant-scoped sequence of
   execution reports. It can recommend monitoring, investigation or quarantine,
   but it never schedules a retry or changes routing.
4. **5.4 human shadow review** requires an explicit manual review reference.
   Shadow observations must prove that routing, activation and export stayed off.
5. **5.5 recovery certification planning** links circuit-breaker and human-review
   evidence. It can complete the engineering evidence chain, but production
   certification and recovery execution remain false.
6. **5.6 bounded autonomy** decomposes a goal, selects allowlisted capabilities,
   critiques results and permits one bounded replan without gaining new powers.
7. **5.7 real-service dual run** compares the bounded control decision with the
   authoritative path while the authoritative path remains unchanged.
8. **5.8 repeated shadow certification** measures agreement, diversity, terminal
   safety coverage and shadow-control latency across unseen goals.
9. **5.9 functional shadow execution** invokes the real safe Module 1–5 services
   against an isolated historical feature snapshot and compares minimized
   outputs, latency and Python allocation without writes or delivery.
10. **5.10 agent security authorization** enforces a short-lived tenant,
    scope, capability, role and separation-of-duties decision before every real
    functional handler, then certifies positive and adversarial cases without
    granting approval, delivery or production-effect authority.

## Persistence and isolation

Migration `0022_module5_governed_agent_control_plane.sql` provides immutable
PostgreSQL evidence tables. All tables force tenant row-level security using
`app.tenant_id`, revoke public privileges, block updates and deletes, and enforce
non-release constraints in SQL.

## Runtime gates

All Module 5 governed-execution flags default to false. In particular,
`MODULE5_AUTONOMOUS_MUTATION_ENABLED` and
`MODULE5_PRODUCTION_ROUTING_ENABLED`, and
`MODULE5_AGENT_PRODUCTION_EFFECT_AUTHORIZATION_ENABLED` must remain false until
separate live-data, external identity, security, load, failure-recovery and
operator certification is complete.

The authenticated tenant-bound status endpoint is:

`GET /api/audience-intelligence/module-5/status`

`module5_engineering_evidence_ready` means the governed 5.1–5.5 evidence chain is
complete. `module5_bounded_autonomy_functional_shadow_ready` additionally means
that a lineage-matched 5.9 historical functional run passed its strict
comparison and resource gates. `module5_agent_security_authorization_ready`
adds application authorization and adversarial certification with a linked
security posture. None of these values means an external identity provider,
penetration test, fresh data, staging infrastructure, activation connector, or
live production is certified.
