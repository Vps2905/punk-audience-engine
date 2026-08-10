-- Module 5.11: immutable, tenant-scoped scale and recovery evidence.
-- This table records aggregate certification reports only. It has no
-- audience-membership or release-effect capability.

CREATE TABLE IF NOT EXISTS module5_scale_recovery_certifications (
    tenant_id TEXT NOT NULL,
    certification_id TEXT NOT NULL,
    environment TEXT NOT NULL CHECK (
        environment IN ('staging', 'preproduction')
    ),
    policy_version TEXT NOT NULL,
    source_functional_shadow_report_fingerprint CHAR(64) NOT NULL,
    source_agent_security_certification_fingerprint CHAR(64) NOT NULL,
    scale_recovery_certification_fingerprint CHAR(64) NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'engineering_preview_ready',
            'engineering_preview_blocked'
        )
    ),
    total_work_units BIGINT NOT NULL CHECK (total_work_units >= 0),
    invocation_count INTEGER NOT NULL CHECK (invocation_count >= 0),
    maximum_observed_concurrency INTEGER NOT NULL CHECK (
        maximum_observed_concurrency >= 0
    ),
    latency_p95_ms DOUBLE PRECISION NOT NULL CHECK (latency_p95_ms >= 0),
    latency_p99_ms DOUBLE PRECISION NOT NULL CHECK (latency_p99_ms >= 0),
    work_units_per_second DOUBLE PRECISION NOT NULL CHECK (
        work_units_per_second >= 0
    ),
    eligible_for_staging_review BOOLEAN NOT NULL,
    report JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, certification_id),
    UNIQUE (tenant_id, scale_recovery_certification_fingerprint)
);

ALTER TABLE module5_scale_recovery_certifications
    ENABLE ROW LEVEL SECURITY;
ALTER TABLE module5_scale_recovery_certifications
    FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS module5_scale_recovery_tenant_policy
    ON module5_scale_recovery_certifications;
CREATE POLICY module5_scale_recovery_tenant_policy
    ON module5_scale_recovery_certifications
    USING (
        tenant_id = current_setting('app.tenant_id', true)
    )
    WITH CHECK (
        tenant_id = current_setting('app.tenant_id', true)
    );

CREATE OR REPLACE FUNCTION module5_scale_recovery_prevent_update_delete()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Module 5 scale/recovery evidence is immutable';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS module5_scale_recovery_prevent_update_delete
    ON module5_scale_recovery_certifications;
CREATE TRIGGER module5_scale_recovery_prevent_update_delete
BEFORE UPDATE OR DELETE ON module5_scale_recovery_certifications
FOR EACH ROW EXECUTE FUNCTION module5_scale_recovery_prevent_update_delete();
