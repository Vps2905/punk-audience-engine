-- Module 1 control-plane tenant isolation.
--
-- Tenant identity is derived from the authenticated database role, not from
-- a caller-settable custom GUC. Runtime roles cannot edit this mapping.

CREATE TABLE IF NOT EXISTS provider_runtime_tenants (
    role_name NAME PRIMARY KEY,
    tenant_id TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

REVOKE ALL ON provider_runtime_tenants FROM PUBLIC;

CREATE OR REPLACE FUNCTION provider_runtime_tenant()
RETURNS TEXT
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT mapping.tenant_id
    FROM public.provider_runtime_tenants mapping
    WHERE mapping.role_name = session_user
$$;

REVOKE ALL ON FUNCTION provider_runtime_tenant() FROM PUBLIC;

ALTER TABLE provider_dataset_contracts ENABLE ROW LEVEL SECURITY;
ALTER TABLE provider_dataset_contracts FORCE ROW LEVEL SECURITY;
ALTER TABLE provider_ingestion_objects ENABLE ROW LEVEL SECURITY;
ALTER TABLE provider_ingestion_objects FORCE ROW LEVEL SECURITY;
ALTER TABLE provider_ingestion_replays ENABLE ROW LEVEL SECURITY;
ALTER TABLE provider_ingestion_replays FORCE ROW LEVEL SECURITY;
ALTER TABLE provider_privacy_releases ENABLE ROW LEVEL SECURITY;
ALTER TABLE provider_privacy_releases FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS provider_dataset_contracts_tenant
ON provider_dataset_contracts;
CREATE POLICY provider_dataset_contracts_tenant
ON provider_dataset_contracts
USING (tenant_id = provider_runtime_tenant())
WITH CHECK (tenant_id = provider_runtime_tenant());

DROP POLICY IF EXISTS provider_ingestion_objects_tenant
ON provider_ingestion_objects;
CREATE POLICY provider_ingestion_objects_tenant
ON provider_ingestion_objects
USING (tenant_id = provider_runtime_tenant())
WITH CHECK (tenant_id = provider_runtime_tenant());

DROP POLICY IF EXISTS provider_ingestion_replays_tenant
ON provider_ingestion_replays;
CREATE POLICY provider_ingestion_replays_tenant
ON provider_ingestion_replays
USING (
    EXISTS (
        SELECT 1
        FROM provider_ingestion_objects ingestion
        WHERE ingestion.ingestion_id = provider_ingestion_replays.ingestion_id
          AND ingestion.tenant_id = provider_runtime_tenant()
    )
)
WITH CHECK (
    EXISTS (
        SELECT 1
        FROM provider_ingestion_objects ingestion
        WHERE ingestion.ingestion_id = provider_ingestion_replays.ingestion_id
          AND ingestion.tenant_id = provider_runtime_tenant()
    )
);

DROP POLICY IF EXISTS provider_privacy_releases_tenant
ON provider_privacy_releases;
CREATE POLICY provider_privacy_releases_tenant
ON provider_privacy_releases
USING (tenant_id = provider_runtime_tenant())
WITH CHECK (tenant_id = provider_runtime_tenant());

DROP POLICY IF EXISTS provider_privacy_windows_tenant
ON provider_privacy_windows;
CREATE POLICY provider_privacy_windows_tenant
ON provider_privacy_windows
USING (tenant_id = provider_runtime_tenant())
WITH CHECK (tenant_id = provider_runtime_tenant());

DROP POLICY IF EXISTS provider_privacy_window_partitions_tenant
ON provider_privacy_window_partitions;
CREATE POLICY provider_privacy_window_partitions_tenant
ON provider_privacy_window_partitions
USING (
    EXISTS (
        SELECT 1 FROM provider_privacy_windows AS privacy_window
        WHERE privacy_window.window_key = provider_privacy_window_partitions.window_key
          AND privacy_window.tenant_id = provider_runtime_tenant()
    )
)
WITH CHECK (
    EXISTS (
        SELECT 1 FROM provider_privacy_windows AS privacy_window
        WHERE privacy_window.window_key = provider_privacy_window_partitions.window_key
          AND privacy_window.tenant_id = provider_runtime_tenant()
    )
);

DROP POLICY IF EXISTS provider_canonical_partitions_tenant
ON provider_canonical_partitions;
CREATE POLICY provider_canonical_partitions_tenant
ON provider_canonical_partitions
USING (
    EXISTS (
        SELECT 1 FROM provider_privacy_windows AS privacy_window
        WHERE privacy_window.window_key = provider_canonical_partitions.window_key
          AND privacy_window.tenant_id = provider_runtime_tenant()
    )
)
WITH CHECK (
    EXISTS (
        SELECT 1 FROM provider_privacy_windows AS privacy_window
        WHERE privacy_window.window_key = provider_canonical_partitions.window_key
          AND privacy_window.tenant_id = provider_runtime_tenant()
    )
);

DROP POLICY IF EXISTS provider_data_rights_requests_tenant
ON provider_data_rights_requests;
CREATE POLICY provider_data_rights_requests_tenant
ON provider_data_rights_requests
USING (tenant_id = provider_runtime_tenant())
WITH CHECK (tenant_id = provider_runtime_tenant());

DROP POLICY IF EXISTS provider_scale_acceptance_runs_tenant
ON provider_scale_acceptance_runs;
CREATE POLICY provider_scale_acceptance_runs_tenant
ON provider_scale_acceptance_runs
USING (tenant_id = provider_runtime_tenant())
WITH CHECK (tenant_id = provider_runtime_tenant());
