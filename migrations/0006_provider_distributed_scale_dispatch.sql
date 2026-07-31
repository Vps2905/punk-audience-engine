ALTER TABLE provider_ingestion_objects
DROP CONSTRAINT IF EXISTS provider_ingestion_objects_status_check;

ALTER TABLE provider_ingestion_objects
ADD CONSTRAINT provider_ingestion_objects_status_check
CHECK (
    status IN (
        'received',
        'validating',
        'dispatching',
        'dispatched',
        'processing',
        'completed',
        'blocked',
        'quarantined',
        'failed'
    )
);

CREATE INDEX IF NOT EXISTS idx_provider_ingestion_dispatched
ON provider_ingestion_objects (status, updated_at)
WHERE status IN ('dispatching', 'dispatched');
