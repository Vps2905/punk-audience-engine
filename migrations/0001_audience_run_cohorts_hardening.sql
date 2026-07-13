ALTER TABLE audience_run_cohorts
ADD COLUMN IF NOT EXISTS export_cohort_id TEXT;

ALTER TABLE audience_run_cohorts
ADD COLUMN IF NOT EXISTS lookback_bucket TEXT;

ALTER TABLE audience_run_cohorts
ADD COLUMN IF NOT EXISTS
management_quality_score DOUBLE PRECISION;

ALTER TABLE audience_run_cohorts
ADD COLUMN IF NOT EXISTS privacy_mode TEXT;

ALTER TABLE audience_run_cohorts
ADD COLUMN IF NOT EXISTS data_safety_status TEXT;

UPDATE audience_run_cohorts
SET export_cohort_id =
    'legacy_' || md5(run_id || ':' || id::text)
WHERE export_cohort_id IS NULL
   OR BTRIM(export_cohort_id) = '';

DELETE FROM audience_run_cohorts older
USING audience_run_cohorts newer
WHERE older.run_id = newer.run_id
  AND older.export_cohort_id = newer.export_cohort_id
  AND older.id < newer.id;

ALTER TABLE audience_run_cohorts
ALTER COLUMN export_cohort_id SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS
uq_audience_run_cohorts_run_export
ON audience_run_cohorts(
    run_id,
    export_cohort_id
);

CREATE INDEX IF NOT EXISTS
idx_audience_run_cohorts_filter
ON audience_run_cohorts(
    run_id,
    approval_status,
    created_day_part,
    location_name,
    primary_poi_type
);
