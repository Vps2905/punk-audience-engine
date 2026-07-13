ALTER TABLE audience_vectors
ADD COLUMN IF NOT EXISTS
embedding_norm DOUBLE PRECISION;

UPDATE audience_vectors
SET embedding_norm = SQRT(
    (
        SELECT COALESCE(
            SUM(value * value),
            0
        )
        FROM unnest(embedding) AS value
    )
)
WHERE embedding_norm IS NULL;

ALTER TABLE audience_vectors
ALTER COLUMN embedding_norm SET NOT NULL;

CREATE INDEX IF NOT EXISTS
idx_audience_vectors_filter
ON audience_vectors(
    job_id,
    location_name,
    primary_poi_type,
    created_day_part,
    quality_score
);
