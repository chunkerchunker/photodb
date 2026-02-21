-- Migration 032: Add age/gender aggregate columns to cluster table
-- These columns store aggregated age/gender estimates from member detections

ALTER TABLE cluster ADD COLUMN age_estimate real;
ALTER TABLE cluster ADD COLUMN age_estimate_stddev real;
ALTER TABLE cluster ADD COLUMN gender char(1) CHECK (gender IN ('M', 'F', 'U'));
ALTER TABLE cluster ADD COLUMN gender_confidence real;
ALTER TABLE cluster ADD COLUMN age_gender_sample_count integer DEFAULT 0;
ALTER TABLE cluster ADD COLUMN age_gender_updated_at timestamptz;
