-- Rollback migration 032: Remove age/gender aggregate columns from cluster table

ALTER TABLE cluster DROP COLUMN IF EXISTS age_estimate;
ALTER TABLE cluster DROP COLUMN IF EXISTS age_estimate_stddev;
ALTER TABLE cluster DROP COLUMN IF EXISTS gender;
ALTER TABLE cluster DROP COLUMN IF EXISTS gender_confidence;
ALTER TABLE cluster DROP COLUMN IF EXISTS age_gender_sample_count;
ALTER TABLE cluster DROP COLUMN IF EXISTS age_gender_updated_at;
