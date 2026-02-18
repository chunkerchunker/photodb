-- Migration 028: Add optional name fields to person table
-- middle_name, maiden_name, preferred_name, suffix

ALTER TABLE person ADD COLUMN middle_name text;
ALTER TABLE person ADD COLUMN maiden_name text;
ALTER TABLE person ADD COLUMN preferred_name text;
ALTER TABLE person ADD COLUMN suffix text;
