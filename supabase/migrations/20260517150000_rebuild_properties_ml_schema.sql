-- ============================================================
-- Migration: Rebuild `properties` table with full California
-- Housing dataset schema.
-- Run this in: Supabase Dashboard → SQL Editor
-- ============================================================

-- Step 1: Drop the old table (all old data will be lost — OK,
--         we will re-seed from the Python script)
DROP TABLE IF EXISTS properties;

-- Step 2: Create new table with full California Housing columns
CREATE TABLE properties (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    longitude          DOUBLE PRECISION NOT NULL,
    latitude           DOUBLE PRECISION NOT NULL,
    housing_median_age DOUBLE PRECISION NOT NULL,
    total_rooms        DOUBLE PRECISION NOT NULL,
    total_bedrooms     DOUBLE PRECISION NOT NULL,
    population         DOUBLE PRECISION NOT NULL,
    households         DOUBLE PRECISION NOT NULL,
    median_income      DOUBLE PRECISION NOT NULL,
    median_house_value DOUBLE PRECISION NOT NULL,
    ocean_proximity    TEXT             NOT NULL
);

-- Step 3: Spatial index on (latitude, longitude) for fast geo queries
CREATE INDEX idx_properties_location ON properties (latitude, longitude);

-- Step 4: Index on median_house_value for price range filtering
CREATE INDEX idx_properties_price ON properties (median_house_value);

-- Step 5: Index on ocean_proximity for category filtering
CREATE INDEX idx_properties_ocean ON properties (ocean_proximity);

-- Verify the schema
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'properties'
ORDER BY ordinal_position;
