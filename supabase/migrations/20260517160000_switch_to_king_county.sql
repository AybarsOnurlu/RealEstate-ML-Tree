-- ============================================================
-- Migration: Switch to King County House Sales Dataset
-- ============================================================

-- Step 1: Drop the old table
DROP TABLE IF EXISTS properties;

-- Step 2: Create new table with King County columns
CREATE TABLE properties (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    price       DOUBLE PRECISION NOT NULL,
    bedrooms    DOUBLE PRECISION NOT NULL,
    bathrooms   DOUBLE PRECISION NOT NULL,
    sqft_living DOUBLE PRECISION NOT NULL,
    sqft_lot    DOUBLE PRECISION NOT NULL,
    floors      DOUBLE PRECISION NOT NULL,
    yr_built    DOUBLE PRECISION NOT NULL,
    lat         DOUBLE PRECISION NOT NULL,
    long        DOUBLE PRECISION NOT NULL
);

-- Step 3: Spatial index on (lat, long) for fast geo queries
CREATE INDEX idx_properties_location_kc ON properties (lat, long);

-- Step 4: Index on price for filtering
CREATE INDEX idx_properties_price_kc ON properties (price);
