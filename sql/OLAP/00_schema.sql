-- ====================================================================
-- 00_schema.sql
-- Create the olap schema that every later file writes into.
--
-- Source : schema oltp — read-only, never written to by this layer.
-- Target : schema olap — the analyst-ready star schema.
--
-- This is the layer every analysis reads: notebooks, Power BI, the
-- statistics work and the ML feature frames. It is derived entirely
-- from oltp (plus one geocoding seed, see 05_seed_geo_coordinates),
-- so it is disposable — `python rebuild.py` drops and rebuilds it in
-- seconds without touching the source data underneath.
--
-- Conventions in this layer, all of them deliberate:
--   * every identifier is lowercase snake_case, so nothing anywhere
--     needs double quotes — the source's "Order ID" mess stops here;
--   * every table is fully schema-qualified (rebuild.py sets no
--     search_path);
--   * money is numeric, never float — the source's real columns carry
--     float32 noise that has no business in an analytical layer;
--   * surrogate keys are integers assigned by row_number() over a
--     deterministic ORDER BY, so a rebuild reproduces the same keys.
--
-- Build order (filename order is the execution order):
--   00  schema
--   05  external seed data
--   1x  dimensions
--   2x  facts
--   90  data-quality assertions — the build fails if any of them do
-- ====================================================================
CREATE SCHEMA IF NOT EXISTS olap;

COMMENT ON SCHEMA olap IS
    'Analyst-ready star schema derived from the oltp schema. Rebuilt from '
    'scratch by rebuild.py; holds no state that is not reproducible.';
