-- ====================================================================
-- 00_schema.sql
-- Create the OLTP schema that every later file writes into.
--
-- Source      : sql/OLTP/_raw/Superstore.sql (SQLPro export, 19-03-1402)
-- Target      : schema oltp — raw source data, read-only downstream
-- Modification: qualified into oltp, and the MySQL export's blanket
--               identifier quoting reduced to the names Postgres
--               actually requires it for (the columns). Statements,
--               column definitions and data rows are unchanged.
--
-- This layer is a faithful copy of the source system: loaded once,
-- then never written to again. The analyst-ready star schema is
-- built on top of it in sql/OLAP, never by editing anything here.
-- ====================================================================
CREATE SCHEMA IF NOT EXISTS oltp;

COMMENT ON SCHEMA oltp IS
  'Raw Superstore source data. Loaded once, read-only thereafter; the
   analyst-ready star schema is derived from it in the olap schema.';
