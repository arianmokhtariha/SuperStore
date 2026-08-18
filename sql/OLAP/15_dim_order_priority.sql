-- ====================================================================
-- 15_dim_order_priority.sql
-- How urgent the order was flagged as.
--
-- Source : oltp.order."Order Priority"
-- Grain  : one row per priority level — 4 rows
--
-- The values are an ordinal scale, and nothing in the data says which
-- end is which: Critical outranks High, but no column, count or
-- average will tell you that. That ordering is a business definition,
-- so it is written down once, here, as a lookup — and every report
-- that sorts by urgency reads it from this column instead of
-- rebuilding its own CASE and getting it subtly different.
--
-- The rows themselves still come from the data (SELECT DISTINCT), so
-- the lookup only supplies the rank. A priority the lookup has never
-- seen therefore fails this file immediately on priority_rank's
-- NOT NULL — a loud stop with the new value in hand, rather than a
-- dimension that quietly sorts it last.
-- ====================================================================
DROP TABLE IF EXISTS olap.dim_order_priority CASCADE;

CREATE TABLE olap.dim_order_priority (
    priority_key   integer     NOT NULL,
    priority       varchar(8)  NOT NULL,
    priority_rank  smallint    NOT NULL,   -- 1 = least urgent
    PRIMARY KEY (priority_key),
    UNIQUE (priority),
    UNIQUE (priority_rank)
);

INSERT INTO olap.dim_order_priority (priority_key, priority, priority_rank)
WITH observed AS (
    SELECT DISTINCT btrim("Order Priority") AS priority
    FROM oltp.order
),
ordinal (priority, priority_rank) AS (
    VALUES ('Low', 1), ('Medium', 2), ('High', 3), ('Critical', 4)
)
SELECT o.priority_rank, v.priority, o.priority_rank::smallint
FROM observed v
LEFT JOIN ordinal o ON o.priority = v.priority;

COMMENT ON TABLE  olap.dim_order_priority IS
    'Order urgency levels. Rows come from the data; the rank is the '
    'business ordering, defined once here.';
COMMENT ON COLUMN olap.dim_order_priority.priority_rank IS
    '1 = Low .. 4 = Critical. Sort and compare on this, not on the name.';
