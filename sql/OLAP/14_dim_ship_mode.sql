-- ====================================================================
-- 14_dim_ship_mode.sql
-- The shipping tier an order was sent on.
--
-- Source : oltp.shipping, oltp.order
-- Grain  : one row per ship mode — 4 rows
--
-- Tiny, and still worth a dimension rather than a text column on the
-- fact: it is where the ordering lives. "Which modes are faster than
-- Standard Class" is a question the data can answer but a varchar
-- cannot, and the ML work in ML/Q2 predicts this column, so it needs
-- somewhere to put the tier's own attributes.
--
-- speed_rank is measured, not declared. Ranking the modes by their
-- observed average order-to-ship lag gives 1 = fastest, and it stays
-- correct if the source ever adds a tier or renames one — no CASE
-- listing today's four names, nothing to update by hand. is_expedited
-- falls out of the same measurement: everything quicker than the
-- slowest tier is something the customer paid to upgrade to, which is
-- exactly the population the upsell question in the report is about.
-- ====================================================================
DROP TABLE IF EXISTS olap.dim_ship_mode CASCADE;

CREATE TABLE olap.dim_ship_mode (
    ship_mode_key   integer       NOT NULL,
    ship_mode       varchar(16)   NOT NULL,
    speed_rank      smallint      NOT NULL,   -- 1 = fastest observed
    avg_ship_days   numeric(6, 4) NOT NULL,
    orders_shipped  integer       NOT NULL,
    is_expedited    boolean       NOT NULL,
    PRIMARY KEY (ship_mode_key),
    UNIQUE (ship_mode)
);

INSERT INTO olap.dim_ship_mode (
    ship_mode_key, ship_mode, speed_rank, avg_ship_days,
    orders_shipped, is_expedited
)
WITH observed AS (
    SELECT btrim(s."Ship Mode") AS ship_mode,
           avg(s."Ship Date"::date - o."Order Date"::date)::numeric AS avg_ship_days,
           count(*)::integer AS orders_shipped
    FROM oltp.shipping s
    JOIN oltp.order o ON o."Order ID" = s."Order ID"
    GROUP BY btrim(s."Ship Mode")
),
ranked AS (
    SELECT ship_mode,
           round(avg_ship_days, 4) AS avg_ship_days,
           orders_shipped,
           row_number() OVER (ORDER BY avg_ship_days, ship_mode)::smallint AS speed_rank,
           count(*)    OVER ()                                            AS mode_count
    FROM observed
)
SELECT speed_rank::integer,      -- key and rank coincide; both are the speed order
       ship_mode,
       speed_rank,
       avg_ship_days,
       orders_shipped,
       speed_rank < mode_count
FROM ranked;

COMMENT ON TABLE  olap.dim_ship_mode IS
    'Shipping tiers, ordered by their observed delivery speed.';
COMMENT ON COLUMN olap.dim_ship_mode.speed_rank IS
    'Derived from the average order-to-ship lag: 1 = fastest.';
COMMENT ON COLUMN olap.dim_ship_mode.is_expedited IS
    'True for every tier faster than the slowest one — the paid upgrades.';
