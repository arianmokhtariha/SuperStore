-- ====================================================================
-- 20_fact_sales.sql
-- The centre of the star: one row per line on an order.
--
-- Source : oltp.order_detail, joined out to oltp.order, oltp.shipping
--          and oltp.returned for the attributes that hang off the order
-- Grain  : one order line — 49,670 rows, matching order_detail exactly
--
-- Grain note. 35 (order, product) pairs appear on two lines of the same
-- order. They are not duplicates: the measures differ line by line, so
-- collapsing them would delete revenue. "Row ID" is the source's line
-- identity and is carried through as order_line_id, which keeps every
-- fact row traceable back to the exact source row and makes the
-- one-to-one with order_detail assertable — 90_checks.sql does assert
-- it, on both the row count and the summed measures.
--
-- Types. Every money column arrives as real (float32) and leaves as
-- numeric. Float32 carries about seven significant digits, which is why
-- the old CSV export shows 263.92 written out as 263.92001342773438;
-- summed over 49,670 rows that noise is real money. numeric is exact.
--
-- Derived measures. Five of them, all computed once here rather than
-- re-derived in every notebook:
--
--   sales           is already net of discount in this source
--   gross_sales     = sales / (1 - discount), the list value
--   discount_amount = gross_sales - sales, the money actually given up
--   cost            = sales - profit, the implied COGS
--   unit_price      = sales / quantity, net price per unit
--   profit_margin   = profit / sales
--
-- discount_amount is worth spelling out. The obvious sales * discount
-- is wrong here, and the bootcamp ML code used exactly that: sales is
-- the post-discount figure, so multiplying it by the rate understates
-- the giveaway by a factor of (1 - discount) — 15% low at a 0.15
-- discount, 85% low at 0.85. Discounts run up to 0.85 in this data and
-- never reach 1, so the division is always safe.
-- ====================================================================
DROP TABLE IF EXISTS olap.fact_sales CASCADE;

CREATE TABLE olap.fact_sales (
    sales_key        integer        NOT NULL,
    -- degenerate dimensions: order identity, carried on the fact
    order_line_id    integer        NOT NULL,   -- oltp.order_detail."Row ID"
    order_id         varchar(16)    NOT NULL,
    -- foreign keys
    order_date_key   integer        NOT NULL,
    ship_date_key    integer        NOT NULL,
    customer_key     integer        NOT NULL,
    product_key      integer        NOT NULL,
    geo_key          integer        NOT NULL,
    ship_mode_key    integer        NOT NULL,
    priority_key     integer        NOT NULL,
    -- measures, as recorded
    quantity         integer        NOT NULL,
    discount         numeric(5, 3)  NOT NULL,
    sales            numeric(14, 5) NOT NULL,
    profit           numeric(14, 5) NOT NULL,
    shipping_cost    numeric(14, 5) NOT NULL,
    -- measures, derived
    gross_sales      numeric(14, 5) NOT NULL,
    discount_amount  numeric(14, 5) NOT NULL,
    cost             numeric(14, 5) NOT NULL,
    unit_price       numeric(14, 5) NOT NULL,
    profit_margin    numeric(12, 6) NOT NULL,
    -- order-level facts, repeated on every line of the order
    ship_lag_days    smallint       NOT NULL,
    is_returned      boolean        NOT NULL,
    PRIMARY KEY (sales_key),
    UNIQUE (order_line_id),
    FOREIGN KEY (order_date_key) REFERENCES olap.dim_date (date_key),
    FOREIGN KEY (ship_date_key)  REFERENCES olap.dim_date (date_key),
    FOREIGN KEY (customer_key)   REFERENCES olap.dim_customer (customer_key),
    FOREIGN KEY (product_key)    REFERENCES olap.dim_product (product_key),
    FOREIGN KEY (geo_key)        REFERENCES olap.dim_geography (geo_key),
    FOREIGN KEY (ship_mode_key)  REFERENCES olap.dim_ship_mode (ship_mode_key),
    FOREIGN KEY (priority_key)   REFERENCES olap.dim_order_priority (priority_key)
);

INSERT INTO olap.fact_sales (
    sales_key, order_line_id, order_id,
    order_date_key, ship_date_key, customer_key, product_key, geo_key,
    ship_mode_key, priority_key,
    quantity, discount, sales, profit, shipping_cost,
    gross_sales, discount_amount, cost, unit_price, profit_margin,
    ship_lag_days, is_returned
)
WITH line AS (
    SELECT
        d."Row ID"                                     AS order_line_id,
        d."Order ID"                                   AS order_id,
        to_char(o."Order Date", 'YYYYMMDD')::integer   AS order_date_key,
        to_char(s."Ship Date",  'YYYYMMDD')::integer   AS ship_date_key,
        dc.customer_key,
        dp.product_key,
        dg.geo_key,
        dsm.ship_mode_key,
        dop.priority_key,
        d."Quantity"                                   AS quantity,
        d."Discount"::numeric                          AS discount,
        d."Sales"::numeric                             AS sales,
        d."Profit"::numeric                            AS profit,
        d."Shipping Cost"::numeric                     AS shipping_cost,
        (s."Ship Date"::date - o."Order Date"::date)   AS ship_lag_days,
        r."Order ID" IS NOT NULL                       AS is_returned
    FROM oltp.order_detail d
    JOIN oltp.order    o ON o."Order ID" = d."Order ID"
    JOIN oltp.shipping s ON s."Order ID" = d."Order ID"
    LEFT JOIN oltp.returned r ON r."Order ID" = d."Order ID"
    -- Inner joins throughout: every one of these is a declared foreign
    -- key in oltp with no nulls, so a row that fails to match is a
    -- broken source load, not a business case to be preserved. The row
    -- count assertion in 90_checks.sql is what turns that into an error
    -- rather than a quietly shorter fact table.
    JOIN olap.dim_customer       dc  ON dc.customer_id  = o."Customer ID"
    JOIN olap.dim_product        dp  ON dp.product_id   = d."Product ID"
    JOIN olap.dim_ship_mode      dsm ON dsm.ship_mode   = btrim(s."Ship Mode")
    JOIN olap.dim_order_priority dop ON dop.priority    = btrim(o."Order Priority")
    JOIN olap.dim_geography      dg  ON dg.city         = btrim(s."City")
                                    AND dg.state        = btrim(s."State")
                                    AND dg.country      = btrim(s."Country")
                                    AND dg.region       = btrim(s."Region")
)
SELECT
    row_number() OVER (ORDER BY order_line_id)::integer,
    order_line_id,
    order_id,
    order_date_key,
    ship_date_key,
    customer_key,
    product_key,
    geo_key,
    ship_mode_key,
    priority_key,
    quantity,
    discount,
    sales,
    profit,
    shipping_cost,
    round(sales / (1 - discount), 5),
    round(sales / (1 - discount) - sales, 5),
    round(sales - profit, 5),
    round(sales / quantity, 5),
    round(profit / sales, 6),
    ship_lag_days::smallint,
    is_returned
FROM line;

-- Indexes on every foreign key. A star query filters the dimensions and
-- then hits the fact through them, so these are the only access paths
-- that matter; the surrogate primary key is almost never the entry point.
CREATE INDEX idx_fact_sales_order_date ON olap.fact_sales (order_date_key);
CREATE INDEX idx_fact_sales_ship_date  ON olap.fact_sales (ship_date_key);
CREATE INDEX idx_fact_sales_customer   ON olap.fact_sales (customer_key);
CREATE INDEX idx_fact_sales_product    ON olap.fact_sales (product_key);
CREATE INDEX idx_fact_sales_geo        ON olap.fact_sales (geo_key);
CREATE INDEX idx_fact_sales_ship_mode  ON olap.fact_sales (ship_mode_key);
CREATE INDEX idx_fact_sales_priority   ON olap.fact_sales (priority_key);
CREATE INDEX idx_fact_sales_order_id   ON olap.fact_sales (order_id);

ANALYZE olap.fact_sales;

COMMENT ON TABLE  olap.fact_sales IS
    'Transaction fact at order-line grain. One row per oltp.order_detail row.';
COMMENT ON COLUMN olap.fact_sales.order_id IS
    'Degenerate dimension — the order number itself, no dim_order to join.';
COMMENT ON COLUMN olap.fact_sales.sales IS
    'Revenue after discount, as recorded by the source.';
COMMENT ON COLUMN olap.fact_sales.gross_sales IS
    'sales / (1 - discount): what the line would have billed at list.';
COMMENT ON COLUMN olap.fact_sales.discount_amount IS
    'gross_sales - sales. NOT sales * discount, which understates it.';
COMMENT ON COLUMN olap.fact_sales.cost IS
    'sales - profit. Implied cost of goods; the source has no cost column.';
COMMENT ON COLUMN olap.fact_sales.profit_margin IS
    'Ratio, non-additive. Average it weighted, or recompute from sums.';
COMMENT ON COLUMN olap.fact_sales.ship_lag_days IS
    'Order-level, repeated on every line of the order — do not SUM it. '
    'Aggregate shipping speed from fact_order, which holds it once.';
COMMENT ON COLUMN olap.fact_sales.is_returned IS
    'Order-level: the source records returns per order, not per line.';
