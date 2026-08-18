-- ====================================================================
-- 21_fact_order.sql
-- The same business process rolled up to the whole order.
--
-- Source : olap.fact_sales (not oltp — see below)
-- Grain  : one order — 25,033 rows, matching oltp.order exactly
--
-- Why a second fact table. Several of the questions this project has to
-- answer are about orders, not lines: does a bigger order pay more to
-- ship, how long does an order take to arrive, which shipping tier will
-- a customer pick. Answering those from the line-grain fact means
-- averaging a value that is repeated once per line, so a five-line
-- order counts five times and the answer skews toward big baskets.
-- Two facts sharing one set of dimensions is the standard fix, and it
-- makes the trap impossible rather than merely documented.
--
-- Built from fact_sales rather than from oltp a second time, so the two
-- cannot drift: every roll-up here is a SUM of rows that are already in
-- the star, and 90_checks.sql asserts the totals tie out to the cent.
--
-- Ratios (discount_rate, profit_margin) are recomputed from the summed
-- money, never averaged from the line ratios — averaging ratios weights
-- a $3 line the same as a $3,000 one.
-- ====================================================================
DROP TABLE IF EXISTS olap.fact_order CASCADE;

CREATE TABLE olap.fact_order (
    order_key        integer        NOT NULL,
    order_id         varchar(16)    NOT NULL,   -- degenerate dimension
    -- foreign keys (no product_key: an order spans products)
    order_date_key   integer        NOT NULL,
    ship_date_key    integer        NOT NULL,
    customer_key     integer        NOT NULL,
    geo_key          integer        NOT NULL,
    ship_mode_key    integer        NOT NULL,
    priority_key     integer        NOT NULL,
    -- basket size
    line_count       smallint       NOT NULL,
    product_count    smallint       NOT NULL,
    quantity         integer        NOT NULL,
    -- money, summed from the lines
    sales            numeric(14, 5) NOT NULL,
    gross_sales      numeric(14, 5) NOT NULL,
    discount_amount  numeric(14, 5) NOT NULL,
    profit           numeric(14, 5) NOT NULL,
    cost             numeric(14, 5) NOT NULL,
    shipping_cost    numeric(14, 5) NOT NULL,
    -- ratios, recomputed from the sums
    discount_rate    numeric(12, 6) NOT NULL,
    profit_margin    numeric(12, 6) NOT NULL,
    shipping_cost_pct numeric(12, 6) NOT NULL,
    -- fulfilment
    ship_lag_days    smallint       NOT NULL,
    is_returned      boolean        NOT NULL,
    PRIMARY KEY (order_key),
    UNIQUE (order_id),
    FOREIGN KEY (order_date_key) REFERENCES olap.dim_date (date_key),
    FOREIGN KEY (ship_date_key)  REFERENCES olap.dim_date (date_key),
    FOREIGN KEY (customer_key)   REFERENCES olap.dim_customer (customer_key),
    FOREIGN KEY (geo_key)        REFERENCES olap.dim_geography (geo_key),
    FOREIGN KEY (ship_mode_key)  REFERENCES olap.dim_ship_mode (ship_mode_key),
    FOREIGN KEY (priority_key)   REFERENCES olap.dim_order_priority (priority_key)
);

INSERT INTO olap.fact_order (
    order_key, order_id,
    order_date_key, ship_date_key, customer_key, geo_key,
    ship_mode_key, priority_key,
    line_count, product_count, quantity,
    sales, gross_sales, discount_amount, profit, cost, shipping_cost,
    discount_rate, profit_margin, shipping_cost_pct,
    ship_lag_days, is_returned
)
WITH rolled AS (
    SELECT
        order_id,
        -- These seven are constant within an order (shipping is 1:1 with
        -- order in the source, and the rest hang off the order row), so
        -- min() is just a way to carry them through the GROUP BY. The
        -- assertion in 90_checks.sql proves the constancy rather than
        -- assuming it.
        min(order_date_key)             AS order_date_key,
        min(ship_date_key)              AS ship_date_key,
        min(customer_key)               AS customer_key,
        min(geo_key)                    AS geo_key,
        min(ship_mode_key)              AS ship_mode_key,
        min(priority_key)               AS priority_key,
        min(ship_lag_days)              AS ship_lag_days,
        bool_or(is_returned)            AS is_returned,
        count(*)::smallint              AS line_count,
        count(DISTINCT product_key)::smallint AS product_count,
        sum(quantity)::integer          AS quantity,
        sum(sales)                      AS sales,
        sum(gross_sales)                AS gross_sales,
        sum(discount_amount)            AS discount_amount,
        sum(profit)                     AS profit,
        sum(cost)                       AS cost,
        sum(shipping_cost)              AS shipping_cost
    FROM olap.fact_sales
    GROUP BY order_id
)
SELECT
    row_number() OVER (ORDER BY order_id)::integer,
    order_id,
    order_date_key,
    ship_date_key,
    customer_key,
    geo_key,
    ship_mode_key,
    priority_key,
    line_count,
    product_count,
    quantity,
    sales,
    gross_sales,
    discount_amount,
    profit,
    cost,
    shipping_cost,
    round(discount_amount / gross_sales, 6),
    round(profit / sales, 6),
    round(shipping_cost / sales, 6),
    ship_lag_days,
    is_returned
FROM rolled;

CREATE INDEX idx_fact_order_order_date ON olap.fact_order (order_date_key);
CREATE INDEX idx_fact_order_ship_date  ON olap.fact_order (ship_date_key);
CREATE INDEX idx_fact_order_customer   ON olap.fact_order (customer_key);
CREATE INDEX idx_fact_order_geo        ON olap.fact_order (geo_key);
CREATE INDEX idx_fact_order_ship_mode  ON olap.fact_order (ship_mode_key);
CREATE INDEX idx_fact_order_priority   ON olap.fact_order (priority_key);

ANALYZE olap.fact_order;

COMMENT ON TABLE  olap.fact_order IS
    'Order-grain fact rolled up from fact_sales. Use it for anything '
    'measured per order — shipping speed, basket size, order value.';
COMMENT ON COLUMN olap.fact_order.product_count IS
    'Distinct products on the order. Below line_count for the 35 orders '
    'that list the same product on two lines.';
COMMENT ON COLUMN olap.fact_order.discount_rate IS
    'discount_amount / gross_sales, from the sums — not the mean of the '
    'line-level discount rates.';
COMMENT ON COLUMN olap.fact_order.shipping_cost_pct IS
    'shipping_cost / sales: what fraction of the order value shipping ate.';
COMMENT ON COLUMN olap.fact_order.ship_lag_days IS
    'Order date to ship date, in days. Additive-safe at this grain.';
