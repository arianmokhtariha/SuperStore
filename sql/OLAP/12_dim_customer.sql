-- ====================================================================
-- 12_dim_customer.sql
-- Who bought, and the shape of their relationship with the store.
--
-- Source : oltp.customer, oltp.order
-- Grain  : one row per customer id — 1,589 rows
--
-- customer_id is the identity, not customer_name: 794 names are shared
-- by more than one id in this data, so keying on the name would merge
-- unrelated people.
--
-- first_order_date / last_order_date / order_count are denormalised
-- onto the dimension on purpose. They are aggregates of the fact, so
-- they are strictly redundant — but they turn cohort, tenure and churn
-- questions ("customers acquired in 2012", "dormant since Q3") into a
-- filter on one table instead of a subquery against 49,670 fact rows.
-- They are a build-time snapshot: they change only when this layer is
-- rebuilt, which for a fixed historical extract is never a problem.
-- Anything that must tie out to the penny belongs in the fact.
--
-- No customer is orphaned (every id in oltp.customer has at least one
-- order), so all three are NOT NULL rather than defaulted.
-- ====================================================================
DROP TABLE IF EXISTS olap.dim_customer CASCADE;

CREATE TABLE olap.dim_customer (
    customer_key      integer     NOT NULL,
    customer_id       varchar(8)  NOT NULL,
    customer_name     varchar(32) NOT NULL,
    segment           varchar(16) NOT NULL,
    first_order_date  date        NOT NULL,
    last_order_date   date        NOT NULL,
    order_count       integer     NOT NULL,
    active_span_days  integer     NOT NULL,
    PRIMARY KEY (customer_key),
    UNIQUE (customer_id)
);

INSERT INTO olap.dim_customer (
    customer_key, customer_id, customer_name, segment,
    first_order_date, last_order_date, order_count, active_span_days
)
WITH order_history AS (
    SELECT "Customer ID"          AS customer_id,
           min("Order Date")::date AS first_order_date,
           max("Order Date")::date AS last_order_date,
           count(*)::integer       AS order_count
    FROM oltp.order
    GROUP BY "Customer ID"
)
SELECT
    row_number() OVER (ORDER BY c."Customer ID")::integer,
    c."Customer ID",
    btrim(c."Customer Name"),
    btrim(c."Segment"),
    h.first_order_date,
    h.last_order_date,
    h.order_count,
    (h.last_order_date - h.first_order_date)::integer
FROM oltp.customer c
JOIN order_history h ON h.customer_id = c."Customer ID";

CREATE INDEX idx_dim_customer_segment ON olap.dim_customer (segment);

COMMENT ON TABLE  olap.dim_customer IS
    'One row per customer, with build-time relationship attributes.';
COMMENT ON COLUMN olap.dim_customer.order_count IS
    'Orders placed across the whole extract. Snapshot of the fact, '
    'denormalised for cohort filtering — not a substitute for it.';
COMMENT ON COLUMN olap.dim_customer.active_span_days IS
    'Days between first and last order. 0 = bought exactly once, or '
    'everything on one day.';
