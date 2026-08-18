-- ====================================================================
-- 90_checks.sql
-- Assertions the star has to satisfy. Runs last; failing stops the build.
--
-- rebuild.py wraps the whole target schema in one transaction, so a
-- RAISE EXCEPTION here rolls back every file before it. That is the
-- point: a star that has quietly lost rows, double-counted money or
-- broken its own grain never becomes the thing the reports read from.
-- Nothing here is optional or advisory — if it fails, the layer is
-- wrong, and it is better to have no olap schema than a plausible one.
--
-- Every check is one boolean. They are evaluated together and the
-- failures reported as a list, so a bad change shows all of its damage
-- in one run instead of one symptom per rebuild.
--
-- Tolerances: comparisons on stored numeric values are exact, because
-- numeric arithmetic is. The three checks on derived measures allow
-- 0.001, which is slack for the last decimal place of two independently
-- rounded values, and far tighter than any real error would be.
-- ====================================================================
DO $$
DECLARE
    failures text[];
BEGIN
    WITH checks (label, ok) AS (
        VALUES

        -- ── grain: the facts have exactly the rows they should ──────
        ('fact_sales row count matches oltp.order_detail',
            (SELECT count(*) FROM olap.fact_sales)
          = (SELECT count(*) FROM oltp.order_detail)),

        ('fact_sales.order_line_id is 1:1 with oltp.order_detail."Row ID"',
            NOT EXISTS (
                SELECT 1 FROM oltp.order_detail d
                FULL JOIN olap.fact_sales f ON f.order_line_id = d."Row ID"
                WHERE d."Row ID" IS NULL OR f.order_line_id IS NULL
            )),

        ('fact_order row count matches oltp.order',
            (SELECT count(*) FROM olap.fact_order)
          = (SELECT count(*) FROM oltp.order)),

        -- ── money: nothing gained or lost on the way in ─────────────
        ('fact_sales money ties out to oltp.order_detail',
            (SELECT (sum(sales), sum(profit), sum(shipping_cost), sum(quantity))
             FROM olap.fact_sales)
          = (SELECT (sum("Sales"::numeric), sum("Profit"::numeric),
                     sum("Shipping Cost"::numeric), sum("Quantity"))
             FROM oltp.order_detail)),

        ('fact_order money ties out to fact_sales',
            (SELECT (sum(sales), sum(profit), sum(shipping_cost),
                     sum(gross_sales), sum(discount_amount), sum(quantity))
             FROM olap.fact_order)
          = (SELECT (sum(sales), sum(profit), sum(shipping_cost),
                     sum(gross_sales), sum(discount_amount), sum(quantity))
             FROM olap.fact_sales)),

        -- ── derived measures agree with the measures they came from ─
        ('gross_sales - discount_amount = sales on every line',
            NOT EXISTS (SELECT 1 FROM olap.fact_sales
                        WHERE abs(gross_sales - discount_amount - sales) > 0.001)),

        ('sales - cost = profit on every line',
            NOT EXISTS (SELECT 1 FROM olap.fact_sales
                        WHERE abs(sales - cost - profit) > 0.001)),

        ('unit_price * quantity = sales on every line',
            NOT EXISTS (SELECT 1 FROM olap.fact_sales
                        WHERE abs(unit_price * quantity - sales) > 0.001)),

        -- ── grain integrity: order attributes really are per-order ──
        -- fact_order carries these forward with min(); this proves there
        -- was only ever one value to carry.
        ('order-level attributes are constant within an order',
            NOT EXISTS (
                SELECT 1 FROM olap.fact_sales
                GROUP BY order_id
                HAVING count(DISTINCT order_date_key) > 1
                    OR count(DISTINCT ship_date_key)  > 1
                    OR count(DISTINCT customer_key)   > 1
                    OR count(DISTINCT geo_key)        > 1
                    OR count(DISTINCT ship_mode_key)  > 1
                    OR count(DISTINCT priority_key)   > 1
                    OR count(DISTINCT ship_lag_days)  > 1
                    OR count(DISTINCT is_returned)    > 1
            )),

        -- ── dimensions: complete, and no wider than the source ──────
        ('dim_customer covers oltp.customer exactly',
            (SELECT count(*) FROM olap.dim_customer)
          = (SELECT count(*) FROM oltp.customer)),

        ('dim_product covers oltp.product exactly',
            (SELECT count(*) FROM olap.dim_product)
          = (SELECT count(*) FROM oltp.product)),

        ('dim_ship_mode covers every ship mode in the source',
            (SELECT count(*) FROM olap.dim_ship_mode)
          = (SELECT count(DISTINCT btrim("Ship Mode")) FROM oltp.shipping)),

        ('dim_order_priority covers every priority in the source',
            (SELECT count(*) FROM olap.dim_order_priority)
          = (SELECT count(DISTINCT btrim("Order Priority")) FROM oltp.order)),

        ('dim_geography covers every distinct shipping location',
            (SELECT count(*) FROM olap.dim_geography)
          = (SELECT count(*) FROM (
                SELECT DISTINCT s."City", s."State", s."Country", s."Region"
                FROM oltp.shipping s) x)),

        ('every geography row was matched to coordinates',
            (SELECT count(*) FROM olap.dim_geography)
          = (SELECT count(*) FROM olap.seed_geo_coordinates)),

        ('dim_customer.order_count sums to the order count',
            (SELECT sum(order_count) FROM olap.dim_customer)
          = (SELECT count(*) FROM oltp.order)),

        -- ── the modelling decisions this layer is built on ──────────
        -- Market lives inside dim_geography instead of its own dimension
        -- only because (country, region) determines it. If a reload ever
        -- breaks that, the hierarchy is wrong and this says so.
        ('(country, region) still determines market',
            NOT EXISTS (
                SELECT 1 FROM olap.dim_geography
                GROUP BY country, region
                HAVING count(DISTINCT market) > 1
            )),

        -- ── calendar ────────────────────────────────────────────────
        ('dim_date has no gaps',
            (SELECT count(*) FROM olap.dim_date)
          = (SELECT max(full_date) - min(full_date) + 1 FROM olap.dim_date)),

        -- make_date, not date_trunc: date_trunc on a date resolves to the
        -- timestamptz overload and drifts with the server's DST. See the
        -- header of 10_dim_date.sql — this check is what caught it.
        ('dim_date covers whole calendar years',
            (SELECT min(full_date)
                    = make_date(extract(year FROM min(full_date))::integer, 1, 1)
                AND max(full_date)
                    = make_date(extract(year FROM max(full_date))::integer, 12, 31)
             FROM olap.dim_date)),

        -- ── value ranges the analyses assume ────────────────────────
        ('no non-positive sales or quantities',
            NOT EXISTS (SELECT 1 FROM olap.fact_sales
                        WHERE sales <= 0 OR quantity <= 0)),

        ('discounts are a fraction below 1',
            NOT EXISTS (SELECT 1 FROM olap.fact_sales
                        WHERE discount < 0 OR discount >= 1)),

        ('nothing ships before it is ordered',
            NOT EXISTS (SELECT 1 FROM olap.fact_order WHERE ship_lag_days < 0))
    )
    SELECT array_agg(label ORDER BY label) INTO failures
    FROM checks
    WHERE ok IS NOT TRUE;   -- NULL counts as a failure, not a pass

    IF failures IS NOT NULL THEN
        RAISE EXCEPTION E'olap data-quality checks failed:\n  - %',
            array_to_string(failures, E'\n  - ');
    END IF;
END
$$;
