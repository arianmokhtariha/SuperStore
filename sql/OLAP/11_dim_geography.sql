-- ====================================================================
-- 11_dim_geography.sql
-- Where an order ships to, from city up to market.
--
-- Source : oltp.shipping (city, state, country, region)
--          oltp.order    (market)
--          olap.seed_geo_coordinates (latitude, longitude)
-- Grain  : one row per distinct shipping location — 3,772 rows
--
-- Market lives here, and that is the one real change from the old CSV
-- model, which carried a separate Dim_Market joined to the fact by its
-- own key. Profiling the source shows (country, region) determines
-- market with zero violations, and adding market to the location grain
-- adds zero rows: 3,772 either way. So market is not an independent
-- axis of analysis — it is the top of the geography hierarchy:
--
--     market > region > country > state > city
--
-- Folding it in removes a key from the fact, removes a join from every
-- market query, and makes the hierarchy legible in one table, which is
-- the whole point of a denormalised dimension.
--
-- Region has to stay part of the grain, though. Seven cities in
-- Austria and one in Mongolia appear under two regions in the source
-- (Austria: Central/EU and EMEA/EMEA; Mongolia: North Asia/APAC and
-- EMEA/EMEA), so (city, state, country) is not unique. Those rows are
-- kept exactly as the source filed them — reassigning them to one
-- "correct" market would silently move revenue between the markets the
-- manager is being asked to compare. is_market_ambiguous flags them
-- instead, so the ambiguity is visible in the data rather than buried.
-- ====================================================================
DROP TABLE IF EXISTS olap.dim_geography CASCADE;

CREATE TABLE olap.dim_geography (
    geo_key             integer         NOT NULL,
    city                varchar(64)     NOT NULL,
    state               varchar(64)     NOT NULL,
    country             varchar(32)     NOT NULL,
    region              varchar(16)     NOT NULL,
    market              varchar(8)      NOT NULL,
    latitude            numeric(10, 7)  NOT NULL,
    longitude           numeric(11, 7)  NOT NULL,
    is_market_ambiguous boolean         NOT NULL,
    PRIMARY KEY (geo_key),
    UNIQUE (city, state, country, region)
);

INSERT INTO olap.dim_geography (
    geo_key, city, state, country, region, market,
    latitude, longitude, is_market_ambiguous
)
WITH source_location AS (
    SELECT DISTINCT
           btrim(s."City")    AS city,
           btrim(s."State")   AS state,
           btrim(s."Country") AS country,
           btrim(s."Region")  AS region,
           btrim(o."Market")  AS market
    FROM oltp.shipping s
    JOIN oltp.order o ON o."Order ID" = s."Order ID"
),
ambiguous AS (
    -- A place filed under more than one market by the source.
    SELECT city, state, country
    FROM source_location
    GROUP BY city, state, country
    HAVING count(DISTINCT market) > 1
)
SELECT
    -- Deterministic ORDER BY: the same rebuild produces the same keys,
    -- so a saved model or a cached dashboard does not silently repoint.
    row_number() OVER (ORDER BY l.country, l.state, l.city, l.region)::integer,
    l.city,
    l.state,
    l.country,
    l.region,
    l.market,
    g.latitude,
    g.longitude,
    a.city IS NOT NULL
FROM source_location l
JOIN olap.seed_geo_coordinates g
       ON  g.city    = l.city
       AND g.state   = l.state
       AND g.country = l.country
       AND g.region  = l.region
LEFT JOIN ambiguous a
       ON  a.city    = l.city
       AND a.state   = l.state
       AND a.country = l.country;

CREATE INDEX idx_dim_geography_country ON olap.dim_geography (country);
CREATE INDEX idx_dim_geography_market  ON olap.dim_geography (market, region);

COMMENT ON TABLE  olap.dim_geography IS
    'Shipping location, one row per city/state/country/region. Holds the '
    'full market > region > country > state > city hierarchy.';
COMMENT ON COLUMN olap.dim_geography.market IS
    'Top of the hierarchy. Functionally determined by (country, region) '
    'in the source, verified in 90_checks.sql.';
COMMENT ON COLUMN olap.dim_geography.is_market_ambiguous IS
    'True where the source files this city under more than one market '
    '(7 Austrian cities, 1 Mongolian). Kept as-is, flagged not fixed.';
