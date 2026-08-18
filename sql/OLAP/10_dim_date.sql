-- ====================================================================
-- 10_dim_date.sql
-- Calendar dimension, plus the two role-playing views over it.
--
-- Source : oltp.order."Order Date", oltp.shipping."Ship Date"
--          (used only to size the calendar, not to populate it)
-- Grain  : one row per calendar day
--
-- Generated, not harvested. The old CSV model built this dimension
-- from the dates that happened to appear in the data, which left it
-- with holes on every day nobody ordered anything — 1,468 rows for a
-- four-year span. A date dimension with holes silently drops those
-- days from any report that joins through it, and Power BI's time
-- intelligence (DAX) refuses to work at all unless the date table is
-- contiguous and covers whole years. So the span is derived from the
-- data and then rounded outward to whole years, and every day inside
-- it exists whether or not it saw a sale.
--
-- day_of_week is ISO: 1 = Monday .. 7 = Sunday. Stated because the
-- old model used pandas' 0-based convention and the two disagree by
-- one, which is exactly the kind of off-by-one that turns "Saturday
-- is our best day" into "Sunday is".
--
-- Everything below stays in date and timestamp arithmetic and never
-- touches timestamptz, which is not fussiness. date_trunc('year', d)
-- on a date resolves to the timestamptz overload — timestamptz is the
-- preferred type in its category, so it wins the cast — and the result
-- then advances in absolute time, not local calendar time. Building
-- the series that way on a server set to Asia/Tehran silently produced
-- 1,825 days instead of 1,826: the local clock drifts an hour at the
-- DST boundary and the last day falls past the stop bound. Dates carry
-- no clock, so the arithmetic here cannot drift, on any server. The
-- explicit ::timestamp casts on to_char keep the same guarantee for
-- the label columns.
-- ====================================================================
DROP TABLE IF EXISTS olap.dim_date CASCADE;

CREATE TABLE olap.dim_date (
    date_key            integer     NOT NULL,   -- YYYYMMDD
    full_date           date        NOT NULL,
    year                smallint    NOT NULL,
    quarter             smallint    NOT NULL,
    month               smallint    NOT NULL,
    day_of_month        smallint    NOT NULL,
    day_of_year         smallint    NOT NULL,
    iso_year            smallint    NOT NULL,
    iso_week            smallint    NOT NULL,
    day_of_week         smallint    NOT NULL,   -- 1 = Monday .. 7 = Sunday
    day_name            varchar(9)  NOT NULL,
    day_abbrev          varchar(3)  NOT NULL,
    month_name          varchar(9)  NOT NULL,
    month_abbrev        varchar(3)  NOT NULL,
    quarter_label       varchar(2)  NOT NULL,   -- Q1 .. Q4
    year_month          integer     NOT NULL,   -- YYYYMM, sorts correctly
    year_month_label    varchar(8)  NOT NULL,   -- '2011-01'
    year_quarter_label  varchar(8)  NOT NULL,   -- '2011-Q1'
    month_start_date    date        NOT NULL,
    month_end_date      date        NOT NULL,
    days_in_month       smallint    NOT NULL,
    is_weekend          boolean     NOT NULL,
    is_month_end        boolean     NOT NULL,
    PRIMARY KEY (date_key),
    UNIQUE (full_date)
);

INSERT INTO olap.dim_date (
    date_key, full_date, year, quarter, month, day_of_month, day_of_year,
    iso_year, iso_week, day_of_week, day_name, day_abbrev, month_name,
    month_abbrev, quarter_label, year_month, year_month_label,
    year_quarter_label, month_start_date, month_end_date, days_in_month,
    is_weekend, is_month_end
)
WITH span AS (
    -- Widest date the model has to be able to represent. Ship dates run
    -- past the last order date, so both tables have a say.
    SELECT least(
               (SELECT min("Order Date")::date FROM oltp.order),
               (SELECT min("Ship Date")::date  FROM oltp.shipping)
           ) AS first_day,
           greatest(
               (SELECT max("Order Date")::date FROM oltp.order),
               (SELECT max("Ship Date")::date  FROM oltp.shipping)
           ) AS last_day
),
bounds AS (
    SELECT make_date(extract(year FROM first_day)::integer, 1, 1)   AS first_day,
           make_date(extract(year FROM last_day)::integer, 12, 31)  AS last_day
    FROM span
),
calendar AS (
    -- date + integer = date. An integer series, not a timestamp one.
    SELECT b.first_day + offset_days AS full_date
    FROM bounds b,
         generate_series(0, b.last_day - b.first_day) AS offset_days
),
parts AS (
    SELECT full_date,
           extract(year    FROM full_date)::integer AS year,
           extract(quarter FROM full_date)::integer AS quarter,
           extract(month   FROM full_date)::integer AS month,
           extract(day     FROM full_date)::integer AS day_of_month,
           make_date(extract(year  FROM full_date)::integer,
                     extract(month FROM full_date)::integer, 1) AS month_start_date
    FROM calendar
),
derived AS (
    -- date + interval = timestamp (exact operator match, no timestamptz,
    -- so no timezone and no DST anywhere in this file).
    SELECT p.*,
           (month_start_date + interval '1 month' - interval '1 day')::date
               AS month_end_date
    FROM parts p
)
SELECT
    year * 10000 + month * 100 + day_of_month,
    full_date,
    year::smallint,
    quarter::smallint,
    month::smallint,
    day_of_month::smallint,
    extract(doy     FROM full_date)::smallint,
    extract(isoyear FROM full_date)::smallint,
    extract(week    FROM full_date)::smallint,
    extract(isodow  FROM full_date)::smallint,
    -- to_char pads names to a fixed width; without the TM prefix it is
    -- always English, so labels do not drift with the server locale.
    btrim(to_char(full_date::timestamp, 'Day')),
    btrim(to_char(full_date::timestamp, 'Dy')),
    btrim(to_char(full_date::timestamp, 'Month')),
    btrim(to_char(full_date::timestamp, 'Mon')),
    'Q' || quarter::text,
    year * 100 + month,
    to_char(full_date::timestamp, 'YYYY-MM'),
    year::text || '-Q' || quarter::text,
    month_start_date,
    month_end_date,
    extract(day FROM month_end_date)::smallint,
    extract(isodow FROM full_date) >= 6,
    full_date = month_end_date
FROM derived;

CREATE INDEX idx_dim_date_full_date  ON olap.dim_date (full_date);
CREATE INDEX idx_dim_date_year_month ON olap.dim_date (year_month);

COMMENT ON TABLE  olap.dim_date IS
    'Gap-free calendar covering whole years around the data span.';
COMMENT ON COLUMN olap.dim_date.day_of_week IS
    'ISO day number: 1 = Monday .. 7 = Sunday.';
COMMENT ON COLUMN olap.dim_date.iso_week IS
    'ISO week number; pair it with iso_year, not year, at year boundaries.';

-- ── role-playing views ──────────────────────────────────────────────
-- Both facts carry two date keys, order and ship. A BI tool can only
-- keep one active relationship to a single table, so the standard fix
-- is to expose the same dimension twice under different names. These
-- are views, not copies: one definition, no chance of the two drifting.
CREATE VIEW olap.dim_order_date AS SELECT * FROM olap.dim_date;
CREATE VIEW olap.dim_ship_date  AS SELECT * FROM olap.dim_date;

COMMENT ON VIEW olap.dim_order_date IS
    'Role-playing alias of dim_date, joined on <fact>.order_date_key.';
COMMENT ON VIEW olap.dim_ship_date IS
    'Role-playing alias of dim_date, joined on <fact>.ship_date_key.';
