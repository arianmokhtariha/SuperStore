-- ====================================================================
-- 13_dim_product.sql
-- What was sold, with its category hierarchy.
--
-- Source : oltp.product
-- Grain  : one row per product id — 10,246 rows
--
-- Hierarchy: category > sub_category > product. Verified functional in
-- the source (no sub-category appears under two categories), so the
-- flat denormalised form below loses nothing.
--
-- product_id is the identity. 1,937 product names are shared by more
-- than one id — the same catalogue item re-listed per market — so the
-- name is a label, never a key. Three names carry stray whitespace in
-- the source; btrim fixes them here, which is the whole reason this
-- layer exists rather than editing the raw export.
--
-- Deliberately no price or profitability attributes: those vary line
-- by line with discount and market, so they are measures of the fact,
-- not properties of the product.
-- ====================================================================
DROP TABLE IF EXISTS olap.dim_product CASCADE;

CREATE TABLE olap.dim_product (
    product_key   integer      NOT NULL,
    product_id    varchar(16)  NOT NULL,
    product_name  varchar(128) NOT NULL,
    category      varchar(16)  NOT NULL,
    sub_category  varchar(16)  NOT NULL,
    PRIMARY KEY (product_key),
    UNIQUE (product_id)
);

INSERT INTO olap.dim_product (
    product_key, product_id, product_name, category, sub_category
)
SELECT
    row_number() OVER (ORDER BY "Product ID")::integer,
    "Product ID",
    btrim("Product Name"),
    btrim("Category"),
    btrim("Sub-Category")
FROM oltp.product;

CREATE INDEX idx_dim_product_category ON olap.dim_product (category, sub_category);

COMMENT ON TABLE  olap.dim_product IS
    'One row per product id, with the category > sub_category hierarchy.';
COMMENT ON COLUMN olap.dim_product.product_name IS
    'Display label only — 1,937 names are shared across product ids.';
