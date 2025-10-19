import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_customer_features(
    fact_sales: pd.DataFrame,
    dim_customer: pd.DataFrame,
    dim_ship_mode: pd.DataFrame,
    dim_product: pd.DataFrame,
    dim_priority: pd.DataFrame,
    dim_market: pd.DataFrame,
    dim_geography: pd.DataFrame,
    dim_date: pd.DataFrame,
) -> pd.DataFrame:
    fact = (
        fact_sales.merge(dim_customer, on="Customer Key", how="inner")
        .merge(dim_ship_mode, on="Ship Mode Key", how="inner")
        .merge(dim_product, on="Product Key", how="inner")
        .merge(dim_priority, on="Priority Key", how="inner")
        .merge(dim_market, on="Market Key", how="inner")
        .merge(dim_geography, on="Geo Key", how="inner")
        .merge(dim_date, left_on="Order Date Key", right_on="Date Key", how="inner")
    )

    fact = fact[fact["Is Returned"] == 0].drop(
        columns=[
            "Product Key",
            "Customer Key",
            "Geo Key",
            "Priority Key",
            "Ship Mode Key",
            "Market Key",
            "Ship Date Key",
            "Date Key",
        ],
        errors="ignore",
    )

    metrics = fact.groupby("Customer ID", as_index=False)[
        ["Sales", "Quantity", "Discount", "Profit", "Shipping Cost"]
    ].sum()

    yearly = (
        fact.groupby(["Customer ID", "Year"], as_index=False)["Row ID"]
        .count()
        .rename(columns={"Row ID": "Orders_Per_Year"})
        .pivot(index="Customer ID", columns="Year", values="Orders_Per_Year")
        .fillna(0.0)
        .rename(columns=lambda y: f"Year_{int(y)}")
        .reset_index()
    )

    return metrics.merge(yearly, on="Customer ID", how="inner")


def add_clusters(features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "Sales",
        "Quantity",
        "Discount",
        "Profit",
        "Shipping Cost",
        "Year_2011",
        "Year_2012",
        "Year_2013",
        "Year_2014",
    ]
    X = features[feature_cols]

    pipeline_raw = Pipeline(
        [("kmeans", KMeans(n_clusters=3, max_iter=1000, random_state=0))]
    )
    features["Cluster_K3_Raw"] = pipeline_raw.fit_predict(X)

    pipeline_scaled = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=5, max_iter=1000, random_state=0)),
        ]
    )
    features["Cluster_K5_Scaled"] = pipeline_scaled.fit_predict(X)

    return features


result_df = add_clusters(
    build_customer_features(
        Fact_Sales,
        Dim_Customer,
        Dim_Ship_Mode,
        Dim_Product,
        Dim_Priority,
        Dim_Market,
        Dim_Geography,
        Dim_Date,
    )
)  ##powerbi takes care of this
