import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


def build_regression_frame(
    fact_sales: pd.DataFrame,
    dim_priority: pd.DataFrame,
    dim_product: pd.DataFrame,
    dim_ship_mode: pd.DataFrame,
    dim_market: pd.DataFrame,
    dim_geography: pd.DataFrame,
    dim_date: pd.DataFrame,
    dim_customer: pd.DataFrame,
) -> pd.DataFrame:
    merged = (
        fact_sales.merge(dim_priority, on="Priority Key", how="inner")
        .merge(dim_product, on="Product Key", how="inner")
        .merge(dim_ship_mode, on="Ship Mode Key", how="inner")
        .merge(dim_market, on="Market Key", how="inner")
        .merge(dim_geography, on="Geo Key", how="inner")
        .merge(dim_date, left_on="Ship Date Key", right_on="Date Key", how="inner")
        .merge(dim_customer, on="Customer Key", how="inner")
    )

    merged = merged[merged["Is Returned"] == 0]

    base_cols = [
        "Sales",
        "Quantity",
        "Discount",
        "Profit",
        "Shipping Cost",
        "Priority Name",
        "Sub Category",
        "Ship Mode Name",
        "Market",
        "Country",
        "Year",
        "Segment",
    ]
    data = merged[base_cols].copy()

    top_countries = data["Country"].value_counts().nlargest(10).index
    data["Country_Grouped"] = np.where(
        data["Country"].isin(top_countries), data["Country"], "Other"
    )

    dummied = data.copy()
    for column, prefix in [
        ("Priority Name", "priority"),
        ("Sub Category", "sub_cat"),
        ("Ship Mode Name", "SMN"),
        ("Market", "market"),
        ("Segment", "segment"),
        ("Country_Grouped", "country"),
    ]:
        dummies = pd.get_dummies(
            dummied[column],
            prefix=prefix,
            prefix_sep="_",
            drop_first=True,
            dtype=int,
        )
        dummied = dummied.join(dummies)

    dummied = dummied.drop(
        columns=[
            "Priority Name",
            "Sub Category",
            "Ship Mode Name",
            "Market",
            "Segment",
            "Country",
            "Country_Grouped",
        ]
    )

    featured = dummied.copy()
    start_year = featured["Year"].min()
    featured["Years_Since_Start"] = featured["Year"] - start_year
    featured = featured.drop(columns="Year")

    sales_threshold = featured["Sales"].quantile(0.6)
    featured["Is_High_Value_Order"] = (featured["Sales"] > sales_threshold).astype(int)

    discount_threshold = featured["Discount"].quantile(0.65)
    featured["Is_high_discount"] = (featured["Discount"] > discount_threshold).astype(
        int
    )
    featured["powered_discount"] = featured["Discount"] ** 2

    market_cols = [col for col in featured.columns if col.startswith("market_")]
    if market_cols:
        original_market = (
            featured[market_cols].idxmax(axis=1).str.replace("market_", "")
        )
        sales_map = (
            featured.assign(market_original=original_market)
            .groupby("market_original")["Sales"]
            .sum()
            .to_dict()
        )
        featured["sum_sale_of_each_market"] = original_market.map(sales_map)

    drop_list = [
        col
        for col in ["market_US", "market_EMEA", "SMN_Second Class", "Discount"]
        if col in featured.columns
    ]
    featured = featured.drop(columns=drop_list)

    return featured


prepared_df = build_regression_frame(
    Fact_Sales,
    Dim_Priority,
    Dim_Product,
    Dim_Ship_Mode,
    Dim_Market,
    Dim_Geography,
    Dim_Date,
    Dim_Customer,
)

feature_cols = [
    col for col in prepared_df.columns if col not in ("Profit", "Shipping Cost")
]
X = prepared_df[feature_cols]
y = prepared_df["Profit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.75, random_state=44
)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb_params = dict(
    booster="gbtree",
    tree_method="hist",
    objective="reg:squarederror",
    random_state=44,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    gamma=0,
    early_stopping_rounds=50,
    # device="cuda",
)

xgb_model = XGBRegressor(**xgb_params)
xgb_model.fit(
    X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False
)

y_train_pred = xgb_model.predict(X_train_scaled)
y_test_pred = xgb_model.predict(X_test_scaled)

regression_metrics_df = pd.DataFrame(
    {
        "Metric": ["R2_train", "R2_test", "RMSE_test", "MAE_test"],
        "Value": [
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred),
            mean_squared_error(y_test, y_test_pred, squared=False),
            mean_absolute_error(y_test, y_test_pred),
        ],
    }
)

X_full_scaled = scaler.transform(X)
prepared_df = prepared_df.copy()
prepared_df["Predicted_Profit"] = xgb_model.predict(X_full_scaled)
result_df = prepared_df

feature_importance_df = pd.DataFrame(
    {
        "Feature": feature_cols,
        "Importance": xgb_model.feature_importances_,
    }
).sort_values("Importance", ascending=False)
