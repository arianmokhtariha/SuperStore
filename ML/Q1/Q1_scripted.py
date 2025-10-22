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
        'Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost',
       'Order Date Key', 'Ship Date Key', 'Priority Name', 'Sub Category',
       'Ship Mode Name', 'Market', 'Country', 'Year', 'Quarter', 'Day of Week',
       'Segment'
    ]

    data = merged[base_cols].copy()

    top_countries = data["Country"].value_counts().nlargest(10).index
    data["Country_Grouped"] = np.where(
        data["Country"].isin(top_countries), data["Country"], "Other"
    )

    data['Order Date'] = pd.to_datetime(data['Order Date Key'], format='%Y%m%d')
    data['Ship Date'] = pd.to_datetime(data['Ship Date Key'], format='%Y%m%d')

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

    featured = dummied.copy()

    featured['Price_per_Unit'] = featured['Sales'] / featured['Quantity']

    featured['Discount_Amount'] = featured['Sales'] * featured['Discount']

    featured['Order_Month'] = featured['Order Date'].dt.month

    featured["powered_discount"] = featured["Discount"] ** 2

    sub_cat_cols = [col for col in featured.columns if col.startswith('sub_cat_')]
    original_sub_cat_col = featured[sub_cat_cols].idxmax(axis=1)
    featured['sub_cat_original'] = original_sub_cat_col.str.replace('sub_cat_', '')
    sub_cat_sales_map = featured.groupby('sub_cat_original')['Sales'].mean().to_dict()
    featured['mean_sale_of_each_sub_cat'] = featured['sub_cat_original'].map(sub_cat_sales_map)
    featured = featured.drop('sub_cat_original', axis=1)

    featured['mean_sub_cat_sale_powered'] = featured['mean_sale_of_each_sub_cat'] ** 2

    drop_list = [
        col
        for col in ["market_US", "market_EMEA", "SMN_Second Class", "Shipping Cost" , 'Segment' ,
                     'Market' , 'Ship Mode Name' , 'Sub Category' , 'Year',
                    'Priority Name' , 'Country_Grouped' ,'Country', 'Day of Week',
                    'Quarter', 'Order Date Key' , 'Ship Date Key' , 'Ship Date']
        if col in featured.columns
    ]
    featured = featured.drop(columns=drop_list)

    featured = featured.sort_values(by='Order Date').reset_index(drop=True)
    featured.drop(columns=['Order Date'] , inplace=True)

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
    col for col in prepared_df.columns if col not in ("Profit")
]
X = prepared_df[feature_cols]
y = prepared_df["Profit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.85, shuffle=False, random_state=44
)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb_params = dict(
    booster='gbtree',
    tree_method = "hist",
    objective='reg:squarederror',
    random_state=440,
    n_estimators=687,
    max_depth=2,
    min_child_weight=2,
    reg_alpha=0.05,
    learning_rate=0.04,
    reg_lambda=0.5,
    colsample_bytree=0.7,
    subsample=0.8,
)

xgb_model = XGBRegressor(**xgb_params)
xgb_model.fit(
    X_train_scaled, y_train, verbose=False
)

y_train_pred = xgb_model.predict(X_train_scaled)
y_test_pred = xgb_model.predict(X_test_scaled)

regression_metrics_df = pd.DataFrame(
    {
        "Metric": ["R2_train", "R2_test", "RMSE_test", "MAE_test"],
        "Value": [
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred),
            mean_squared_error(y_test, y_test_pred),
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
