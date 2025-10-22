import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score
)
from catboost import CatBoostClassifier, Pool, cv
from imblearn.over_sampling import SMOTENC

cat_cols = [
    "Priority Name",
    "Market",
    "Segment",
    "City",
    "State",
    "Region",
    "Year",
    "Month",
    "Day",
    "Day of Week",
    "Sales Label",
    "Discount Label",
]
selected_features = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
    "Priority Name",
    "Market",
    "Segment",
    "City",
    "State",
    "Region",
    "Year",
    "Month",
    "Day",
    "Day of Week",
]


def get_data_features(
    fact_sales_df: pd.DataFrame,
    dim_ship_mode_df: pd.DataFrame,
    dim_priority_df: pd.DataFrame,
    dim_market_df: pd.DataFrame,
    dim_customer_df: pd.DataFrame,
    dim_geography_df: pd.DataFrame,
    # dim_order_df: pd.DataFrame,
    dim_date_df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        fact_sales_df.merge(dim_ship_mode_df, on="Ship Mode Key", how="inner")
        .merge(dim_priority_df, on="Priority Key", how="inner")
        .merge(dim_market_df, on="Market Key", how="inner")
        .merge(dim_customer_df, on="Customer Key", how="inner")
        .merge(dim_geography_df, on="Geo Key", how="inner")
        # .merge(
        #     dim_order_df,
        #     on="Order Key",
        #     how="inner",
        # )
        .merge(dim_date_df, left_on="Order Date Key", right_on="Date Key", how="inner")
    )

    prepared_df = result[selected_features]

    prepared_df["profit_margin"] = prepared_df["Profit"] / (prepared_df["Sales"] + 1e-6)
    prepared_df["sales_per_unit"] = prepared_df["Sales"] / (
        prepared_df["Quantity"] + 1e-6
    )
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    labels = ["Low", "Potential Low", "Potential High", "High"]

    prepared_df["Sales Label"] = pd.cut(
        prepared_df["Sales"] / prepared_df["Sales"].max(),
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    prepared_df["Discount Label"] = pd.cut(
        prepared_df["Discount"] / prepared_df["Discount"].max(),
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    # features_to_cal = prepared_df.columns.to_list()

    for c in cat_cols:
        if c in prepared_df.columns:
            prepared_df[c] = prepared_df[c].astype("category")

    return prepared_df


prepared_df = get_data_features(
    fact_sales_df=Fact_Sales,
    dim_ship_mode_df=Dim_Ship_Mode,
    dim_priority_df=Dim_Priority,
    dim_market_df=Dim_Market,
    dim_customer_df=Dim_Customer,
    dim_geography_df=Dim_Geography,
    # dim_order_df: pd.DataFrame,
    dim_date_df=Dim_Date,
)
final_res = prepared_df.copy()

XCat = prepared_df.drop(columns=["Ship Mode Name"])
yCat = prepared_df["Ship Mode Name"].astype("category")

XCat_encoded = XCat.copy()
XCat_encoded[cat_cols] =  XCat_encoded[cat_cols].apply(LabelEncoder().fit_transform)
cat_features = [XCat_encoded.columns.get_loc(col) for col in cat_cols]
smote_nc = SMOTENC(categorical_features=cat_features, random_state=42)
XCat_balanced,yCat_balanced = smote_nc.fit_resample(XCat_encoded, yCat)

XC_train, XC_test, yC_train, yC_test = train_test_split(
    XCat_balanced,yCat_balanced, test_size=0.25, random_state=42, stratify=yCat_balanced
)

params = dict(
    iterations=2000,                         
    loss_function="MultiClass",
    eval_metric="TotalF1:average=Macro",
    depth=7,
    learning_rate=0.04,                     
    random_seed=42,
    verbose=False,
)
train_pool = Pool(XC_train, yC_train, cat_features=cat_cols)
cv_result = cv(
    pool=train_pool,
    params=params,
    fold_count=3,
    shuffle=True,
    stratified=True,
    partition_random_seed=42,
    verbose=False
)
best_iter = cv_result['test-TotalF1:average=Macro-mean'].idxmax()
# best_iter = cv_result['test-TotalF1:average=Macro-mean'].idxmax()
final_model = CatBoostClassifier(
    iterations=best_iter,
    loss_function="MultiClass",
    eval_metric="TotalF1:average=Macro",
    depth=7,
    learning_rate=0.04,                      
    random_seed=42,
    verbose=False,
)

final_model.fit(train_pool)

y_train_p = final_model.predict(XC_train)
y_test_t = final_model.predict(XC_test)


classification_metrics_df = pd.DataFrame(
    {
        "Metric": ["F1-MACRO-TEST","F1-MACRO-TRAIN"],
        "Value": [
            f1_score(yC_train, y_train_p),
            f1_score(yC_test, y_test_t)
        ],
    }
)

final_res["predicted_ship_mode"] = final_model.predict(XCat_encoded)
result_df = final_res

feature_importance_df = pd.DataFrame(
    final_model.get_feature_importance(prettified=True)
)
