import os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,RobustScaler
from catboost import CatBoostClassifier, Pool, cv
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(FILE_DIR, "Fact&dim-csv")
MODEL_DIR = os.path.join(BASE_DIR, "ML_models")
os.makedirs(MODEL_DIR, exist_ok=True)

FACT_SALES_CSV = os.path.join(BASE_DIR, "Fact_Sales.csv")
DIM_SHIP_MODE_CSV = os.path.join(BASE_DIR, "Dim_Ship_Mode.csv")
DIM_PRIORITY_CSV = os.path.join(BASE_DIR, "Dim_Priority.csv")
DIM_MARKET_CSV = os.path.join(BASE_DIR, "Dim_Market.csv")
DIM_CUSTOMER_CSV = os.path.join(BASE_DIR, "Dim_Customer.csv")
DIM_GEOGRAPHY_CSV = os.path.join(BASE_DIR, "Dim_Geography.csv")
DIM_DATE_CSV = os.path.join(BASE_DIR, "Dim_Date.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "shipmode.cbm")
META_PATH = os.path.join(MODEL_DIR, "shipmode_meta.json")


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
    dim_date_df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        fact_sales_df.merge(dim_ship_mode_df, on="Ship Mode Key", how="inner")
        .merge(dim_priority_df, on="Priority Key", how="inner")
        .merge(dim_market_df, on="Market Key", how="inner")
        .merge(dim_customer_df, on="Customer Key", how="inner")
        .merge(dim_geography_df, on="Geo Key", how="inner")
        .merge(dim_date_df, left_on="Order Date Key", right_on="Date Key", how="inner")
    )

    prepared_df = result[selected_features + ["Ship Mode Name"]].copy()

    prepared_df["profit_margin"] = prepared_df["Profit"] / (prepared_df["Sales"] + 1e-6)
    prepared_df["sales_per_unit"] = prepared_df["Sales"] / (
        prepared_df["Quantity"] + 1e-6
    )

    bins = [0, 0.25, 0.5, 0.75, 1.0]
    labels = ["Low", "Potential Low", "Potential High", "High"]

    max_sales = max(prepared_df["Sales"].max(), 1.0)
    max_disc = max(prepared_df["Discount"].max(), 1.0)

    prepared_df["Sales Label"] = pd.cut(
        prepared_df["Sales"] / max_sales, bins=bins, labels=labels, include_lowest=True
    )
    prepared_df["Discount Label"] = pd.cut(
        prepared_df["Discount"] / max_disc,
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    for c in cat_cols:
        if c in prepared_df.columns:
            prepared_df[c] = prepared_df[c].astype("category")

    return prepared_df


def main():

    fact_sales = pd.read_csv(FACT_SALES_CSV, low_memory=False)
    dim_ship_mode = pd.read_csv(DIM_SHIP_MODE_CSV, low_memory=False)
    dim_priority = pd.read_csv(DIM_PRIORITY_CSV, low_memory=False)
    dim_market = pd.read_csv(DIM_MARKET_CSV, low_memory=False)
    dim_customer = pd.read_csv(DIM_CUSTOMER_CSV, low_memory=False)
    dim_geography = pd.read_csv(DIM_GEOGRAPHY_CSV, low_memory=False)
    dim_date = pd.read_csv(DIM_DATE_CSV, low_memory=False)

    prepared_df = get_data_features(
        fact_sales_df=fact_sales,
        dim_ship_mode_df=dim_ship_mode,
        dim_priority_df=dim_priority,
        dim_market_df=dim_market,
        dim_customer_df=dim_customer,
        dim_geography_df=dim_geography,
        dim_date_df=dim_date,
    )

    XCat = prepared_df.drop(columns=["Ship Mode Name"])
    yCat = prepared_df["Ship Mode Name"].astype("category")

    encoders = {}
    XCat_encoded = XCat.copy()
    present_cat_cols = [c for c in cat_cols if c in XCat_encoded.columns]
    robost_scaler = RobustScaler()


    cat_features_idx = [XCat_encoded.columns.get_loc(c) for c in present_cat_cols]
    numeric_cols =  XCat.select_dtypes(exclude=['category','object']).columns.to_list()

    XC_train_n, XC_test, yC_train_n, yC_test = train_test_split(
        XCat_encoded, yCat, test_size=0.25, random_state=42, stratify=yCat
    )

    XC_train, yC_train = XC_train_n, yC_train_n

    XC_train[numeric_cols] = robost_scaler.fit_transform(XC_train[numeric_cols])
    XC_test[numeric_cols] = robost_scaler.transform(XC_test[numeric_cols])

    final_model = CatBoostClassifier(
        iterations=1000,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        depth=2,
        learning_rate=0.06,
        verbose=False,
        random_state=42,
    )
    final_model.fit(
        Pool(XC_train, yC_train,cat_features=cat_features_idx),
        eval_set=Pool(XC_test,yC_test,cat_features=cat_features_idx),
        use_best_model=True,
        early_stopping_rounds=200,
    )

    y_pred_train = np.array(final_model.predict(XC_train)).ravel()
    y_pred_test = np.array(final_model.predict(XC_test)).ravel()

    metrics = {
        "train": {
            "accuracy": float(accuracy_score(yC_train, y_pred_train)),
            "f1_macro": float(f1_score(yC_train, y_pred_train, average="macro")),
        },
        "test": {
            "accuracy": float(accuracy_score(yC_test, y_pred_test)),
            "f1_macro": float(f1_score(yC_test, y_pred_test, average="macro")),
        },
    }

    labels = yCat.cat.categories.tolist()
    cm_train = confusion_matrix(yC_train, y_pred_train, labels=labels)
    cm_test = confusion_matrix(yC_test, y_pred_test, labels=labels)
    confusion = {
        "labels": labels,
        "train": cm_train.tolist(),
        "test": cm_test.tolist(),
    }

    final_model.save_model(MODEL_PATH)
    meta = {
        "feature_order": list(XCat_encoded.columns),
        "cat_cols": present_cat_cols,
        "label_encoders": encoders,
        "target_classes": yCat.cat.categories.tolist(),
        "metrics": metrics,
        "confusion matrix": confusion,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:")
    print("  Model:", MODEL_PATH)
    print("  Meta :", META_PATH)


if __name__ == "__main__":
    main()
