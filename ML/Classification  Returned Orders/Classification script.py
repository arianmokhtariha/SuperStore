import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler , MinMaxScaler , OneHotEncoder 
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Cats = ['Segment' , 'Ship Mode Name' , 'Category' , 'Priority Name' , 'Region' , 'Order Day of Week' , 'Ship Day of Week']
Nums = ['Sales' , 'Quantity' , 'Discount' , 'Profit' , 'Shipping Cost' ,'Time Diffrence']
Columns = ['Sales' , 'Quantity' , 'Discount' , 'Profit' , 'Shipping Cost' ,'Segment' ,'Ship Mode Name' , 'Category' ,'Priority Name' ,'Region' , 'Order Day of Week' , 'Ship Day of Week' , 'Time Diffrence' , 'Is Returned']
log_scales=['Sales', 'Shipping Cost' ]

def build_order_features(
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
    fact_sales['Order Date']=pd.to_datetime(fact_sales['Order Date'],format='%Y-%m-%d %H:%M:%S.%f')
    fact_sales['Ship Date']=pd.to_datetime(fact_sales['Ship Date'],format='%Y-%m-%d %H:%M:%S.%f')
    fact_sales['Time Diffrence']=fact_sales['Ship Date']-fact_sales['Order Date']
   
    Orders = fact_sales[Columns]
    Orders['Time Diffrence']=Orders['Time Diffrence'].dt.days
    return Orders
Orders= build_order_features( 
    Fact_Sales,
    Dim_Customer,
    Dim_Ship_Mode,
    Dim_Product,
    Dim_Priority,
    Dim_Market,
    Dim_Geography,
    Dim_Date
    )

def classifier(Orders :pd.DataFrame):
    Y=Orders['Is Returned']
    X=Orders.drop(columns=["Is Returned"])
    X_train , X_test, Y_train , Y_test =train_test_split(X, Y, train_size=0.8 , test_size=0.2 , random_state=42)

    encoder = OneHotEncoder(sparse_output=False)

    one_hot_encoded = encoder.fit_transform(X_train[Cats])

    one_hot_df = pd.DataFrame(one_hot_encoded, 
                            columns=encoder.get_feature_names_out(Cats))

    Xtrain_encoded = pd.concat([X_train.drop(Cats, axis=1).reset_index(), one_hot_df], axis=1)
    Xtrain_encoded=Xtrain_encoded.set_index('index')

    Xtrain_transformed=Xtrain_encoded.copy()

    scaler_logMinMax = MinMaxScaler(feature_range=(0, 1))
    Quantity_MinMax = MinMaxScaler()
    Discount_MinMax = MinMaxScaler()
    Time_MinMax = MinMaxScaler()
    scaler_Std = StandardScaler()

    def log_and_scale(X, scaler=scaler_logMinMax, shift:int=0):
        X_log = np.log(X + shift)
        scaled = scaler.fit_transform(X_log)
        return pd.DataFrame(scaled, index=X.index, columns=X.columns)


    def inv_log_and_scale(X, scaler=scaler_logMinMax, shift:int=0):
        unscaled = scaler.inverse_transform(X)
        return pd.DataFrame(np.exp(unscaled) - shift, index=X.index, columns=X.columns)

    log_and_scale_transformer = FunctionTransformer(
        func=log_and_scale, 
        inverse_func=inv_log_and_scale, 
        validate=False,  
        check_inverse=True,
    )

    Xtrain_transformed[log_scales]=log_and_scale_transformer.fit_transform(Xtrain_transformed[log_scales])
    Xtrain_transformed['Quantity']=Quantity_MinMax.fit_transform(Xtrain_transformed[['Quantity']])
    Xtrain_transformed['Discount']=Discount_MinMax.fit_transform(Xtrain_transformed[['Discount']])
    Xtrain_transformed['Time Diffrence']=Time_MinMax.fit_transform(Xtrain_transformed[['Time Diffrence']])
    Xtrain_transformed['Profit']=scaler_Std.fit_transform(Xtrain_transformed[['Profit']])


    sm = SMOTE(k_neighbors=5)
    X_res, y_res = sm.fit_resample(Xtrain_transformed, Y_train)
    classifier = RandomForestClassifier(n_estimators=10, random_state=42)
    classifier.fit(X_res, y_res)


    one_hot_encoded_test = encoder.transform(X_test[Cats])

    one_hot_test_df = pd.DataFrame(one_hot_encoded_test, 
                            columns=encoder.get_feature_names_out(Cats))

    Xtest_encoded = pd.concat([X_test.drop(Cats, axis=1).reset_index(), one_hot_test_df], axis=1)
    Xtest_encoded=Xtest_encoded.set_index('index')

    Xtest_transformed=Xtest_encoded.copy()

    Xtest_transformed['Quantity']=Quantity_MinMax.transform(Xtest_transformed[['Quantity']])
    Xtest_transformed['Discount']=Discount_MinMax.transform(Xtest_transformed[['Discount']])
    Xtest_transformed['Time Diffrence']=Time_MinMax.transform(Xtest_transformed[['Time Diffrence']])
    Xtest_transformed['Profit']=scaler_Std.transform(Xtest_transformed[['Profit']])

    y_pred = classifier.predict(Xtest_transformed)

    result_df = pd.DataFrame({
    'True Labels' :Y_test,
    'Predicted Labels' : y_pred})

    return result_df

classifier(Orders)