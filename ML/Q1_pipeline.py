import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
import pickle

class SuperstoreProfitPipeline:
    def __init__(self):
        self.scaler = RobustScaler()
        self.model = None
        self.feature_names_in_ = None
        self.top_countries_ = None 

    def _engineer_features(self, df, is_training=False):
        df_copy = df.copy()

        if 'Order Date Key' in df_copy.columns:
            df_copy['Order Date'] = pd.to_datetime(df_copy['Order Date Key'], format='%Y%m%d')
        
        df_copy['Price_per_Unit'] = df_copy['Sales'] / df_copy['Quantity']
        df_copy['Discount_Amount'] = df_copy['Sales'] * df_copy['Discount']
        if 'Order Date' in df_copy.columns:
            df_copy['Order_Month'] = df_copy['Order Date'].dt.month
        df_copy['powered_discount'] = df_copy['Discount'] ** 2

        if 'Sub Category' in df_copy.columns:
            if is_training:
                self.sub_cat_sales_map_ = df_copy.groupby('Sub Category')['Sales'].mean().to_dict()
            if self.sub_cat_sales_map_:
                df_copy['mean_sale_of_each_sub_cat'] = df_copy['Sub Category'].map(self.sub_cat_sales_map_)
                fill_value = np.mean(list(self.sub_cat_sales_map_.values()))
                df_copy['mean_sale_of_each_sub_cat'].fillna(fill_value, inplace=True)
                df_copy['mean_sub_cat_sale_powered'] = df_copy['mean_sale_of_each_sub_cat'] ** 2
        
        if 'Country' in df_copy.columns:
            if is_training:
                self.top_countries_ = df_copy['Country'].value_counts().nlargest(10).index
            if self.top_countries_ is not None:
                df_copy['Country'] = np.where(df_copy['Country'].isin(self.top_countries_), df_copy['Country'], 'Other')

        categorical_cols = ['Priority Name', 'Sub Category', 'Ship Mode Name', 'Market', 'Segment', 'Country', 'Category']
        
        df_dummied = pd.get_dummies(df_copy, columns=[col for col in categorical_cols if col in df_copy.columns], drop_first=True, dtype=int)
        
        final_cols_to_drop = [
            'Order Date Key', 'Ship Date Key', 'Ship Date', 'Order Date', 'Customer Key', 'Product Key',
            'Geo Key', 'Product ID', 'Product Name', 'Customer ID', 'Customer Name', 'City', 'State',
            'Region', 'Date', 'Day of Week', 'Quarter', 'Is Returned', 'Year',
            'market_US', 'SMN_Second Class', 'market_EMEA', 'Shipping Cost'
        ]
        
        df_final = df_dummied.drop(columns=[col for col in final_cols_to_drop if col in df_dummied.columns], errors='ignore')
        
        return df_final

    def fit(self, X_train_raw, y_train):
        X_train_featured = self._engineer_features(X_train_raw, is_training=True)
        
        self.feature_names_in_ = X_train_featured.columns.tolist()

        X_train_scaled = self.scaler.fit_transform(X_train_featured)
        
        self.model = XGBRegressor(
            booster='gbtree',
            tree_method="hist",
            objective='reg:squarederror',
            random_state=440,
            n_estimators=687,
            max_depth=2,
            min_child_weight=2,
            reg_alpha=0.05,
            learning_rate=0.04,
            reg_lambda=0.5,
            colsample_bytree=0.7,
            subsample=0.8
        )
        self.model.fit(X_train_scaled, y_train, verbose=False)
        return self

    def predict(self, X_new_raw):
        if self.model is None:
            raise RuntimeError("pipeline has not been fitted yet.")
            
        X_new_featured = self._engineer_features(X_new_raw, is_training=False)
        
        X_new_aligned = X_new_featured.reindex(columns=self.feature_names_in_, fill_value=0)
        
        X_new_scaled = self.scaler.transform(X_new_aligned)
        
        predictions = self.model.predict(X_new_scaled)
        return predictions

if __name__ == '__main__':
    Fact_Sales = pd.read_csv(r"D:\Data Analysis BC\Projects\Project4\Data-Analysis-Projects-3-SuperStore\Fact&dim-csv\Fact_Sales.csv")
    Ship_Mode = pd.read_csv(r"D:\Data Analysis BC\Projects\Project4\Data-Analysis-Projects-3-SuperStore\Fact&dim-csv\Dim_Ship_Mode.csv")
    Product = pd.read_csv(r"D:\Data Analysis BC\Projects\Project4\Data-Analysis-Projects-3-SuperStore\Fact&dim-csv\Dim_Product.csv")
    Priority = pd.read_csv(r"D:\Data Analysis BC\Projects\Project4\Data-Analysis-Projects-3-SuperStore\Fact&dim-csv\Dim_Priority.csv")
    Market = pd.read_csv(r"D:\Data Analysis BC\Projects\Project4\Data-Analysis-Projects-3-SuperStore\Fact&dim-csv\Dim_Market.csv")
    Geography = pd.read_csv(r"D:\Data Analysis BC\Projects\Project4\Data-Analysis-Projects-3-SuperStore\Fact&dim-csv\Dim_Geography.csv")
    Date = pd.read_csv(r"D:\Data Analysis BC\Projects\Project4\Data-Analysis-Projects-3-SuperStore\Fact&dim-csv\Dim_Date.csv")
    Customer = pd.read_csv(r"D:\Data Analysis BC\Projects\Project4\Data-Analysis-Projects-3-SuperStore\Fact&dim-csv\Dim_Customer.csv")

    merged = pd.merge(Fact_Sales , Priority , how='inner' , on='Priority Key')
    merged = pd.merge(merged , Product , how='inner' , on='Product Key')
    merged = pd.merge(merged , Ship_Mode , how='inner' , on='Ship Mode Key')
    merged = pd.merge(merged , Market , how='inner' , on='Market Key')
    merged = pd.merge(merged , Geography , how='inner' , on='Geo Key')
    merged = pd.merge(merged, Date, left_on='Order Date Key', right_on='Date Key', how='inner')
    merged = pd.merge(merged , Customer , how='inner' , on='Customer Key')
    merged = merged[merged['Is Returned'] == 0]
    
    merged['Order Date'] = pd.to_datetime(merged['Order Date Key'], format='%Y%m%d')
    final_data_sorted = merged.sort_values(by='Order Date').reset_index(drop=True)
    
    X_base = final_data_sorted.drop('Profit', axis=1)
    Y_base = final_data_sorted['Profit']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_base, Y_base, train_size=0.8, shuffle=False
    )
    
    pipeline = SuperstoreProfitPipeline()
    pipeline.fit(X_train_raw, y_train)

    with open('Q1_profit_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    print("pipeline saved to 'Q1_profit_pipeline.pkl'")