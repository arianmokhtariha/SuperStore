# 🛒 SuperStore Data Analysis & Predictive Modeling

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PowerBI](https://img.shields.io/badge/Visualization-PowerBI%20Dashboard-f2c811)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20CatBoost%20%7C%20RandomForest-orange)
![Stats](https://img.shields.io/badge/Statistical_Analysis-Mann--Whitney%20%7C%20Shapiro--Wilk%20%7C%20Yeo--Johnson-blueviolet)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

> **Credit & Collaboration Note:**  
> This project was originally developed as a group effort for Quera's Data Analysis bootcamp. This repository is a **continuation/fork** containing my own refactored code, additional machine learning optimizations, and extended documentation.  
> **Original Team:**
  [@AlirezaNyi](https://github.com/AlirezaNyi)
  [@arianmokhtariha](https://github.com/arianmokhtariha)
  [@mohsen20roohi-hue](https://github.com/mohsen20roohi-hue)
  [@MonaKheirieh](https://github.com/MonaKheirieh)
  [@anooshanth](https://github.com/anooshanth)

---

## 📖 Project Overview
This project performs an end-to-end analysis of a retail "SuperStore" dataset. The goal was to identify key drivers of profitability, predict order returns to reduce logistics costs, and segment customers for targeted marketing strategies.

The solution utilizes a **Star Schema** data model, rigorous statistical hypothesis testing to validate assumptions, and multiple advanced Machine Learning pipelines (XGBoost, CatBoost) integrated into a **Power BI** dashboard.

## 🏗️ Advanced Machine Learning Pipelines

This project features three distinct predictive modeling modules, engineered to solve specific business problems.

### 1. 📉 Return Prevention System (Binary Classification)
*   **Objective:** Predict the probability of an order being returned (`Is Returned = 1`) to proactively flag high-risk transactions.
*   **Model:** `RandomForestClassifier` (Ensemble Learning).
*   **Class Imbalance Handling:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) with $k=5$ neighbors to generate synthetic examples for the minority class (Returns), ensuring the model doesn't bias towards non-returns.
*   **Feature Engineering:**
    *   Calculated `Time Difference` (Shipping Date - Order Date).
    *   One-Hot Encoding for categorical variables (`Segment`, `Region`).
*   **Preprocessing:**
    *   Custom **Log-Transformation** pipeline for skewed numerical features (`Sales`, `Shipping Cost`).
    *   `MinMaxScaler` for time-based features.

### 2. 💰 Profit forecasting Engine (Regression)
*   **Objective:** Predict the exact profit of sales transactions to optimize pricing strategies.
*   **Model:** `XGBoost Regressor` (Gradient Boosting).
*   **Hyperparameter Optimization:** Tuned using Histogram-based tree method (`tree_method='hist'`) with regularization (`reg_alpha=0.05`, `reg_lambda=0.5`) to prevent overfitting.
*   **Advanced Feature Engineering:**
    *   **Target Encoding:** Created `mean_sale_of_each_sub_cat` to capture category value without high-cardinality dimensionality expansion.
    *   **Polynomial Features:** Generated interaction terms like `powered_discount` ($Discount^2$) to capture non-linear relationships between price cuts and profit.
*   **Scaling:** utilized **RobustScaler** instead of Standard Scaler to minimize the impact of extreme outliers in the sales data.

### 3. 🚢 Logistics Optimization (Multi-Class Classification)
*   **Objective:** Classify/Predict the optimal `Ship Mode` based on order characteristics.
*   **Model:** `CatBoostClassifier`.
*   **Configuration:** Optimized for multi-class loss with `TotalF1:average=Macro` metric.
*   **Features:** Engineered `profit_margin` and `sales_per_unit` ratios to give the model context on unit economics.

### 4. 👥 Customer Segmentation (Clustering)
*   **Algorithm:** **K-Means Clustering** ($k=5$).
*   **Dimensionality Reduction:** Applied **PCA** (Principal Component Analysis) and **t-SNE** to project high-dimensional customer data into 2D space for visualization.
*   **Pipeline:** `StandardScaler` $\to$ `KMeans` $\to$ Power BI Integration.

---

## 🔬 Statistical Analysis & Methodology
*   **Tools:** `SciPy`, `Statsmodels`
*   **Normalization:** Applied **Yeo-Johnson transformations** to normalize skewed continuous variables (e.g., Sales Quantity) before modeling.
*   **Normality Testing:** Used **Shapiro-Wilk** tests to check distribution assumptions.
*   **Hypothesis Testing:** Conducted **Mann-Whitney U Tests** (non-parametric) to determine if discounted items significantly impact order quantity compared to non-discounted items.

---

## 📂 Project Structure

```text
├── Fact&dim-csv/                # The Database: CSV files representing the Star Schema
├── Hypothesis testing/          # Statistical analysis scripts (SciPy/Statsmodels)
│   ├── stats.ipynb              # Main statistical notebook
│   └── stats_fpr_PBI.py         # Script optimized for Power BI integration
├── ML/
│   ├── Classification Returned Orders/  # Random Forest & SMOTE implementation
│   ├── Clustering/                      # KMeans, PCA, and t-SNE scripts
│   ├── Q1/                              # Profit Regression (XGBoost)
│   ├── Q2/                              # Ship Mode Analysis (CatBoost)
└── Final_ProjectSuperstore.pbix # Main Power BI Dashboard file
```

## 📊 Data Schema Details
Although stored as CSVs, the data is structured relationally:

| Table | Type | Description |
| :--- | :--- | :--- |
| **Fact_Sales** | FACT | Contains foreign keys (Product Key, Customer Key) and metrics (Sales, Profit, Quantity). |
| **Dim_Customer** | DIM | Customer demographics and segmentation (Consumer, Corporate, Home Office). |
| **Dim_Product** | DIM | Product hierarchy (Category > Sub-Category > Product Name). |
| **Dim_Geography** | DIM | Spatial data (City, State, Region, Country). |
| **Dim_Date** | DIM | Calendar attributes for time-series analysis. |

---

## 🚀 How to Run

### 1. Prerequisites
*   Python 3.8+
*   Microsoft Power BI Desktop (to view `.pbix` files).
*   Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `catboost`, `scipy`, `imblearn`.

### 2. Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/YourUsername/SuperStore-Analysis.git
pip install pandas numpy scikit-learn xgboost catboost scipy imbalanced-learn
```

### 3. Running the Analysis
**To Run the Machine Learning Models:**
Navigate to the specific folder and run the python script. For example, to run the Classification model:

```bash
cd "ML/Classification Returned Orders"
python "Classification script.py"
```

**To View the Dashboard:**
1.  Open `Final_ProjectSuperstore.pbix` in Power BI Desktop.
2.  *Note: You may need to update the data source settings in Power BI to point to the `Fact&dim-csv` folder on your local machine.*

---

## 📈 Future Improvements
*   **Optimization:** Perform Hyperparameter tuning (GridSearchCV) on the XGBoost model to further minimize RMSE.