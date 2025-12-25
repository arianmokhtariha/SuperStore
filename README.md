# 🛒 SuperStore Data Analysis & Predictive Modeling

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PowerBI](https://img.shields.io/badge/PowerBI-Dashboard-yellow)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20CatBoost%20%7C%20Sklearn-orange)
![Hypothesis Testing](https://img.shields.io/badge/Hypothesis%20Testing-SciPy%20%7C%20Shapiro%20%7C%20Mann--Whitney-purple)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

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

The solution utilizes a **Star Schema** data model, statistical hypothesis testing to validate assumptions, and multiple Machine Learning models integrated into a **Power BI** dashboard for stakeholder reporting.

## 🛠️ Tech Stack & Tools
*   **Languages:** Python (Pandas, NumPy)
*   **Machine Learning:** Scikit-Learn, XGBoost, CatBoost, Imbalanced-Learn (SMOTE)
*   **Statistics:** SciPy (Hypothesis testing: Shapiro-Wilk, Mann-Whitney U, Yeo-Johnson)
*   **Visualization:** Power BI (DAX, Dashboarding), Matplotlib, Seaborn, Plotly
*   **Data Modeling:** Star Schema Design (Fact & Dimension Tables)

---

## 📊 Key Features & Analysis

### 1. Data Modeling (Star Schema)
The raw data is structured into a relational model to ensure efficient querying and analysis:
*   **Fact Table:** `Fact_Sales`
*   **Dimension Tables:** `Dim_Customer`, `Dim_Product`, `Dim_Date`, `Dim_Geography`, `Dim_Ship_Mode`, etc.

### 2. Machine Learning Modules
The project implements three distinct ML tasks to solve business problems:

*   **📉 Classification (Returned Orders):**
    *   **Goal:** Predict if an order will be returned to minimize shipping losses.
    *   **Model:** Random Forest Classifier.
    *   **Techniques:** Handling class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique), OneHotEncoding, and Feature Scaling.
    *   **Metrics:** Confusion Matrix, Accuracy Score.

*   **💰 Regression (Profit Prediction):**
    *   **Goal:** Predict the profit of specific sales transactions based on discounts, shipping costs, and product categories.
    *   **Model:** XGBoost Regressor.
    *   **Techniques:** RobustScaler for outliers, Feature Importance analysis.
    *   **Metrics:** R2 Score, RMSE, MAE.

*   **👥 Clustering (Customer Segmentation):**
    *   **Goal:** Group customers based on purchasing behavior (RFM-style metrics).
    *   **Model:** K-Means Clustering.
    *   **Visualization:** PCA (Principal Component Analysis) and t-SNE for dimensionality reduction and cluster visualization.

### 3. Statistical Hypothesis Testing
Rigorous statistical tests were conducted to validate business assumptions:
*   **Discount vs. Quantity:** Used **Mann-Whitney U Test** to determine if discounts significantly impact the quantity of items purchased.
*   **Normality Checks:** Applied **Shapiro-Wilk** tests and **Yeo-Johnson** transformations to check data distribution before modeling.

### 4. Interactive Dashboard (Power BI)
The final insights and model outputs are visualized in Power BI:
*   Executive summary of Sales and Profit.
*   Geospatial analysis of shipping routes.
*   Visualizing customer clusters and predicted return rates.

---

## 📂 Project Structure

```text
├── Fact&dim-csv/                # CSV files representing the Database Schema
├── Hypothesis testing/          # Statistical analysis scripts (SciPy)
├── ML/
│   ├── Classification Returned Orders/  # Random Forest & SMOTE implementation
│   ├── Clustering/                      # KMeans, PCA, and t-SNE scripts
│   ├── Q1/                              # Profit Regression (XGBoost)
│   ├── Q2/                              # Ship Mode/Analysis (CatBoost)
└── Final_ProjectSuperstore.pbix # Main Power BI Dashboard