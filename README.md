# Fraud_Transaction_Detection
🚨 Fraud Transaction Detection Using Machine Learning
📌 Project Overview

This project focuses on detecting fraudulent financial transactions using machine learning models on a large-scale synthetic dataset. The goal is to build, evaluate, and compare multiple tree-based models that are commonly used in real-world fraud detection systems.

The project emphasizes:

Robust data preprocessing

Memory-efficient feature engineering

Handling high-cardinality categorical features

Addressing severe class imbalance

Comparing XGBoost, LightGBM, and CatBoost

All steps are implemented in Jupyter Notebooks, structured for clarity and reproducibility.

📂 Repository Structure
├── fraud_trans_dete_preprocessing.ipynb
├── fraud_trans_dete_xg_light.ipynb
├── fraud_trans_dete_catboost.ipynb
├── README.md


📁 Dataset
The dataset is stored on Google Drive due to its large size (~1GB).
A download link is provided in this repository.

🧠 Dataset Description

Synthetic transaction dataset

Millions of rows

Highly imbalanced target variable (is_fraud)

Includes:

Transaction metadata

Merchant information

Device & channel details

Velocity-based behavioral features

🔧 1. Data Preprocessing & Feature Engineering

Notebook: fraud_trans_dete_preprocessing.ipynb

🔹 Key Steps
1️⃣ Initial Data Cleaning

Removed duplicate transactions using transaction_id

Checked for:

Missing values

Invalid transaction amounts

Logical inconsistencies (e.g., card_present = yes but channel ≠ POS)

2️⃣ Exploratory Validation

Examined value distributions of all categorical features

Ensured consistency across transaction attributes

3️⃣ Feature Reduction

Removed identifiers and leakage-prone columns such as:

transaction_id

customer_id

card_number

ip_address

timestamp

This ensures models learn behavioral patterns, not identities.

🚀 Velocity Feature Engineering

The column velocity_last_hour contained dictionary-like behavioral summaries.
These were safely extracted into numerical features:

num_transactions

total_amount

unique_merchants

unique_countries

max_single_amount

This step significantly improves fraud detection performance by capturing short-term behavioral risk.

🔄 Encoding Strategy (Critical Design Choice)

To support different model architectures, two separate datasets were created:

🔹 For XGBoost & LightGBM

Target Encoding

Applied to high-cardinality features:

merchant

merchant_type

Label Encoding

Applied to remaining categorical variables

Saved as:

fraud_data_lgbm_xgb_train.csv

fraud_data_lgbm_xgb_test.csv

🔹 For CatBoost

No manual encoding

Raw categorical columns preserved

Saved as:

fraud_data_catboost.csv

This separation avoids unnecessary memory usage and ensures each model uses its optimal input format.

🤖 2. XGBoost & LightGBM Modeling

Notebook: fraud_trans_dete_xg_light.ipynb

🔹 Why These Models?

Industry-standard for tabular fraud detection

Excellent handling of nonlinear interactions

Scalable to millions of rows

⚖️ Handling Class Imbalance

Fraud is rare → incorrect handling leads to biased models.

Solutions used:

XGBoost: scale_pos_weight

LightGBM: class_weight='balanced'

📊 Evaluation Metrics

Models are evaluated using:

Confusion Matrix

Precision, Recall, F1-score

ROC-AUC

ROC Curve visualization

ROC-AUC is emphasized since it reflects ranking quality in imbalanced datasets.

🏆 Results Summary
Model	ROC-AUC	Fraud Recall
XGBoost	0.9956	0.97
LightGBM	0.9940	0.96

XGBoost achieved the best overall performance, especially in identifying fraudulent transactions.

🐱 3. CatBoost Modeling

Notebook: fraud_trans_dete_catboost.ipynb

🔹 Why CatBoost?

Native categorical feature handling

Avoids target leakage via ordered boosting

Strong performance with minimal preprocessing

⚙️ Key Implementation Details

Raw categorical features passed directly using cat_features

scale_pos_weight used for class imbalance

Early stopping applied to prevent overfitting

📈 CatBoost Results

ROC-AUC: 0.9941

Performance comparable to LightGBM

Slightly below XGBoost due to dominance of numerical velocity features

🧪 Final Model Comparison
Model	Strength
XGBoost	Best overall detection & ranking
LightGBM	Fast training & scalable
CatBoost	Clean handling of categorical data
🎯 Key Takeaways

Velocity features are extremely powerful for fraud detection

Encoding strategy must align with the chosen model

High ROC-AUC confirms strong discrimination capability

The pipeline is memory-aware, scalable, and production-aligned

📌 Skills Demonstrated

✔ Large-scale data processing
✔ Feature engineering for fraud detection
✔ Handling imbalanced datasets
✔ Model comparison & evaluation
✔ Memory-efficient ML pipelines

🚀 Future Improvements

Precision-Recall curve analysis

Cost-sensitive threshold optimization

SHAP-based explainability

Time-based validation split

📎 Dataset Access

Due to size constraints, the dataset is hosted on Google Drive.
📥 [Download Link – Provided in Repository]
