# 🚨 Fraud Transaction Detection

## 📌 Project Overview

This project focuses on detecting fraudulent financial transactions using machine learning models on a **large-scale synthetic dataset**. The objective is to design an end-to-end fraud detection pipeline that includes **data preprocessing, feature engineering, class imbalance handling, and model comparison** using industry-standard algorithms.

The project demonstrates practical skills required in real-world fraud detection systems, including memory-efficient processing and model evaluation on highly imbalanced data.

---

## 📂 Repository Structure

├── fraud_trans_dete_preprocessing.ipynb  
├── fraud_trans_dete_xg_light.ipynb  
├── fraud_trans_dete_catboost.ipynb  
├── README.md  


📁 **Dataset**  
Due to the large size, the dataset is hosted on **Google Drive**.  
A download link is provided in this repository.

---

## 🧠 Dataset Description

- Synthetic financial transaction dataset
- 7.4 million transaction records
- Highly imbalanced target variable (`is_fraud`)
- Includes:
  - Transaction metadata
  - Merchant information
  - Device and channel details
  - Velocity-based behavioral features

---

## 🔧 1. Data Preprocessing & Feature Engineering  
**Notebook:** `fraud_trans_dete_preprocessing.ipynb`

### 🔹 Data Cleaning & Validation

- Removed duplicate transactions using `transaction_id`
- Checked for missing values and invalid amounts
- Identified logical inconsistencies (e.g., `card_present = yes` but channel ≠ POS)
- Verified distributions of all categorical variables

---

### 🔹 Feature Reduction

Removed identifier and leakage-prone columns such as:
- `transaction_id`
- `customer_id`
- `card_number`
- `ip_address`
- `timestamp`
- `device_fingerprint`

This ensures models learn **behavioral patterns**, not unique identities.

---

### 🚀 Velocity Feature Engineering

The `velocity_last_hour` column contained dictionary-like behavioral summaries.  
It was decomposed into the following numerical features:

- `num_transactions`
- `total_amount`
- `unique_merchants`
- `unique_countries`
- `max_single_amount`

These features capture short-term transactional behavior and significantly improve fraud detection performance.

---

### 🔄 Encoding Strategy

To support different modeling approaches, **two separate datasets** were created:

#### 🔹 XGBoost & LightGBM Dataset
- **Target Encoding**
  - Applied to high-cardinality features:
    - `merchant`
    - `merchant_type`
- **Label Encoding**
  - Applied to remaining categorical variables
- Train-test split with stratification
- Saved as:
  - `fraud_data_lgbm_xgb_train.csv`
  - `fraud_data_lgbm_xgb_test.csv`

#### 🔹 CatBoost Dataset
- No manual encoding applied
- Raw categorical features preserved
- Saved as:
  - `fraud_data_catboost.csv`

This design ensures optimal performance and memory efficiency for each model.

---

## 🤖 2. XGBoost & LightGBM Modeling  
**Notebook:** `fraud_trans_dete_xg_light.ipynb`

### 🔹 Models Used
- **XGBoost**
- **LightGBM**

Both models are widely used in production fraud detection systems due to their performance on tabular data.

---

### ⚖️ Handling Class Imbalance

Fraud detection datasets are extremely imbalanced.  
This was addressed using:

- **XGBoost:** `scale_pos_weight`
- **LightGBM:** `class_weight = 'balanced'`

---

### 📊 Evaluation Metrics

Each model is evaluated using:
- Confusion Matrix
- Precision, Recall, F1-score
- ROC-AUC score
- ROC Curve visualization

ROC-AUC is emphasized due to its robustness on imbalanced datasets.

---

### 🏆 Results

#### XGBoost
- ROC-AUC: **0.9956**
- Fraud Recall: **0.97**

#### LightGBM
- ROC-AUC: **0.9940**
- Fraud Recall: **0.96**

XGBoost achieved the best overall performance, particularly in identifying fraudulent transactions.

---

## 🐱 3. CatBoost Modeling  
**Notebook:** `fraud_trans_dete_catboost.ipynb`

### 🔹 Why CatBoost?
- Native categorical feature handling
- Prevents target leakage using ordered boosting
- Minimal preprocessing required

---

### ⚙️ Implementation Details

- Raw categorical features passed directly via `cat_features`
- Class imbalance handled using `scale_pos_weight`
- Early stopping applied to prevent overfitting

---

### 📈 CatBoost Results

- ROC-AUC: **0.9941**
- Performance comparable to LightGBM
- Slightly below XGBoost due to dominance of engineered numerical features

---

## 🧪 Final Model Comparison

| Model     | ROC-AUC |
|-----------|--------|
| XGBoost   | **0.9956** |
| LightGBM  | 0.9940 |
| CatBoost  | 0.9941 |

---

## 🎯 Key Takeaways

- Velocity-based behavioral features are critical for fraud detection
- Encoding strategy must align with the chosen model
- Proper class imbalance handling significantly improves recall
- XGBoost delivered the strongest overall performance
- The pipeline is scalable and production-oriented

---

## 📌 Skills Demonstrated

- Large-scale data preprocessing
- Feature engineering for fraud detection
- Handling highly imbalanced datasets
- Model comparison and evaluation
- Memory-efficient machine learning workflows

---

## 🚀 Future Improvements

- Precision-Recall curve analysis
- Cost-sensitive threshold tuning
- Model explainability using SHAP
- Time-based cross-validation

---

## 📎 Dataset Access

Due to size constraints, the dataset is hosted externally.  
📥 **Download Dataset:**  
https://drive.google.com/file/d/1oV0gDsgLzXwKIxv156sear_PWMPhOnDi/view?usp=drive_link

---

⭐ If you found this project useful, feel free to star the repository!
