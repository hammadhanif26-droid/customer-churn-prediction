# Customer Churn Prediction — SaaS Clients

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-1E90FF)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

An end-to-end machine learning project that builds a **predictive model to identify when a client is likely to churn**, so the business can step in early with customer-success outreach, discounts or renewal campaigns.

---

## 📌 Problem Statement

Losing clients is expensive. The goal of this project is to use client usage, engagement, contract and pricing data to predict which corporate clients of a SaaS business are likely to **churn**, and to evaluate whether that prediction is reliable enough to drive retention decisions.

## 📂 Repository Structure

```
customer-churn-prediction/
├── data/
│   └── data_file.csv                         # 10,000 SaaS clients, 18 features + churn label
├── notebooks/
│   └── customer_churn_prediction.ipynb       # Full analysis: EDA → features → models → evaluation
├── requirements.txt
└── README.md
```

## 📊 Dataset

10,000 client records, 18 features, no missing values or duplicates. Target: `churned` (Yes / No), with a churn rate of about **20.4%**.

| Group | Features |
|---|---|
| **Usage & engagement** | `monthly_usage_hours`, `num_logins`, `days_since_last_login`, `email_open_rate`, `webinar_attendance_last_3mo` |
| **Account** | `company_size`, `account_age_months`, `num_products_used`, `num_admin_users`, `has_custom_integration` |
| **Support** | `num_support_tickets` |
| **Commercial** | `plan_type`, `base_price_usd`, `discount_rate`, `is_annual_contract` |
| **Firmographics** | `region`, `industry` |

## 🔬 Approach

1. **Data preparation:** dropped the index column and checked for missing values, duplicates and out-of-range values.
2. **Exploratory data analysis:** looked at class balance, churner vs. non-churner distributions, churn rate by category, correlations and multicollinearity.
3. **Feature engineering:** added `tickets_per_login`, `usage_per_login`, `admin_to_size_ratio`, `products_per_admin`, `effective_price` and `is_dormant_admin`, then tested with cross-validation whether they helped.
4. **Preprocessing pipeline:** `StandardScaler` and `OneHotEncoder` inside a `ColumnTransformer` and `Pipeline`, so the test set can't leak into training.
5. **Train/test split:** a stratified 80/20 split, so both sets keep the same ~20% churn rate.
6. **Model comparison:** Dummy baseline, Logistic Regression, Random Forest, Gradient Boosting and XGBoost, compared with 5-fold stratified cross-validation and class weighting for the imbalance.
7. **Hyperparameter tuning:** `GridSearchCV` on Logistic Regression and Random Forest.
8. **Final evaluation:** the held-out test set is used **once**, with a classification report, confusion matrix, ROC and Precision-Recall curves.
9. **Sanity check:** a synthetic, deliberately predictive feature is fed through the same pipeline to prove the pipeline can detect signal when it exists.
10. **Recommendation:** the business trade-offs and the final model choice.

## 📈 Results

**Cross-validation (training set, 5-fold stratified):**

| Model | ROC-AUC | F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|---|
| Dummy (baseline) | 0.506 | 0.218 | 0.213 | 0.223 | 0.207 |
| **Logistic Regression** | **0.504** | **0.290** | 0.208 | **0.478** | 0.210 |
| Random Forest | 0.502 | 0.018 | 0.212 | 0.009 | 0.210 |
| Gradient Boosting | 0.503 | 0.006 | 0.185 | 0.003 | 0.209 |
| XGBoost | 0.484 | 0.171 | 0.205 | 0.147 | 0.205 |

**Held-out test set (tuned Logistic Regression):** ROC-AUC **0.494** · churn recall 0.48 · churn precision 0.21

**Sanity check:** with an injected predictive feature, the same pipeline scores ROC-AUC **0.983**, which confirms the code works.

## 💡 Key Findings

- **The features don't carry predictive signal for churn.** Correlations with churn are all near zero (|r| < 0.02), the distributions for churners and non-churners overlap, and churn rates are flat (~19–21%) across every region, industry and plan.
- **More complex models don't fix weak data.** Random Forest, Gradient Boosting and XGBoost did no better than a random baseline.
- **The pipeline isn't broken.** The sanity check (AUC 0.98 on a synthetic signal) shows the weak results describe the dataset, not a bug in the code.

## ✅ Recommendation

**Tuned Logistic Regression** is the preferred model because it is fast, easy to explain to stakeholders and performed at least as well as the ensemble models. With a test ROC-AUC of about 0.50, however, it should **not be deployed** to drive retention decisions, because it would give false confidence. Next steps:

- Collect richer behavioural signals, such as feature-level usage trends, NPS/CSAT scores, support ticket sentiment and billing events.
- Use time-based features, such as month-over-month changes in usage and logins, instead of point-in-time snapshots.
- Re-run this same validated pipeline once the new data is available.

## 🚀 How to Run

```bash
git clone https://github.com/hammadhanif26-droid/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook notebooks/customer_churn_prediction.ipynb
```

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · Jupyter
