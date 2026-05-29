# Loan Default Prediction

A binary classification system that predicts whether an Indian lending applicant will default on their loan before disbursement. Built to reduce credit risk exposure by flagging high-risk borrowers early.

---

## Problem Statement

Loan defaults cause significant financial losses for lenders. This project trains a machine learning model on 51,000 real-world-style Indian lending records to identify high-risk applicants at the time of application — before any money is disbursed.

**Target variable:** `loan_default` (0 = No Default, 1 = Default)  
**Class distribution:** 85% non-default / 15% default

---

## Results

| Metric | Value |
|---|---|
| Model | XGBoost (Tuned) |
| Recall (defaulters caught) | **92.2%** |
| ROC-AUC | **96.3%** |
| Precision | 59.6% |
| F1 Score | 72.4% |
| Missed defaulters (FN) | 117 / 1,500 (7.8%) |
| Estimated net benefit | ₹61 crore per 10,000 applicants |

---

## Project Structure

```
loan_default_prediction/
├── data/
│   ├── raw/
│   │   └── loan_default_risk_prediction_dataset(in).csv   # 51,000 records, 43 columns
│   └── processed/
│       ├── X_train.csv    # 40,000 × 50 (scaled + encoded)
│       ├── X_test.csv     # 10,000 × 50
│       ├── y_train.csv    # 40,000 labels
│       └── y_test.csv     # 10,000 labels
├── notebooks/
│   ├── data_understanding.ipynb   # EDA, class imbalance, data profiling
│   ├── preprocessing.ipynb        # Cleaning, encoding, scaling pipeline
│   └── model_training.ipynb       # Baseline models, tuning, evaluation
├── src/
│   ├── preprocessing.py           # Reusable preprocessing functions
│   ├── train_model.py             # Model training script
│   └── predict.py                 # Inference script
├── apps/
│   └── streamlit_app.py           # 4-page interactive Streamlit UI
├── models/
│   ├── final_pipeline.pkl         # Complete inference pipeline
│   ├── xgboost_tuned.pkl          # Final tuned model
│   ├── scaler.pkl                 # StandardScaler (fit on train)
│   ├── feature_columns.pkl        # Ordered list of 50 model features
│   └── numeric_columns.pkl        # List of 30 numeric columns
├── reports/
│   ├── data_quality_report.md     # Full data cleaning audit
│   └── model_evaluation_report.md # Model metrics, confusion matrix, business impact
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

| Attribute | Detail |
|---|---|
| Source | Indian lending dataset |
| Records | 51,000 |
| Raw Features | 43 |
| Model Features (post-processing) | 50 |
| Date range | 2021–2024 |
| Target | Binary (loan_default: 0/1) |

Key feature groups: applicant demographics, income metrics, loan details, credit history, repayment behavior, and engineered date features.

---

## Methodology

### 1. Data Cleaning
- Stripped ₹ and % symbols from 7 numeric columns stored as strings
- Imputed ~5% missing values (median for numeric, `'Unknown'` for employment type)
- Removed 1,000 duplicate rows
- Fixed 436 logically invalid entries (negative ages, credit scores > 900, etc.)
- Capped outliers using IQR method across 8 financial columns
- Standardized casing and aliases in 7 categorical columns
- Removed 4 data leakage columns (post-default event flags)

### 2. Feature Engineering
- Derived 5 time-based features from 4 date columns (`loan_age_days`, `days_since_last_pay`, etc.)
- One-hot encoded 7 categorical columns (50 final features)
- Scaled 30 numeric columns with `StandardScaler` (fit on train, applied to test)
- Stratified 80/20 train-test split preserving the 85:15 class ratio

### 3. Modeling
- Trained 3 baselines: Logistic Regression, Random Forest, XGBoost
- Addressed class imbalance via `scale_pos_weight=5.67` (XGBoost) and `class_weight='balanced'`
- Selected XGBoost based on recall + ROC-AUC superiority across 5-fold CV
- Tuned with GridSearchCV (27 combinations × 5 folds = 135 fits)
- Best params: `learning_rate=0.05`, `max_depth=4`, `n_estimators=300`

### 4. Risk Stratification
| Risk Tier | Probability | Default Rate |
|---|---|---|
| Low Risk | < 0.31 | 0.6% |
| Medium Risk | 0.31 – 0.66 | 16.5% |
| High Risk | > 0.66 | 70.1% |

---

## Streamlit App

A 4-page interactive web application for loan risk assessment.

**Pages:**
1. **Overview** — Project summary, dataset stats, class distribution
2. **Predict** — Input applicant details, get real-time risk score and recommendation
3. **Model Performance** — Confusion matrix, ROC curve, metric comparisons
4. **Feature Insights** — Feature importance charts, risk tier breakdowns

**Run the app:**
```bash
streamlit run apps/streamlit_app.py
```

Navigate to [http://localhost:8501](http://localhost:8501)

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd loan_default_prediction

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Key Findings

- **Past payment behavior is the strongest predictor** — `previous_defaults`, `missed_payments_last_12_months`, and `late_payment_count` together drive ~36% of model decisions.
- **Employment status matters** — `employment_type_Unemployed` is the 3rd most important feature, directly linking job stability to default likelihood.
- **Credit score is overrated** — Despite being the traditional benchmark, credit score ranks 9th; behavioral and capacity features dominate.
- **Financial leverage is critical** — `loan_to_income_ratio` and `debt_to_income_ratio` rank 5th and 8th respectively.
- **The model errs toward caution** — Deliberately biased toward high recall, it flags 11% false alarms to catch 92.2% of real defaulters — the correct trade-off for a risk-averse lender.

---

## Reports

- [Data Quality Report](reports/data_quality_report.md) — Full audit of cleaning steps, imputation values, and feature engineering decisions
- [Model Evaluation Report](reports/model_evaluation_report.md) — All model metrics, confusion matrix, CV results, feature importance, and business impact analysis
