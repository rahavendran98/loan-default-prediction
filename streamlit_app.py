"""
Loan Default Prediction — Streamlit Application
Dataset: 51,000 Indian lending records | Model: XGBoost (Tuned)
Run: streamlit run streamlit_app.py
"""

import os
import warnings
import joblib
from datetime import date, datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #0f3460;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e94560; }
    .metric-label { font-size: 0.85rem; color: #a0aec0; margin-top: 4px; }
    .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }
    .low-risk    { background: #1e4d2b; color: #48bb78; }
    .medium-risk { background: #4d3800; color: #f6c543; }
    .high-risk   { background: #4d1a1a; color: #fc8181; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #63b3ed;
        border-bottom: 2px solid #2d3748;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Artifact loading ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    try:
        return {
            "model":           joblib.load(os.path.join(MODEL_DIR, "xgboost_tuned.pkl")),
            "scaler":          joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
            "feature_columns": joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl")),
            "numeric_columns": joblib.load(os.path.join(MODEL_DIR, "numeric_columns.pkl")),
            "loaded": True,
        }
    except Exception as exc:
        return {"loaded": False, "error": str(exc)}


artifacts = load_artifacts()

# ── Hard-coded reference data from notebooks ──────────────────────────────────
MODEL_RESULTS = {
    "Logistic Regression": {"accuracy": 0.8575, "precision": 0.5149, "recall": 0.8653, "f1": 0.6456, "roc_auc": 0.9366},
    "Random Forest":       {"accuracy": 0.8830, "precision": 0.8526, "recall": 0.2660, "f1": 0.4055, "roc_auc": 0.9459},
    "XGBoost (Baseline)":  {"accuracy": 0.9014, "precision": 0.6175, "recall": 0.9007, "f1": 0.7326, "roc_auc": 0.9612},
    "XGBoost (Tuned)":     {"accuracy": 0.8944, "precision": 0.5956, "recall": 0.9220, "f1": 0.7237, "roc_auc": 0.9629},
}

FEATURE_IMPORTANCE = {
    "previous_defaults":               0.1350,
    "missed_payments_last_12_months":  0.1260,
    "employment_type_Unemployed":      0.1258,
    "late_payment_count":              0.1030,
    "loan_to_income_ratio":            0.0948,
    "monthly_income":                  0.0767,
    "repayment_history_score":         0.0517,
    "debt_to_income_ratio":            0.0476,
    "credit_score":                    0.0332,
    "on_time_payment_ratio":           0.0313,
    "average_delay_days":              0.0180,
    "employment_type_Unknown":         0.0107,
    "loan_purpose_Medical":            0.0077,
    "savings_ratio":                   0.0074,
    "residential_status_Rented":       0.0065,
}

CM = {"tn": 7561, "fp": 939, "fn": 117, "tp": 1383}

REF_DATE = pd.Timestamp("2026-05-27")

# ── Helpers ────────────────────────────────────────────────────────────────────

def categorize_risk(prob: float):
    if prob < 0.31:
        return "Low Risk", "#48bb78", "low-risk"
    elif prob < 0.66:
        return "Medium Risk", "#f6c543", "medium-risk"
    return "High Risk", "#fc8181", "high-risk"


def generate_recommendations(prob, credit_score, debt_to_income, previous_defaults, missed_payments):
    label, _, _ = categorize_risk(prob)
    recs = []
    if label == "Low Risk":
        recs.append("✅ Loan can be approved — borrower profile looks healthy.")
    elif label == "Medium Risk":
        recs.append("⚠️ Consider approval with additional verification or a reduced loan amount.")
    else:
        recs.append("🚨 Manual review required — elevated default probability detected.")
    if credit_score < 600:
        recs.append("📉 Credit score is below 600; request a guarantor or collateral.")
    if debt_to_income > 40:
        recs.append("💳 Debt-to-income ratio exceeds 40% — consider lowering the approved amount.")
    if previous_defaults > 0:
        recs.append(f"⛔ {int(previous_defaults)} prior default(s) on record — strict verification required.")
    if missed_payments > 2:
        recs.append(f"📅 {int(missed_payments)} missed payments in the last 12 months — high-risk signal.")
    return recs


def preprocess_input(data: dict) -> pd.DataFrame:
    """Replicates the notebook preprocessing pipeline for a single applicant."""
    df = pd.DataFrame([data])

    app_dt   = pd.to_datetime(data["application_date"])
    start_dt = pd.to_datetime(data["loan_start_date"])
    pay_dt   = pd.to_datetime(data["last_payment_date"])
    due_dt   = pd.to_datetime(data["next_due_date"])

    df["loan_age_days"]         = max((REF_DATE - start_dt).days, 0)
    df["days_since_last_pay"]   = max((REF_DATE - pay_dt).days, 0)
    df["days_since_next_due"]   = max((REF_DATE - due_dt).days, 0)
    df["application_month"]     = app_dt.month
    df["application_dayofweek"] = app_dt.dayofweek

    drop_cols = ["application_date", "loan_start_date", "last_payment_date", "next_due_date"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    cat_cols = ["gender", "marital_status", "education_level",
                "residential_status", "employment_type", "location_type", "loan_purpose"]
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype="int")

    feat_cols = artifacts["feature_columns"]
    df_enc = df_enc.reindex(columns=feat_cols, fill_value=0)

    num_cols = artifacts["numeric_columns"]
    valid_num = [c for c in num_cols if c in df_enc.columns]
    df_enc[valid_num] = artifacts["scaler"].transform(df_enc[valid_num])

    return df_enc

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🏦 Loan Default Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    ["🏠  Overview", "🔍  Predict Default", "📊  Model Performance", "📈  Feature Insights"],
)

st.sidebar.markdown("---")
if artifacts["loaded"]:
    st.sidebar.success("Model loaded successfully")
else:
    st.sidebar.error(f"Model load error: {artifacts.get('error', 'unknown')}")

st.sidebar.markdown(
    """
    **Dataset:** 51,000 loan records
    **Final Model:** XGBoost (Tuned)
    **Recall:** 92.2% | **ROC-AUC:** 96.3%
    """
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.title("Loan Default Prediction Dashboard")
    st.markdown(
        "A machine-learning system that flags high-risk loan applicants before disbursement, "
        "built on 51,000 real-world Indian lending records with 43 features."
    )
    st.markdown("---")

    # ── Key metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "51,000",  "Total Records"),
        (c2, "43",      "Raw Features"),
        (c3, "15%",     "Default Rate"),
        (c4, "92.2%",   "Recall (Tuned XGB)"),
        (c5, "96.3%",   "ROC-AUC Score"),
    ]
    for col, val, lbl in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{val}</div>'
                f'<div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Class distribution + model snapshot
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<p class="section-header">Class Distribution</p>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["No Default (85%)", "Default (15%)"],
            values=[43342, 7658],
            hole=0.55,
            marker_colors=["#2d6a4f", "#e63946"],
            textfont_size=13,
        ))
        fig_pie.update_layout(
            height=300,
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)
        df_results = pd.DataFrame(MODEL_RESULTS).T.reset_index()
        df_results.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        df_results = df_results.round(4)
        st.dataframe(
            df_results.style
                .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                               color="#1a3a4a")
                .format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # ── Key insights
    st.subheader("Key Findings")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(
            "**Payment behaviour is the strongest signal.**  \n"
            "Previous defaults, missed payments, and late payment count together explain "
            "over 36% of the model's decisions."
        )
    with col_b:
        st.warning(
            "**Unemployment carries significant risk.**  \n"
            "Being unemployed is the third most important feature (12.6%), directly linking "
            "job stability to the probability of default."
        )
    with col_c:
        st.success(
            "**The model catches 92% of defaulters.**  \n"
            "Out of 1,500 actual defaulters in the test set, only 117 were missed — "
            "protecting the lender from the majority of bad loans."
        )

    st.markdown("---")
    st.subheader("Financial Impact (Test Set Estimate)")
    ci1, ci2, ci3, ci4 = st.columns(4)
    ci1.metric("Defaulters Caught", "1,383", "92.2% recall")
    ci2.metric("Defaulters Missed",   "117",  "-7.8%")
    ci3.metric("False Alarms",        "939",  "11% of non-defaulters")
    ci4.metric("Estimated Net Benefit", "₹60.95 Cr", "per 10,000 applicants")

#
        )
