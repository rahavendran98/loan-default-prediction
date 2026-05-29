"""
Loan Default Risk Prediction - Streamlit UI
=============================================

"""

import datetime
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
.stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    base = os.path.join(os.path.dirname(__file__), "..", "models")
    model       = joblib.load(os.path.join(base, "xgboost_tuned.pkl"))
    scaler      = joblib.load(os.path.join(base, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(base, "feature_columns.pkl"))
    num_cols    = joblib.load(os.path.join(base, "numeric_columns.pkl"))
    return model, scaler, feature_cols, num_cols


try:
    model, scaler, FEATURE_COLUMNS, NUMERIC_COLUMNS = load_artifacts()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR   = str(e)


# ============================================================
# DOMAIN HELPERS
# ============================================================
def categorize_risk(probability):
    if probability < 0.31:
        return "Low Risk", "🟢", "low"
    if probability < 0.66:
        return "Medium Risk", "🟡", "medium"
    return "High Risk", "🔴", "high"


def generate_recommendations(probability, credit_score, debt_to_income,
                              previous_defaults, missed_payments):
    _, _, level = categorize_risk(probability)
    recs = []
    if level == "low":
        recs.append(("✅", "Loan can be approved with standard terms"))
    elif level == "medium":
        recs.append(("⚠️", "Approve with additional verification or reduced amount"))
    else:
        recs.append(("🛑", "Manual review required — high default risk"))

    if credit_score < 600:
        recs.append(("📉", "Low credit score — request guarantor or collateral"))
    if debt_to_income > 40:
        recs.append(("💸", "High debt-to-income — consider reducing loan amount"))
    if previous_defaults > 0:
        recs.append(("🚨", "Previous defaults detected — strict verification required"))
    if missed_payments > 2:
        recs.append(("📌", "Multiple missed payments — high-risk review needed"))
    return recs


def build_feature_row(user_input, feature_columns):
    row = {col: 0 for col in feature_columns}

    numeric_map = {
        "age":                          user_input["age"],
        "years_of_employment":          user_input["years_of_employment"],
        "monthly_income":               user_input["monthly_income"],
        "income_stability_score":       user_input["income_stability_score"],
        "additional_income":            user_input["additional_income"],
        "loan_amount":                  user_input["loan_amount"],
        "loan_term_months":             user_input["loan_term_months"],
        "interest_rate":                user_input["interest_rate"],
        "emi_amount":                   user_input["emi_amount"],
        "loan_to_income_ratio":         user_input["loan_to_income_ratio"],
        "credit_score":                 user_input["credit_score"],
        "existing_loans":               user_input["existing_loans"],
        "credit_card_usage":            user_input["credit_card_usage"],
        "debt_to_income_ratio":         user_input["debt_to_income_ratio"],
        "previous_defaults":            user_input["previous_defaults"],
        "missed_payments_last_12_months": user_input["missed_payments"],
        "on_time_payment_ratio":        user_input["on_time_payment_ratio"],
        "late_payment_count":           user_input["late_payment_count"],
        "average_delay_days":           user_input["average_delay_days"],
        "repayment_history_score":      user_input["repayment_history_score"],
        "bank_balance":                 user_input["bank_balance"],
        "savings_amount":               user_input["savings_amount"],
        "monthly_expense":              user_input["monthly_expense"],
        "savings_ratio":                user_input["savings_ratio"],
        "transaction_frequency":        user_input["transaction_frequency"],
        "loan_age_days":                user_input["loan_age_days"],
        "days_since_last_pay":          user_input["days_since_last_pay"],
        "days_since_next_due":          user_input["days_since_next_due"],
        "application_month":            user_input["application_month"],
        "application_dayofweek":        user_input["application_dayofweek"],
    }
    for k, v in numeric_map.items():
        if k in row:
            row[k] = v

    for dummy in [
        f"gender_{user_input['gender']}",
        f"marital_status_{user_input['marital_status']}",
        f"education_level_{user_input['education_level']}",
        f"residential_status_{user_input['residential_status']}",
        f"employment_type_{user_input['employment_type']}",
        f"location_type_{user_input['location_type']}",
        f"loan_purpose_{user_input['loan_purpose']}",
    ]:
        if dummy in row:
            row[dummy] = 1

    return pd.DataFrame([row])[feature_columns]


def run_predict(user_input):
    X = build_feature_row(user_input, FEATURE_COLUMNS)
    X_scaled = X.copy()
    cols = [c for c in NUMERIC_COLUMNS if c in X_scaled.columns]
    X_scaled[cols] = scaler.transform(X_scaled[cols])
    prob = model.predict_proba(X_scaled)[0, 1]
    return float(prob)


# ============================================================
# CHART HELPERS
# ============================================================
def gauge_chart(probability):
    pct   = probability * 100
    color = "#28a745" if probability < 0.31 else "#ffc107" if probability < 0.66 else "#dc3545"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        delta={"reference": 50, "valueformat": ".1f"},
        title={"text": "Default Probability", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
            "bar":  {"color": color, "thickness": 0.28},
            "steps": [
                {"range": [0,  31], "color": "#d4edda"},
                {"range": [31, 66], "color": "#fff3cd"},
                {"range": [66, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": pct,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30))
    return fig


def importance_chart(mdl, feature_columns, top_n=15):
    imp = mdl.feature_importances_
    idx = np.argsort(imp)[-top_n:]
    labels = [feature_columns[i].replace("_", " ").title() for i in idx]
    values = imp[idx]
    max_v  = values.max()
    colors = [f"rgba(220,53,69,{0.35 + 0.65 * v / max_v:.2f})" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        height=420,
        margin=dict(t=40, b=20, l=10, r=70),
        xaxis_title="Importance Score",
        yaxis={"categoryorder": "array", "categoryarray": labels},
    )
    return fig


# ============================================================
# SIDEBAR — ALL INPUTS
# ============================================================
with st.sidebar:
    st.title("🏦 Loan Risk Predictor")
    st.caption("Fill in applicant details, then click Predict.")
    st.markdown("---")

    with st.expander("👤 Personal Info", expanded=True):
        age              = st.number_input("Age", 18, 75, 35)
        gender           = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status   = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
        education_level  = st.selectbox("Education", ["Graduate", "Post Graduate", "School", "Diploma"])
        residential_status = st.selectbox("Residence Type", ["Owned", "Rented", "Family"])
        location_type    = st.selectbox("Location", ["Urban", "Semi-Urban", "Rural"])

    with st.expander("💼 Employment & Income", expanded=True):
        employment_type      = st.selectbox("Employment Type",
                                            ["Salaried", "Self-Employed", "Business", "Unemployed", "Unknown"])
        years_of_employment  = st.number_input("Years of Employment", 0, 50, 5)
        monthly_income       = st.number_input("Monthly Income (₹)", 0, 10_000_000, 50_000, step=1_000)
        additional_income    = st.number_input("Additional Income (₹)", 0, 10_000_000, 5_000, step=1_000)
        income_stability_score = st.slider("Income Stability Score", 0.0, 10.0, 7.0)
        bank_balance         = st.number_input("Bank Balance (₹)", 0, 100_000_000, 80_000, step=1_000)
        savings_amount       = st.number_input("Savings (₹)", 0, 100_000_000, 50_000, step=1_000)
        monthly_expense      = st.number_input("Monthly Expense (₹)", 0, 10_000_000, 30_000, step=1_000)
        transaction_frequency = st.number_input("Transactions / Month", 0, 500, 25)

    with st.expander("💰 Loan Details", expanded=True):
        loan_amount      = st.number_input("Loan Amount (₹)", 0, 100_000_000, 500_000, step=10_000)
        loan_term_months = st.number_input("Loan Term (months)", 1, 360, 60)
        interest_rate    = st.slider("Interest Rate (%)", 5.0, 30.0, 14.0, step=0.1)
        emi_amount       = st.number_input("EMI Amount (₹)", 0, 1_000_000, 12_000, step=500)
        loan_purpose     = st.selectbox("Loan Purpose",
                                        ["Personal", "Vehicle", "Home", "Business", "Medical", "Education"])

    with st.expander("📊 Credit Profile", expanded=True):
        credit_score        = st.slider("Credit Score", 300, 900, 700)
        existing_loans      = st.number_input("Existing Loans", 0, 20, 1)
        credit_card_usage   = st.slider("Credit Card Usage (%)", 0.0, 100.0, 30.0)
        debt_to_income_ratio = st.slider("Debt-to-Income Ratio (%)", 0.0, 100.0, 35.0)

    with st.expander("🔄 Repayment History", expanded=True):
        previous_defaults      = st.number_input("Previous Defaults", 0, 20, 0)
        missed_payments        = st.number_input("Missed Payments (last 12m)", 0, 50, 0)
        on_time_payment_ratio  = st.slider("On-Time Payment Ratio", 0.0, 1.0, 0.9, step=0.01)
        late_payment_count     = st.number_input("Late Payment Count", 0, 100, 2)
        average_delay_days     = st.number_input("Avg Delay (days)", 0, 365, 5)
        repayment_history_score = st.slider("Repayment History Score", 0.0, 10.0, 7.0)

    with st.expander("📅 Application Date & Timing", expanded=False):
        app_date              = st.date_input("Application Date", datetime.date.today())
        loan_age_days         = st.number_input("Loan Age (days)", 0, 3_650, 30)
        days_since_last_pay   = st.number_input("Days Since Last Payment", 0, 365, 30)
        days_since_next_due   = st.number_input("Days Until Next Due", 0, 365, 0)

    st.markdown("---")
    predict_btn = st.button("🎯 Predict Default Risk", type="primary", use_container_width=True)


# ============================================================
# MAIN AREA
# ============================================================
st.title("🏦 Loan Default Risk Predictor")
st.markdown("Machine-learning powered applicant risk assessment using a tuned **XGBoost** model.")

if not MODEL_LOADED:
    st.error(f"Could not load model artifacts: {LOAD_ERROR}")
    st.stop()

# Derived / auto-computed features
savings_ratio           = (savings_amount / monthly_income * 100) if monthly_income > 0 else 0.0
loan_to_income_ratio_v  = (loan_amount / (monthly_income * 12))   if monthly_income > 0 else 0.0
application_month       = app_date.month
application_dayofweek   = app_date.weekday()

with st.expander("📐 Auto-Computed Derived Features", expanded=False):
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Savings Ratio",       f"{savings_ratio:.1f}%")
    d2.metric("Loan-to-Income",      f"{loan_to_income_ratio_v:.2f}×")
    d3.metric("Application Month",   app_date.strftime("%B"))
    d4.metric("Day of Week",         app_date.strftime("%A"))

st.markdown("---")

# ============================================================
# PREDICTION RESULTS
# ============================================================
if predict_btn:
    user_input = {
        "age":                   age,
        "gender":                gender,
        "marital_status":        marital_status,
        "education_level":       education_level,
        "residential_status":    residential_status,
        "location_type":         location_type,
        "employment_type":       employment_type,
        "years_of_employment":   years_of_employment,
        "monthly_income":        monthly_income,
        "additional_income":     additional_income,
        "income_stability_score": income_stability_score,
        "loan_amount":           loan_amount,
        "loan_term_months":      loan_term_months,
        "interest_rate":         interest_rate,
        "emi_amount":            emi_amount,
        "loan_to_income_ratio":  loan_to_income_ratio_v,
        "loan_purpose":          loan_purpose,
        "credit_score":          credit_score,
        "existing_loans":        existing_loans,
        "credit_card_usage":     credit_card_usage,
        "debt_to_income_ratio":  debt_to_income_ratio,
        "previous_defaults":     previous_defaults,
        "missed_payments":       missed_payments,
        "on_time_payment_ratio": on_time_payment_ratio,
        "late_payment_count":    late_payment_count,
        "average_delay_days":    average_delay_days,
        "repayment_history_score": repayment_history_score,
        "bank_balance":          bank_balance,
        "savings_amount":        savings_amount,
        "monthly_expense":       monthly_expense,
        "savings_ratio":         savings_ratio,
        "transaction_frequency": transaction_frequency,
        "loan_age_days":         loan_age_days,
        "days_since_last_pay":   days_since_last_pay,
        "days_since_next_due":   days_since_next_due,
        "application_month":     application_month,
        "application_dayofweek": application_dayofweek,
    }

    try:
        probability = run_predict(user_input)
        risk_label, risk_emoji, risk_level = categorize_risk(probability)

        # ---- KPI row ----
        st.subheader("📊 Prediction Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Default Probability", f"{probability*100:.1f}%")
        k2.metric("Risk Category",       f"{risk_emoji} {risk_label}")
        k3.metric("Credit Score",        str(credit_score))
        k4.metric("Loan-to-Income",      f"{loan_to_income_ratio_v:.2f}×")

        st.markdown("---")

        # ---- Gauge + Recommendations ----
        gc, rc = st.columns([1, 1])
        with gc:
            st.plotly_chart(gauge_chart(probability), use_container_width=True)

        with rc:
            st.subheader("💡 Recommendations")
            recs = generate_recommendations(probability, credit_score, debt_to_income_ratio,
                                            previous_defaults, missed_payments)
            for emoji, text in recs:
                st.write(f"{emoji} {text}")

            st.subheader("⚠️ Risk Factors")
            risk_factors = []
            if credit_score < 600:
                risk_factors.append(f"Low credit score ({credit_score})")
            if debt_to_income_ratio > 40:
                risk_factors.append(f"High debt-to-income ({debt_to_income_ratio:.0f}%)")
            if previous_defaults > 0:
                risk_factors.append(f"Previous defaults ({previous_defaults})")
            if missed_payments > 2:
                risk_factors.append(f"Missed payments ({missed_payments})")
            if loan_to_income_ratio_v > 3:
                risk_factors.append(f"High loan-to-income ({loan_to_income_ratio_v:.1f}×)")

            if risk_factors:
                for rf in risk_factors:
                    st.warning(rf, icon="🚩")
            else:
                st.success("No major risk factors detected", icon="✅")

        st.markdown("---")

        # ---- Feature importance ----
        st.subheader("🔑 Model Feature Importances")
        st.plotly_chart(importance_chart(model, FEATURE_COLUMNS), use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("This usually means a feature column name mismatch. "
                "Check that feature_columns.pkl matches the categorical options in the sidebar.")

else:
    # Landing state
    st.info("👈 Fill in the applicant details in the sidebar and click **Predict Default Risk** to begin.")
    col_a, col_b, col_c = st.columns(3)
    col_a.info("**Model**\nTuned XGBoost Classifier")
    col_b.info("**Risk Bands**\nLow < 31% · Medium 31–66% · High > 66%")
    col_c.info("**Features**\n30+ applicant & financial attributes")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Loan Default Risk Predictor · Model: Tuned XGBoost · For educational purposes only")



##Run with: streamlit run apps/streamlit_app.py