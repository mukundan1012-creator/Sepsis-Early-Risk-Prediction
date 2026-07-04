"""
app.py

Streamlit frontend for the
Sepsis Early Risk Prediction System.

Author: Mukundan D
"""

import streamlit as st
import pandas as pd
from src.predict import predict_sepsis


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Sepsis Early Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Do NOT hide header — it contains the sidebar toggle button */

    [data-testid="stSidebar"] {
        background-color: #0d0f18 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #c8cce0 !important;
    }

    .result-card {
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 6px 0;
    }
    .card-label { font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .card-value { font-size: 28px; font-weight: 700; margin-top: 6px; }

    .risk-low    { background:#0d2b1a; border:1px solid #2d7a4f; color:#4dbb7a; border-radius:10px; padding:16px; text-align:center; font-size:20px; font-weight:700; }
    .risk-medium { background:#2b1e0a; border:1px solid #c47c1a; color:#f0a83a; border-radius:10px; padding:16px; text-align:center; font-size:20px; font-weight:700; }
    .risk-high   { background:#2b0d0d; border:1px solid #cc3333; color:#ff6b6b; border-radius:10px; padding:16px; text-align:center; font-size:20px; font-weight:700; }

    .footer { text-align:center; color:#555; font-size:13px; border-top:1px solid #222; padding-top:20px; margin-top:40px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Demo Values
# ============================================================

HEALTHY = {
    "age": 45, "iculos": 3,
    "HR":    [68.0,  70.0,  69.0],
    "O2Sat": [99.0,  98.0,  99.0],
    "Temp":  [36.6, 36.7, 36.6],
    "SBP":   [125.0, 122.0, 120.0],
    "MAP":   [88.0,  86.0,  85.0],
    "DBP":   [72.0,  70.0,  70.0],
    "Resp":  [14.0,  15.0,  14.0],
}

HIGH_RISK = {
    # Extreme septic shock pattern
    # Every vital trending hard in wrong direction across 3 hours
    "age": 85, "iculos": 72,
    "HR":    [138.0, 148.0, 158.0],
    "O2Sat": [80.0,  76.0,  72.0],
    "Temp":  [40.0,  40.4,  40.8],
    "SBP":   [68.0,  60.0,  52.0],
    "MAP":   [44.0,  38.0,  32.0],
    "DBP":   [28.0,  22.0,  18.0],
    "Resp":  [30.0,  33.0,  36.0],
}

VITALS_CONFIG = {
    "HR":    ("Heart Rate (bpm)",        40.0,  200.0,  75.0),
    "O2Sat": ("Oxygen Saturation (%)",   70.0,  100.0,  98.0),
    "Temp":  ("Temperature (°C)",        34.0,  42.0,   37.0),
    "SBP":   ("Systolic BP (mmHg)",      60.0,  200.0,  120.0),
    "MAP":   ("Mean Arterial Pressure",  40.0,  150.0,  85.0),
    "DBP":   ("Diastolic BP (mmHg)",     30.0,  130.0,  70.0),
    "Resp":  ("Respiratory Rate (/min)", 5.0,   50.0,   16.0),
}

VITAL_STEPS = {
    "HR": 1.0, "O2Sat": 1.0, "Temp": 0.1,
    "SBP": 1.0, "MAP": 1.0, "DBP": 1.0, "Resp": 1.0,
}


# ============================================================
# Session State Init
# ============================================================

if "demo" not in st.session_state:
    st.session_state["demo"] = None

for key, (_, mn, mx, default) in VITALS_CONFIG.items():
    for i, suffix in enumerate(["_h1", "_h2", "_h3"]):
        skey = key + suffix
        if skey not in st.session_state:
            st.session_state[skey] = default

if "age_val" not in st.session_state:
    st.session_state["age_val"] = 55
if "iculos_val" not in st.session_state:
    st.session_state["iculos_val"] = 5


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 🏥 Sepsis Risk Predictor")
    st.markdown("---")

    st.markdown("### 📋 About")
    st.markdown(
        "Predicts sepsis onset from **3 consecutive hours** of ICU vitals "
        "using a trained XGBoost model. Built on PhysioNet 2019 — "
        "**40,336 real ICU patients.**"
    )

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.markdown("""
- **Algorithm:** XGBoost  
- **Recall:** 89.65%  
- **ROC-AUC:** 0.790  
- **Features:** 53 engineered  
- **Explainability:** SHAP  
    """)

    st.markdown("---")
    st.markdown("### 📖 How To Use")
    st.markdown("""
1. Click a **demo button** or enter vitals manually  
2. Click **Predict Sepsis Risk**  
3. Read the risk level and probability  
    """)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("`XGBoost` · `SHAP` · `Pandas` · `Streamlit` · `scikit-learn`")

    st.markdown("---")
    st.warning(
        "⚠️ **Medical Disclaimer:** For educational and research "
        "purposes only. Not a substitute for clinical judgment."
    )

    st.markdown("---")
    st.markdown("**👨‍💻 Mukundan D**")
    st.markdown("Electronics and Communication Engineering")


# ============================================================
# Header
# ============================================================

st.markdown("# 🏥 Sepsis Early Risk Prediction System")
st.markdown(
    "Enter **three consecutive hourly ICU readings** to predict "
    "sepsis risk using a trained XGBoost model with SHAP explainability."
)
st.markdown("---")


# ============================================================
# Demo Buttons — update session state on click
# ============================================================

st.markdown("#### ⚡ Quick Demo")
st.caption("Instantly fill all inputs with pre-set patient vitals.")

col1, col2, _ = st.columns([1.3, 1.3, 3])

if col1.button("✅ Load Healthy Patient", use_container_width=True):
    st.session_state["demo"] = "healthy"
    st.session_state["age_val"]    = HEALTHY["age"]
    st.session_state["iculos_val"] = HEALTHY["iculos"]
    for key in VITALS_CONFIG:
        st.session_state[key + "_h1"] = HEALTHY[key][0]
        st.session_state[key + "_h2"] = HEALTHY[key][1]
        st.session_state[key + "_h3"] = HEALTHY[key][2]

if col2.button("🚨 Load High-Risk Patient", use_container_width=True):
    st.session_state["demo"] = "high_risk"
    st.session_state["age_val"]    = HIGH_RISK["age"]
    st.session_state["iculos_val"] = HIGH_RISK["iculos"]
    for key in VITALS_CONFIG:
        st.session_state[key + "_h1"] = HIGH_RISK[key][0]
        st.session_state[key + "_h2"] = HIGH_RISK[key][1]
        st.session_state[key + "_h3"] = HIGH_RISK[key][2]

if st.session_state["demo"] == "healthy":
    st.success("✅ Healthy patient vitals loaded — stable vitals, normal ranges.")
elif st.session_state["demo"] == "high_risk":
    st.error("🚨 High-risk patient vitals loaded — crashing BP, rising fever, low SpO2.")


# ============================================================
# Patient Info
# ============================================================

st.markdown("---")
st.markdown("#### 👤 Patient Information")

c1, c2, _ = st.columns([1, 1, 2])
age    = c1.number_input("Age (years)",              18,  100, key="age_val")
iculos = c2.number_input("ICU Length of Stay (hrs)", 1,   200, key="iculos_val")


# ============================================================
# Vital Signs Grid
# ============================================================

st.markdown("---")
st.markdown("#### 💓 Vital Signs — Three Consecutive Hours")

hc1, hc2, hc3, hc4 = st.columns([1.5, 1, 1, 1])
hc2.markdown("🕐 **2 hrs ago**")
hc3.markdown("🕑 **1 hr ago**")
hc4.markdown("🕒 **Now**")

vital_inputs = {}

for key, (label, mn, mx, default) in VITALS_CONFIG.items():
    step = VITAL_STEPS[key]
    lc, c1, c2, c3 = st.columns([1.5, 1, 1, 1])
    lc.markdown(f"**{label}**")

    v1 = c1.number_input("", mn, mx, step=step, label_visibility="collapsed", key=key + "_h1")
    v2 = c2.number_input("", mn, mx, step=step, label_visibility="collapsed", key=key + "_h2")
    v3 = c3.number_input("", mn, mx, step=step, label_visibility="collapsed", key=key + "_h3")

    vital_inputs[key] = [v1, v2, v3]


# ============================================================
# Predict Button
# ============================================================

st.markdown("---")
_, btn_col, _ = st.columns([1, 1, 1])

if btn_col.button("🔍 Predict Sepsis Risk", use_container_width=True, type="primary"):

    with st.spinner("Running prediction pipeline..."):

        patient_df = pd.DataFrame({
            "Patient_ID": [99999, 99999, 99999],
            "ICULOS":     [max(1, iculos - 2), max(1, iculos - 1), iculos],
            "Age":        [age, age, age],
            **{k: vital_inputs[k] for k in vital_inputs}
        })

        result = predict_sepsis(patient_df)

    st.markdown("---")
    st.markdown("#### 📊 Prediction Result")

    r1, r2, r3 = st.columns(3)

    status = "🚨 Sepsis Risk" if result["prediction"] == 1 else "✅ No Sepsis"
    r1.markdown(f"""
    <div class="result-card" style="background:#1a1d27;border:1px solid #2e3350;">
        <div class="card-label">Model Prediction</div>
        <div class="card-value" style="font-size:20px;">{status}</div>
    </div>""", unsafe_allow_html=True)

    r2.markdown(f"""
    <div class="result-card" style="background:#1a1d27;border:1px solid #2e3350;">
        <div class="card-label">Sepsis Probability</div>
        <div class="card-value">{result['probability'] * 100:.1f}%</div>
    </div>""", unsafe_allow_html=True)

    risk = result["risk_level"]
    icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[risk]
    r3.markdown(
        f'<div class="risk-{risk.lower()}">{icon} {risk} Risk</div>',
        unsafe_allow_html=True
    )

    if risk == "High":
        st.error("⚠️ High risk detected. Immediate clinical review recommended.")
    elif risk == "Medium":
        st.warning("⚠️ Medium risk. Continue monitoring and consider clinical assessment.")
    else:
        st.success("✅ Low risk. Vital signs appear stable. Continue standard monitoring.")


# ============================================================
# Explainers
# ============================================================

st.markdown("---")

with st.expander("📐 How Are Features Engineered?"):
    st.markdown("""
    | Feature Type | Example | What It Captures |
    |---|---|---|
    | **Lag** | HR_Lag1 | Previous hour's reading |
    | **Change** | HR_Change | Hourly delta |
    | **Rolling Mean** | HR_RollingMean_3 | 3-hour trend |
    | **Rolling Std** | HR_RollingStd_3 | Instability / variability |
    | **Rolling Max/Min** | Temp_RollingMax_3 | Worst-case seen |
    | **Pulse Pressure** | SBP − DBP | Cardiac output indicator |
    | **Shock Index** | HR / SBP | Classic septic shock marker |
    """)

with st.expander("🔍 What is SHAP?"):
    st.markdown("""
    **SHAP** explains *why* the model predicted sepsis for a patient.  
    Each feature gets a value showing how much it pushed the result  
    toward or away from sepsis.

    **Top model drivers:**  
    `Temp_RollingMax_3` · `MAP_RollingMin_3` · `ShockIndex` · `Resp_RollingMean_3`
    """)


# ============================================================
# Footer
# ============================================================

st.markdown("""
<div class="footer">
    <b>Mukundan D</b> · Electronics and Communication Engineering · Batch 2028<br>
    Sepsis Early Risk Prediction System · XGBoost · SHAP · Streamlit<br>
    <i>For educational and research purposes only. Not for clinical use.</i>
</div>
""", unsafe_allow_html=True)