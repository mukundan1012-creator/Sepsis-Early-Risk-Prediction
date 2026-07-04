"""
predict.py

Loads the trained XGBoost model and predicts
the risk of Sepsis for a patient.
"""

import joblib
import pandas as pd

from src.feature_engineering import engineer_features

# ============================================================
# Load Trained Model
# ============================================================

MODEL_PATH = "models/final_sepsis_model.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# Prediction Function
# ============================================================

def predict_sepsis(patient_df):
    """
    Predicts Sepsis risk for a patient.

    Parameters
    ----------
    patient_df : pandas.DataFrame
        DataFrame containing three consecutive
        hourly readings of one patient.

    Returns
    -------
    dict
        {
            "prediction": 0 or 1,
            "probability": float,
            "risk_level": str
        }
    """

    # ========================================================
    # Feature Engineering
    # ========================================================

    patient_df = engineer_features(patient_df)

    # ========================================================
    # Keep Only Latest Hour
    # ========================================================

    latest_row = patient_df.iloc[[-1]]

    # ========================================================
    # Remove Patient_ID
    # ========================================================

    latest_row = latest_row.drop(columns=["Patient_ID"])

    latest_row = latest_row[
    [
        "HR","O2Sat","Temp","SBP","MAP","DBP","Resp",
        "Age","ICULOS",
        "HR_RollingMean_3","O2Sat_RollingMean_3","Temp_RollingMean_3",
        "SBP_RollingMean_3","MAP_RollingMean_3","DBP_RollingMean_3","Resp_RollingMean_3",
        "HR_RollingStd_3","O2Sat_RollingStd_3","Temp_RollingStd_3",
        "SBP_RollingStd_3","MAP_RollingStd_3","DBP_RollingStd_3","Resp_RollingStd_3",
        "HR_Change","O2Sat_Change","Temp_Change",
        "SBP_Change","MAP_Change","DBP_Change","Resp_Change",
        "HR_Lag1","O2Sat_Lag1","Temp_Lag1",
        "SBP_Lag1","MAP_Lag1","DBP_Lag1","Resp_Lag1",
        "HR_RollingMax_3","O2Sat_RollingMax_3","Temp_RollingMax_3",
        "SBP_RollingMax_3","MAP_RollingMax_3","DBP_RollingMax_3","Resp_RollingMax_3",
        "HR_RollingMin_3","O2Sat_RollingMin_3","Temp_RollingMin_3",
        "SBP_RollingMin_3","MAP_RollingMin_3","DBP_RollingMin_3","Resp_RollingMin_3",
        "PulsePressure","ShockIndex"
    ]
]

    # ========================================================
    # Predict Class
    # ========================================================

    prediction = int(model.predict(latest_row)[0])

    # ========================================================
    # Predict Probability
    # ========================================================

    probability = float(model.predict_proba(latest_row)[0][1])

    # ========================================================
    # Risk Level
    # ========================================================

    if probability >= 0.70:
        risk_level = "High"

    elif probability >= 0.45:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    # ========================================================
    # Return Prediction
    # ========================================================

    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level
    }