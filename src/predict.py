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

    if probability >= 0.80:
        risk_level = "High"

    elif probability >= 0.50:
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