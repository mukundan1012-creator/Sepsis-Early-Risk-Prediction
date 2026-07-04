"""
feature_engineering.py

Contains all feature engineering functions used for
Sepsis Early Risk Prediction.

Author: Mukundan D
"""

import pandas as pd

# ============================================================
# Vital Sign Columns
# ============================================================

VITALS = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp"
]


# ============================================================
# Lag Features
# ============================================================

def create_lag_features(df):
    """
    Creates previous-hour (Lag1) features
    for every vital sign.
    """

    for col in VITALS:
        df[f"{col}_Lag1"] = (
            df.groupby("Patient_ID")[col]
            .shift(1)
        )

    return df


# ============================================================
# Change Features
# ============================================================

def create_change_features(df):
    """
    Calculates the difference between
    current and previous readings.
    """

    for col in VITALS:
        df[f"{col}_Change"] = (
            df.groupby("Patient_ID")[col]
            .diff()
        )

    return df


# ============================================================
# Rolling Mean Features
# ============================================================

def create_rolling_mean_features(df):
    """
    Creates rolling mean using the
    last three readings.
    """

    for col in VITALS:
        df[f"{col}_RollingMean_3"] = (
            df.groupby("Patient_ID")[col]
            .transform(
                lambda x: x.rolling(
                    window=3,
                    min_periods=1
                ).mean()
            )
        )

    return df


# ============================================================
# Rolling Standard Deviation Features
# ============================================================

def create_rolling_std_features(df):
    """
    Measures stability of
    the last three readings.
    """

    for col in VITALS:
        df[f"{col}_RollingStd_3"] = (
            df.groupby("Patient_ID")[col]
            .transform(
                lambda x: x.rolling(
                    window=3,
                    min_periods=2
                ).std()
            )
        )

    return df


# ============================================================
# Rolling Maximum Features
# ============================================================

def create_rolling_max_features(df):
    """
    Creates rolling maximum
    of last three readings.
    """

    for col in VITALS:
        df[f"{col}_RollingMax_3"] = (
            df.groupby("Patient_ID")[col]
            .transform(
                lambda x: x.rolling(
                    window=3,
                    min_periods=1
                ).max()
            )
        )

    return df


# ============================================================
# Rolling Minimum Features
# ============================================================

def create_rolling_min_features(df):
    """
    Creates rolling minimum
    of last three readings.
    """

    for col in VITALS:
        df[f"{col}_RollingMin_3"] = (
            df.groupby("Patient_ID")[col]
            .transform(
                lambda x: x.rolling(
                    window=3,
                    min_periods=1
                ).min()
            )
        )

    return df


# ============================================================
# Clinical Features
# ============================================================

def create_clinical_features(df):
    """
    Creates medically important features.
    """

    df["PulsePressure"] = (
        df["SBP"] - df["DBP"]
    )

    df["ShockIndex"] = (
        df["HR"] / df["SBP"]
    )

    return df


# ============================================================
# Main Feature Engineering Pipeline
# ============================================================

def engineer_features(df):
    """
    Runs the complete feature
    engineering pipeline.
    """

    df = create_lag_features(df)

    df = create_change_features(df)

    df = create_rolling_mean_features(df)

    df = create_rolling_std_features(df)

    df = create_rolling_max_features(df)

    df = create_rolling_min_features(df)

    df = create_clinical_features(df)

    df.fillna(0, inplace=True)

    return df