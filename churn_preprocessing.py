"""Shared preprocessing helpers for ChurnXplain.

This module intentionally performs encoding only. The numerical scaler is fitted
inside the training split in 02_model_training.py to avoid data leakage.
"""
from pathlib import Path
from typing import Tuple

import pandas as pd

BINARY_COLUMNS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COLUMN = "Churn"
DEFAULT_DATA_PATH = Path("data") / "telco_churn.csv"


def load_and_encode_data(data_path=DEFAULT_DATA_PATH) -> Tuple[pd.DataFrame, pd.Series, int]:
    """Load the raw Telco dataset, clean it, and one-hot encode predictors.

    Returns X, y, and the number of rows dropped because TotalCharges was blank.
    The returned X contains unscaled numeric values; fit the scaler only on X_train.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Run this script from the project root."
        )

    df = pd.read_csv(data_path)
    required = {"Churn", "TotalCharges", *BINARY_COLUMNS[:-1]}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    rows_before = len(df)
    df = df.dropna(subset=["TotalCharges"]).copy()
    dropped_rows = rows_before - len(df)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    for column in BINARY_COLUMNS:
        if column in df.columns:
            mapped = df[column].map({"Yes": 1, "No": 0})
            # Churn and the other listed binary columns are expected to contain
            # Yes/No. Fail clearly if the source data changes unexpectedly.
            if mapped.isna().any():
                bad_values = df.loc[mapped.isna(), column].drop_duplicates().tolist()
                raise ValueError(f"Unexpected values in {column}: {bad_values}")
            df[column] = mapped.astype("int8")

    df = pd.get_dummies(df, drop_first=True, dtype="int8")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing after encoding.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype("int8")

    # Keep feature order deterministic and numeric for model training.
    X = X.reindex(sorted(X.columns), axis=1)
    return X, y, dropped_rows
