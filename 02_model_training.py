"""Train and evaluate the ChurnXplain model with leakage-free preprocessing."""
import json
import os

import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_preprocessing import load_and_encode_data, NUMERIC_FEATURES

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_churn_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "standard_scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.json")


def main():
    X, y, dropped_rows = load_and_encode_data()
    print(f"Dataset prepared. Dropped {dropped_rows} rows with blank TotalCharges.")
    print(f"Features: {X.shape[1]} | Rows: {X.shape[0]}")

    # Split BEFORE fitting the scaler so the test set cannot influence scaling.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    numeric_features = [col for col in NUMERIC_FEATURES if col in X_train.columns]
    # Use a floating dtype for the full matrix so scaled continuous values can
    # be assigned without pandas dtype warnings.
    X_train = X_train.astype("float64")
    X_test = X_test.astype("float64")
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nModel evaluation on held-out test data")
    print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"], zero_division=0))
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURES_PATH, "w", encoding="utf-8") as file:
        json.dump(list(X.columns), file, indent=2)

    print("\nSaved model artifacts:")
    print(f"- Model: {MODEL_PATH}")
    print(f"- Fitted scaler: {SCALER_PATH}")
    print(f"- Ordered feature list: {FEATURES_PATH}")
    print("NOTE: Update 04_backend_api.py to load and reuse standard_scaler.pkl before running the app.")


if __name__ == "__main__":
    main()
