
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap

from flask import Flask, request, jsonify
from flask_cors import CORS

from churn_preprocessing import NUMERIC_FEATURES


# ==================================================
# 1. Initialize Flask
# ==================================================

app = Flask(__name__)
CORS(app)


# ==================================================
# 2. Load Model and Preprocessing Artifacts
# ==================================================

MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_churn_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "standard_scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.json")


def load_artifacts():
    """Load the trained model and saved preprocessing artifacts."""

    for path in (MODEL_PATH, SCALER_PATH, FEATURES_PATH):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required model artifact not found: {path}. "
                "Run 02_model_training.py first."
            )

    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        loaded_features = json.load(file)

    if not isinstance(loaded_features, list) or not loaded_features:
        raise ValueError("feature_columns.json must contain a non-empty list.")

    if len(loaded_features) != loaded_model.n_features_in_:
        raise ValueError(
            "The saved feature count does not match the trained model."
        )

    if len(set(loaded_features)) != len(loaded_features):
        raise ValueError("Duplicate feature names found in feature_columns.json.")

    if not set(NUMERIC_FEATURES).issubset(loaded_features):
        raise ValueError(
            "The saved feature list does not contain all expected numeric features."
        )

    if hasattr(loaded_scaler, "feature_names_in_"):
        scaler_features = list(loaded_scaler.feature_names_in_)
        if scaler_features != list(NUMERIC_FEATURES):
            raise ValueError(
                "The saved scaler's feature names do not match NUMERIC_FEATURES."
            )

    return loaded_model, loaded_scaler, loaded_features


model, scaler, TRAINING_FEATURES = load_artifacts()

# Initialize SHAP once rather than once per prediction request.
explainer = shap.TreeExplainer(model)

print("ChurnXplain API initialized successfully.")
print(f"Model loaded: {MODEL_PATH}")
print(f"Scaler loaded: {SCALER_PATH}")
print(f"Feature count: {len(TRAINING_FEATURES)}")


# ==================================================
# 3. Home Route
# ==================================================

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Welcome to ChurnXplain API.",
        "endpoints": ["/health", "/predict"]
    })


# ==================================================
# 4. Health Check
# ==================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "feature_count": len(TRAINING_FEATURES)
    }), 200


# ==================================================
# 5. Prediction Route
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        # Accept one customer or a list of customers.
        if isinstance(data, dict):
            records = [data]

        elif isinstance(data, list) and data:
            if not all(isinstance(row, dict) for row in data):
                return jsonify({
                    "error": "Every item in the input list must be an object."
                }), 400

            records = data

        else:
            return jsonify({
                "error": "Send a JSON object or a non-empty list of objects."
            }), 400

        df = pd.DataFrame(records)

        # Reject missing features rather than silently inventing values.
        missing_features = [
            feature
            for feature in TRAINING_FEATURES
            if feature not in df.columns
        ]

        if missing_features:
            return jsonify({
                "error": "Required model features are missing.",
                "missing_features": missing_features
            }), 400

        # Enforce exactly the feature order used during training.
        df = df[TRAINING_FEATURES].copy()

        # Validate numeric inputs.
        for column in TRAINING_FEATURES:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

        if df.isna().any().any():
            invalid_features = df.columns[
                df.isna().any()
            ].tolist()

            return jsonify({
                "error": "Inputs contain missing or non-numeric values.",
                "invalid_features": invalid_features
            }), 400

        if not np.isfinite(df.to_numpy(dtype=float)).all():
            return jsonify({
                "error": "Inputs must contain finite numeric values."
            }), 400

        # Scale using the fitted scaler from model training.
        df.loc[:, NUMERIC_FEATURES] = scaler.transform(
            df[NUMERIC_FEATURES]
        )

        # Generate predictions and churn probabilities.
        churn_probabilities = model.predict_proba(df)[:, 1]
        churn_predictions = model.predict(df)

        # Generate SHAP explanations.
        shap_values = explainer.shap_values(df)

        # Handle different SHAP output formats.
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        shap_values = np.asarray(shap_values)

        # Some SHAP versions return rows x features x classes.
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]

        if shap_values.shape != (len(df), len(TRAINING_FEATURES)):
            app.logger.error(
                "Unexpected SHAP output shape: %s",
                shap_values.shape
            )
            return jsonify({
                "error": "Could not generate explanations for these predictions."
            }), 500

        results = []

        for i in range(len(df)):
            feature_importance = [
                {
                    "feature": TRAINING_FEATURES[j],
                    "shap_value": float(shap_values[i, j])
                }
                for j in range(len(TRAINING_FEATURES))
            ]

            top_features = sorted(
                feature_importance,
                key=lambda item: abs(item["shap_value"]),
                reverse=True
            )[:5]

            results.append({
                "prediction": int(churn_predictions[i]),
                "churn_probability": round(
                    float(churn_probabilities[i]), 4
                ),
                "top_features": top_features
            })

        return jsonify(results), 200

    except Exception:
        app.logger.exception("Prediction request failed.")

        return jsonify({
            "error": "Prediction failed. Check the API server logs."
        }), 500


# ==================================================
# 6. Start the Flask Server
# ==================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
