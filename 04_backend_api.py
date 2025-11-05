from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import shap
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return "Welcome to ChurnXplain API. Use /predict to get predictions."

# ----------------------
# 1. Load Model & Features
# ----------------------
model = joblib.load("models/xgb_churn_model.pkl")
scaler = StandardScaler()

# Get feature names directly from the trained model
try:
    TRAINING_FEATURES = list(model.get_booster().feature_names)
except:
    raise ValueError("Could not extract features from the model.")

print("✅ Loaded model features:", TRAINING_FEATURES)

# ----------------------
# 2. Prediction Route
# ----------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format"}), 400

        print("📥 Raw Input DataFrame:")
        print(df.head())

        # Ensure all model features are present
        for col in TRAINING_FEATURES:
            if col not in df.columns:
                df[col] = 0  # Missing column → fill with 0

        # Remove any extra columns not in training
        df = df[TRAINING_FEATURES]

        # Scale numeric columns that were scaled during training
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = scaler.fit_transform(df[[col]])

        # Predict
        churn_probs = model.predict_proba(df)[:, 1]
        churn_preds = model.predict(df)

        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)

        results = []
        for i in range(len(df)):
            feature_importance = [
                {"feature": TRAINING_FEATURES[j], "shap_value": float(shap_values[i][j])}
                for j in range(len(TRAINING_FEATURES))
            ]
            top_features = sorted(feature_importance, key=lambda x: abs(x["shap_value"]), reverse=True)[:5]

            results.append({
                "prediction": int(churn_preds[i]),
                "churn_probability": round(float(churn_probs[i]), 4),
                "top_features": top_features
            })

        return jsonify(results)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
