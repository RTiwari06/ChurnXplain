import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import json
import os
from datetime import datetime

# ----------------------
# 1. Config
# ----------------------
st.set_page_config(page_title="ChurnXplain", layout="wide")

API_URL = "http://127.0.0.1:5000/predict"
USERS_FILE = "users.json"
HISTORY_FILE = "prediction_history.json"


# The model expects the exact encoded feature schema saved during training.
FEATURES_FILE = os.path.join("models", "feature_columns.json")
NUMERIC_FEATURES = {"tenure", "MonthlyCharges", "TotalCharges"}


def load_expected_features():
    """Load the exact ordered feature schema used by the trained model."""
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(
            f"Feature schema not found: {FEATURES_FILE}. "
            "Run the app from the ChurnXplain project root."
        )
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        features = json.load(f)
    if not isinstance(features, list) or not features or not all(
        isinstance(name, str) and name for name in features
    ):
        raise ValueError("models/feature_columns.json is not a valid feature list.")
    if len(features) != len(set(features)):
        raise ValueError("Feature schema contains duplicate column names.")
    return features


def validate_prediction_csv(uploaded_df):
    """Validate and order an uploaded CSV without silently filling missing data."""
    expected = load_expected_features()
    actual = list(uploaded_df.columns)

    missing = [name for name in expected if name not in actual]
    unexpected = [name for name in actual if name not in expected]
    if missing or unexpected:
        details = []
        if missing:
            details.append("Missing required columns: " + ", ".join(missing))
        if unexpected:
            details.append("Unexpected columns: " + ", ".join(unexpected))
        raise ValueError("\\n".join(details))

    if uploaded_df.empty:
        raise ValueError("The uploaded CSV contains no customer rows.")

    df = uploaded_df[expected].copy()

    for column in expected:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_numeric = [
        column for column in expected
        if df[column].isna().any() or not np.isfinite(df[column].to_numpy(dtype=float)).all()
    ]
    if invalid_numeric:
        raise ValueError(
            "Missing, non-numeric, or non-finite values found in: "
            + ", ".join(invalid_numeric)
        )

    binary_features = [column for column in expected if column not in NUMERIC_FEATURES]
    invalid_binary = [
        column for column in binary_features
        if not df[column].isin([0, 1]).all()
    ]
    if invalid_binary:
        raise ValueError(
            "Binary features must contain only 0 or 1. Check: "
            + ", ".join(invalid_binary)
        )

    if (df["tenure"] < 0).any() or (df["tenure"] > 72).any():
        raise ValueError("tenure must be between 0 and 72 months.")
    if (df["MonthlyCharges"] < 0).any():
        raise ValueError("MonthlyCharges cannot be negative.")
    if (df["TotalCharges"] < 0).any():
        raise ValueError("TotalCharges cannot be negative.")

    # Check basic consistency between internet-service and related service flags.
    no_internet = df["InternetService_No"].eq(1)
    no_internet_flags = [
        name for name in (
            "DeviceProtection_No internet service",
            "OnlineBackup_No internet service",
            "OnlineSecurity_No internet service",
            "StreamingMovies_No internet service",
            "StreamingTV_No internet service",
            "TechSupport_No internet service",
        ) if name in df.columns
    ]
    for column in no_internet_flags:
        if (df.loc[no_internet, column] != 1).any():
            raise ValueError(
                f"When InternetService_No is 1, '{column}' must also be 1."
            )

    return df

# ----------------------
# 2. Helper Functions
# ----------------------
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def validate_username(username):
    return username.startswith("admin_") and len(username) >= 8

def validate_password(password):
    return len(password) >= 6 and any(c.isdigit() for c in password)

def save_to_history(entry):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

def generate_retention_recommendations(customer, churn_probability):
    """Return transparent, rule-based retention actions for one encoded customer."""
    actions = []

    # Contract flags use month-to-month as the all-zero baseline.
    if not customer.get("Contract_One year", 0) and not customer.get("Contract_Two year", 0):
        actions.append("Offer an optional longer-term plan with a clear price comparison or loyalty incentive.")

    tenure = float(customer.get("tenure", 0) or 0)
    if tenure < 12:
        actions.append("Provide onboarding support and check whether the customer has unresolved setup issues.")

    monthly_charges = float(customer.get("MonthlyCharges", 0) or 0)
    if monthly_charges >= 80:
        actions.append("Review the current bill and explain suitable lower-cost plans or bundles, if available.")

    if customer.get("InternetService_No", 0) != 1:
        if not customer.get("OnlineSecurity_Yes", 0):
            actions.append("Offer information about online security options, without assuming the customer needs an add-on.")
        if not customer.get("TechSupport_Yes", 0):
            actions.append("Explain available technical-support options and how to contact support when needed.")

    if customer.get("PaymentMethod_Electronic check", 0):
        actions.append("Offer optional automatic-payment methods and explain their convenience; do not pressure the customer to switch.")

    if churn_probability >= 0.60:
        actions.insert(0, "Prioritize a respectful, personalized retention check-in to understand the customer's needs.")
    elif churn_probability >= 0.30:
        actions.insert(0, "Consider a proactive customer check-in before offering a discount or plan change.")

    if not actions:
        actions.append("Maintain service quality and use routine customer-satisfaction check-ins; no specific intervention is indicated by these rules.")

    return actions


def plot_shap_bar(top_features):
    df = pd.DataFrame(top_features, columns=["feature", "shap_value"])
    df = df.sort_values(by="shap_value", key=abs, ascending=True)
    fig = px.bar(
        df,
        x="shap_value",
        y="feature",
        orientation="h",
        title="Top Contributing Features",
        color="shap_value",
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------
# 3. Authentication
# ----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

users = load_users()

if not st.session_state.logged_in:
    st.title("🔐 Admin Login / Signup")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if login_user in users and users[login_user]["password"] == login_pass:
                st.session_state.logged_in = True
                st.session_state.username = login_user
                st.success("✅ Logged in successfully!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

    with tab2:
        signup_user = st.text_input("New Username (must start with 'admin_')", key="signup_user")
        signup_pass = st.text_input("New Password (min 6 chars, 1 digit)", type="password", key="signup_pass")
        if st.button("Signup"):
            if not validate_username(signup_user):
                st.error("❌ Username must start with 'admin_' and be at least 8 characters.")
            elif not validate_password(signup_pass):
                st.error("❌ Password must be at least 6 characters long and contain a digit.")
            elif signup_user in users:
                st.error("❌ Username already exists.")
            else:
                users[signup_user] = {"password": signup_pass}
                save_users(users)
                st.success("✅ Signup successful! Please login.")

# ----------------------
# 4. Dashboard (After Login)
# ----------------------
else:
    st.sidebar.title(f"👋 Welcome, {st.session_state.username}")
    menu = st.sidebar.radio("Navigation", ["Predict", "Prediction History", "Logout"])

    # ----------------------
    # Prediction
    # ----------------------
    if menu == "Predict":
        st.title("📊 ChurnXplain - Customer Churn Prediction")

        st.subheader("Upload CSV Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of the Uploaded Data:")
            st.dataframe(df)

            if st.button("Get Predictions for File"):
                try:
                    validated_df = validate_prediction_csv(df)
                    with st.spinner("Validating customer data and generating predictions..."):
                        response = requests.post(
                            API_URL,
                            json=validated_df.to_dict(orient="records"),
                            timeout=60,
                        )
                    if response.status_code != 200:
                        try:
                            api_message = response.json().get("error", response.text)
                        except (ValueError, AttributeError):
                            api_message = response.text or "Unknown API error"
                        st.error(f"Prediction API error ({response.status_code}): {api_message}")
                    else:
                        predictions = response.json()
                        if not isinstance(predictions, list) or len(predictions) != len(validated_df):
                            st.error(
                                "The API response did not contain exactly one prediction "
                                "for each uploaded customer."
                            )
                        else:
                            results_table = []
                            export_df = validated_df.copy()
                            customer_labels = []
                            predicted_churn = []
                            churn_probabilities_pct = []
                            recommendation_texts = []
                            for i, pred in enumerate(predictions):
                                customer_label = f"Customer {i+1}"
                                probability = float(pred["churn_probability"])
                                customer_record = validated_df.iloc[i].to_dict()
                                actions = generate_retention_recommendations(
                                    customer_record, probability
                                )
                                customer_labels.append(customer_label)
                                predicted_churn.append(
                                    "Yes" if int(pred["prediction"]) == 1 else "No"
                                )
                                churn_probabilities_pct.append(round(probability * 100, 2))
                                recommendation_texts.append(" | ".join(actions))
                                results_table.append({
                                    "Customer": customer_label,
                                    "Predicted Churn": "Yes" if int(pred["prediction"]) == 1 else "No",
                                    "Churn Probability (%)": round(probability * 100, 2),
                                    "Recommended Actions": " | ".join(actions),
                                })

                            # Put useful output columns first; retain all model input features.
                            export_df.insert(0, "Customer", customer_labels)
                            export_df.insert(1, "Predicted Churn", predicted_churn)
                            export_df["Churn Probability (%)"] = churn_probabilities_pct
                            export_df["Recommended Retention Actions"] = recommendation_texts
                            st.subheader("Prediction Summary & Retention Actions")
                            st.download_button(
                                label="Download Full Prediction Report (CSV)",
                                data=export_df.to_csv(index=False).encode("utf-8-sig"),
                                file_name="churn_predictions_with_retention_actions.csv",
                                mime="text/csv",
                                help="Downloads customer labels, predicted churn status, probability, retention actions, and all 30 input features.",
                            )
                            st.caption("Use the download button above for the full report. The table toolbar's download icon may export only the displayed summary.")
                            st.dataframe(results_table, use_container_width=True)

                            for i, pred in enumerate(predictions):
                                st.markdown(
                                    f"### Customer {i+1} - Churn Probability: "
                                    f"**{float(pred['churn_probability'])*100:.2f}%**"
                                )
                                st.write("Top Features (SHAP explanation):")
                                st.dataframe(pred["top_features"], use_container_width=True)
                                plot_shap_bar(pred["top_features"])
                                st.write("**Recommended Retention Actions**")
                                for action in generate_retention_recommendations(
                                    validated_df.iloc[i].to_dict(), float(pred["churn_probability"])
                                ):
                                    st.markdown(f"- {action}")

                                save_to_history({
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "admin": st.session_state.username,
                                    "customer": f"Customer {i+1}",
                                    "probability": float(pred["churn_probability"]) * 100,
                                    "features": pred["top_features"],
                                })
                            st.success(f"Successfully predicted {len(predictions)} customers.")
                except ValueError as exc:
                    st.error(f"CSV validation failed: {exc}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to the ChurnXplain API. Make sure the Flask server "
                        "is running at http://127.0.0.1:5000."
                    )
                except requests.exceptions.Timeout:
                    st.error("The prediction API timed out. Try a smaller CSV file.")
                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not complete the API request: {exc}")
                except (KeyError, TypeError, ValueError) as exc:
                    st.error(f"Unexpected prediction response: {exc}")

        # Manual Data Entry: collect original telecom fields and encode them
        # into the exact 30-feature schema expected by the trained model.
        st.subheader("Manual Entry for a Single Customer")
        st.caption("Enter telecom account details. These fields are encoded to match the trained model.")

        with st.form("manual_entry_form"):
            customer_id = st.text_input("Customer ID", value="CUSTOMER-001")

            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Has Partner", ["No", "Yes"])
                dependents = st.selectbox("Has Dependents", ["No", "Yes"])
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
            with col2:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            with col3:
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

            col4, col5, col6 = st.columns(3)
            with col4:
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            with col5:
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            with col6:
                payment_method = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Credit card (automatic)", "Bank transfer (automatic)"],
                )

            col7, col8 = st.columns(2)
            with col7:
                monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
            with col8:
                total_charges = st.number_input("Total Charges", min_value=0.0, value=840.0, step=10.0)

            submit_manual = st.form_submit_button("Predict Churn")

        if submit_manual:
            if not customer_id.strip():
                st.error("Please enter a customer ID.")
            elif phone_service == "No" and multiple_lines != "No phone service":
                st.error("If Phone Service is No, Multiple Lines must be 'No phone service'.")
            elif phone_service == "Yes" and multiple_lines == "No phone service":
                st.error("'No phone service' is only valid when Phone Service is No.")
            else:
                # Start with every model feature at zero, then set the relevant encoded flags.
                input_row = {feature: 0 for feature in load_expected_features()}
                input_row.update({
                    "SeniorCitizen": int(senior_citizen == "Yes"),
                    "Partner": int(partner == "Yes"),
                    "Dependents": int(dependents == "Yes"),
                    "tenure": int(tenure),
                    "PhoneService": int(phone_service == "Yes"),
                    "PaperlessBilling": int(paperless_billing == "Yes"),
                    "MonthlyCharges": float(monthly_charges),
                    "TotalCharges": float(total_charges),
                    "gender_Male": int(gender == "Male"),
                })

                # Contract (month-to-month is the all-zero baseline).
                if contract == "One year":
                    input_row["Contract_One year"] = 1
                elif contract == "Two year":
                    input_row["Contract_Two year"] = 1

                # Internet service (DSL is the all-zero baseline).
                if internet_service == "Fiber optic":
                    input_row["InternetService_Fiber optic"] = 1
                elif internet_service == "No":
                    input_row["InternetService_No"] = 1

                # Multiple lines (No is the baseline when phone service is available).
                if multiple_lines == "Yes":
                    input_row["MultipleLines_Yes"] = 1
                elif multiple_lines == "No phone service":
                    input_row["MultipleLines_No phone service"] = 1

                # Internet-dependent services. If there is no internet, encode that category.
                service_fields = {
                    "DeviceProtection": device_protection,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                }
                for prefix, choice in service_fields.items():
                    if internet_service == "No":
                        input_row[f"{prefix}_No internet service"] = 1
                    elif choice == "Yes":
                        input_row[f"{prefix}_Yes"] = 1
                    # When internet is available, "No" is the all-zero baseline.

                # Payment method (Bank transfer is the all-zero baseline).
                payment_columns = {
                    "Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
                    "Electronic check": "PaymentMethod_Electronic check",
                    "Mailed check": "PaymentMethod_Mailed check",
                }
                if payment_method in payment_columns:
                    input_row[payment_columns[payment_method]] = 1

                # Put fields in the precise training order before calling the API.
                expected_features = load_expected_features()
                manual_df = pd.DataFrame([input_row])[expected_features]
                try:
                    with st.spinner("Generating prediction and SHAP explanation..."):
                        response = requests.post(
                            API_URL,
                            json=manual_df.to_dict(orient="records"),
                            timeout=60,
                        )
                    if response.status_code != 200:
                        try:
                            message = response.json().get("error", response.text)
                        except (ValueError, AttributeError):
                            message = response.text or "Unknown API error"
                        st.error(f"Prediction API error ({response.status_code}): {message}")
                    else:
                        predictions = response.json()
                        if not isinstance(predictions, list) or len(predictions) != 1:
                            st.error("The API did not return exactly one prediction for this customer.")
                        else:
                            prediction = predictions[0]
                            probability = float(prediction["churn_probability"]) * 100
                            st.markdown(f"### Prediction for Customer {customer_id.strip()}")
                            st.metric("Churn Probability", f"{probability:.2f}%")
                            st.write("Top Features (SHAP explanation):")
                            st.dataframe(prediction["top_features"], use_container_width=True)
                            plot_shap_bar(prediction["top_features"])
                            st.write("**Recommended Retention Actions**")
                            for action in generate_retention_recommendations(
                                manual_df.iloc[0].to_dict(), float(prediction["churn_probability"])
                            ):
                                st.markdown(f"- {action}")
                            st.caption("Recommendations are rule-based suggestions, not proven causes of churn or guaranteed outcomes.")
                            save_to_history({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "admin": st.session_state.username,
                                "customer": customer_id.strip(),
                                "probability": probability,
                                "features": prediction["top_features"],
                            })
                            st.success("Prediction completed and saved to history.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the ChurnXplain API. Make sure Flask is running on port 5000.")
                except requests.exceptions.Timeout:
                    st.error("The prediction API timed out. Please try again.")
                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not complete the API request: {exc}")
                except (KeyError, TypeError, ValueError) as exc:
                    st.error(f"Unexpected prediction response: {exc}")

    # ----------------------
    # Prediction History
    # ----------------------
    elif menu == "Prediction History":
        st.title("📜 Prediction History")
        history = load_history()

        if history:
            df_history = pd.DataFrame(history)
            st.dataframe(df_history[["timestamp", "admin", "customer", "probability"]])

            # Download as CSV
            csv_data = df_history.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download History as CSV",
                data=csv_data,
                file_name="prediction_history.csv",
                mime="text/csv"
            )

            # Clear history option
            if st.button("🗑️ Clear History"):
                clear_history()
                st.warning("Prediction history cleared!")
                st.rerun()

            # Latest details
            latest = history[-1]
            st.markdown(f"### Latest Prediction: {latest['customer']}")
            st.write(f"Churn Probability: {latest['probability']:.2f}%")
            st.write("Top Features (Table):")
            st.dataframe(latest["features"])
            plot_shap_bar(latest["features"])
        else:
            st.info("No prediction history found.")

    # ----------------------
    # Logout
    # ----------------------
    elif menu == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("✅ Logged out successfully.")
        st.rerun()
