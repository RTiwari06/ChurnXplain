import streamlit as st
import pandas as pd
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
                response = requests.post(API_URL, json=df.fillna(0).to_dict(orient="records"))
                if response.status_code == 200:
                    predictions = response.json()

                    results_table = [
                        {"Customer": f"Customer {i+1}", "Churn Probability (%)": pred["churn_probability"] * 100}
                        for i, pred in enumerate(predictions)
                    ]
                    st.subheader("Prediction Summary")
                    st.dataframe(results_table)

                    for i, pred in enumerate(predictions):
                        st.markdown(f"### Customer {i+1} - Churn Probability: **{pred['churn_probability']*100:.2f}%**")
                        st.write("Top Features (Table):")
                        st.dataframe(pred["top_features"])
                        plot_shap_bar(pred["top_features"])

                        # Save history
                        save_to_history({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "admin": st.session_state.username,
                            "customer": f"Customer {i+1}",
                            "probability": pred["churn_probability"] * 100,
                            "features": pred["top_features"]
                        })
                else:
                    st.error(f"Error: {response.status_code}")

        # Manual Data Entry
        st.subheader("Manual Entry for a Single Customer")
        with st.form("manual_entry_form"):
            customer_id = st.text_input("Customer ID")
            data_used = st.number_input("Data Used (MB)", min_value=0.0, step=0.1)
            call_minutes = st.number_input("Call Minutes", min_value=0.0, step=0.1)
            plan_type = st.selectbox("Plan Type", ["Basic", "Premium", "Unlimited"])
            submit_manual = st.form_submit_button("Predict Churn")

        if submit_manual:
            input_data = [{
                "customer_id": customer_id,
                "data_used": data_used,
                "call_minutes": call_minutes,
                "plan_type": plan_type
            }]
            response = requests.post(API_URL, json=input_data)
            if response.status_code == 200:
                prediction = response.json()[0]
                st.write(f"Prediction for Customer {customer_id}:")
                st.write(f"Churn Probability: {prediction['churn_probability']*100:.2f}%")

                st.write("Top Features (Table):")
                st.dataframe(prediction["top_features"])
                plot_shap_bar(prediction["top_features"])

                # Save history
                save_to_history({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "admin": st.session_state.username,
                    "customer": customer_id,
                    "probability": prediction["churn_probability"] * 100,
                    "features": prediction["top_features"]
                })
            else:
                st.error(f"Error: {response.status_code}")

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
