import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

#Loading processed data
df = pd.read_csv('data/telco_churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop(['customerID'], axis=1, inplace=True)

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes':1, 'No':0})
    
df = pd.get_dummies(df, drop_first=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
    df[['tenure', 'MonthlyCharges', 'TotalCharges']]
)

X = df.drop('Churn', axis=1)
y = df['Churn']

#Loading trained model
model = joblib.load('models/xgb_churn_model.pkl')

#SHAP Explainer
explainer = shap.Explainer(model)
shap_values = explainer(X)

#Show summary plot
print("Generating SHAP summary plot...")
shap.summary_plot(shap_values, X)

