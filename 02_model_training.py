import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#1. Loading the preprocessed data from the preprocessing step
df = pd.read_csv("data/telco_churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop(['customerID'], axis=1, inplace=True)
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
    df[['tenure', 'MonthlyCharges', 'TotalCharges']]
)

X = df.drop('Churn', axis=1)
y = df['Churn']

#2. Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#3. Creating and training the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

#4. Making predictions and evalue
y_pred = model.predict(X_test)

print("\n Model Training Complete!")
print("\n Classification Report: ")
print(classification_report(y_test, y_pred))

print(f" Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")

#5. Saving the model to the models folder
joblib.dump(model, "models/xgb_churn_model.pkl")
print("\n Model saved to models/xgb_churn_model.pkl")