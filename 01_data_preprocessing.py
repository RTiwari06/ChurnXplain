import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#1. Loading the dataset
df = pd.read_csv('data/telco_churn.csv')
print("Dataset loaded sucessfully.")

#2. Basic info 
print("\n First 5 rows of data: ")
print(df.head())

print("\n Dataset shape: ", df.shape)
print("\n Null values: ")
print(df.isnull().sum())

#3. Convert TotalCharges to numeric (some are blank strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#4. Drops rows with missing TotalCharges
df.dropna(inplace=True)
print("\n Dropped rows with missing TotalCharges. New shape: ", df.shape)

#5. Drop irrelevant columns
df.drop(['customerID'], axis=1, inplace=True)

#6. Encode binary categorical columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    
#7. One-hot encode other categorical features
df = pd.get_dummies(df, drop_first=True)
print("\n Applied one-hot encoding. Current columns: ")
print(df.columns)

#8. Normalize numerical columns
scaler = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
    df[['tenure', 'MonthlyCharges', 'TotalCharges']]
)

#9. Split into features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

print("\n Data preprocessing complete.")
print(f" Features shape: {X.shape}")
print(f" Target shape: {y.shape}")