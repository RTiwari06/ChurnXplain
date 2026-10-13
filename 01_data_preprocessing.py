"""Inspect and encode the Telco churn dataset without fitting a scaler."""
from churn_preprocessing import load_and_encode_data, NUMERIC_FEATURES


def main():
    X, y, dropped_rows = load_and_encode_data()
    print("Dataset loaded successfully.")
    print(f"Rows removed because TotalCharges was blank: {dropped_rows}")
    print(f"Encoded feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print("\nTarget distribution:")
    print(y.value_counts().rename(index={0: "No churn", 1: "Churn"}))
    print("\nNumerical features that will be scaled after the train/test split:")
    print([column for column in NUMERIC_FEATURES if column in X.columns])
    print("\nEncoded feature columns:")
    print(list(X.columns))
    print("\nPreprocessing inspection complete. No scaler was fitted on the full dataset.")


if __name__ == "__main__":
    main()
