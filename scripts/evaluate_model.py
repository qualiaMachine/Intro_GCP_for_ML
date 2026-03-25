"""Evaluate a trained XGBoost model on the Titanic test set.

Usage:
    python evaluate_model.py
    python evaluate_model.py --model xgboost-model --test titanic_test.csv
"""

import argparse
import json

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score


def preprocess_data(df):
    """Apply the same preprocessing as train_xgboost.py."""
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model")
    parser.add_argument("--model", type=str, default="xgboost-model")
    parser.add_argument("--test", type=str, default="titanic_test.csv")
    args = parser.parse_args()

    # Load model
    model = joblib.load(args.model)

    # Load and preprocess test data
    test_df = pd.read_csv(args.test)
    X_test, y_test = preprocess_data(test_df)

    # Predict
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Compute accuracy
    acc = accuracy_score(y_test, y_pred_binary)
    print(f"Test accuracy: {acc:.4f}")

    # Save results
    results = {"test_accuracy": float(acc), "test_samples": int(len(y_test))}
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to evaluation_results.json")


if __name__ == "__main__":
    main()
