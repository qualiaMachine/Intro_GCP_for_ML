#!/usr/bin/env python3
"""
XGBoost trainer for the Titanic dataset.

Trains an XGBoost classifier on the Titanic dataset. All file I/O uses plain
local paths — HTCondor handles file transfer to/from worker machines.
"""

import argparse
import os
from pathlib import Path
from time import time

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib


def preprocess_data(df):
    """Preprocess the Titanic dataset for model training."""
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    return X, y


def train_model(dtrain, params, num_round):
    """Train the XGBoost model with the specified parameters."""
    start_time = time()
    model = xgb.train(params, dtrain, num_boost_round=num_round)
    print(f"Training time: {time() - start_time:.2f} seconds")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost on Titanic data (CHTC version)")
    parser.add_argument("--train", type=str, default="titanic_train.csv",
                        help="Path to training data CSV")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory for output artifacts")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--num_round", type=int, default=100)
    args = parser.parse_args()

    # Output path
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_model_path = os.path.join(output_dir, "xgboost-model")

    # Load and preprocess data
    df = pd.read_csv(args.train)
    X, y = preprocess_data(df)

    # Split data for evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape}")
    print(f"Val size:   {X_val.shape}")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # XGBoost parameters
    params = {
        "objective": "binary:logistic",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "eval_metric": "logloss",
        "seed": 42,
    }

    # Train & save
    model = train_model(dtrain, params, args.num_round)
    joblib.dump(model, output_model_path)
    print(f"Model saved to {output_model_path}")

    # Evaluate on validation set
    val_preds = model.predict(dval)
    val_acc = ((val_preds > 0.5) == y_val.values).mean()
    print(f"Validation accuracy: {val_acc:.4f}")
