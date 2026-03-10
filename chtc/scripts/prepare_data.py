#!/usr/bin/env python3
"""
prepare_data.py — Prepare Titanic CSV data into train/val/test .npz files for PyTorch training.

Preprocesses the Titanic CSV and splits into train/val/test sets saved as .npz files.
Can be run on the submit node or as an HTCondor job.

Default split: 60% train, 20% validation, 20% test.
- Train: used for model fitting
- Validation: used for early stopping and hyperparameter selection
- Test: held out until the very end for unbiased performance estimate
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_titanic(df):
    """Preprocess the Titanic dataset: handle missing values, encode categoricals."""
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare Titanic data for PyTorch training")
    parser.add_argument("--input", type=str, default="titanic_train.csv",
                        help="Path to raw Titanic CSV")
    parser.add_argument("--output_train", type=str, default="train_data.npz",
                        help="Output path for training .npz")
    parser.add_argument("--output_val", type=str, default="val_data.npz",
                        help="Output path for validation .npz")
    parser.add_argument("--output_test", type=str, default="test_data.npz",
                        help="Output path for test .npz")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Fraction of total data for validation (default 0.2)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of total data for test (default 0.2)")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # Load and preprocess
    df = pd.read_csv(args.input)
    df = preprocess_titanic(df)

    X = df.drop(columns=["Survived"]).values.astype("float32")
    y = df["Survived"].values.astype("float32")

    # Two-stage split: first hold out test set, then split remainder into train/val
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    val_fraction = args.val_size / (1.0 - args.test_size)  # fraction of trainval
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction, random_state=args.random_state
    )

    # Save as .npz
    np.savez(args.output_train, X_train=X_train, y_train=y_train)
    np.savez(args.output_val, X_val=X_val, y_val=y_val)
    np.savez(args.output_test, X_test=X_test, y_test=y_test)

    total = len(y)
    print(f"Total samples:   {total}")
    print(f"Training data:   {args.output_train} — {X_train.shape[0]} rows ({X_train.shape[0]/total:.0%})")
    print(f"Validation data: {args.output_val} — {X_val.shape[0]} rows ({X_val.shape[0]/total:.0%})")
    print(f"Test data:       {args.output_test} — {X_test.shape[0]} rows ({X_test.shape[0]/total:.0%})")


if __name__ == "__main__":
    main()
