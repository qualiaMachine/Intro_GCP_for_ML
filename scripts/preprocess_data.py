"""Preprocess Titanic dataset into .npz files for PyTorch training.

This script reads raw CSV files, applies preprocessing (encoding, scaling,
imputation), and saves train/val arrays as compressed .npz files.

Usage:
    python preprocess_data.py
    python preprocess_data.py --train titanic_train.csv --test titanic_test.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def main():
    parser = argparse.ArgumentParser(description="Preprocess Titanic data to .npz")
    parser.add_argument("--train", type=str, default="titanic_train.csv")
    parser.add_argument("--test", type=str, default="titanic_test.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.train)

    # Encode categorical features
    sex_enc = LabelEncoder().fit(df["Sex"])
    df["Sex"] = sex_enc.transform(df["Sex"])
    df["Embarked"] = df["Embarked"].fillna("S")
    emb_enc = LabelEncoder().fit(df["Embarked"])
    df["Embarked"] = emb_enc.transform(df["Embarked"])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values
    y = df["Survived"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    np.savez("train_data.npz", X_train=X_train, y_train=y_train)
    np.savez("val_data.npz", X_val=X_val, y_val=y_val)

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    print("Saved: train_data.npz, val_data.npz")


if __name__ == "__main__":
    main()
