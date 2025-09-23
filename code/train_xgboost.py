#!/usr/bin/env python3
"""
XGBoost training script adapted for Google Cloud Vertex AI.

- Data path is taken as-is; if it starts with 'gs://', we stream from GCS.
- Model artifacts are saved into AIP_MODEL_DIR so Vertex AI can upload them.
- Pandas ops avoid chained-assignment / inplace patterns.
"""

import argparse
import os
import io
import tempfile
from time import time

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


# ---------- Utilities ----------

def _parse_gcs_uri(uri: str):
    """Return (bucket, key) from a 'gs://bucket/key' URI."""
    assert uri.startswith("gs://"), f"Not a GCS URI: {uri}"
    path = uri[5:]  # strip 'gs://'
    bucket, _, key = path.partition("/")
    if not bucket or not key:
        raise ValueError(f"Malformed GCS URI: {uri}")
    return bucket, key


def read_csv_any(path: str) -> pd.DataFrame:
    """
    Read CSV from a local path or a GCS URI.
    Prefers fsspec/gcsfs for GCS; falls back to google-cloud-storage if needed.
    """
    if path.startswith("gs://"):
        # Try fsspec/gcsfs first (simplest / streaming)
        try:
            import fsspec  # gcsfs is auto-loaded by fsspec if installed
            with fsspec.open(path, mode="rb") as f:
                return pd.read_csv(f)
        except Exception:
            # Fallback: download via google-cloud-storage
            from google.cloud import storage  # noqa: WPS433 (import inside)
            bucket_name, key = _parse_gcs_uri(path)
            client = storage.Client()
            blob = client.bucket(bucket_name).blob(key)
            data = blob.download_as_bytes()
            return pd.read_csv(io.BytesIO(data))
    else:
        return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame):
    """Preprocess Titanic dataset for model training (pandas-3.0 safe)."""
    df = df.copy()

    # Fill missing values
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Embarked" in df.columns:
        mode = df["Embarked"].mode(dropna=True)
        df["Embarked"] = df["Embarked"].fillna(mode.iloc[0] if not mode.empty else "S")

    # Drop unused columns if present
    drop_cols = [c for c in ["Name", "Ticket", "Cabin"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Encode categoricals if present
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 1, "female": 0}).fillna(0).astype(int)
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0).astype(int)

    # Target / features
    if "Survived" not in df.columns:
        raise ValueError("Expected 'Survived' column in training data.")
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    return X, y


# ---------- Training ----------

def train_model(dtrain: xgb.DMatrix, dval: xgb.DMatrix, params: dict, num_round: int):
    """Train XGBoost and report timing."""
    start_time = time()
    model = xgb.train(params, dtrain, num_boost_round=num_round, evals=[(dval, "val")])
    print(f"Training time: {time() - start_time:.2f} seconds")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True,
                        help="Path to training CSV (local path or gs://bucket/obj)")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--num_round", type=int, default=100)
    args = parser.parse_args()

    # IMPORTANT: do not os.path.join with the train path; respect 'gs://' if present
    input_data_path = args.train

    # Vertex AI uploads models from AIP_MODEL_DIR if present
    model_dir = os.environ.get("AIP_MODEL_DIR", ".")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgboost-model.joblib")

    # Load & preprocess
    df = read_csv_any(input_data_path)
    X, y = preprocess_data(df)

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape}")
    print(f"Val size:   {X_val.shape}")

    # DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Params
    params = {
        "objective": "binary:logistic",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "eval_metric": "logloss",
        # "tree_method": "hist",  # uncomment for larger datasets / speed
    }

    # Train
    model = train_model(dtrain, dval, params, args.num_round)

    # Save artifact where Vertex AI will pick it up
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
