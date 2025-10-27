import argparse
import io
import os
import tempfile
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from time import time
import joblib

# Optional: only needed if you pass gs:// paths
try:
    from google.cloud import storage
    _HAS_GCS = True
except Exception:
    _HAS_GCS = False


def _is_gcs(p: str) -> bool:
    return str(p).startswith("gs://")


def _gcs_client():
    if not _HAS_GCS:
        raise RuntimeError("google-cloud-storage is required to read/write gs:// paths. Install it with: pip install google-cloud-storage")
    return storage.Client()


def read_csv_any(path: str) -> pd.DataFrame:
    if _is_gcs(path):
        client = _gcs_client()
        bucket, _, key = path[5:].partition("/")
        blob = client.bucket(bucket).blob(key)
        return pd.read_csv(io.BytesIO(blob.download_as_bytes()))
    return pd.read_csv(path)


def save_model_any(model, dest_path: str):
    # joblib.dump needs a local file; upload if gs://
    if _is_gcs(dest_path):
        client = _gcs_client()
        bucket, _, key = dest_path[5:].partition("/")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "model.joblib"
            joblib.dump(model, tmp)
            client.bucket(bucket).blob(key).upload_from_filename(str(tmp))
    else:
        Path(os.path.dirname(dest_path) or ".").mkdir(parents=True, exist_ok=True)
        joblib.dump(model, dest_path)


def preprocess_data(df):
    """Preprocess the Titanic dataset for model training."""
    df = df.copy()
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True, errors="ignore")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train.csv", help="Path to training data CSV (local or gs://...)")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--num_round", type=int, default=100)
    args = parser.parse_args()

    # GCP-style env vars (analogous to SageMaker's)
    # AIP_MODEL_DIR may be gs:// (Vertex AI CustomJob) or empty (Workbench/local)
    input_data_path = args.train  # pass local path or gs:// explicitly
    output_dir = os.environ.get("AIP_MODEL_DIR", ".")
    output_data_path = os.path.join(output_dir, "xgboost-model")  

    # Load and preprocess data
    df = read_csv_any(input_data_path)
    X, y = preprocess_data(df)

    # Split data for evaluation purposes
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
    }

    # Train & save
    model = train_model(dtrain, params, args.num_round)
    save_model_any(model, output_data_path)
    print(f"Model saved to {output_data_path}")
