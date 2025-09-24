#!/usr/bin/env python3
"""
XGBoost training script adapted for Google Cloud Vertex AI.

All outputs are colocated with the model artifact:
- <artifact_dir>/model.joblib  (or your --model_out file)
- <artifact_dir>/metrics.json
- <artifact_dir>/eval_history.csv
- <artifact_dir>/training.log

Defaults:
- If --model_out is not set and AIP_MODEL_DIR exists (Vertex AI), saves to:
  AIP_MODEL_DIR/xgboost-model.joblib and writes logs/metrics alongside it.
- If run locally without AIP_MODEL_DIR and no --model_out, saves in the CWD.
"""

import argparse
import io
import json
import os
import sys
from time import time

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


# ---------- Path & I/O helpers ----------

def _parse_gcs_uri(uri: str):
    assert uri.startswith("gs://"), f"Not a GCS URI: {uri}"
    path = uri[5:]
    bucket, _, key = path.partition("/")
    if not bucket or not key:
        raise ValueError(f"Malformed GCS URI: {uri}")
    return bucket, key


def _parent_dir(path: str) -> str:
    if path.startswith("gs://"):
        return path.rsplit("/", 1)[0] if "/" in path[5:] else path
    return os.path.dirname(path) or "."


def _join_dir(path_dir: str, name: str) -> str:
    if path_dir.startswith("gs://"):
        return path_dir.rstrip("/") + "/" + name
    return os.path.join(path_dir, name)


def _write_bytes(path: str, data: bytes):
    """Write bytes to local or GCS path (fsspec first, then storage fallback)."""
    if path.startswith("gs://"):
        try:
            import fsspec  # uses gcsfs if installed
            with fsspec.open(path, "wb") as f:
                f.write(data)
            return
        except Exception:
            from google.cloud import storage  # type: ignore
            bucket_name, key = _parse_gcs_uri(path)
            client = storage.Client()
            client.bucket(bucket_name).blob(key).upload_from_string(data)
            return
    os.makedirs(_parent_dir(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _write_text(path: str, text: str):
    _write_bytes(path, text.encode("utf-8"))


def read_csv_any(path: str) -> pd.DataFrame:
    if path.startswith("gs://"):
        try:
            import fsspec
            with fsspec.open(path, "rb") as f:
                return pd.read_csv(f)
        except Exception:
            from google.cloud import storage  # type: ignore
            bucket, key = _parse_gcs_uri(path)
            data = storage.Client().bucket(bucket).blob(key).download_as_bytes()
            return pd.read_csv(io.BytesIO(data))
    else:
        return pd.read_csv(path)


# ---------- Data prep ----------

def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Embarked" in df.columns:
        mode = df["Embarked"].mode(dropna=True)
        df["Embarked"] = df["Embarked"].fillna(mode.iloc[0] if not mode.empty else "S")

    drop_cols = [c for c in ["Name", "Ticket", "Cabin"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 1, "female": 0}).fillna(0).astype(int)
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0).astype(int)

    if "Survived" not in df.columns:
        raise ValueError("Expected 'Survived' column in training data.")
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    return X, y


# ---------- Training ----------

def train_model(dtrain: xgb.DMatrix, dval: xgb.DMatrix, params: dict, num_round: int):
    evals_result = {}
    start_time = time()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_round,
        evals=[(dval, "val")],
        evals_result=evals_result,
    )
    print(f"Training time: {time() - start_time:.2f} seconds")
    return booster, evals_result


# ---------- Simple stdout/stderr capture ----------

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True,
                        help="Path to training CSV (local path or gs://bucket/obj)")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--model_out", default="",
                        help="Exact output path for trained model (local or gs://...). "
                             "If empty, defaults to AIP_MODEL_DIR/xgboost-model.joblib or local CWD.")
    args = parser.parse_args()

    # Default model path selection
    aip_model_dir = os.environ.get("AIP_MODEL_DIR", "")
    if args.model_out.strip():
        model_path = args.model_out.strip()
    else:
        # Use AIP_MODEL_DIR if present; else save in the current directory
        model_dir = aip_model_dir if aip_model_dir else "."
        if not model_dir.startswith("gs://"):
            os.makedirs(model_dir, exist_ok=True)
        model_path = _join_dir(model_dir, "xgboost-model.joblib")

    # Artifact directory = parent of the model_path (local or gs://)
    artifact_dir = _parent_dir(model_path)

    # Capture stdout/stderr; flush to <artifact_dir>/training.log at the end
    buf = io.StringIO()
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(sys.stdout, buf)
    sys.stderr = _Tee(sys.stderr, buf)

    log_path = _join_dir(artifact_dir, "training.log")
    metrics_path = _join_dir(artifact_dir, "metrics.json")
    history_path = _join_dir(artifact_dir, "eval_history.csv")

    try:
        print(f"[INFO] AIP_MODEL_DIR = {aip_model_dir or '(not set)'}")
        print(f"[INFO] Model path    = {model_path}")
        print(f"[INFO] Artifact dir  = {artifact_dir}")
        print(f"[INFO] Log file      = {log_path}")

        # Load & preprocess
        df = read_csv_any(args.train)
        X, y = preprocess_data(df)

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Train size: {X_train.shape}")
        print(f"Val size:   {X_val.shape}")

        # DMatrix + params
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            "objective": "binary:logistic",
            "max_depth": args.max_depth,
            "eta": args.eta,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "eval_metric": "logloss",
            # "tree_method": "hist",  # uncomment for large datasets
        }
        print("Params:", params)

        # Train
        model, evals_result = train_model(dtrain, dval, params, args.num_round)

        # Save model artifact (joblib)
        model_bytes = io.BytesIO()
        joblib.dump(model, model_bytes)
        _write_bytes(model_path, model_bytes.getvalue())
        print(f"[INFO] Saved model to: {model_path}")

        # Metrics summary
        final_val = None
        if "val" in evals_result and "logloss" in evals_result["val"]:
            vals = evals_result["val"]["logloss"]
            final_val = float(vals[-1]) if vals else None

        metrics = {
            "final_val_logloss": final_val,
            "num_boost_round": int(args.num_round),
            "params": params,
            "train_rows": int(X_train.shape[0]),
            "val_rows": int(X_val.shape[0]),
            "features": list(map(str, X_train.columns)),
            "model_uri": model_path,
        }
        _write_text(metrics_path, json.dumps(metrics, indent=2))
        print(f"[INFO] Wrote metrics to: {metrics_path}")

        # Per-iteration history
        if "val" in evals_result and "logloss" in evals_result["val"]:
            vals = evals_result["val"]["logloss"]
            csv = "iter,val_logloss\n" + "\n".join(f"{i+1},{v}" for i, v in enumerate(vals))
            _write_text(history_path, csv)
            print(f"[INFO] Wrote eval history to: {history_path}")
        else:
            print("[WARN] No validation eval history captured.")

    finally:
        # Restore std streams, then write the captured combined log next to the model
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = orig_out, orig_err
        try:
            _write_text(log_path, buf.getvalue())
            print(f"[INFO] Training log saved to: {log_path}")
        except Exception as e:
            print(f"[WARN] Could not write training log to {log_path}: {e}")


if __name__ == "__main__":
    main()
