# GCP-friendly PyTorch trainer for Vertex AI or local use.
# Reads from local or gs:// paths, writes model + metrics + logs together under AIP_MODEL_DIR.
# Mirrors the I/O conventions of the XGBoost example (safe for CustomTrainingJob use).

import os
import io
import json
import argparse
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Google Cloud Storage helpers
# -----------------------------
try:
    from google.cloud import storage
    _HAS_GCS = True
except Exception:
    _HAS_GCS = False  # Don't fail import if running locally

def _is_gcs(p: str) -> bool:
    """Return True if path starts with gs://."""
    return str(p).startswith("gs://")

def _gcs_client():
    """Initialize a GCS client or raise error if unavailable."""
    if not _HAS_GCS:
        raise RuntimeError("google-cloud-storage is required for gs:// I/O.")
    return storage.Client()

def _split_gs(p: str):
    """Split a GCS URI into (bucket, key). Example: gs://my-bucket/path/to/file -> ('my-bucket', 'path/to/file')"""
    bucket, _, key = p[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid GCS URI: {p}")
    return bucket, key

def read_npz_any(path: str) -> dict:
    """Read a .npz file from either local disk or GCS."""
    if _is_gcs(path):
        client = _gcs_client()
        bkt, key = _split_gs(path)
        data = client.bucket(bkt).blob(key).download_as_bytes()
        return np.load(io.BytesIO(data))
    return np.load(path)

def _makedirs_local(p: str):
    """Create local directories if they don’t exist."""
    Path(p or ".").mkdir(parents=True, exist_ok=True)

def save_bytes_any(data: bytes, dest_path: str):
    """Write binary data to local or GCS destination."""
    if _is_gcs(dest_path):
        client = _gcs_client()
        bkt, key = _split_gs(dest_path)
        client.bucket(bkt).blob(key).upload_from_string(data)
    else:
        _makedirs_local(os.path.dirname(dest_path))
        with open(dest_path, "wb") as f:
            f.write(data)

def save_text_any(text: str, dest_path: str):
    """Write UTF-8 text to local or GCS destination."""
    save_bytes_any(text.encode("utf-8"), dest_path)

def save_torch_model_any(model: nn.Module, dest_path: str):
    """Save a PyTorch model (state_dict) locally or to GCS via temporary staging file."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "model.pt"
        torch.save(model.state_dict(), tmp)
        if _is_gcs(dest_path):
            client = _gcs_client()
            bkt, key = _split_gs(dest_path)
            client.bucket(bkt).blob(key).upload_from_filename(str(tmp))
        else:
            _makedirs_local(os.path.dirname(dest_path))
            Path(dest_path).write_bytes(tmp.read_bytes())

# -----------------------------
# Model definition
# -----------------------------
class TitanicNet(nn.Module):
    """Simple feed-forward neural network for binary survival prediction."""
    def __init__(self, d_in=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),   # Input layer → hidden 1
            nn.Linear(64, 32), nn.ReLU(),     # Hidden 1 → hidden 2
            nn.Linear(32, 1), nn.Sigmoid(),   # Output layer → probability (0–1)
        )
    def forward(self, x):
        return self.net(x)

def accuracy_binary(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute simple binary accuracy (rounded predictions vs labels)."""
    return ((preds.round() == labels).float().mean().item())

# -----------------------------
# Training entry point
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Train TitanicNet on .npz data (local or gs://)")
    ap.add_argument("--train", required=True, help="Path to training npz (local or gs://)")
    ap.add_argument("--val",   required=True, help="Path to validation npz (local or gs://)")
    ap.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    ap.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    args = ap.parse_args()

    # Vertex AI convention: AIP_MODEL_DIR is where artifacts go.
    # Fallback to current directory for local runs.
    output_dir = os.environ.get("AIP_MODEL_DIR", ".")
    model_path = os.path.join(output_dir, "model.pt")
    metrics_path = os.path.join(output_dir, "metrics.json")
    history_path = os.path.join(output_dir, "eval_history.csv")
    log_path = os.path.join(output_dir, "training.log")

    # Capture stdout/stderr to both console and training.log
    import sys
    log_buf = io.StringIO()
    class _Tee:
        """Duplicated stream writer for capturing logs."""
        def __init__(self, *streams): self.streams = streams
        def write(self, d): 
            for s in self.streams: s.write(d); s.flush()
        def flush(self):
            for s in self.streams: s.flush()
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(sys.stdout, log_buf)
    sys.stderr = _Tee(sys.stderr, log_buf)

    try:
        # -----------------------------
        # Load pre-split .npz datasets
        # -----------------------------
        tr = read_npz_any(args.train)
        va = read_npz_any(args.val)
        Xtr, ytr = tr["X_train"].astype("float32"), tr["y_train"].astype("float32")
        Xva, yva = va["X_val"].astype("float32"),   va["y_val"].astype("float32")

        # Choose device (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert numpy arrays → PyTorch tensors and move to device
        Xtr_t = torch.from_numpy(Xtr).to(device)
        ytr_t = torch.from_numpy(ytr).view(-1, 1).to(device)
        Xva_t = torch.from_numpy(Xva).to(device)
        yva_t = torch.from_numpy(yva).view(-1, 1).to(device)

        # -----------------------------
        # Initialize model, optimizer, loss
        # -----------------------------
        model = TitanicNet(d_in=Xtr.shape[1]).to(device)
        opt = optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_fn = nn.BCELoss()  # Binary cross-entropy loss

        hist = []  # track val_loss over epochs

        # -----------------------------
        # Training loop
        # -----------------------------
        for ep in range(1, args.epochs + 1):
            model.train()  # enable dropout/batchnorm (not used here but good habit)
            opt.zero_grad()  # reset gradients
            pred = model(Xtr_t)  # forward pass
            loss = loss_fn(pred, ytr_t)  # compute training loss
            loss.backward()  # backpropagation
            opt.step()       # optimizer step (parameter update)

            # Evaluate on validation data
            model.eval()
            with torch.no_grad():
                val_pred = model(Xva_t)
                val_loss = loss_fn(val_pred, yva_t).item()
                val_acc  = accuracy_binary(val_pred, yva_t)
            hist.append(val_loss)

            # Print progress every 10 epochs (or first/last)
            if ep == 1 or ep % 10 == 0 or ep == args.epochs:
                print(f"epoch={ep} loss={loss.item():.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

        # -----------------------------
        # Save artifacts: model + metrics + logs
        # -----------------------------
        save_torch_model_any(model, model_path)
        print(f"[INFO] Saved model: {model_path}")

        # Prepare metrics.json
        metrics = {
            "final_val_loss": float(hist[-1]) if hist else None,
            "final_val_accuracy": float(val_acc),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "train_rows": int(Xtr.shape[0]),
            "val_rows": int(Xva.shape[0]),
            "features": list(range(Xtr.shape[1])),
            "model_uri": model_path,
            "device": str(device),
        }
        save_text_any(json.dumps(metrics, indent=2), metrics_path)

        # Save validation history as CSV
        csv = "iter,val_loss\n" + "\n".join(f"{i+1},{v}" for i, v in enumerate(hist))
        save_text_any(csv, history_path)

    finally:
        # Save training.log even if job crashes early
        try:
            save_text_any(log_buf.getvalue(), log_path)
        except Exception as e:
            print(f"[WARN] Could not write log: {e}", file=orig_err)
        sys.stdout, sys.stderr = orig_out, orig_err  # restore original streams

    # Final summary for the console
    print(f"Artifacts written to: {output_dir}")
    print(f"- model:   {model_path}")
    print(f"- metrics: {metrics_path}")
    print(f"- history: {history_path}")
    print(f"- log:     {log_path}")

if __name__ == "__main__":
    main()
