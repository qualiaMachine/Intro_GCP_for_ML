# GCP-friendly PyTorch trainer for Vertex AI or local use.
# Reads from local or gs:// paths, writes model + metrics + logs together under AIP_MODEL_DIR.
# Now includes early stopping with optional best-weight restore.

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
import random

# -----------------------------
# Set seeds for reproducibility
# -----------------------------
SEED = 42                      # Fixed seed value (can be changed if needed)
np.random.seed(SEED)            # Ensures NumPy operations give consistent results
random.seed(SEED)               # Makes Python's random module reproducible
torch.manual_seed(SEED)         # Sets the seed for CPU-based PyTorch operations
torch.cuda.manual_seed_all(SEED)  # Ensures reproducibility across all available GPUs
torch.backends.cudnn.deterministic = True  # Forces deterministic behavior for CuDNN ops (slower but consistent)
torch.backends.cudnn.benchmark = False     # Disables auto-optimizations that may change results between runs

# -----------------------------
# Google Cloud Storage helpers
# -----------------------------
try:
    from google.cloud import storage
    _HAS_GCS = True
except Exception:
    _HAS_GCS = False  # Don't fail import if running locally

def _is_gcs(p: str) -> bool:
    return str(p).startswith("gs://")

def _gcs_client():
    if not _HAS_GCS:
        raise RuntimeError("google-cloud-storage is required for gs:// I/O.")
    return storage.Client()

def _split_gs(p: str):
    bucket, _, key = p[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid GCS URI: {p}")
    return bucket, key

def read_npz_any(path: str) -> dict:
    if _is_gcs(path):
        client = _gcs_client()
        bkt, key = _split_gs(path)
        data = client.bucket(bkt).blob(key).download_as_bytes()
        return np.load(io.BytesIO(data))
    return np.load(path)

def _makedirs_local(p: str):
    Path(p or ".").mkdir(parents=True, exist_ok=True)

def save_bytes_any(data: bytes, dest_path: str):
    if _is_gcs(dest_path):
        client = _gcs_client()
        bkt, key = _split_gs(dest_path)
        client.bucket(bkt).blob(key).upload_from_string(data)
    else:
        _makedirs_local(os.path.dirname(dest_path))
        with open(dest_path, "wb") as f:
            f.write(data)

def save_text_any(text: str, dest_path: str):
    save_bytes_any(text.encode("utf-8"), dest_path)

def save_torch_model_any(model: nn.Module, dest_path: str):
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
    def __init__(self, d_in=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

def accuracy_binary(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return ((preds.round() == labels).float().mean().item())

# -----------------------------
# Training entry point
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Train TitanicNet on .npz data (local or gs://)")
    ap.add_argument("--train", required=True, help="Path to training npz (local or gs://)")
    ap.add_argument("--val",   required=True, help="Path to validation npz (local or gs://)")
    ap.add_argument("--epochs", type=int, default=1000, help="Max number of epochs")
    ap.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    # Early stopping args
    ap.add_argument("--patience", type=int, default=10, help="Epochs to wait for val_loss improvement")
    ap.add_argument("--min_delta", type=float, default=1e-3, help="Minimum improvement in val_loss to reset patience")
    ap.add_argument("--restore_best", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True,
                    help="Restore best weights before saving (default True)")
    args = ap.parse_args()

    # Vertex AI convention: AIP_MODEL_DIR is where artifacts go (fallback "." for local runs)
    output_dir = os.environ.get("AIP_MODEL_DIR", ".")
    model_path = os.path.join(output_dir, "model.pt")
    metrics_path = os.path.join(output_dir, "metrics.json")
    history_path = os.path.join(output_dir, "eval_history.csv")
    log_path = os.path.join(output_dir, "training.log")

    # Capture stdout/stderr to both console and training.log
    import sys
    log_buf = io.StringIO()
    class _Tee:
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Xtr_t = torch.from_numpy(Xtr).to(device)
        ytr_t = torch.from_numpy(ytr).view(-1, 1).to(device)
        Xva_t = torch.from_numpy(Xva).to(device)
        yva_t = torch.from_numpy(yva).view(-1, 1).to(device)

        # -----------------------------
        # Initialize model, optimizer, loss
        # -----------------------------
        model = TitanicNet(d_in=Xtr.shape[1]).to(device)
        opt = optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_fn = nn.BCELoss()

        hist = []            # validation loss per epoch
        best_val = float("inf")
        best_epoch = 0
        epochs_no_improve = 0
        best_state_dict = None

        # -----------------------------
        # Training loop with early stopping
        # -----------------------------
        for ep in range(1, args.epochs + 1):
            model.train()
            opt.zero_grad()
            pred = model(Xtr_t)
            loss = loss_fn(pred, ytr_t)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(Xva_t)
                val_loss = loss_fn(val_pred, yva_t).item()
                val_acc  = accuracy_binary(val_pred, yva_t)

            hist.append(val_loss)

            improved = (best_val - val_loss) > args.min_delta
            if improved:
                best_val = val_loss
                best_epoch = ep
                epochs_no_improve = 0
                # Keep a copy of the best weights in memory
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1


            if ep == 1 or ep % 10 == 0 or ep == args.epochs or improved:
                    
                print(f"epoch={ep} loss={loss.item():.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

                # print these on their own lines for HyperparamterTuning jobs to detect
                print(f"validation_loss: {val_loss:.6f}", flush=True)
                print(f"validation_accuracy: {val_acc:.6f}", flush=True)


            if epochs_no_improve >= args.patience:
                print(f"[EARLY STOP] No improvement in val_loss for {args.patience} epochs (best at epoch {best_epoch}).", flush=True)
                break

        final_epoch = ep  # last executed epoch (may be early-stopped)

        # Optionally restore best weights before saving
        if args.restore_best and best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            final_val = best_val
            final_epoch = best_epoch
        else:
            final_val = hist[-1] if hist else None

        # Recompute final val metrics with the weights we are saving
        model.eval()
        with torch.no_grad():
            val_pred = model(Xva_t)
            final_val_loss = loss_fn(val_pred, yva_t).item()
            final_val_acc  = accuracy_binary(val_pred, yva_t)

        #  Guarantee Vertex sees a compliant metric line 
        print(f"validation_accuracy: {final_val_acc:.6f}", flush=True)
        print(f"validation_loss: {final_val_loss:.6f}", flush=True)


        # -----------------------------
        # Save artifacts: model + metrics + logs
        # -----------------------------
        save_torch_model_any(model, model_path)
        print(f"[INFO] Saved model: {model_path}")

        metrics = {
            "final_val_loss": float(final_val_loss),
            "final_val_accuracy": float(final_val_acc),
            "best_val_loss": float(best_val) if best_val < float('inf') else None,
            "best_epoch": int(best_epoch),
            "stopped_epoch": int(final_epoch),
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "restore_best": bool(args.restore_best),
            "max_epochs_requested": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "train_rows": int(Xtr.shape[0]),
            "val_rows": int(Xva.shape[0]),
            "features": list(range(Xtr.shape[1])),
            "model_uri": model_path,
            "device": str(device),
        }
        save_text_any(json.dumps(metrics, indent=2), metrics_path)

        csv = "iter,val_loss\n" + "\n".join(f"{i+1},{v}" for i, v in enumerate(hist))
        save_text_any(csv, history_path)

    finally:
        try:
            save_text_any(log_buf.getvalue(), log_path)
        except Exception as e:
            print(f"[WARN] Could not write log: {e}", file=orig_err)
        sys.stdout, sys.stderr = orig_out, orig_err

    print(f"Artifacts written to: {output_dir}")
    print(f"- model:   {model_path}")
    print(f"- metrics: {metrics_path}")
    print(f"- history: {history_path}")
    print(f"- log:     {log_path}")

if __name__ == "__main__":
    main()
