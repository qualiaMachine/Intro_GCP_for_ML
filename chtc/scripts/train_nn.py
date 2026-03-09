#!/usr/bin/env python3
"""
CHTC-simplified PyTorch trainer for the Titanic dataset.

This is a streamlined version of the GCP workshop's train_nn.py with
cloud-specific code (GCS, Vertex AI, hypertune) removed. All file I/O
is plain local paths — HTCondor handles file transfer to/from workers.
Includes early stopping with optional best-weight restore.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
    return (preds.round() == labels).float().mean().item()


def main():
    ap = argparse.ArgumentParser(description="Train TitanicNet on .npz data (CHTC version)")
    ap.add_argument("--train", required=True, help="Path to training npz file")
    ap.add_argument("--val", required=True, help="Path to validation npz file")
    ap.add_argument("--output_dir", type=str, default=".",
                    help="Directory for output artifacts")
    ap.add_argument("--epochs", type=int, default=1000, help="Max number of epochs")
    ap.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--patience", type=int, default=10,
                    help="Epochs to wait for val_loss improvement")
    ap.add_argument("--min_delta", type=float, default=1e-3,
                    help="Minimum improvement in val_loss to reset patience")
    ap.add_argument("--restore_best", type=lambda s: s.lower() in {"1", "true", "yes", "y"},
                    default=True, help="Restore best weights before saving (default True)")
    args = ap.parse_args()

    # Output paths
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pt")
    metrics_path = os.path.join(output_dir, "metrics.json")
    history_path = os.path.join(output_dir, "eval_history.csv")

    # Load pre-split .npz datasets
    tr = np.load(args.train)
    va = np.load(args.val)
    Xtr, ytr = tr["X_train"].astype("float32"), tr["y_train"].astype("float32")
    Xva, yva = va["X_val"].astype("float32"), va["y_val"].astype("float32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).view(-1, 1).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva).view(-1, 1).to(device)

    # Initialize model, optimizer, loss
    model = TitanicNet(d_in=Xtr.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCELoss()

    hist = []
    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state_dict = None

    # Training loop with early stopping
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
            val_acc = accuracy_binary(val_pred, yva_t)

        hist.append(val_loss)

        improved = (best_val - val_loss) > args.min_delta
        if improved:
            best_val = val_loss
            best_epoch = ep
            epochs_no_improve = 0
            best_state_dict = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if ep == 1 or ep % 10 == 0 or ep == args.epochs or improved:
            print(f"epoch={ep} loss={loss.item():.4f} val_loss={val_loss:.4f} "
                  f"val_acc={val_acc:.4f}", flush=True)

        if epochs_no_improve >= args.patience:
            print(f"[EARLY STOP] No improvement in val_loss for {args.patience} "
                  f"epochs (best at epoch {best_epoch}).", flush=True)
            break

    final_epoch = ep

    # Optionally restore best weights before saving
    if args.restore_best and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        final_epoch = best_epoch

    # Recompute final val metrics with the weights we are saving
    model.eval()
    with torch.no_grad():
        val_pred = model(Xva_t)
        final_val_loss = loss_fn(val_pred, yva_t).item()
        final_val_acc = accuracy_binary(val_pred, yva_t)

    print(f"validation_accuracy: {final_val_acc:.6f}", flush=True)
    print(f"validation_loss: {final_val_loss:.6f}", flush=True)

    # Save artifacts
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Saved model: {model_path}")

    metrics = {
        "final_val_loss": float(final_val_loss),
        "final_val_accuracy": float(final_val_acc),
        "best_val_loss": float(best_val) if best_val < float("inf") else None,
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
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    csv = "iter,val_loss\n" + "\n".join(f"{i+1},{v}" for i, v in enumerate(hist))
    with open(history_path, "w") as f:
        f.write(csv)

    print(f"\nArtifacts written to: {output_dir}")
    print(f"- model:   {model_path}")
    print(f"- metrics: {metrics_path}")
    print(f"- history: {history_path}")


if __name__ == "__main__":
    main()
