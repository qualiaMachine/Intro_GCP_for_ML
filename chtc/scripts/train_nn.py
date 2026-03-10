#!/usr/bin/env python3
"""
PyTorch trainer for the Titanic dataset.

Trains a small neural network (TitanicNet) with early stopping and optional
best-weight restore. All file I/O uses plain local paths — HTCondor handles
file transfer to/from workers.

Supports HTCondor self-checkpointing via --checkpoint_every and
--checkpoint_exit_code flags. When a checkpoint is due, the script saves
full training state (model, optimizer, epoch, best weights, history) and
exits with the specified code (default 85). HTCondor transfers the checkpoint
file back to the submit node, then restarts the job on the same or a different
machine. On restart, the script detects the checkpoint and resumes training
from where it left off.
"""

import os
import json
import argparse
import sys
import time
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

CHECKPOINT_FILE = "checkpoint.pt"


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


def save_checkpoint(path, model, optimizer, epoch, hist, best_val, best_epoch,
                    epochs_no_improve, best_state_dict):
    """Save full training state for resumption."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "hist": hist,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "epochs_no_improve": epochs_no_improve,
        "best_state_dict": best_state_dict,
    }, path)
    print(f"[CHECKPOINT] Saved training state at epoch {epoch} to {path}")


def load_checkpoint(path, model, optimizer, device):
    """Load training state from a checkpoint file."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return (
        ckpt["epoch"],
        ckpt["hist"],
        ckpt["best_val"],
        ckpt["best_epoch"],
        ckpt["epochs_no_improve"],
        ckpt["best_state_dict"],
    )


def main():
    ap = argparse.ArgumentParser(description="Train TitanicNet on .npz data (CHTC version)")
    ap.add_argument("--train", required=True, help="Path to training npz file")
    ap.add_argument("--val", required=True, help="Path to validation npz file")
    ap.add_argument("--test", type=str, default=None,
                    help="Path to test npz file (for final evaluation after training)")
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
    ap.add_argument("--combine_train_val", action="store_true",
                    help="Combine train+val data for final retraining (uses fixed epoch count, no early stopping)")
    # Checkpointing arguments
    ap.add_argument("--checkpoint_every", type=int, default=0,
                    help="Save checkpoint and exit every N seconds (0=disabled). "
                         "For CHTC, set to 3600 for hourly checkpoints.")
    ap.add_argument("--checkpoint_exit_code", type=int, default=85,
                    help="Exit code to signal HTCondor to restart from checkpoint (default 85)")
    args = ap.parse_args()

    # Output paths
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pt")
    metrics_path = os.path.join(output_dir, "metrics.json")
    history_path = os.path.join(output_dir, "eval_history.csv")
    checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILE)

    # Load pre-split .npz datasets
    tr = np.load(args.train)
    va = np.load(args.val)
    Xtr, ytr = tr["X_train"].astype("float32"), tr["y_train"].astype("float32")
    Xva, yva = va["X_val"].astype("float32"), va["y_val"].astype("float32")

    # For final retraining: combine train+val into one training set
    if args.combine_train_val:
        print("[INFO] Combining train + val data for final retraining")
        Xtr = np.concatenate([Xtr, Xva], axis=0)
        ytr = np.concatenate([ytr, yva], axis=0)
        # Val set is still used for monitoring but NOT for early stopping
        print(f"[INFO] Combined training set: {Xtr.shape[0]} rows")

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
    start_epoch = 1

    # Resume from checkpoint if available
    if os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] Found {checkpoint_path}, resuming training...")
        start_epoch_prev, hist, best_val, best_epoch, epochs_no_improve, best_state_dict = \
            load_checkpoint(checkpoint_path, model, opt, device)
        start_epoch = start_epoch_prev + 1
        print(f"[CHECKPOINT] Resuming from epoch {start_epoch} "
              f"(best_val={best_val:.6f} at epoch {best_epoch})")

    # Track wall-clock time for checkpointing
    wall_start = time.monotonic()

    # Training loop with early stopping
    for ep in range(start_epoch, args.epochs + 1):
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

        if ep == start_epoch or ep % 10 == 0 or ep == args.epochs or improved:
            print(f"epoch={ep} loss={loss.item():.4f} val_loss={val_loss:.4f} "
                  f"val_acc={val_acc:.4f}", flush=True)

        if not args.combine_train_val and epochs_no_improve >= args.patience:
            print(f"[EARLY STOP] No improvement in val_loss for {args.patience} "
                  f"epochs (best at epoch {best_epoch}).", flush=True)
            break

        # Checkpoint if wall-clock timeout is approaching
        if args.checkpoint_every > 0:
            elapsed = time.monotonic() - wall_start
            if elapsed >= args.checkpoint_every:
                save_checkpoint(checkpoint_path, model, opt, ep, hist,
                                best_val, best_epoch, epochs_no_improve,
                                best_state_dict)
                print(f"[CHECKPOINT] Exiting with code {args.checkpoint_exit_code} "
                      f"after {elapsed:.0f}s for HTCondor restart.")
                sys.exit(args.checkpoint_exit_code)

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

    # Clean up checkpoint file — training is complete
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("[CHECKPOINT] Removed checkpoint file (training complete)")

    # Optional: evaluate on held-out test set
    if args.test:
        te = np.load(args.test)
        Xte = te["X_test"].astype("float32")
        yte = te["y_test"].astype("float32")
        Xte_t = torch.from_numpy(Xte).to(device)
        yte_t = torch.from_numpy(yte).view(-1, 1).to(device)

        model.eval()
        with torch.no_grad():
            test_pred = model(Xte_t)
            test_loss = loss_fn(test_pred, yte_t).item()
            test_acc = accuracy_binary(test_pred, yte_t)

        metrics["test_loss"] = float(test_loss)
        metrics["test_accuracy"] = float(test_acc)
        metrics["test_rows"] = int(Xte.shape[0])

        # Re-save metrics with test results
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\ntest_loss: {test_loss:.6f}")
        print(f"test_accuracy: {test_acc:.6f}")

    print(f"\nArtifacts written to: {output_dir}")
    print(f"- model:   {model_path}")
    print(f"- metrics: {metrics_path}")
    print(f"- history: {history_path}")


if __name__ == "__main__":
    main()
