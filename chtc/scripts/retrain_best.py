#!/usr/bin/env python3
"""
retrain_best.py — Retrain TitanicNet with the best hyperparameters on train+val,
then evaluate on held-out test set.

Reads best_config.json (produced by aggregate_results.py) to get the winning
hyperparameters and optimal epoch count, then calls train_nn.py with
--combine_train_val to train on the full training data (train+val combined)
for the known-good number of epochs, and --test to get final test performance.

Usage:
    python3 retrain_best.py --config best_config.json
"""

import argparse
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Retrain with best HP config on train+val, evaluate on test"
    )
    parser.add_argument("--config", type=str, default="best_config.json",
                        help="Path to best_config.json from aggregate_results.py")
    parser.add_argument("--train", type=str, default="train_data.npz",
                        help="Path to training .npz")
    parser.add_argument("--val", type=str, default="val_data.npz",
                        help="Path to validation .npz")
    parser.add_argument("--test", type=str, default="test_data.npz",
                        help="Path to test .npz")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory for output artifacts")
    parser.add_argument("--epoch_multiplier", type=float, default=1.2,
                        help="Multiply best_epoch by this for retrain epoch count (default 1.2)")
    args = parser.parse_args()

    # Load best config
    with open(args.config) as f:
        config = json.load(f)

    print("=== Retraining with Best Configuration ===")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Patience:      {config['patience']}")
    print(f"  Min delta:     {config['min_delta']}")
    print(f"  Best epoch:    {config['best_epoch']}")
    print(f"  Sweep val loss:     {config['final_val_loss']:.6f}")
    print(f"  Sweep val accuracy: {config['final_val_accuracy']:.6f}")
    print()

    # Set epoch count: use best_epoch * multiplier (train on more data = may need slightly more epochs)
    retrain_epochs = max(int(config["best_epoch"] * args.epoch_multiplier), 10)
    print(f"  Retrain epochs: {retrain_epochs} (best_epoch={config['best_epoch']} × {args.epoch_multiplier})")
    print()

    # Build train_nn.py command
    cmd = [
        sys.executable, "train_nn.py",
        "--train", args.train,
        "--val", args.val,
        "--test", args.test,
        "--output_dir", args.output_dir,
        "--epochs", str(retrain_epochs),
        "--learning_rate", str(config["learning_rate"]),
        "--patience", str(config["patience"]),
        "--min_delta", str(config["min_delta"]),
        "--combine_train_val",
        "--restore_best", "false",
    ]

    print(f"Running: {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
