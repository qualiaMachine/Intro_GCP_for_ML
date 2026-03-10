#!/usr/bin/env python3
"""
aggregate_results.py — Collect metrics.json from all HP sweep trial directories.

After an HTCondor hyperparameter sweep, each trial writes its results to a
separate directory (trial_0/, trial_1/, ...). This script reads all metrics.json
files, creates a summary DataFrame, and identifies the best trial.

Usage:
    python3 aggregate_results.py --results_dir . --output_csv hp_summary.csv
"""

import argparse
import json
import os
import glob

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Aggregate HP sweep results")
    parser.add_argument("--results_dir", type=str, default=".",
                        help="Directory containing trial_*/metrics.json files")
    parser.add_argument("--output_csv", type=str, default="hp_summary.csv",
                        help="Output summary CSV")
    parser.add_argument("--output_best", type=str, default="best_config.json",
                        help="Output best hyperparameters as JSON (for retraining)")
    args = parser.parse_args()

    # Find all metrics.json files
    pattern = os.path.join(args.results_dir, "trial_*", "metrics.json")
    metrics_files = sorted(glob.glob(pattern))

    if not metrics_files:
        print(f"No metrics.json files found matching: {pattern}")
        print("Looking for metrics.json in current directory...")
        # Also try flat structure (metrics files named by process ID)
        pattern = os.path.join(args.results_dir, "metrics_*.json")
        metrics_files = sorted(glob.glob(pattern))

    if not metrics_files:
        print("No metrics files found. Check your results directory.")
        return

    # Load all metrics
    rows = []
    for path in metrics_files:
        try:
            with open(path) as f:
                m = json.load(f)
            m["metrics_file"] = path
            rows.append(m)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: could not read {path}: {e}")

    if not rows:
        print("No valid metrics files found.")
        return

    df = pd.DataFrame(rows)

    # Sort by validation loss (ascending — lower is better)
    df = df.sort_values("final_val_loss", ascending=True).reset_index(drop=True)
    df.index.name = "rank"

    # Save summary
    df.to_csv(args.output_csv, index=True)
    print(f"\nSummary saved to: {args.output_csv}")
    print(f"Total trials: {len(df)}")

    # Show top results
    print("\n=== Top 5 Trials (by validation loss) ===")
    cols = ["learning_rate", "patience", "min_delta", "final_val_loss",
            "final_val_accuracy", "best_epoch", "stopped_epoch"]
    display_cols = [c for c in cols if c in df.columns]
    print(df[display_cols].head().to_string())

    # Best trial
    best = df.iloc[0]
    print(f"\n=== Best Trial ===")
    print(f"  Learning rate: {best.get('learning_rate', 'N/A')}")
    print(f"  Patience:      {best.get('patience', 'N/A')}")
    print(f"  Min delta:     {best.get('min_delta', 'N/A')}")
    print(f"  Val loss:      {best.get('final_val_loss', 'N/A'):.6f}")
    print(f"  Val accuracy:  {best.get('final_val_accuracy', 'N/A'):.6f}")
    print(f"  Best epoch:    {best.get('best_epoch', 'N/A')}")
    print(f"  Metrics file:  {best.get('metrics_file', 'N/A')}")

    # Save best config as JSON for retraining
    best_config = {
        "learning_rate": float(best["learning_rate"]),
        "patience": int(best["patience"]),
        "min_delta": float(best["min_delta"]),
        "best_epoch": int(best["best_epoch"]),
        "final_val_loss": float(best["final_val_loss"]),
        "final_val_accuracy": float(best["final_val_accuracy"]),
    }
    with open(args.output_best, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nBest config saved to: {args.output_best}")
    print("Use this to retrain on train+val with: train_nn.py --combine_train_val --test test_data.npz")


if __name__ == "__main__":
    main()
