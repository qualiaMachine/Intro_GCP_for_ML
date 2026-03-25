"""Aggregate hyperparameter tuning results from multiple HTCondor jobs.

This script reads metrics.json files from a results directory and prints
a summary table sorted by validation accuracy. Used in Episode 06.

Usage:
    python aggregate_results.py --results-dir results/
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Aggregate HP tuning results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing metrics_*.json files",
    )
    args = parser.parse_args()

    results = []
    for fname in sorted(os.listdir(args.results_dir)):
        if fname.startswith("metrics_") and fname.endswith(".json"):
            path = os.path.join(args.results_dir, fname)
            with open(path) as f:
                data = json.load(f)
            data["file"] = fname
            results.append(data)

    if not results:
        print(f"No metrics_*.json files found in {args.results_dir}")
        sys.exit(1)

    # Sort by validation accuracy (descending)
    results.sort(key=lambda r: r.get("final_val_accuracy", 0), reverse=True)

    # Print summary table
    print(f"{'File':<40} {'Val Acc':>8} {'Val Loss':>9} {'LR':>10} {'Patience':>9} {'Best Epoch':>11}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['file']:<40} "
            f"{r.get('final_val_accuracy', 'N/A'):>8.4f} "
            f"{r.get('final_val_loss', 'N/A'):>9.4f} "
            f"{r.get('learning_rate', 'N/A'):>10.6f} "
            f"{r.get('patience', 'N/A'):>9} "
            f"{r.get('best_epoch', 'N/A'):>11}"
        )

    print(f"\nBest trial: {results[0]['file']}")
    print(f"  Validation accuracy: {results[0].get('final_val_accuracy', 'N/A'):.4f}")
    print(f"  Learning rate: {results[0].get('learning_rate', 'N/A')}")
    print(f"  Patience: {results[0].get('patience', 'N/A')}")


if __name__ == "__main__":
    main()
