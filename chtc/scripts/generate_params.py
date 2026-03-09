#!/usr/bin/env python3
"""
generate_params.py — Generate hyperparameter combinations for an HTCondor sweep.

Outputs a CSV file where each row is one trial's hyperparameters. HTCondor's
`queue from params.csv` syntax reads this file and launches one job per row.

Usage:
    python3 generate_params.py --output params.csv --mode grid
    python3 generate_params.py --output params.csv --mode random --n_trials 20
"""

import argparse
import csv
import itertools
import random


def grid_search():
    """Generate all combinations of a predefined grid."""
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
    patience_values = [10, 20, 30]
    min_delta_values = [1e-4, 1e-3]

    combos = list(itertools.product(learning_rates, patience_values, min_delta_values))
    return combos


def random_search(n_trials, seed=42):
    """Generate random hyperparameter combinations."""
    random.seed(seed)
    combos = []
    for _ in range(n_trials):
        lr = 10 ** random.uniform(-4, -2)  # 1e-4 to 1e-2
        patience = random.choice([5, 10, 15, 20, 25, 30])
        min_delta = 10 ** random.uniform(-4, -2)  # 1e-4 to 1e-2
        combos.append((round(lr, 6), patience, round(min_delta, 6)))
    return combos


def main():
    parser = argparse.ArgumentParser(description="Generate hyperparameter sweep CSV")
    parser.add_argument("--output", type=str, default="params.csv")
    parser.add_argument("--mode", choices=["grid", "random"], default="grid")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of random trials (only for --mode random)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "grid":
        combos = grid_search()
    else:
        combos = random_search(args.n_trials, args.seed)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        for combo in combos:
            writer.writerow(combo)

    print(f"Generated {len(combos)} hyperparameter combinations → {args.output}")
    print(f"Mode: {args.mode}")
    if args.mode == "grid":
        print("Grid dimensions: 4 learning_rates × 3 patience × 2 min_delta")


if __name__ == "__main__":
    main()
