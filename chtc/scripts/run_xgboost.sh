#!/bin/bash
# run_xgboost.sh — Wrapper script for XGBoost training on CHTC.
# HTCondor runs this script inside the container; it calls the Python trainer.

set -e

echo "=== CHTC XGBoost Training ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Files: $(ls -la)"
echo ""

python3 train_xgboost.py \
    --train titanic_train.csv \
    --output_dir . \
    "$@"

echo ""
echo "=== Training complete ==="
echo "Output files: $(ls -la)"
