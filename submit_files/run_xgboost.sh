#!/bin/bash
# Wrapper script for XGBoost training on CHTC
# This script installs dependencies and runs the training script.

set -euo pipefail

# Install Python dependencies
pip install --quiet -r requirements_xgboost.txt

# Run the training script, passing all arguments through
python train_xgboost.py "$@"

echo "Training complete. Output files:"
ls -la
