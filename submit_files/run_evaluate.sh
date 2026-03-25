#!/bin/bash
# Wrapper script for model evaluation on CHTC
set -euo pipefail

pip install --quiet -r requirements_xgboost.txt
python evaluate_model.py

echo "Evaluation complete."
