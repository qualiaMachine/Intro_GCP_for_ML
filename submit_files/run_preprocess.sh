#!/bin/bash
# Wrapper script for data preprocessing on CHTC
set -euo pipefail

pip install --quiet pandas scikit-learn numpy
python preprocess_data.py

echo "Preprocessing complete. Output files:"
ls -la *.npz
