#!/bin/bash
# run_retrain.sh — Wrapper for retraining the best model on train+val and evaluating on test.
# HTCondor runs this script inside the container.

set -e

echo "=== CHTC Final Retrain + Test Evaluation ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Files: $(ls -la)"
echo ""

python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

python3 retrain_best.py \
    --config best_config.json \
    --train train_data.npz \
    --val val_data.npz \
    --test test_data.npz \
    --output_dir . \
    "$@"

echo ""
echo "=== Retrain + Evaluation complete ==="
echo "Output files: $(ls -la)"
