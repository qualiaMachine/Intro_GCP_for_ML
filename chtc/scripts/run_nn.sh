#!/bin/bash
# run_nn.sh — Wrapper script for PyTorch neural network training on CHTC.
# HTCondor runs this script inside the container; it calls the Python trainer.

set -e

echo "=== CHTC PyTorch Training ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Files: $(ls -la)"
echo ""

# Check for GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

python3 train_nn.py \
    --train train_data.npz \
    --val val_data.npz \
    --output_dir . \
    "$@"

echo ""
echo "=== Training complete ==="
echo "Output files: $(ls -la)"
