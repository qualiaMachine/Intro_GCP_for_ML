#!/bin/bash
# Wrapper script for PyTorch neural network training on CHTC
# This script runs the training script inside a PyTorch Docker container.

set -euo pipefail

# Run the training script, passing all arguments through
python train_nn.py "$@"

echo "Training complete. Output files:"
ls -la
