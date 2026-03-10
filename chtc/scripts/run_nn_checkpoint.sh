#!/bin/bash
# run_nn_checkpoint.sh — Wrapper for PyTorch training with HTCondor checkpointing.
#
# This wrapper enables checkpoint-based restarts:
# - The training script saves a checkpoint every --checkpoint_every seconds
# - It exits with code 85, signaling HTCondor to transfer files and restart
# - On restart, the script detects checkpoint.pt and resumes training

set -e

echo "=== CHTC PyTorch Training (with checkpointing) ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Files: $(ls -la)"
echo ""

# Check for existing checkpoint (indicates a restart)
if [ -f checkpoint.pt ]; then
    echo "[RESUME] Found checkpoint.pt — will resume from saved state"
else
    echo "[START] No checkpoint found — starting fresh"
fi
echo ""

python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

# --checkpoint_every=3600 → save checkpoint and exit every hour
# HTCondor will transfer checkpoint.pt back and restart the job
python3 train_nn.py \
    --train train_data.npz \
    --val val_data.npz \
    --output_dir . \
    --checkpoint_every 3600 \
    "$@"

echo ""
echo "=== Training complete ==="
echo "Output files: $(ls -la)"
