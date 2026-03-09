#!/bin/bash
# cleanup.sh — Clean up old job output files from CHTC workshop exercises.
#
# Usage:
#     bash cleanup.sh              # Dry run (shows what would be deleted)
#     bash cleanup.sh --execute    # Actually delete files

set -e

WORKSHOP_DIR="${1:-.}"
EXECUTE=false

if [[ "$1" == "--execute" ]] || [[ "$2" == "--execute" ]]; then
    EXECUTE=true
    echo "=== EXECUTING CLEANUP ==="
else
    echo "=== DRY RUN (add --execute to actually delete) ==="
fi

echo "Workshop directory: $WORKSHOP_DIR"
echo ""

# Count files to clean
LOG_FILES=$(find "$WORKSHOP_DIR" -name "*.log" -o -name "*.out" -o -name "*.err" | wc -l)
TRIAL_DIRS=$(find "$WORKSHOP_DIR" -maxdepth 1 -name "trial_*" -type d | wc -l)

echo "Found:"
echo "  - $LOG_FILES HTCondor log/out/err files"
echo "  - $TRIAL_DIRS trial directories"
echo ""

if $EXECUTE; then
    # Remove HTCondor output files
    find "$WORKSHOP_DIR" -maxdepth 1 \( -name "*.log" -o -name "*.out" -o -name "*.err" \) -delete
    echo "Removed HTCondor log/out/err files"

    # Remove trial directories from HP sweep
    find "$WORKSHOP_DIR" -maxdepth 1 -name "trial_*" -type d -exec rm -rf {} +
    echo "Removed trial directories"

    # Remove temporary model files
    find "$WORKSHOP_DIR" -maxdepth 1 \( -name "xgboost-model" -o -name "model.pt" -o -name "metrics.json" -o -name "eval_history.csv" \) -delete
    echo "Removed model artifacts"

    echo ""
    echo "Cleanup complete."
else
    echo "Files that would be removed:"
    find "$WORKSHOP_DIR" -maxdepth 1 \( -name "*.log" -o -name "*.out" -o -name "*.err" \) | head -10
    if [ $LOG_FILES -gt 10 ]; then
        echo "  ... and $((LOG_FILES - 10)) more"
    fi
    echo ""
    echo "Run with --execute to actually delete these files."
fi
