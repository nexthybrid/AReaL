#!/bin/bash
# Script to test ensemble of checkpoints using majority voting
# 
# Usage:
#   bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh [checkpoint_dir] [epochs]
#
#   checkpoint_dir: Base directory containing checkpoints
#                    (e.g., /workspace/outputs/grpo/checkpoints/root/experiment/trial/default/)
#   epochs: Space-separated list of epochs to test (default: 4 9 14 19 24)
#
# Examples:
#   # Test with default epochs (4, 9, 14, 19, 24)
#   bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh /workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-2gpu-1000samples-v3-conservative/trial0/default
#
#   # Test with custom epochs
#   bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh /workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-2gpu-1000samples-v3-conservative/trial0/default "4 9 14"
#
#   # Auto-detect from training log
#   bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh "" "" examples/cloud_gsm8k/train_logs/logs_25epochs.txt
#
# Batch Size:
#   - Default: 32 (optimized for A100 80GB)
#   - Can be overridden with TEST_BATCH_SIZE environment variable
#   - For smaller GPUs, reduce to 16 or 8

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CHECKPOINT_DIR="${1:-}"
EPOCHS="${2:-4 9 14 19 24}"
BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
MAX_NEW_TOKENS=512
LOG_FILE="${3:-}"

# Function to find checkpoint directory from log file
find_checkpoint_dir_from_log() {
    local log_file="$1"
    if [ -z "$log_file" ] || [ ! -f "$log_file" ]; then
        return 1
    fi
    
    # Try to extract checkpoint path from log
    # Look for patterns like "Checkpoint saved to: /path/to/checkpoint"
    local checkpoint_path=$(grep -i "checkpoint.*saved\|checkpoint.*path\|model.*checkpoint" "$log_file" | tail -1 | grep -oP '/workspace/outputs/grpo/checkpoints/[^\s]+' | head -1)
    
    if [ -n "$checkpoint_path" ]; then
        # Extract base directory (remove epoch-specific part)
        echo "$checkpoint_path" | sed 's|/epoch[0-9]*epochstep.*||'
        return 0
    fi
    
    return 1
}

# Function to find latest checkpoint directory
find_latest_checkpoint_dir() {
    local base_dir="/workspace/outputs/grpo/checkpoints"
    
    # Find most recently modified checkpoint directory
    local latest_dir=$(find "$base_dir" -type d -name "default" -path "*/checkpoints/*/default" 2>/dev/null | \
        xargs -I {} sh -c 'echo "$(stat -c %Y {}) {}"' 2>/dev/null | \
        sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -n "$latest_dir" ]; then
        echo "$latest_dir"
        return 0
    fi
    
    return 1
}

# Determine checkpoint directory
if [ -z "$CHECKPOINT_DIR" ]; then
    if [ -n "$LOG_FILE" ]; then
        echo "Attempting to extract checkpoint directory from log file: $LOG_FILE"
        CHECKPOINT_DIR=$(find_checkpoint_dir_from_log "$LOG_FILE")
    fi
    
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "Attempting to find latest checkpoint directory..."
        CHECKPOINT_DIR=$(find_latest_checkpoint_dir)
    fi
    
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "Error: Could not determine checkpoint directory."
        echo "Please provide checkpoint directory as first argument, or ensure log file contains checkpoint path."
        exit 1
    fi
fi

# Verify checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

echo "Using checkpoint directory: $CHECKPOINT_DIR"
echo "Testing epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo ""

# Run the ensemble testing script
python3 examples/cloud_gsm8k/test_checkpoint_ensemble.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --epochs $EPOCHS \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --log-dir "/workspace/outputs/grpo/test_logs"

echo ""
echo "Ensemble testing completed!"
echo "Check logs in /workspace/outputs/grpo/test_logs/"

