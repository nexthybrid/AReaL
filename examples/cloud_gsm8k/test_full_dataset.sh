#!/bin/bash
# Script to test trained model on full GSM8K dataset (1319 samples)
# 
# Usage:
#   bash examples/cloud_gsm8k/test_full_dataset.sh [checkpoint_path] [log_file]
#
#   checkpoint_path: Optional. Full path to model checkpoint directory.
#                    If not provided, will try to extract from log_file or find latest checkpoint.
#   log_file: Optional. Path to training log file to extract checkpoint path from.
#             If not provided, will try to find latest checkpoint automatically.
#
# Examples:
#   # Auto-detect from latest checkpoint
#   bash examples/cloud_gsm8k/test_full_dataset.sh
#
#   # Extract from specific log file
#   bash examples/cloud_gsm8k/test_full_dataset.sh "" examples/cloud_gsm8k/train_logs/logs_grpo_1k_v3_15epochs.txt
#
#   # Use specific checkpoint
#   bash examples/cloud_gsm8k/test_full_dataset.sh /workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-2gpu-1000samples-v3-conservative/trial_20251201_091022/default/epoch14epochstep249globalstep3749
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
CHECKPOINT_PATH="${1:-}"
LOG_FILE="${2:-}"
BATCH_SIZE="${TEST_BATCH_SIZE:-32}"  # Optimized for A100 80GB - can handle 32-64 easily
MAX_NEW_TOKENS=512

# Check if checkpoint path is provided
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Using provided checkpoint path: $CHECKPOINT_PATH"
    MODEL_PATH="$CHECKPOINT_PATH"
elif [ -n "$LOG_FILE" ]; then
    echo "Extracting checkpoint path from log file: $LOG_FILE"
    # Try multiple patterns to extract checkpoint path
    if grep -q "Testing TRAINED model:" "$LOG_FILE" 2>/dev/null; then
        # Extract from test log format: "Testing TRAINED model: /path/to/checkpoint"
        MODEL_PATH=$(grep "Testing TRAINED model:" "$LOG_FILE" | head -1 | sed 's/.*Testing TRAINED model: //' | tr -d '[:space:]')
        echo "Extracted checkpoint path from test log: $MODEL_PATH"
    elif grep -q "Trained model checkpoint:" "$LOG_FILE" 2>/dev/null; then
        # Extract from training log format: "Trained model checkpoint: /path/to/checkpoint"
        MODEL_PATH=$(grep "Trained model checkpoint:" "$LOG_FILE" | head -1 | sed 's/.*Trained model checkpoint: //' | tr -d '[:space:]')
        echo "Extracted checkpoint path from training log: $MODEL_PATH"
    elif grep -q "default/epoch" "$LOG_FILE" 2>/dev/null; then
        # Extract any checkpoint path containing "default/epoch"
        MODEL_PATH=$(grep -o "/workspace/outputs/grpo/checkpoints/root/[^[:space:]]*/default/epoch[^[:space:]]*" "$LOG_FILE" | head -1)
        echo "Extracted checkpoint path from epoch pattern: $MODEL_PATH"
    else
        echo "ERROR: Could not find checkpoint path in log file."
        echo "Please provide checkpoint path as first argument."
        exit 1
    fi
else
    echo "No checkpoint path or log file provided. Attempting to find latest checkpoint..."
    
    # Try to find latest checkpoint from common experiment names
    CHECKPOINT_BASE="/workspace/outputs/grpo/checkpoints/root"
    
    if [ ! -d "$CHECKPOINT_BASE" ]; then
        echo "ERROR: Checkpoint base directory not found: $CHECKPOINT_BASE"
        echo "Please provide checkpoint path as first argument."
        exit 1
    fi
    
    # Find latest checkpoint (most recent epoch)
    LATEST_CHECKPOINT=$(find "$CHECKPOINT_BASE" -type d -path "*/default/epoch*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "ERROR: Could not find any checkpoint in $CHECKPOINT_BASE"
        echo "Please provide checkpoint path as first argument."
        exit 1
    fi
    
    MODEL_PATH="$LATEST_CHECKPOINT"
    echo "Found latest checkpoint: $MODEL_PATH"
fi

# Validate checkpoint exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Checkpoint directory does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: Checkpoint directory does not contain config.json: $MODEL_PATH"
    echo "This may not be a valid HuggingFace checkpoint."
    exit 1
fi

echo ""
echo "=================================================================================="
echo "FULL DATASET TEST CONFIGURATION"
echo "=================================================================================="
echo "Checkpoint: $MODEL_PATH"
echo "Dataset: GSM8K test set (1319 samples)"
echo "Batch Size: $BATCH_SIZE (optimized for A100 80GB)"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "=================================================================================="
echo ""

# Determine if this is a reasoning model or standard GRPO model
# Check checkpoint path for "reasoning" keyword
if [[ "$MODEL_PATH" == *"reasoning"* ]]; then
    echo "Detected reasoning model. Using reasoning test script..."
    TEST_SCRIPT="examples/cloud_gsm8k/test_reasoning_model_cloud.py"
    MAX_NEW_TOKENS=1024  # Reasoning models need more tokens
else
    echo "Detected standard GRPO model. Using standard test script..."
    TEST_SCRIPT="examples/cloud_gsm8k/test_trained_model_cloud.py"
fi

# Extract model name from checkpoint path for logging
MODEL_NAME=$(basename "$(dirname "$(dirname "$MODEL_PATH")")")
if [ -z "$MODEL_NAME" ] || [ "$MODEL_NAME" == "." ]; then
    MODEL_NAME="Trained"
fi

echo ""
echo "Starting full dataset test..."
echo "This may take 10-30 minutes depending on GPU and batch size..."
echo ""

# Run test on full dataset
python3 "$TEST_SCRIPT" \
    --model-path "$MODEL_PATH" \
    --all \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --model-name "$MODEL_NAME" \
    --batch-size "$BATCH_SIZE"

TEST_EXIT_CODE=$?

echo ""
echo "=================================================================================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ Full dataset test completed successfully!"
    echo "Check log file in /workspace/outputs/grpo/test_logs/ for detailed results."
else
    echo "⚠️ Test completed with exit code: $TEST_EXIT_CODE"
    echo "Check log file in /workspace/outputs/grpo/test_logs/ for details."
fi
echo "=================================================================================="

exit $TEST_EXIT_CODE

