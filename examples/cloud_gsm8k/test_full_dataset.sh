#!/bin/bash
# Script to test trained model on full GSM8K dataset (1319 samples)
# 
# Usage:
#   bash examples/cloud_gsm8k/test_full_dataset.sh [checkpoint_path] [log_file]
#
#   checkpoint_path: Optional. Full path to model checkpoint directory OR "baseline" to test base model.
#                    If "baseline", tests Qwen/Qwen2.5-0.5B-Instruct on full dataset.
#                    If not provided, will try to extract from log_file or find latest checkpoint.
#   log_file: Optional. Path to training log file to extract checkpoint path from.
#             If not provided, will try to find latest checkpoint automatically.
#
# Examples:
#   # Test baseline model on full dataset
#   bash examples/cloud_gsm8k/test_full_dataset.sh baseline
#
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
#
# Auto-Upload Logs:
#   - Set AUTO_UPLOAD_LOGS_METHOD environment variable to enable auto-upload
#   - Supported methods: email, gdrive, s3, hf, wandb, webhook
#   - Method-specific environment variables (same as training script):
#     - Email: AUTO_UPLOAD_EMAIL_TO (required), EMAIL_FROM, SMTP_PASSWORD
#     - Google Drive: AUTO_UPLOAD_GDRIVE_FOLDER_ID
#     - S3: AUTO_UPLOAD_S3_BUCKET, AUTO_UPLOAD_S3_PREFIX (optional)
#     - Hugging Face: AUTO_UPLOAD_HF_REPO_ID
#     - W&B: AUTO_UPLOAD_WANDB_PROJECT, AUTO_UPLOAD_WANDB_RUN_NAME (optional)
#     - Webhook: AUTO_UPLOAD_WEBHOOK_URL, AUTO_UPLOAD_WEBHOOK_API_KEY (optional)
#   - Example: AUTO_UPLOAD_LOGS_METHOD=email AUTO_UPLOAD_EMAIL_TO=user@example.com bash test_full_dataset.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CHECKPOINT_PATH="${1:-}"
LOG_FILE="${2:-}"
BATCH_SIZE="${TEST_BATCH_SIZE:-32}"  # Optimized for A100 80GB - can handle 32-64 easily
MAX_NEW_TOKENS=512
IS_BASELINE=false
MODEL_NAME=""

# Check if checkpoint path is provided
if [ -n "$CHECKPOINT_PATH" ]; then
    # Special case: "baseline" means test the base model
    if [ "$CHECKPOINT_PATH" = "baseline" ]; then
        echo "Testing BASELINE model (Qwen/Qwen2.5-0.5B-Instruct) on full dataset..."
        MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
        MODEL_NAME="Baseline"
        IS_BASELINE=true
    else
        echo "Using provided checkpoint path: $CHECKPOINT_PATH"
        MODEL_PATH="$CHECKPOINT_PATH"
        IS_BASELINE=false
    fi
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

# Validate checkpoint exists (skip for baseline/HuggingFace model identifiers)
if [ "$IS_BASELINE" != "true" ]; then
    # Check if it's a HuggingFace model identifier (contains /)
    if [[ "$MODEL_PATH" == *"/"* ]] && [[ "$MODEL_PATH" != "/"* ]]; then
        # It's a HuggingFace model identifier, skip directory check
        echo "Using HuggingFace model identifier: $MODEL_PATH"
    elif [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: Checkpoint directory does not exist: $MODEL_PATH"
        exit 1
    elif [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "ERROR: Checkpoint directory does not contain config.json: $MODEL_PATH"
        echo "This may not be a valid HuggingFace checkpoint."
        exit 1
    fi
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
# Check checkpoint path for "reasoning" keyword (skip for baseline)
if [ "$IS_BASELINE" != "true" ] && [[ "$MODEL_PATH" == *"reasoning"* ]]; then
    echo "Detected reasoning model. Using reasoning test script..."
    TEST_SCRIPT="examples/cloud_gsm8k/test_reasoning_model_cloud.py"
    MAX_NEW_TOKENS=1024  # Reasoning models need more tokens
else
    echo "Detected standard GRPO model. Using standard test script..."
    TEST_SCRIPT="examples/cloud_gsm8k/test_trained_model_cloud.py"
fi

# Extract model name from checkpoint path for logging (if not already set)
if [ -z "$MODEL_NAME" ]; then
    # Check if it's a HuggingFace model identifier
    if [[ "$MODEL_PATH" == *"/"* ]] && [[ "$MODEL_PATH" != "/"* ]]; then
        MODEL_NAME="Baseline"
    else
        MODEL_NAME=$(basename "$(dirname "$(dirname "$MODEL_PATH")")")
        if [ -z "$MODEL_NAME" ] || [ "$MODEL_NAME" == "." ]; then
            MODEL_NAME="Trained"
        fi
    fi
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

# Auto-upload logs if configured (same as training script)
if [ -n "$AUTO_UPLOAD_LOGS_METHOD" ]; then
    # Normalize method to lowercase (handle EMAIL -> email, etc.)
    UPLOAD_METHOD=$(echo "$AUTO_UPLOAD_LOGS_METHOD" | tr '[:upper:]' '[:lower:]')
    
    echo ""
    echo "=========================================="
    echo "📤 Auto-uploading test logs via $UPLOAD_METHOD..."
    echo "=========================================="
    
    UPLOAD_SCRIPT="examples/cloud_gsm8k/upload_logs.py"
    UPLOAD_CMD="python3 $UPLOAD_SCRIPT --log-dir /workspace/outputs/grpo/test_logs --method $UPLOAD_METHOD --latest-only"
    
    # Add method-specific arguments from environment
    if [ "$UPLOAD_METHOD" = "email" ] && [ -n "$AUTO_UPLOAD_EMAIL_TO" ]; then
        UPLOAD_CMD="$UPLOAD_CMD --email-to $AUTO_UPLOAD_EMAIL_TO"
    elif [ "$UPLOAD_METHOD" = "gdrive" ] && [ -n "$AUTO_UPLOAD_GDRIVE_FOLDER_ID" ]; then
        UPLOAD_CMD="$UPLOAD_CMD --gdrive-folder-id $AUTO_UPLOAD_GDRIVE_FOLDER_ID"
    elif [ "$UPLOAD_METHOD" = "s3" ] && [ -n "$AUTO_UPLOAD_S3_BUCKET" ]; then
        UPLOAD_CMD="$UPLOAD_CMD --s3-bucket $AUTO_UPLOAD_S3_BUCKET"
        if [ -n "$AUTO_UPLOAD_S3_PREFIX" ]; then
            UPLOAD_CMD="$UPLOAD_CMD --s3-prefix $AUTO_UPLOAD_S3_PREFIX"
        fi
    elif [ "$UPLOAD_METHOD" = "hf" ] && [ -n "$AUTO_UPLOAD_HF_REPO_ID" ]; then
        UPLOAD_CMD="$UPLOAD_CMD --hf-repo-id $AUTO_UPLOAD_HF_REPO_ID"
    elif [ "$UPLOAD_METHOD" = "wandb" ]; then
        if [ -n "$AUTO_UPLOAD_WANDB_PROJECT" ]; then
            UPLOAD_CMD="$UPLOAD_CMD --wandb-project $AUTO_UPLOAD_WANDB_PROJECT"
        fi
        if [ -n "$AUTO_UPLOAD_WANDB_RUN_NAME" ]; then
            UPLOAD_CMD="$UPLOAD_CMD --wandb-run-name $AUTO_UPLOAD_WANDB_RUN_NAME"
        fi
    elif [ "$UPLOAD_METHOD" = "webhook" ] && [ -n "$AUTO_UPLOAD_WEBHOOK_URL" ]; then
        UPLOAD_CMD="$UPLOAD_CMD --webhook-url $AUTO_UPLOAD_WEBHOOK_URL"
        if [ -n "$AUTO_UPLOAD_WEBHOOK_API_KEY" ]; then
            UPLOAD_CMD="$UPLOAD_CMD --webhook-api-key $AUTO_UPLOAD_WEBHOOK_API_KEY"
        fi
    fi
    
    eval $UPLOAD_CMD || echo "⚠️  Log upload failed, but continuing..."
    echo "=========================================="
fi

exit $TEST_EXIT_CODE

