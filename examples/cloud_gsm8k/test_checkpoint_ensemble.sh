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
#   - Example: AUTO_UPLOAD_LOGS_METHOD=email AUTO_UPLOAD_EMAIL_TO=user@example.com bash test_checkpoint_ensemble.sh ...

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CHECKPOINT_DIR="${1:-}"
EPOCHS="${2:-4 9 14 19 24}"
BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
MAX_NEW_TOKENS=512
N_SAMPLES="${N_SAMPLES:-1}"  # Number of samples per checkpoint (default: 1, use >1 for self-consistency)
TEMPERATURE="${TEMPERATURE:-0.0}"  # Temperature for sampling (default: 0.0 = greedy)
SUB_BATCH_SIZE="${SUB_BATCH_SIZE:-}"  # Sub-batch size for multi-sample generation (default: auto, ~16 for A100)
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
echo "Samples per checkpoint: $N_SAMPLES"
echo "Temperature: $TEMPERATURE"
echo ""

# Run the ensemble testing script
# Convert EPOCHS string to array and pass as separate arguments
# This ensures each epoch is passed as a separate argument to Python
# Use IFS to properly split the string into an array
IFS=' ' read -ra EPOCHS_ARRAY <<< "$EPOCHS"
echo "Epochs array: ${EPOCHS_ARRAY[@]}"
echo "Number of epochs: ${#EPOCHS_ARRAY[@]}"
python3 examples/cloud_gsm8k/test_checkpoint_ensemble.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --epochs "${EPOCHS_ARRAY[@]}" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --n-samples "$N_SAMPLES" \
    --temperature "$TEMPERATURE" \
    ${SUB_BATCH_SIZE:+--sub-batch-size "$SUB_BATCH_SIZE"} \
    --log-dir "/workspace/outputs/grpo/test_logs"

TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ Ensemble testing completed successfully!"
else
    echo "⚠️  Ensemble testing completed with exit code $TEST_EXIT_CODE"
fi
echo "Check logs in /workspace/outputs/grpo/test_logs/"

# Collect log files for upload
# Find all ensemble-related log files (individual checkpoints + ensemble voting)
LOG_DIR="/workspace/outputs/grpo/test_logs"
ENSEMBLE_LOG_FILES=()

# Find individual checkpoint logs
# Use the same array expansion as above
IFS=' ' read -ra EPOCHS_ARRAY <<< "$EPOCHS"
for epoch in "${EPOCHS_ARRAY[@]}"; do
    checkpoint_log=$(ls -t "$LOG_DIR"/ensemble_checkpoint_epoch${epoch}.log 2>/dev/null | head -1)
    if [ -n "$checkpoint_log" ] && [ -f "$checkpoint_log" ]; then
        ENSEMBLE_LOG_FILES+=("$checkpoint_log")
    fi
done

# Find ensemble voting log (most recent)
ensemble_voting_log=$(ls -t "$LOG_DIR"/ensemble_majority_voting_*.log 2>/dev/null | head -1)
if [ -n "$ensemble_voting_log" ] && [ -f "$ensemble_voting_log" ]; then
    ENSEMBLE_LOG_FILES+=("$ensemble_voting_log")
fi

# Auto-upload logs if configured (same as training script)
if [ -n "$AUTO_UPLOAD_LOGS_METHOD" ] && [ ${#ENSEMBLE_LOG_FILES[@]} -gt 0 ]; then
    # Normalize method to lowercase (handle EMAIL -> email, etc.)
    UPLOAD_METHOD=$(echo "$AUTO_UPLOAD_LOGS_METHOD" | tr '[:upper:]' '[:lower:]')
    
    echo ""
    echo "=========================================="
    echo "📤 Auto-uploading ensemble test logs via $UPLOAD_METHOD..."
    echo "=========================================="
    
    UPLOAD_SCRIPT="examples/cloud_gsm8k/upload_logs.py"
    
    # Upload all ensemble log files
    echo "Uploading ${#ENSEMBLE_LOG_FILES[@]} ensemble test log file(s)..."
    UPLOAD_CMD="python3 $UPLOAD_SCRIPT --log-dir $LOG_DIR --method $UPLOAD_METHOD --log-files"
    for log_file in "${ENSEMBLE_LOG_FILES[@]}"; do
        if [ -f "$log_file" ]; then
            UPLOAD_CMD="$UPLOAD_CMD $log_file"
        fi
    done
    
    # Add method-specific arguments from environment
    if [ "$UPLOAD_METHOD" = "email" ] && [ -n "$AUTO_UPLOAD_EMAIL_TO" ]; then
        UPLOAD_CMD="$UPLOAD_CMD --email-to $AUTO_UPLOAD_EMAIL_TO"
        # Note: upload_logs.py will use EMAIL_FROM and SMTP_PASSWORD from environment if not provided
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
    
    # Execute upload command
    if eval $UPLOAD_CMD; then
        echo "✅ Successfully uploaded ${#ENSEMBLE_LOG_FILES[@]} log file(s) via $UPLOAD_METHOD!"
    else
        echo "⚠️  Log upload failed, but continuing..."
    fi
    echo "=========================================="
fi

# Exit with test exit code
exit $TEST_EXIT_CODE

