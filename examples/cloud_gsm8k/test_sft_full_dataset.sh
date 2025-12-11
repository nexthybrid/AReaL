#!/bin/bash
# Script to test trained SFT model on full GSM8K dataset (1319 samples)
# 
# Usage:
#   bash examples/cloud_gsm8k/test_sft_full_dataset.sh [checkpoint_path] [log_file] [--test-intervals]
#
#   checkpoint_path: Optional. Full path to model checkpoint directory OR "baseline" to test base model.
#                    If "baseline", tests Qwen/Qwen2.5-0.5B-Instruct on full dataset.
#                    If not provided, will try to extract from log_file or find latest checkpoint.
#   log_file: Optional. Path to training log file to extract checkpoint path from.
#             If not provided, will try to find latest checkpoint automatically.
#   --test-intervals: Optional flag. If provided, tests checkpoints at 5-epoch intervals (4, 9, 14, 19, etc.)
#                    instead of just the latest checkpoint. All interval test logs will be uploaded.
#
# Examples:
#   # Test baseline model on full dataset
#   bash examples/cloud_gsm8k/test_sft_full_dataset.sh baseline
#
#   # Auto-detect from latest checkpoint (test only latest)
#   bash examples/cloud_gsm8k/test_sft_full_dataset.sh
#
#   # Test latest checkpoint from log file
#   bash examples/cloud_gsm8k/test_sft_full_dataset.sh "" examples/cloud_gsm8k/train_logs/logs_sft_1K_10epochs.txt
#
#   # Test at 5-epoch intervals (4, 9, 14, 19, etc.)
#   bash examples/cloud_gsm8k/test_sft_full_dataset.sh "" examples/cloud_gsm8k/train_logs/logs_sft_1K_10epochs.txt --test-intervals
#
#   # Use specific checkpoint (test only that checkpoint)
#   bash examples/cloud_gsm8k/test_sft_full_dataset.sh /workspace/outputs/sft/checkpoints/root/gsm8k-sft-cloud-1gpu-1000samples/trial0/default/epoch1
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
#   - Example: AUTO_UPLOAD_LOGS_METHOD=email AUTO_UPLOAD_EMAIL_TO=user@example.com bash test_sft_full_dataset.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Check for completion markers to prevent re-running interval tests
# This prevents RunPod container restarts from causing infinite loops
COMPLETION_MARKER_PATTERN="/workspace/outputs/sft/test_logs/interval_testing_completed_*.marker"
# Only check if --test-intervals flag is present (check after argument parsing)
# We'll check this after parsing arguments

# Configuration
TEST_INTERVALS=false
# Check for --test-intervals flag FIRST (before processing positional args)
# This allows the flag to be in any position
for arg in "$@"; do
    if [ "$arg" = "--test-intervals" ]; then
        TEST_INTERVALS=true
        break
    fi
done

# Now process positional arguments, skipping --test-intervals
CHECKPOINT_PATH=""
LOG_FILE=""
for arg in "$@"; do
    if [ "$arg" != "--test-intervals" ]; then
        if [ -z "$CHECKPOINT_PATH" ]; then
            CHECKPOINT_PATH="$arg"
        elif [ -z "$LOG_FILE" ]; then
            LOG_FILE="$arg"
        fi
    fi
done
BATCH_SIZE="${TEST_BATCH_SIZE:-32}"  # Optimized for A100 80GB - can handle 32-64 easily
MAX_NEW_TOKENS=512
IS_BASELINE=false
MODEL_NAME=""
INTERVAL_LOG_FILES=()  # Array to store log files from interval testing

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
    if grep -q "Checkpoints saved to:" "$LOG_FILE" 2>/dev/null; then
        # Extract from training summary format
        MODEL_PATH=$(grep -A 5 "Checkpoints saved to:" "$LOG_FILE" | grep "Latest:" | head -1 | sed 's/.*Latest: //' | tr -d '[:space:]')
        echo "Extracted checkpoint path from training summary: $MODEL_PATH"
    elif grep -q "default/epoch" "$LOG_FILE" 2>/dev/null; then
        # Extract any checkpoint path containing "default/epoch"
        MODEL_PATH=$(grep -o "/workspace/outputs/sft/checkpoints/[^[:space:]]*/default/epoch[^[:space:]]*" "$LOG_FILE" | head -1)
        echo "Extracted checkpoint path from epoch pattern: $MODEL_PATH"
    else
        echo "ERROR: Could not find checkpoint path in log file."
        echo "Please provide checkpoint path as first argument."
        exit 1
    fi
else
    echo "No checkpoint path or log file provided. Attempting to find latest checkpoint..."
    
    # Try to find latest checkpoint from common experiment names
    CHECKPOINT_BASE="/workspace/outputs/sft/checkpoints/root"
    
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
        # It's a HuggingFace model ID, skip file check
        echo "Using HuggingFace model identifier: $MODEL_PATH"
    elif [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: Checkpoint directory does not exist: $MODEL_PATH"
        exit 1
    fi
fi

# Function to parse epoch number from checkpoint path
parse_epoch() {
    local ckpt_path="$1"
    # Extract epoch number from path like .../epoch1 or .../epoch14epochstep...
    echo "$ckpt_path" | grep -oE "epoch[0-9]+" | head -1 | sed 's/epoch//'
}

# Function to find interval checkpoints (epochs 4, 9, 14, 19, etc.)
find_interval_checkpoints() {
    local base_dir="$1"
    local checkpoints=()
    
    # Find all epoch checkpoints
    local all_epochs=$(find "$base_dir" -type d -path "*/default/epoch*" -printf '%p\n' 2>/dev/null | sort -V)
    
    if [ -z "$all_epochs" ]; then
        echo "No checkpoints found in $base_dir"
        return
    fi
    
    # Extract epoch numbers and find interval checkpoints
    local max_epoch=0
    while IFS= read -r ckpt; do
        local epoch=$(parse_epoch "$ckpt")
        if [ -n "$epoch" ] && [ "$epoch" -gt "$max_epoch" ]; then
            max_epoch=$epoch
        fi
    done <<< "$all_epochs"
    
    echo "Found maximum epoch: $max_epoch" >&2
    
    # Find checkpoints at intervals: 4, 9, 14, 19, 24, etc. (starting from epoch 4)
    # Also include the final epoch
    local target_epochs=()
    local epoch=4
    while [ "$epoch" -le "$max_epoch" ]; do
        target_epochs+=($epoch)
        epoch=$((epoch + 5))
    done
    
    # Always include the final epoch if it's not already in the list
    if [ "$max_epoch" -gt 0 ] && [[ ! " ${target_epochs[@]} " =~ " ${max_epoch} " ]]; then
        target_epochs+=($max_epoch)
    fi
    
    echo "Target epochs for interval testing: ${target_epochs[@]}" >&2
    
    # Find checkpoints matching target epochs
    while IFS= read -r ckpt; do
        local epoch=$(parse_epoch "$ckpt")
        if [ -n "$epoch" ] && [[ " ${target_epochs[@]} " =~ " ${epoch} " ]]; then
            checkpoints+=("$ckpt")
        fi
    done <<< "$all_epochs"
    
    # Sort by epoch number
    printf '%s\n' "${checkpoints[@]}" | sort -V
}

# Handle interval testing
if [ "$TEST_INTERVALS" = "true" ]; then
    echo ""
    echo "=========================================="
    echo "Interval Testing Mode"
    echo "=========================================="
    
    # Check for completion markers
    if ls $COMPLETION_MARKER_PATTERN 1> /dev/null 2>&1; then
        echo "⚠️  Interval testing already completed!"
        echo "Found completion marker(s):"
        ls -lh $COMPLETION_MARKER_PATTERN
        echo ""
        echo "Exiting to prevent re-running and save costs."
        echo "If you want to run again, delete the completion markers first."
        echo "=========================================="
        exit 0
    fi
    
    # Extract base directory from MODEL_PATH
    if [ "$IS_BASELINE" = "true" ]; then
        echo "ERROR: Cannot do interval testing on baseline model."
        exit 1
    fi
    
    # Find the base directory (up to "default")
    BASE_DIR=$(echo "$MODEL_PATH" | sed 's|/default/.*||')
    if [ -z "$BASE_DIR" ] || [ ! -d "$BASE_DIR" ]; then
        BASE_DIR=$(dirname "$MODEL_PATH")
        # Try to go up to find the experiment directory
        while [ "$BASE_DIR" != "/" ] && [ ! -d "$BASE_DIR/default" ]; do
            BASE_DIR=$(dirname "$BASE_DIR")
        done
    fi
    
    if [ ! -d "$BASE_DIR/default" ]; then
        echo "ERROR: Could not find checkpoint base directory with 'default' subdirectory."
        echo "MODEL_PATH: $MODEL_PATH"
        exit 1
    fi
    
    echo "Base checkpoint directory: $BASE_DIR"
    
    # Find interval checkpoints
    INTERVAL_CHECKPOINTS=($(find_interval_checkpoints "$BASE_DIR/default"))
    
    if [ ${#INTERVAL_CHECKPOINTS[@]} -eq 0 ]; then
        echo "ERROR: No interval checkpoints found."
        exit 1
    fi
    
    echo "Found ${#INTERVAL_CHECKPOINTS[@]} checkpoint(s) for interval testing:"
    for ckpt in "${INTERVAL_CHECKPOINTS[@]}"; do
        epoch=$(parse_epoch "$ckpt")
        echo "  - Epoch $epoch: $ckpt"
    done
    echo ""
    
    # Test each checkpoint
    TEST_EXIT_CODE=0
    for i in "${!INTERVAL_CHECKPOINTS[@]}"; do
        ckpt="${INTERVAL_CHECKPOINTS[$i]}"
        epoch=$(parse_epoch "$ckpt")
        checkpoint_num=$((i + 1))
        total_checkpoints=${#INTERVAL_CHECKPOINTS[@]}
        
        echo ""
        echo "=========================================="
        echo "CHECKPOINT $checkpoint_num/$total_checkpoints: Testing Epoch $epoch"
        echo "=========================================="
        echo "Checkpoint: $ckpt"
        echo ""
        
        # Check for completion marker for this specific checkpoint
        checkpoint_marker="/workspace/outputs/sft/test_logs/interval_testing_epoch${epoch}_completed.marker"
        if [ -f "$checkpoint_marker" ]; then
            echo "⚠️  Checkpoint epoch $epoch already tested. Skipping..."
            echo "   Marker: $checkpoint_marker"
            echo "   To re-test, delete the marker file."
            continue
        fi
        
        # Run test
        log_file="/workspace/outputs/sft/test_logs/test_sft_model_epoch${epoch}_$(date +%Y%m%d_%H%M%S).log"
        python3 examples/cloud_gsm8k/test_sft_model_cloud.py \
            --model-path "$ckpt" \
            --test-all \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --batch-size "$BATCH_SIZE" \
            --model-name "SFT-Epoch${epoch}" \
            --log-dir /workspace/outputs/sft/test_logs 2>&1 | tee "$log_file"
        
        test_exit=$?
        if [ $test_exit -ne 0 ]; then
            TEST_EXIT_CODE=$test_exit
            echo "⚠️  Test failed for epoch $epoch with exit code $test_exit"
        else
            echo "✅ Test completed for epoch $epoch"
        fi
        
        # Store log file for later upload
        if [ -f "$log_file" ]; then
            INTERVAL_LOG_FILES+=("$log_file")
        fi
        
        # Create completion marker for this checkpoint
        echo "Epoch $epoch tested at $(date)" > "$checkpoint_marker"
        echo "Checkpoint: $ckpt" >> "$checkpoint_marker"
        echo "Log file: $log_file" >> "$checkpoint_marker"
        echo "✅ Created completion marker: $checkpoint_marker"
        
        echo ""
    done
    
    # Create overall completion marker
    COMPLETION_MARKER="/workspace/outputs/sft/test_logs/interval_testing_completed_$(date +%Y%m%d_%H%M%S).marker"
    echo "Interval testing completed at $(date)" > "$COMPLETION_MARKER"
    echo "Tested ${#INTERVAL_CHECKPOINTS[@]} checkpoint(s)" >> "$COMPLETION_MARKER"
    echo "Exit code: $TEST_EXIT_CODE" >> "$COMPLETION_MARKER"
    for ckpt in "${INTERVAL_CHECKPOINTS[@]}"; do
        epoch=$(parse_epoch "$ckpt")
        echo "  - Epoch $epoch: $ckpt" >> "$COMPLETION_MARKER"
    done
    echo "✅ Created completion marker: $COMPLETION_MARKER"
    
else
    # Single checkpoint testing
    echo ""
    echo "=========================================="
    echo "Testing SFT Model"
    echo "=========================================="
    echo "Model: $MODEL_PATH"
    echo "Batch size: $BATCH_SIZE"
    echo "Max new tokens: $MAX_NEW_TOKENS"
    echo "=========================================="
    echo ""
    
    python3 examples/cloud_gsm8k/test_sft_model_cloud.py \
        --model-path "$MODEL_PATH" \
        --test-all \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --batch-size "$BATCH_SIZE" \
        --model-name "$MODEL_NAME" \
        --log-dir /workspace/outputs/sft/test_logs
    
    TEST_EXIT_CODE=$?
fi

# Auto-upload logs if configured (same as training script)
if [ -n "$AUTO_UPLOAD_LOGS_METHOD" ]; then
    # Normalize method to lowercase (handle EMAIL -> email, etc.)
    UPLOAD_METHOD=$(echo "$AUTO_UPLOAD_LOGS_METHOD" | tr '[:upper:]' '[:lower:]')
    
    echo ""
    echo "=========================================="
    echo "📤 Auto-uploading test logs via $UPLOAD_METHOD..."
    echo "=========================================="
    
    UPLOAD_SCRIPT="examples/cloud_gsm8k/upload_logs.py"
    
    # If interval testing was used, upload all interval logs; otherwise just latest
    if [ "$TEST_INTERVALS" = "true" ] && [ ${#INTERVAL_LOG_FILES[@]} -gt 0 ]; then
        echo "Uploading ${#INTERVAL_LOG_FILES[@]} interval test log(s)..."
        # Upload all interval log files - build command with all log files
        UPLOAD_CMD="python3 $UPLOAD_SCRIPT --log-dir /workspace/outputs/sft/test_logs --method $UPLOAD_METHOD --log-files"
        for log_file in "${INTERVAL_LOG_FILES[@]}"; do
            if [ -f "$log_file" ]; then
                UPLOAD_CMD="$UPLOAD_CMD $log_file"
            fi
        done
    else
        UPLOAD_CMD="python3 $UPLOAD_SCRIPT --log-dir /workspace/outputs/sft/test_logs --method $UPLOAD_METHOD --pattern 'test_sft_model_*.log'"
    fi
    
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

