#!/bin/bash
# Cloud-optimized training script for SFT
#
# Usage:
#   bash examples/cloud_gsm8k/run_sft_training_cloud.sh [config_name] [epochs_override]
#
#   config_name: Name of the config preset or path to YAML file (default: 1k)
#   epochs_override: Optional number of epochs to override YAML setting (e.g., 5, 6)
#
# Examples:
#   bash examples/cloud_gsm8k/run_sft_training_cloud.sh 1k
#   bash examples/cloud_gsm8k/run_sft_training_cloud.sh 1k 5
#   bash examples/cloud_gsm8k/run_sft_training_cloud.sh examples/cloud_gsm8k/gsm8k_sft_1000samples_1GPU.yaml 3
#
# Config options:
#   - 1k: 1K samples, 1 GPU (~1-2 hours)
#   - 2k: 2K samples, 3 GPUs (~2-3 hours)
#   - full: Full dataset, 3 GPUs (~1-2 days)

set -e

# Configuration
CONFIG_NAME="${1:-1k}"
EPOCHS_OVERRIDE="${2:-}"  # Optional: override total_train_epochs from YAML
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Check if we're in the right directory
if [ ! -f "examples/cloud_gsm8k/gsm8k_sft_train.py" ]; then
    echo "ERROR: Not in AReaL project root or cloud_gsm8k files not found"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check for completion markers to prevent re-running
COMPLETION_MARKER_PATTERN="/workspace/outputs/sft/training_completed_*.marker"
if ls $COMPLETION_MARKER_PATTERN 1> /dev/null 2>&1; then
    echo "=========================================="
    echo "⚠️  Training already completed!"
    echo "=========================================="
    echo "Found completion marker(s):"
    ls -lh $COMPLETION_MARKER_PATTERN
    echo ""
    echo "Exiting to prevent re-running and save costs."
    echo "If you want to run again, delete the completion markers first."
    echo "=========================================="
    exit 0
fi

# Check WandB API key
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. WandB logging will be disabled."
    echo "Set it with: export WANDB_API_KEY=your-api-key"
fi

# Verify AReaL is installed (only install if not already installed)
echo "Checking AReaL installation..."
if ! python3 -c "import areal" 2>/dev/null; then
    echo "AReaL not found. Installing..."
    pip install -e .
else
    echo "AReaL already installed. Skipping installation."
fi

# Clean up any leftover processes that might be using the GPU
echo "Cleaning up any leftover GPU processes..."

# Kill any Python processes that might be holding the GPU
pkill -9 -f "areal.launcher" 2>/dev/null || true
pkill -9 -f "torchrun" 2>/dev/null || true
pkill -9 -f "python.*gsm8k_sft" 2>/dev/null || true

# Wait a moment for processes to terminate
sleep 3

# Get GPU information and check status
echo "Checking GPU..."
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)
if [ -z "$GPU_INFO" ]; then
    echo "WARNING: nvidia-smi not available. GPU may not be accessible."
    GPU_NAME=""
    GPU_MEMORY=""
    GPU_COUNT=0
else
    echo "$GPU_INFO"
    GPU_NAME=$(echo "$GPU_INFO" | head -1 | cut -d',' -f1 | xargs)
    GPU_MEMORY=$(echo "$GPU_INFO" | head -1 | cut -d',' -f2 | xargs | grep -oE '[0-9]+' | head -1)
    GPU_COUNT=$(echo "$GPU_INFO" | wc -l | xargs)
    # Set PyTorch memory allocator for better memory management
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for better memory management"
    echo "Detected $GPU_COUNT GPU(s)"
fi

# Function to check if GPU is suitable for full training
check_full_training_gpu() {
    if [ -z "$GPU_NAME" ]; then
        echo "ERROR: Cannot detect GPU. Full training requires H200/H100/A100-80GB or equivalent."
        return 1
    fi
    
    # Check for high-end GPUs suitable for full training
    if echo "$GPU_NAME" | grep -qiE "H200|H100|A100.*80|A100.*80GB"; then
        return 0
    fi
    
    # Check memory (H200-class GPUs have 80GB+)
    if [ -n "$GPU_MEMORY" ] && [ "$GPU_MEMORY" -ge 80000 ]; then
        return 0
    fi
    
    return 1
}

# Check if CONFIG_NAME is a file path (ends with .yaml or .yml)
# If so, use it directly instead of looking up a preset
if [[ "$CONFIG_NAME" == *".yaml" ]] || [[ "$CONFIG_NAME" == *".yml" ]]; then
    # It's a file path - use it directly
    CONFIG_FILE="$CONFIG_NAME"
    # Extract experiment name from filename
    EXPERIMENT_NAME=$(basename "$CONFIG_FILE" .yaml | sed 's/_/-/g')
    TRAIN_SCRIPT="examples/cloud_gsm8k/gsm8k_sft_train.py"
    echo "Using config file directly: $CONFIG_FILE"
    echo "Experiment name: $EXPERIMENT_NAME"
else
    # It's a preset name - look it up in the case statement
    case "$CONFIG_NAME" in
    1k)
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_sft_1000samples_1GPU.yaml"
        TRAIN_SCRIPT="examples/cloud_gsm8k/gsm8k_sft_train.py"
        EXPERIMENT_NAME="gsm8k-sft-cloud-1gpu-1000samples"
        echo "Using 1K SAMPLES configuration (1 GPU)"
        echo "Note: Requires 1 A100 80GB or equivalent GPU"
        echo "Expected training time: ~1-2 hours"
        
        # Check GPU count
        if [ -z "$GPU_COUNT" ] || [ "$GPU_COUNT" -lt 1 ]; then
            echo "ERROR: This config requires at least 1 GPU"
            echo "Detected: $GPU_COUNT GPU(s)"
            exit 1
        fi
        ;;
    2k)
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_sft_2000samples_3GPUs.yaml"
        TRAIN_SCRIPT="examples/cloud_gsm8k/gsm8k_sft_train.py"
        EXPERIMENT_NAME="gsm8k-sft-cloud-3gpu-2000samples"
        echo "Using 2K SAMPLES configuration (3 GPUs)"
        echo "Note: Requires 3 A100 80GB or equivalent GPUs"
        echo "Expected training time: ~2-3 hours"
        
        # Check GPU count
        if [ -z "$GPU_COUNT" ] || [ "$GPU_COUNT" -lt 3 ]; then
            echo "ERROR: This config requires 3 GPUs"
            echo "Detected: $GPU_COUNT GPU(s)"
            echo ""
            echo "This config is optimized for 3x A100 80GB/H100/H200 for distributed training."
            echo "Please use a pod with at least 3 GPUs or use the '1k' config for 1 GPU."
            exit 1
        fi
        ;;
    full)
        # Full training requires high-end GPUs
        if ! check_full_training_gpu; then
            echo "ERROR: Full training requires H200, H100, A100-80GB, or equivalent GPU (80GB+ memory)"
            echo "Detected GPU: $GPU_NAME ($GPU_MEMORY MB)"
            echo ""
            echo "For full training, please use:"
            echo "  - H200 (141GB memory)"
            echo "  - H100 (80GB memory)"
            echo "  - A100 80GB (80GB memory)"
            exit 1
        fi
        
        # Check GPU count
        if [ -z "$GPU_COUNT" ] || [ "$GPU_COUNT" -lt 3 ]; then
            echo "ERROR: This config requires 3 GPUs"
            echo "Detected: $GPU_COUNT GPU(s)"
            echo ""
            echo "This config is optimized for 3x A100 80GB/H100/H200 for distributed training."
            echo "Please use a pod with at least 3 GPUs."
            exit 1
        fi
        
        CONFIG_FILE="examples/cloud_gsm8k/gsm8k_sft_full_3GPUs.yaml"
        TRAIN_SCRIPT="examples/cloud_gsm8k/gsm8k_sft_train.py"
        EXPERIMENT_NAME="gsm8k-sft-cloud-3gpu-full"
        echo "Using FULL DATASET configuration (3 GPUs)"
        echo "Note: Full GSM8K dataset (7473 samples)"
        echo "GPU allocation: 3 GPUs for distributed training"
        echo "Expected training time: ~1-2 days"
        echo "GPU: $GPU_NAME ($GPU_MEMORY MB) - suitable for full training"
        echo "GPU count: $GPU_COUNT (required: 3)"
        ;;
    *)
        echo "ERROR: Unknown config name: $CONFIG_NAME"
        echo ""
        echo "Valid options:"
        echo "  - 1k: 1K samples, 1 GPU (~1-2 hours)"
        echo "  - 2k: 2K samples, 3 GPUs (~2-3 hours)"
        echo "  - full: Full dataset, 3 GPUs (~1-2 days)"
        echo ""
        echo "Or provide a path to a YAML config file:"
        echo "  bash examples/cloud_gsm8k/run_sft_training_cloud.sh examples/cloud_gsm8k/gsm8k_sft_1000samples_1GPU.yaml"
        exit 1
        ;;
    esac
fi

# Handle epochs override
OVERRIDE_ARGS=()
if [ -n "$EPOCHS_OVERRIDE" ]; then
    echo "Overriding total_train_epochs to: $EPOCHS_OVERRIDE"
    OVERRIDE_ARGS=("total_train_epochs=$EPOCHS_OVERRIDE")
fi

# Build the training command
echo ""
echo "=========================================="
echo "Starting SFT Training"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Script: $TRAIN_SCRIPT"
echo "Experiment: $EXPERIMENT_NAME"
if [ -n "$EPOCHS_OVERRIDE" ]; then
    echo "Epochs override: $EPOCHS_OVERRIDE"
fi
echo "=========================================="
echo ""

# Run training using AReaL's local launcher
python -m areal.launcher.local "$TRAIN_SCRIPT" \
    --config "$CONFIG_FILE" \
    "${OVERRIDE_ARGS[@]}"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Training completed successfully!"
    echo "=========================================="
    
    # Create completion marker
    MARKER_FILE="/workspace/outputs/sft/training_completed_$(date +%Y%m%d_%H%M%S).marker"
    mkdir -p "$(dirname "$MARKER_FILE")"
    echo "Training completed at $(date)" > "$MARKER_FILE"
    echo "Config: $CONFIG_FILE" >> "$MARKER_FILE"
    echo "Experiment: $EXPERIMENT_NAME" >> "$MARKER_FILE"
    echo "Completion marker created: $MARKER_FILE"
    
    echo ""
    echo "Checkpoints saved to: /workspace/outputs/sft/checkpoints/"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "=========================================="
    exit $TRAIN_EXIT_CODE
fi

