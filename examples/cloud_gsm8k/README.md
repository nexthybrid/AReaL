# Cloud Deployment Guide for GRPO Training

This directory contains scripts and configurations for running AReaL GRPO training on cloud GPU platforms, **optimized for RunPod** (most economical option).

## Recommended Platform: RunPod

**Why RunPod?**
- 💰 **Best pricing**: RTX 4090 at $0.29/hour (spot: $0.09/hour!)
- 📦 **Network volumes**: Persistent storage for checkpoints
- 🚀 **Easy setup**: Template system
- ⚡ **Fast startup**: Pods ready in seconds

**Quick Start**: See `RUNPOD_QUICK_START.md` or `RUNPOD_COMPLETE_GUIDE.md`

## Quick Start (RunPod)

### 1. Create RunPod Account
- Go to https://runpod.io
- Sign up and add credits

### 2. Create Network Volume
- Go to "Volumes" → "Create Volume"
- Name: `areal-outputs`, Size: 50GB

### 3. Deploy Pod
- Use template (see `RUNPOD_QUICK_START.md`)
- Or manual deployment (see `RUNPOD_COMPLETE_GUIDE.md`)

### 4. Run Training
```bash
# Inside pod - choose appropriate config for your GPU:
bash examples/cloud_gsm8k/run_training_cloud.sh fast      # 20-30 min, any GPU
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour     # 1-2 hours, any GPU (default)
bash examples/cloud_gsm8k/run_training_cloud.sh 3hour     # 3-4 hours, any GPU
bash examples/cloud_gsm8k/run_training_cloud.sh full       # 5 days, REQUIRES H200/H100/A100-80GB

# Optional: Override number of epochs without modifying YAML files
bash examples/cloud_gsm8k/run_training_cloud.sh standard_2000samples_2GPUs_v3 6  # Use 6 epochs instead of YAML default
bash examples/cloud_gsm8k/run_training_cloud.sh standard_2000samples_2GPUs_v3 5  # Use 5 epochs instead of YAML default
```

**⚠️ Important**: The `full` config requires H200, H100, or A100-80GB (80GB+ memory). The script will automatically validate your GPU and reject full training on smaller GPUs.

**💡 Important**: Set your WandB API key as an environment variable in RunPod (see `RUNPOD_COMPLETE_GUIDE.md` for details)

## Files

### Main Training Script
- `run_training_cloud.sh` - **Main training script** - Use this to start training
  - Supports: `fast`, `1hour`, `3hour`, `full`, `standard_1000samples_2GPUs`, `standard_2000samples_2GPUs_v3`, etc.
  - Auto-detects GPU type (A40, RTX 5090, H200, etc.) and uses appropriate optimized config
  - Validates GPU requirements for `full` training (requires 80GB+ memory)
  - **Epoch override**: Optional second parameter to override epochs without modifying YAML files
  - Usage: `bash examples/cloud_gsm8k/run_training_cloud.sh [config_name] [epochs_override]`
  - Examples:
    - `bash examples/cloud_gsm8k/run_training_cloud.sh standard_2000samples_2GPUs_v3` (uses YAML epochs)
    - `bash examples/cloud_gsm8k/run_training_cloud.sh standard_2000samples_2GPUs_v3 6` (overrides to 6 epochs)

### RunPod Documentation
- `RUNPOD_COMPLETE_GUIDE.md` - ⭐ **Complete RunPod guide** - Setup, troubleshooting, and all procedures
- `runpod_template.json` - RunPod template configuration (optional)

### Training Configurations

**Standard GRPO Configs (2x A100 GPUs):**
- `gsm8k_grpo_1000samples_2GPUs.yaml` - 1000 samples, ~3 hours
- `gsm8k_grpo_1000samples_2GPUs_v3_conservative.yaml` - 1000 samples, v3 conservative settings, ~3-4 hours
- `gsm8k_grpo_1000samples_2GPUs_v4.yaml` - 1000 samples, v4 improved settings, ~4-5 hours
- `gsm8k_grpo_2000samples_2GPUs.yaml` - 2000 samples, ~6 hours
- `gsm8k_grpo_2000samples_2GPUs_v3.yaml` - 2000 samples, v3 conservative settings, ~6-7 hours
- `gsm8k_grpo_4000samples_2GPUs.yaml` - 4000 samples, ~12 hours

**Full Training:**
- `gsm8k_grpo_cloud.yaml` - **Full training** (REQUIRES H200/H100/A100-80GB, 80GB+ memory)
  - Full dataset (7473 samples), 5 epochs, ~5 days training time
  - Auto-validated: script checks GPU before allowing full training

**Quick Training (Single GPU):**
- `gsm8k_grpo_1hour.yaml` - 1-hour training (works on all GPUs)
  - Memory-optimized settings: works on RTX 4090, RTX 5090, A40, A100, H200, etc.
- `gsm8k_grpo_3hour.yaml` - 3-hour training (works on all GPUs)
  - Memory-optimized settings: works on RTX 4090, RTX 5090, A40, A100, H200, etc.
- `gsm8k_grpo_fast.yaml` - Fast training (20-30 min)

### Training Scripts
- `gsm8k_grpo_train.py` - **Consolidated training script** (used by `run_training_cloud.sh`)
  - Handles all training configurations (fast, 1hour, 3hour, full)
  - Configuration is controlled via YAML files and command-line overrides

### Recovery and Checkpoint Management
- `list_checkpoints.py` - List available checkpoints
- `setup_recovery.py` - Set up recovery from a checkpoint
- `resume_training.py` - Interactive recovery guide
- `CHECKPOINT_AND_RECOVERY_GUIDE.md` - Detailed recovery documentation
- `RECOVERY_QUICK_START.md` - Quick recovery reference
- `CIRCUIT_BREAKER_AND_RECOVERY_SUMMARY.md` - Circuit breaker implementation summary

### GPU-Specific Documentation
- `H200_SETUP.md` - H200 GPU setup and configuration
- `H200_STEP188_DIAGNOSIS.md` - Analysis of H200 training crash at step 188
- `A40_GPU_FIX.md` - A40 GPU memory optimization guide
- `A40_3HOUR_SUMMARY.md` - A40 3-hour training summary
- `1HOUR_VS_3HOUR_COMPARISON.md` - Comparison of 1-hour vs 3-hour configs

### Testing Scripts
- `test_trained_model_cloud.py` - Model evaluation script (standard GRPO models)
- `test_reasoning_model_cloud.py` - Model evaluation script (reasoning models)
- `test_full_dataset.sh` - **Full dataset test script** - Test trained model on all 1319 GSM8K test samples
  - Automatically extracts checkpoint path from training logs
  - Optimized batch size (32) for A100 80GB GPUs
  - Usage: `bash examples/cloud_gsm8k/test_full_dataset.sh [checkpoint_path] [log_file]`
  - Examples:
    - `bash examples/cloud_gsm8k/test_full_dataset.sh` (auto-detect latest checkpoint)
    - `bash examples/cloud_gsm8k/test_full_dataset.sh "" examples/cloud_gsm8k/train_logs/logs_grpo_1k_v3_15epochs.txt` (extract from log)
    - `bash examples/cloud_gsm8k/test_full_dataset.sh /path/to/checkpoint` (use specific checkpoint)
  - Batch size can be overridden: `TEST_BATCH_SIZE=48 bash examples/cloud_gsm8k/test_full_dataset.sh`

### Other Documentation
- `CHECKPOINT_SAVING_FIX.md` - Checkpoint saving configuration fixes

## Cost Comparison

For 3-hour training:
- **RunPod RTX 4090 Spot**: ~$0.27 ⭐ **Best Value**
- **RunPod RTX 4090**: ~$0.87
- **RunPod A40**: ~$1.20-1.50 (if RTX 4090 unavailable)
- **RunPod RTX 5090**: ~$1.50-2.00

For full training (5 days):
- **RunPod H200**: ~$540-720 (for full dataset, 5 epochs)
- **RunPod H100**: ~$360-480
- **RunPod A100 80GB**: ~$300-400

**Note**: All configs use memory-optimized settings that work across all GPUs. No GPU-specific configs needed.

## Key Features

1. **Universal Configs**: All configs use memory-optimized settings that work on all GPUs
   - No need for GPU-specific configs - one config per time length works everywhere
   - Optimized for memory efficiency while maintaining training quality
2. **Full Training Validation**: `full` config automatically validates GPU (requires 80GB+ memory)
3. **Circuit Breaker**: Training stops automatically if task reward is zero for 10 consecutive steps
4. **Checkpoint Recovery**: Easy recovery from checkpoints using provided scripts
5. **Network Volumes**: Persistent storage for checkpoints across pod restarts
6. **Spot Instances**: Enable for 50-70% cost savings

## Important: Using Your Forked Branch

**This setup uses your forked repository**: `nexthybrid/AReaL` branch `DL4Math`

- ✅ **Docker Image**: Uses `ghcr.io/inclusionai/areal-runtime:v0.3.4` (provides runtime environment)
- ✅ **Code**: Clones from `https://github.com/nexthybrid/AReaL` branch `DL4Math`
- ✅ **Your custom scripts**: Uses your modified `examples/cloud_gsm8k/` scripts

**Why the original Docker image works**: The image only provides the runtime environment (CUDA, PyTorch, dependencies). Your actual code gets cloned from GitHub, so you can use your forked branch without building a new image.

## Troubleshooting

- **Pip installation fails**: See `RUNPOD_COMPLETE_GUIDE.md` - "Pip Installation Fails" section
- **CUDA Out of Memory on A40**: See `A40_GPU_FIX.md` ⚠️ or `RUNPOD_COMPLETE_GUIDE.md` - "CUDA Out of Memory" section
- **Checkpoints not saving**: See `CHECKPOINT_SAVING_FIX.md` ⚠️
- **Training crashes**: See `RUNPOD_COMPLETE_GUIDE.md` - "Training Crashes" section
- **Task reward drops to zero**: See `H200_STEP188_DIAGNOSIS.md` and `CIRCUIT_BREAKER_AND_RECOVERY_SUMMARY.md`
- **Resuming training**: See `CHECKPOINT_AND_RECOVERY_GUIDE.md` or `RECOVERY_QUICK_START.md`

## Next Steps

1. **For RunPod Setup**: See `RUNPOD_COMPLETE_GUIDE.md` ⭐ (includes quick start section)
2. **For Training Best Practices**: See `TRAINING_LEARNINGS.md` (includes tuning guide, epoch analysis, troubleshooting)
3. **For A40 GPU Issues**: See `RUNPOD_COMPLETE_GUIDE.md` - "CUDA Out of Memory" section
4. **For H200 Setup**: See `H200_SETUP.md` (if exists)
5. **For Recovery**: See `TRAINING_LEARNINGS.md` - "Checkpoint and Recovery" section
