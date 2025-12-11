# Training Learnings and Best Practices

This document consolidates all key learnings, fixes, and best practices from training GRPO on cloud GPUs (RunPod).

## Table of Contents

1. [Spot Instances](#spot-instances)
2. [Checkpoint and Recovery](#checkpoint-and-recovery)
3. [Circuit Breaker](#circuit-breaker)
4. [GPU Configuration](#gpu-configuration)
5. [Common Issues and Fixes](#common-issues-and-fixes)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)
8. [GRPO Training Tuning Guide](#grpo-training-tuning-guide)
9. [Epoch Analysis: When Would More Epochs Help?](#epoch-analysis-when-would-more-epochs-help)
10. [Troubleshooting: Reasoning Model 4-GPU Training](#troubleshooting-reasoning-model-4-gpu-training)
11. [GRPO Training Learnings from Experiments](#grpo-training-learnings-from-experiments)

## Spot Instances

### Understanding Spot Instances

**Spot instances** on RunPod are 50-70% cheaper but can be **interrupted** at any time when:
- Someone else claims the GPU
- Provider needs the GPU for a regular customer
- Spot price changes

### What Happens During Interruption

- ✅ Training stops immediately (no graceful shutdown)
- ✅ Checkpoints are saved (if using network volume)
- ⚠️ WandB shows "running" (no clean shutdown signal)
- ⚠️ No error messages (external interruption, not training error)

### Spot vs Regular Instances

| Type | Cost | Stability | Best For |
|------|------|-----------|----------|
| **Spot** | 50-70% cheaper | Can be interrupted | Experiments, testing, short runs |
| **Regular** | Full price | Guaranteed | Long training runs, production |

**Example**: RTX 4090 spot ($0.09/hour) vs regular ($0.29/hour) = 69% savings

### Best Practices for Spot Instances

1. **Always use network volumes** - Mount to `/workspace/outputs` for checkpoint persistence
2. **Frequent checkpoints** - Ensure config saves checkpoints regularly
3. **Monitor training** - Check WandB regularly to detect interruptions
4. **Set up recovery** - Know how to resume from checkpoints
5. **Use circuit breaker** - Training script includes automatic error detection
6. **Test with short runs** - Use `fast` or `1hour` configs first

### Detecting Spot Interruptions

**Signs**:
- WandB run shows "running" but hasn't updated in hours
- RunPod pod status shows "Stopped" or "Failed"
- Training was healthy (no errors) before stopping
- Last step was mid-training (not at epoch end)

**Verify**:
```bash
# Check WandB run status
python examples/cloud_gsm8k/check_wandb_run.py --trial-name trial_YYYYMMDD_HHMMSS
```

## Checkpoint and Recovery

### Checkpoint Strategy

**Current Configuration**:
```yaml
saver:
  freq_epochs: 1  # Save after each epoch
  freq_steps: null
  freq_secs: null

recover:
  mode: disabled  # Set to 'auto' to enable
  freq_epochs: 1
  freq_steps: null
  freq_secs: 3600  # Save recovery info every hour
```

**Checkpoint Location**:
```
/workspace/outputs/grpo/checkpoints/{user}/{experiment_name}/{trial_name}/default/epoch{epoch}epochstep{step}globalstep{global_step}/
```

**Recovery Info Location**:
```
/workspace/outputs/grpo/checkpoints/{user}/{experiment_name}/{trial_name}/recover_checkpoint/
```

### Recovery Tools

#### 1. List Checkpoints

```bash
# List all checkpoints
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS

# List checkpoints before a specific step
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS \
    --before-step 188
```

#### 2. Set Up Recovery

```bash
# Automatically find and set up latest checkpoint
python examples/cloud_gsm8k/setup_recovery.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS \
    --before-step 188
```

#### 3. Resume Training

```bash
# After setting up recovery, resume with:
python -m areal.launcher.local examples/cloud_gsm8k/gsm8k_grpo_train.py \
    --config examples/cloud_gsm8k/gsm8k_grpo_1hour.yaml \
    experiment_name=gsm8k-grpo-cloud-1hour \
    trial_name=trial_YYYYMMDD_HHMMSS \
    recover.mode=auto
```

#### 4. Interactive Recovery Guide

```bash
python examples/cloud_gsm8k/resume_training.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS \
    --before-step 188
```

### Recovery Workflow

1. **Detect Interruption**: Check WandB or RunPod dashboard
2. **Verify Checkpoints**: Use `list_checkpoints.py` to find saved checkpoints
3. **Set Up Recovery**: Use `setup_recovery.py` to prepare recovery
4. **Resume Training**: Start new pod and resume with `recover.mode=auto`
5. **Monitor**: Watch WandB to ensure training continues from correct step

### What Gets Recovered

When resuming with `recover.mode=auto`:
- Model weights from checkpoint
- Optimizer state (if saved)
- Dataloader state (continues from same data position)
- Training step counter (starts from `global_step + 1`)
- Stats logger state (for WandB continuity)

## Circuit Breaker

### Overview

The training script (`gsm8k_grpo_train.py`) includes a **circuit breaker** that prevents model corruption from training on invalid data.

### How It Works

- **Monitors** `grpo_actor/task_reward/avg` after each training step
- **Tracks** consecutive zero-reward steps
- **Stops training** if reward is zero for **10 consecutive steps**
- **Saves checkpoint** before stopping
- **Provides detailed error message** with recovery instructions

### Configuration

```python
CIRCUIT_BREAKER_THRESHOLD = 10  # Stop after 10 consecutive zero-reward steps
CIRCUIT_BREAKER_ENABLED = True
```

### Benefits

- Prevents model corruption from training on invalid data
- Stops training immediately when SGLang server fails
- Saves checkpoint before stopping for easy recovery
- Provides clear error message with recovery instructions

### Common Causes of Zero Rewards

- SGLang server crashed or disconnected
- Inference server not responding
- Network connectivity issues
- All rollouts failed

## GPU Configuration

### Memory-Optimized Configs

All time-based configs (`1hour`, `3hour`) use memory-optimized settings that work on all GPUs:
- **Batch size**: 4 (reduced from 8)
- **Gradient checkpointing**: Enabled
- **SGLang memory fraction**: 0.5 (reduced from 0.8)
- **Max new tokens**: 256 (reduced from 512)
- **Max tokens per microbatch**: 4096 (reduced from 5120)
- **Max concurrent rollouts**: 16 (reduced from 32)

### GPU Requirements

| Config | GPU Memory | Recommended GPUs |
|--------|------------|------------------|
| `fast` | Any | RTX 4090, RTX 5090, A40, etc. |
| `1hour` | Any | RTX 4090, RTX 5090, A40, etc. |
| `3hour` | Any | RTX 4090, RTX 5090, A40, etc. |
| `full` | 80GB+ | H200, H100, A100-80GB only |

### Attention Backend

- **RTX 5090 (SM 100+)**: Requires `flashinfer` backend
- **Other GPUs**: Work with `flashinfer` (universal compatibility)
- **H200 (SM 90)**: Works with default `fa3` or `flashinfer`

### H200 Full Training

H200 (141GB memory) is ideal for full dataset training:
- **No memory optimizations needed**
- **Gradient checkpointing disabled** (faster training)
- **Larger batch sizes** (better stability)
- **More concurrent rollouts** (higher throughput)
- **Full dataset** (7473 samples, 5 epochs)
- **Training time**: ~5 days

### Memory Optimization Details

**Why Memory Optimization is Needed**:
- Both SGLang inference server and PPO trainer share the same GPU
- Default configs allocate 80% of GPU memory to SGLang (e.g., 36GB on A40)
- This leaves insufficient memory for the trainer, causing OOM errors

**Memory Breakdown (Example: A40 GPU)**:

| Config | SGLang | Trainer | Total | Status |
|--------|--------|---------|-------|--------|
| Default | ~36GB (80%) | ~8GB | ~44GB | ❌ OOM |
| Optimized | ~22GB (50%) | ~6GB | ~28GB | ✅ Fits |

**Performance Impact**:
- **Training speed**: ~20-30% slower due to gradient checkpointing
- **Inference throughput**: ~30-40% lower due to reduced parallelism
- **Total time**: 3-hour config may take ~4-4.5 hours with optimizations

## Common Issues and Fixes

### Issue: Checkpoints Not Saving to Persistent Volume

**Problem**: Checkpoints saved to container disk (lost on pod restart) instead of network volume.

**Root Cause**: Config used relative path `./outputs/grpo` which resolved to `/workspace/AReaL/outputs/grpo` (container disk).

**Fix**: Changed all configs to use absolute path `/workspace/outputs/grpo` (mounted volume).

**Verification**:
```bash
# Check mounted volume
ls -lh /workspace/outputs/grpo/checkpoints/
```

### Issue: CUDA Out of Memory on A40/RTX 5090

**Problem**: OOM errors on GPUs with 32-48GB memory (A40, RTX 5090, RTX 4090).

**Solution**: Use memory-optimized configs (automatically selected by `run_training_cloud.sh`):
- Reduced SGLang memory fraction: 0.8 → 0.5
- Enabled gradient checkpointing
- Reduced batch size: 8 → 4
- Reduced max new tokens: 512 → 256
- Reduced concurrent rollouts: 32 → 16

**Additional Optimizations** (if still OOM):
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Issue: SGLang Server Disconnect (Step 188 Example)

**Problem**: Task reward dropped to zero and stayed zero for hundreds of steps, corrupting the model.

**Root Cause**: SGLang server crashed/disconnected, but training continued with failed rollouts.

**Solution**: Circuit breaker now stops training after 10 consecutive zero-reward steps, preventing model corruption.

**Prevention**:
- Circuit breaker monitors `task_reward/avg` and stops on sustained zero rewards
- Increased `rollout.request_timeout` to 7200 seconds
- Increased `rollout.request_retries` to 5
- Increased `rollout.setup_timeout` to 300 seconds

### Issue: RTX 5090 "Unknown config name" Error

**Problem**: Training failed with "Unknown config name: h200" error on RTX 5090.

**Root Cause**: Script tried to use H200-specific config on RTX 5090.

**Solution**: Consolidated configs - `full` config now validates GPU requirements and works on all high-memory GPUs.

### Issue: Spot Instance Interruption

**Problem**: Training stopped abruptly, WandB shows "running" but no updates.

**Root Cause**: Spot instance was preempted by another user or provider needs.

**Solution**: 
- Always use network volumes for checkpoint persistence
- Resume from checkpoint using recovery workflow
- Monitor WandB to detect interruptions early

## Troubleshooting

### Training Crashes

**Symptoms**:
- Pod status changes to "Failed" or "Stopped"
- Terminal output ends abruptly
- GPU usage drops to zero
- No clear error message

**Solutions**:
1. **Check logs**: Look for `CUDA out of memory` or SGLang server errors
2. **Reduce memory usage**:
   - Reduce `sglang.mem_fraction_static` to 0.5 or 0.7
   - Enable `actor.gradient_checkpointing: true`
   - Reduce `train_dataset.batch_size`
   - Reduce `gconfig.max_new_tokens`
3. **Increase resilience**:
   - Increase `rollout.request_timeout` to 7200
   - Increase `rollout.request_retries` to 5
4. **Check circuit breaker**: May have triggered due to zero rewards

### Checkpoints Not Saving

**Symptoms**:
- No `outputs/grpo/checkpoints` directory after restart
- Cannot resume training from previous state

**Solutions**:
1. **Create network volume**: Mount to `/workspace/outputs`
2. **Verify fileroot**: Ensure config points to `/workspace/outputs/grpo`
3. **Check permissions**: Ensure write access to volume

### WandB Shows "Running" But Training Stopped

**Symptoms**:
- WandB run shows "running" state
- No updates in hours
- Training was healthy before stopping

**Likely Cause**: Spot instance interruption or external termination

**Solutions**:
1. **Check RunPod dashboard**: Verify pod status
2. **Check for checkpoints**: Use `list_checkpoints.py`
3. **Resume from checkpoint**: Use recovery workflow

### No Checkpoints Found

**Solutions**:
1. **Verify path**: Check that fileroot is correct (`/workspace/outputs/grpo`)
2. **Check user**: Checkpoints saved under user who ran training (usually `root`)
3. **Verify experiment/trial names**: Must match exactly
4. **Check network volume**: Ensure volume is mounted correctly

### Recovery Fails

**Solutions**:
1. **Check recover_checkpoint directory**: Must have necessary files
2. **Verify checkpoint integrity**: Check that model files exist
3. **Check logs**: Look for errors in recovery process
4. **Verify step_info.json**: Must contain valid step information

## Best Practices

### Before Training

1. **Set up network volume** - Mount to `/workspace/outputs` for persistence
2. **Configure WandB API key** - Set as environment variable in RunPod
3. **Choose appropriate config** - Match GPU capabilities to config
4. **Test with short run** - Use `fast` config first to verify setup

### During Training

1. **Monitor WandB** - Check training progress regularly
2. **Watch for zero rewards** - Circuit breaker will stop if detected
3. **Check GPU memory** - Ensure not approaching limits
4. **Verify checkpoints** - Confirm checkpoints are being saved

### After Interruption

1. **Check WandB status** - Use `check_wandb_run.py` to diagnose
2. **List checkpoints** - Use `list_checkpoints.py` to find saved checkpoints
3. **Set up recovery** - Use `setup_recovery.py` to prepare recovery
4. **Resume training** - Use `recover.mode=auto` to continue

### Cost Optimization

1. **Use spot instances** - 50-70% savings for experiments
2. **Use regular instances** - For production/long runs
3. **Monitor usage** - Stop pods when not in use
4. **Choose right GPU** - Match GPU to training needs

### Configuration Tips

1. **Memory-constrained GPUs**: Use `1hour` or `3hour` configs (already optimized)
2. **High-memory GPUs**: Can use `full` config for complete training
3. **Adjust checkpoint frequency**: Set `saver.freq_steps` for more frequent saves
4. **Enable recovery**: Set `recover.mode=auto` for automatic recovery

## Recovery Tools

### 1. List Checkpoints

```bash
# List all checkpoints
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS

# List checkpoints before a specific step
python examples/cloud_gsm8k/list_checkpoints.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS \
    --before-step 188
```

### 2. Set Up Recovery

```bash
# Automatically find and set up latest checkpoint
python examples/cloud_gsm8k/setup_recovery.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS \
    --before-step 188
```

### 3. Interactive Recovery Guide

```bash
# Get step-by-step recovery instructions
python examples/cloud_gsm8k/resume_training.py \
    --experiment-name gsm8k-grpo-cloud-1hour \
    --trial-name trial_YYYYMMDD_HHMMSS \
    --before-step 188
```

### 4. Check WandB Run Status

```bash
# Diagnose WandB run (check if interrupted, find last step, etc.)
python examples/cloud_gsm8k/check_wandb_run.py \
    --trial-name trial_YYYYMMDD_HHMMSS
```

## GRPO Training Tuning Guide

This section provides recommendations to improve GRPO training accuracy on GSM8K.

### 🔴 Critical Issues (Likely Causing Degradation)

#### 1. Max New Tokens Too Short
**Current**: `max_new_tokens: 256`  
**Problem**: Many math problems need longer reasoning chains. Truncation prevents complete solutions.

**Recommendation**:
```yaml
gconfig:
  max_new_tokens: 512  # For 0.5B model (or 1024 if you have memory)
```

#### 2. No KL Penalty (KL Control = 0)
**Current**: `kl_ctl: 0.0`  
**Problem**: Model can deviate too far from reference, causing instability and degradation.

**Recommendation**:
```yaml
actor:
  kl_ctl: 0.01  # Start small, can increase to 0.05-0.1 if needed
```

#### 3. Epsilon Clip Too High
**Current**: `eps_clip: 0.4`  
**Problem**: Allows too aggressive policy updates, leading to instability.

**Recommendation**:
```yaml
actor:
  eps_clip: 0.2  # Standard PPO value, more conservative
```

#### 4. PPO Minibatches = 1
**Current**: `ppo_n_minibatches: 1`  
**Problem**: No data reuse, less stable updates.

**Recommendation**:
```yaml
actor:
  ppo_n_minibatches: 2  # Data reuse for stability
```

### 🟡 Important Issues

#### 5. Reward Bias Too Negative
**Current**: `reward_bias: -0.5`  
**Problem**: Shifts all rewards negative, making learning harder.

**Recommendation**:
```yaml
actor:
  reward_bias: 0.0  # No bias, let rewards be natural
```

#### 6. Learning Rate May Be Too High
**Current**: `lr: 1.70e-5`  
**Problem**: May cause instability with other aggressive settings.

**Recommendation**:
```yaml
actor:
  optimizer:
    lr: 1.30e-5  # Slightly reduced for stability
```

#### 7. Warmup Too Short
**Current**: `warmup_steps_proportion: 0.001`  
**Problem**: Model doesn't have time to adapt gradually.

**Recommendation**:
```yaml
actor:
  optimizer:
    warmup_steps_proportion: 0.01  # 1% warmup (or 0.05 for 5%)
```

#### 8. Epochs May Be Too Few
**Current**: `total_train_epochs: 3`  
**Problem**: Model may not have enough training to learn patterns.

**Recommendation**:
```yaml
total_train_epochs: 5  # More epochs for better convergence
```

**💡 Tip**: You can override epochs without modifying YAML files:
```bash
bash examples/cloud_gsm8k/run_training_cloud.sh standard_2000samples_2GPUs_v3 6
```

### Monitoring & Validation

#### Key Metrics to Watch in W&B

1. **`grpo_actor/task_reward/avg`**: Should increase over time
2. **`grpo_actor/behave_approx_kl/avg`**: Should stay small (< 0.1) - if too high, increase `kl_ctl`
3. **`grpo_actor/behave_imp_weight/avg`**: Should stay around 1.0 - if too high (> 2.0), reduce `eps_clip` or increase `kl_ctl`
4. **`grpo_actor/loss/avg`**: Should decrease over time
5. **Test accuracy**: Should improve after training

#### Early Stopping Indicators

**Stop training if**:
- `behave_approx_kl` > 0.5 (model deviating too far)
- `task_reward/avg` drops below baseline for > 10 steps
- Test accuracy decreases significantly

### Testing Strategy

1. **Start with high-priority changes only**:
   - `max_new_tokens: 512`
   - `kl_ctl: 0.01`
   - `eps_clip: 0.2`
   - `ppo_n_minibatches: 2`

2. **Run a test training** (use `fastest` or `fast` config first)

3. **Monitor W&B metrics** - check if:
   - Task reward increases
   - KL divergence stays low
   - Test accuracy improves

4. **If working, add medium-priority changes**:
   - Adjust learning rate
   - Remove reward bias
   - Increase warmup

5. **If still improving, add low-priority changes**:
   - Increase group size
   - More epochs (or use epoch override)

### Interval Testing: Comprehensive Checkpoint Evaluation

For long training runs (20+ epochs), use **interval testing** to evaluate checkpoints at regular intervals instead of just the final checkpoint. This provides:

- **Training progression tracking**: See how accuracy improves over epochs
- **Optimal checkpoint identification**: Find the best-performing epoch (may not be the latest)
- **Overfitting detection**: Observe if accuracy plateaus or decreases after a certain epoch
- **Comprehensive analysis**: Get a complete picture of model performance throughout training

#### How to Use Interval Testing

**Basic usage:**
```bash
# Test checkpoints at 5-epoch intervals (4, 9, 14, 19, 24, etc.)
bash examples/cloud_gsm8k/test_full_dataset.sh "" examples/cloud_gsm8k/train_logs/logs_grpo_1k_v3_25epochs.txt --test-intervals
```

**With auto-detection:**
```bash
# Script automatically finds latest checkpoint
bash examples/cloud_gsm8k/test_full_dataset.sh "" "" --test-intervals
```

**Important**: The `--test-intervals` flag must be passed to `test_full_dataset.sh`, not to `run_training_cloud.sh`.

#### Interval Testing Details

- **Interval pattern**: Tests at epochs 4, 9, 14, 19, 24, etc. (5-epoch intervals starting from epoch 4, since epoch 0 is the initial state)
- **Full dataset evaluation**: Each checkpoint is tested on all 1319 GSM8K test samples
- **Resume capability**: If interrupted (e.g., container restart), the script resumes from where it left off using completion markers
- **Skip already-tested**: Automatically skips checkpoints that have already been tested (log file exists with "FINAL ACCURACY")
- **Auto-upload**: All interval test logs are automatically uploaded if auto-upload is configured

#### Example Results from Interval Testing

From a 25-epoch training run:
- **Epoch 4**: 54.28% accuracy (716/1319)
- **Epoch 9**: 56.33% accuracy (743/1319)
- **Epoch 14**: 55.72% accuracy (735/1319)
- **Epoch 19**: **56.48% accuracy (745/1319)** ← Best performance
- **Epoch 24**: 55.88% accuracy (737/1319)

This shows that the best checkpoint was at epoch 19, not the final epoch 24, indicating potential overfitting in later epochs.

#### Best Practices for Interval Testing

1. **Use for long training runs**: Especially valuable for 20+ epoch training
2. **Run after training completes**: Add to container starter code to run automatically
3. **Monitor progress**: Each interval test takes time (full dataset evaluation), so be patient
4. **Check completion markers**: If script seems stuck, verify completion markers exist
5. **Analyze results**: Compare accuracy across epochs to identify optimal checkpoint and detect overfitting

### Expected Improvements

With these changes, you should see:
- **Task reward**: Increase from ~0.2-0.3 to ~0.4-0.6 (or higher)
- **Test accuracy**: Improve from baseline instead of degrading
- **Training stability**: More consistent improvements, less variance

### Troubleshooting

#### If accuracy still decreases:

1. **Increase KL penalty**: Try `kl_ctl: 0.05` or `0.1`
2. **Reduce learning rate further**: Try `lr: 1.00e-5`
3. **Reduce epsilon clip more**: Try `eps_clip: 0.15`
4. **Check reward signal**: Verify rewards are being computed correctly
5. **More data**: Increase `max_train_samples` to 2000 or 4000

#### If training is unstable:

1. **Increase minibatches**: `ppo_n_minibatches: 4`
2. **Reduce learning rate**: `lr: 1.00e-5`
3. **Increase KL penalty**: `kl_ctl: 0.05`
4. **Check for OOM**: Reduce batch size or max_new_tokens

## Epoch Analysis: When Would More Epochs Help?

### Key Metrics from Training

Based on training logs and W&B metrics:

- **Training reward**: 50.3% accuracy during training (`eval-rollout/reward`)
- **Test accuracy**: 38% (final test)
- **KL divergence**: 0.24 (well below 0.5 threshold = no overfitting)
- **Task reward variance**: High (0.125 to 0.9375) = model still learning

### Insights: Would More Epochs Help?

**✅ YES - More epochs likely to help** because:

1. **No overfitting signs**:
   - KL divergence is very small (< 0.24)
   - Model is staying close to reference policy
   - No degradation in training metrics

2. **Training reward vs test accuracy gap**:
   - Training shows 50.3% accuracy
   - Test shows 38% accuracy
   - **12% gap** suggests model needs more training to generalize

3. **Conservative hyperparameters**:
   - Learning rate: 1.00e-5 (very conservative)
   - KL control: 0.02 (strong regularization)
   - Epochs: 4 (relatively few)
   - **Room for more training without overfitting**

4. **Task reward variability**:
   - High variance suggests model is still learning
   - Not yet converged to stable high performance

### Recommendations

**Option 1: Increase to 5-6 epochs** (Recommended)
- **Rationale**: No overfitting, training reward higher than test
- **Expected improvement**: +2-5% accuracy (40-43% total)
- **Time cost**: +1.5-3 hours
- **Risk**: Low (KL divergence well-controlled)

**Option 2: Increase to 5-6 epochs + slightly higher LR**
- **Rationale**: Current LR (1.00e-5) is very conservative
- **Changes**: Epochs: 5-6, LR: 1.10e-5 to 1.15e-5
- **Expected improvement**: +3-6% accuracy (41-44% total)
- **Risk**: Low-Medium (monitor KL divergence)

### When Would We Hit Overfitting?

**Overfitting Indicators to Watch:**
1. **KL divergence > 0.5**: Model deviating too far from reference
2. **Training reward increases but test accuracy decreases**: Classic overfitting
3. **Task reward variance decreases but test accuracy plateaus**: Model memorizing
4. **Importance weights > 2.0**: Model making large policy changes

**Estimated Overfitting Threshold:**
- **Conservative estimate**: 6-7 epochs (with current settings)
- **Aggressive estimate**: 8-10 epochs (if KL divergence stays low)
- **With more data (4000 samples)**: 7-9 epochs

**Recommended**: Start with **5-6 epochs** and monitor W&B metrics. If KL divergence stays low and test accuracy improves, continue to 7 epochs.

## Troubleshooting: Reasoning Model 4-GPU Training

### Problem

When running the `reasoning_2000samples_4GPUs` config on RunPod with 4x A40 GPUs, the training failed consistently.

**Initial Error:** `CUDA error: CUDA-capable device(s) is/are busy or unavailable` on SGLang initialization (GPU 0).

**Root Cause:** Physical GPU 0 is dead/zombie. Regardless of cleanup or reordering, any process attempting to initialize a CUDA context on Physical GPU 0 fails.

### Solution

We must exclude Physical GPU 0 entirely from the training job. Since 0 is dead, we use **3 GPUs (3, 2, 1)**.

**Infrastructure Logic:**
- **Reordering**: `CUDA_VISIBLE_DEVICES=3,2,1,0` exposes Physical GPUs [3, 2, 1, 0] as Logical GPUs [0, 1, 2, 3]
- **Selection**: The launcher picks the first N logical GPUs required by the config
- If config needs 3 GPUs, it picks Logical 0, 1, 2 → Maps to Physical 3, 2, 1
- Physical 0 (Logical 3) remains unused

**Configuration Fixes:**
- Changed `allocation_mode` from `sglang.d1t1p1+d3t1p1` (4 GPUs) to `sglang.d1t1p1+d2t1p1` (3 GPUs)
- 1 GPU for SGLang (Logical 0 → Physical 3)
- 2 GPUs for Training (Logical 1, 2 → Physical 2, 1)

## GRPO Training Learnings from Experiments

This section documents key learnings from actual GRPO training experiments on GSM8K with Qwen2.5-0.5B-Instruct.

### Experiment Timeline and Results

#### Baseline Performance
- **Baseline model accuracy**: ~38% (untrained Qwen2.5-0.5B-Instruct)
- **Initial goal**: Improve accuracy through GRPO training

#### Experiment 1: Improved Config (1000 samples)
- **Settings**: `lr: 1.30e-5`, `kl_ctl: 0.01`, `eps_clip: 0.2`, `ppo_n_minibatches: 2`, `total_train_epochs: 5`
- **Result**: Model solved problems correctly but didn't consistently use `\boxed{}` format
- **Key learning**: Model needs more training to learn format requirements

#### Experiment 2: v2 Config (1000 samples) - Model Collapse
- **Settings**: `lr: 1.50e-5` (increased), `total_train_epochs: 6` (increased), `warmup_steps_proportion: 0.02`
- **Result**: **0% accuracy** - Complete model collapse
- **Symptoms**: 
  - Model stopped generating coherent answers
  - Lost all reasoning capability
  - Generated gibberish or incomplete responses
- **Root cause**: Too aggressive learning rate combined with more epochs
- **Key learning**: ⚠️ **Aggressive hyperparameters can cause catastrophic model collapse**

#### Experiment 3: v3 Conservative Config (1000 samples) - Success
- **Settings**: `lr: 1.00e-5` (reduced), `total_train_epochs: 4` (reduced), `warmup_steps_proportion: 0.05` (increased), `kl_ctl: 0.02` (increased)
- **Result**: **38% accuracy** - Maintained baseline performance
- **Key learning**: ✅ **Conservative hyperparameters prevent collapse and maintain stability**

#### Experiment 4: v4 Config (1000 samples) - Regression
- **Settings**: `lr: 1.20e-5` (slightly increased from v3), `total_train_epochs: 5` (increased from v3)
- **Result**: **12% accuracy** - Significant regression from v3's 38%
- **Symptoms**:
  - Model lost `\boxed{}` format usage
  - Made more reasoning errors
  - Performance worse than baseline
- **Key learning**: ⚠️ **Even small increases in aggressiveness can cause regression**

#### Experiment 5: v3 Config with 2000 Samples
- **Settings**: Same as v3 conservative, but `max_train_samples: 2000`
- **Result**: **38% accuracy** - Same as 1000-sample v3
- **Key learning**: ⚠️ **More data alone doesn't guarantee improvement** - Other factors (reward function, training dynamics, model capacity) may be limiting

### Critical Learnings

#### 1. Model Collapse is Real and Catastrophic
- **What we learned**: Aggressive hyperparameters (high LR, many epochs) can cause complete model collapse
- **Evidence**: v2 config resulted in 0% accuracy, model lost all reasoning capability
- **Prevention**: Always start with conservative hyperparameters
- **Recovery**: Once collapsed, model cannot recover - must restart from baseline

#### 2. Conservative Hyperparameters Are Essential
- **What we learned**: Conservative settings (low LR, moderate epochs, high warmup, strong KL control) maintain stability
- **Evidence**: v3 conservative maintained 38% accuracy (baseline level)
- **Recommended starting point**:
  - `lr: 1.00e-5` (very conservative)
  - `kl_ctl: 0.02` (strong regularization)
  - `eps_clip: 0.2` (standard PPO value)
  - `warmup_steps_proportion: 0.05` (5% warmup for stability)
  - `total_train_epochs: 4` (moderate, can increase if no overfitting)

#### 3. Small Changes Can Have Large Effects
- **What we learned**: Even small increases in aggressiveness (e.g., LR from 1.00e-5 to 1.20e-5) can cause significant regression
- **Evidence**: v4 showed 12% accuracy (down from v3's 38%) with only slight hyperparameter increases
- **Implication**: Hyperparameter tuning requires careful, incremental changes
- **Best practice**: Change one hyperparameter at a time, monitor closely

#### 4. More Data Doesn't Always Help
- **What we learned**: Doubling data (1000 → 2000 samples) with same hyperparameters didn't improve accuracy
- **Evidence**: 2000-sample v3 config achieved same 38% as 1000-sample v3
- **Possible reasons**:
  - Reward function may not be providing sufficient signal
  - Training dynamics may be limiting (conservative hyperparameters)
  - Model capacity may be limiting (0.5B model)
  - Data quality/diversity may be the bottleneck
- **Implication**: Before increasing data, consider:
  - Improving reward function
  - Adjusting training dynamics
  - Using larger model
  - Ensuring data diversity

#### 5. Format Learning is Challenging
- **What we learned**: Model can solve problems correctly but fail to use required format (`\boxed{}`)
- **Evidence**: Improved config showed correct answers but format extraction failed
- **Possible solutions**:
  - More epochs (with conservative settings)
  - Format-specific reward shaping
  - Format examples in prompt
  - Post-processing to add format

#### 6. Training-Test Gap Indicates Underfitting
- **What we learned**: Training accuracy (50.3%) > test accuracy (38%) suggests model needs more training
- **Evidence**: 2000-sample v3 training showed 50.3% training accuracy but 38% test accuracy
- **Implication**: More epochs (with conservative settings) may help close the gap
- **Risk**: Must monitor KL divergence to avoid overfitting

#### 7. KL Divergence is a Reliable Overfitting Indicator
- **What we learned**: KL divergence < 0.24 indicates no overfitting, room for more training
- **Evidence**: v3 config showed KL divergence of 0.24, well below 0.5 threshold
- **Best practice**: Monitor KL divergence in W&B, stop if > 0.5

### Hyperparameter Sensitivity Ranking

Based on experiments, hyperparameters ranked by sensitivity (most sensitive first):

1. **Learning Rate** (`lr`) - ⚠️ **Most critical**
   - Small changes (1.00e-5 → 1.20e-5) caused 26% accuracy drop
   - Start very conservative (1.00e-5), increase slowly

2. **Total Training Epochs** (`total_train_epochs`) - ⚠️ **Very sensitive**
   - More epochs with high LR caused collapse
   - Start moderate (4 epochs), increase if no overfitting

3. **KL Control** (`kl_ctl`) - ⚠️ **Important for stability**
   - Higher values (0.02) prevent collapse
   - Start at 0.02, can reduce if too conservative

4. **Warmup Steps Proportion** (`warmup_steps_proportion`) - ⚠️ **Important for stability**
   - Higher values (0.05) help prevent collapse
   - Start at 0.05, can reduce if training too slow

5. **Epsilon Clip** (`eps_clip`) - ✅ **Moderate sensitivity**
   - Standard value (0.2) works well
   - Can adjust slightly (0.15-0.25) if needed

6. **PPO Minibatches** (`ppo_n_minibatches`) - ✅ **Low sensitivity**
   - Value of 2 works well
   - Can increase to 4 for more stability

### Recommended Training Strategy

#### Phase 1: Establish Baseline Stability
1. **Start with v3 conservative settings**:
   - `lr: 1.00e-5`
   - `kl_ctl: 0.02`
   - `eps_clip: 0.2`
   - `warmup_steps_proportion: 0.05`
   - `total_train_epochs: 4`
   - `ppo_n_minibatches: 2`

2. **Verify stability**:
   - Check accuracy maintains or improves from baseline
   - Monitor KL divergence stays < 0.3
   - Verify no model collapse

#### Phase 2: Incremental Improvements
1. **If stable, try more epochs**:
   - Increase to 5-6 epochs (use epoch override)
   - Monitor KL divergence and test accuracy
   - Stop if overfitting signs appear

2. **If still stable, try slightly higher LR**:
   - Increase to 1.10e-5 (10% increase)
   - Monitor closely for regression
   - Revert if accuracy drops

3. **If still improving, try more data**:
   - Increase to 2000-4000 samples
   - Monitor if accuracy improves
   - Consider if other factors are limiting

#### Phase 3: Advanced Tuning
1. **Fine-tune reward function**:
   - Consider format-specific rewards
   - Adjust reward scaling if needed

2. **Experiment with model capacity**:
   - Try larger models if available
   - Consider if 0.5B is limiting factor

3. **Explore alternative algorithms**:
   - Consider reasoning model format
   - Try different RL algorithms

### Red Flags to Watch For

**Stop training immediately if you see:**
1. **Accuracy drops to 0%** - Model collapse, cannot recover
2. **Accuracy drops significantly** (>20% drop) - Hyperparameters too aggressive
3. **KL divergence > 0.5** - Model deviating too far, overfitting risk
4. **Training reward increases but test accuracy decreases** - Classic overfitting
5. **Model generates gibberish** - Collapse or corruption

**Warning signs (reduce aggressiveness):**
1. **Accuracy plateaus or decreases** - Reduce LR or epochs
2. **KL divergence increasing** - Increase `kl_ctl` or reduce LR
3. **High variance in task rewards** - Model unstable, reduce LR
4. **Importance weights > 2.0** - Policy changes too large, reduce `eps_clip`

### Success Metrics

**Good training run indicators:**
- ✅ Accuracy maintains or improves from baseline
- ✅ KL divergence stays < 0.3
- ✅ Training reward increases over time
- ✅ Test accuracy improves or maintains
- ✅ Model generates coherent, formatted answers
- ✅ No collapse or regression

**Target improvements:**
- **Baseline**: 38% accuracy
- **Conservative target**: 40-45% accuracy (5-18% improvement)
- **Optimistic target**: 50%+ accuracy (30%+ improvement)
- **Current best**: 38% (maintained baseline with v3 conservative)

### Lessons for Future Experiments

1. **Always start conservative** - Better to be too conservative than too aggressive
2. **Change one thing at a time** - Isolate effects of each hyperparameter
3. **Monitor closely** - Watch W&B metrics, test accuracy, and model outputs
4. **Have recovery plan** - Know how to revert if things go wrong
5. **Document everything** - Record all hyperparameters, results, and observations
6. **Test on small subset first** - Use `fastest` or `fast` configs before long runs
7. **Don't assume more is better** - More data, epochs, or LR doesn't always help

## Summary

✅ **Always use network volumes** for checkpoint persistence
✅ **Monitor training** via WandB to detect issues early
✅ **Use circuit breaker** to prevent model corruption
✅ **Set up recovery** before long training runs
✅ **Choose appropriate config** for your GPU
✅ **Use spot instances** for cost savings (with recovery setup)
✅ **Resume seamlessly** from checkpoints after interruptions
✅ **Memory-optimized configs** work on all GPUs (no GPU-specific configs needed)
✅ **Use epoch override** to experiment without modifying YAML files
✅ **Monitor KL divergence** to detect overfitting early
✅ **Start with conservative hyperparameters** - Better safe than sorry
✅ **Change hyperparameters incrementally** - Small changes can have large effects
✅ **More data doesn't always help** - Consider other limiting factors
✅ **Model collapse is catastrophic** - Once collapsed, cannot recover

