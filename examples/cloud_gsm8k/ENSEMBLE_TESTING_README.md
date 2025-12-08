# Checkpoint Ensemble Testing

This document describes how to use the checkpoint ensemble testing script to improve accuracy by pooling answers from multiple checkpoints using majority voting.

## Overview

The ensemble testing approach:
1. Loads multiple checkpoints (default: epochs 4, 9, 14, 19, 24)
2. For each checkpoint, runs inference on the full GSM8K test set (1319 samples)
3. Collects all answers from all checkpoints
4. For each question, performs majority voting to select the most frequent answer
5. If there's a tie, defaults to the answer from the latest checkpoint (epoch 24)
6. Calculates overall accuracy of the ensemble approach

## Why Ensemble Testing?

- **Improved Accuracy**: Combining predictions from multiple checkpoints can improve accuracy over any single checkpoint
- **Robustness**: Reduces the impact of individual checkpoint errors
- **Optimal Checkpoint Selection**: Helps identify which checkpoints contribute most to accuracy

## Usage

### Basic Usage

```bash
# Test with default epochs (4, 9, 14, 19, 24)
bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh /workspace/outputs/grpo/checkpoints/root/experiment/trial0/default
```

### Custom Epochs

```bash
# Test with custom epochs
bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh /workspace/outputs/grpo/checkpoints/root/experiment/trial0/default "4 9 14"
```

### Auto-Detect from Log File

```bash
# Extract checkpoint directory from training log
bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh "" "" examples/cloud_gsm8k/train_logs/logs_25epochs.txt
```

### Direct Python Script Usage

```bash
python3 examples/cloud_gsm8k/test_checkpoint_ensemble.py \
    --checkpoint-dir /workspace/outputs/grpo/checkpoints/root/experiment/trial0/default \
    --epochs 4 9 14 19 24 \
    --batch-size 32 \
    --max-new-tokens 512
```

## Arguments

### Bash Script Arguments

1. **checkpoint_dir** (positional, optional): Base directory containing checkpoints
   - If not provided, script will try to extract from log file or find latest checkpoint
   - Example: `/workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-2gpu-1000samples-v3-conservative/trial0/default`

2. **epochs** (positional, optional): Space-separated list of epochs to test
   - Default: `4 9 14 19 24`
   - Example: `"4 9 14"` for testing only epochs 4, 9, and 14

3. **log_file** (positional, optional): Training log file to extract checkpoint path from

### Python Script Arguments

- `--checkpoint-dir`: Base directory containing checkpoints (required)
- `--epochs`: Epochs to test (default: 4 9 14 19 24)
- `--max-new-tokens`: Maximum new tokens to generate (default: 512)
- `--batch-size`: Batch size for inference (default: 32)
- `--log-dir`: Directory for log files (default: /workspace/outputs/grpo/test_logs)
- `--checkpoint-pattern`: Pattern to match checkpoint directories (default: epoch{epoch}epochstep*globalstep*)

### Environment Variables

- `TEST_BATCH_SIZE`: Override default batch size (default: 32)
  - For smaller GPUs, reduce to 16 or 8
  - For A100 80GB, can increase to 64

## How It Works

### 1. Checkpoint Loading

The script loads one checkpoint at a time to minimize GPU memory usage:
- Loads model and tokenizer from checkpoint
- Processes all 1319 test samples
- Saves results and frees GPU memory
- Moves to next checkpoint

### 2. Answer Extraction

For each checkpoint, the script:
- Generates answers for all test samples
- Extracts numerical answers using AReaL's math parser
- Handles various answer formats (\\boxed{}, plain numbers, etc.)

### 3. Majority Voting

For each question:
- Collects answers from all checkpoints
- Counts occurrences of each answer
- Selects the most frequent answer
- If there's a tie, uses the answer from the latest checkpoint (epoch 24)

### 4. Accuracy Calculation

- Compares ensemble answer with ground truth using AReaL's math parser
- Calculates overall accuracy
- Also reports individual checkpoint accuracies for comparison

## Output

The script generates several log files:

1. **Individual checkpoint logs**: `ensemble_checkpoint_epoch{N}.log`
   - Contains detailed results for each checkpoint
   - Saved in `/workspace/outputs/grpo/test_logs/`

2. **Ensemble voting log**: `ensemble_majority_voting_{timestamp}.log`
   - Contains majority voting results
   - Includes per-question breakdown
   - Shows which checkpoint answers were selected

3. **Console output**: Summary printed to stdout
   - Ensemble accuracy
   - Individual checkpoint accuracies
   - Log file locations

## Example Output

```
================================================================================
ENSEMBLE TESTING SUMMARY
================================================================================
Ensemble Accuracy: 58.45% (771/1319)

Individual Checkpoint Accuracies:
  Epoch 4: 54.28% (716/1319)
  Epoch 9: 56.33% (743/1319)
  Epoch 14: 55.72% (735/1319)
  Epoch 19: 56.48% (745/1319)
  Epoch 24: 55.88% (737/1319)

Detailed log: /workspace/outputs/grpo/test_logs/ensemble_majority_voting_20251206_123456.log
================================================================================
```

## Performance Considerations

### GPU Memory

- Script loads one checkpoint at a time to minimize memory usage
- Batch size can be adjusted based on available GPU memory
- Default batch size (32) is optimized for A100 80GB

### Runtime

- Each checkpoint takes ~10-15 minutes to process all 1319 samples (on A100)
- Total runtime: ~50-75 minutes for 5 checkpoints
- Can be run in background on RunPod

### Efficiency

- Processes all samples with one checkpoint before moving to next
- Avoids repeated model loading/unloading
- Clears GPU memory between checkpoints

## Best Practices

1. **Use interval checkpoints**: Test checkpoints at regular intervals (e.g., every 5 epochs)
2. **Include latest checkpoint**: Always include the latest checkpoint for tie-breaking
3. **Monitor individual accuracies**: Compare ensemble accuracy with individual checkpoint accuracies
4. **Check log files**: Review detailed logs to understand which checkpoints contribute most

## Troubleshooting

### Checkpoint Not Found

If the script can't find checkpoints:
- Verify checkpoint directory path is correct
- Check that checkpoint pattern matches your checkpoint naming convention
- Use `--checkpoint-pattern` to customize the pattern

### Out of Memory

If you get OOM errors:
- Reduce batch size: `TEST_BATCH_SIZE=16 bash test_checkpoint_ensemble.sh ...`
- Or use smaller batch size in Python script: `--batch-size 16`

### Slow Performance

If testing is too slow:
- Reduce number of epochs tested
- Increase batch size (if GPU memory allows)
- Use smaller max_new_tokens if answers are typically short

## Auto-Upload Logs (Email Notification)

After ensemble testing completes, the script can automatically upload logs via email (or other methods) if configured.

### Quick Setup: Email (Easiest)

Set these environment variables in your RunPod pod:

```bash
export AUTO_UPLOAD_LOGS_METHOD=email
export AUTO_UPLOAD_EMAIL_TO=your-email@example.com
export EMAIL_FROM=your-sender@example.com
export SMTP_PASSWORD=your-app-password  # Gmail: use App Password, not regular password
```

**For Gmail:**
1. Enable 2-factor authentication
2. Generate an App Password: https://myaccount.google.com/apppasswords
3. Use the App Password as `SMTP_PASSWORD`

### Other Upload Methods

The script supports the same upload methods as the training script:
- **Email** (SMTP)
- **Google Drive** (via rclone)
- **AWS S3**
- **Hugging Face Hub**
- **Weights & Biases** (as artifacts)
- **Generic webhook/API endpoint**

See `RUNPOD_COMPLETE_GUIDE.md` for detailed setup instructions for each method.

### What Gets Uploaded

The script uploads:
- Individual checkpoint logs: `ensemble_checkpoint_epoch{N}.log` (one per checkpoint)
- Ensemble voting log: `ensemble_majority_voting_{timestamp}.log` (contains final results)

## Integration with Training

You can add ensemble testing to your RunPod container starter code:

```bash
# After training completes, run ensemble testing with email notification
export AUTO_UPLOAD_LOGS_METHOD=email
export AUTO_UPLOAD_EMAIL_TO=your-email@example.com
export EMAIL_FROM=your-sender@example.com
export SMTP_PASSWORD=your-app-password

bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh \
    /workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-2gpu-1000samples-v3-conservative/trial0/default \
    "4 9 14 19 24"
```

Or use it in a separate script that runs after training:

```bash
#!/bin/bash
# Run after training completes
EXPERIMENT_NAME="gsm8k-grpo-cloud-2gpu-1000samples-v3-conservative"
TRIAL_NAME="trial0"
CHECKPOINT_DIR="/workspace/outputs/grpo/checkpoints/root/${EXPERIMENT_NAME}/${TRIAL_NAME}/default"

# Set up email notification
export AUTO_UPLOAD_LOGS_METHOD=email
export AUTO_UPLOAD_EMAIL_TO=your-email@example.com
export EMAIL_FROM=your-sender@example.com
export SMTP_PASSWORD=your-app-password

bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh "$CHECKPOINT_DIR" "4 9 14 19 24"
```

## Comparison with Single Checkpoint Testing

The ensemble approach typically provides:
- **2-5% accuracy improvement** over the best individual checkpoint
- **More robust predictions** by reducing impact of individual errors
- **Better understanding** of which checkpoints are most valuable

However, it requires:
- **More compute time** (5x for 5 checkpoints)
- **More storage** for intermediate results
- **More complex evaluation** process

Use ensemble testing when:
- You want maximum accuracy
- You have multiple checkpoints available
- You want to understand checkpoint contributions

Use single checkpoint testing when:
- You need quick results
- You have limited compute resources
- You're confident in a specific checkpoint's performance

