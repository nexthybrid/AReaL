# Self-Consistency Voting: Multiple Samples Per Checkpoint

## Overview

The ensemble testing script now supports **self-consistency voting** - generating multiple samples from the same checkpoint and voting among them. This can improve accuracy even further by leveraging the diversity of answers from a single checkpoint.

## How It Works

### Traditional Ensemble (Default)
- **1 sample per checkpoint** (greedy decoding, temperature=0.0)
- Epoch 24 will **always produce the same answer** to the same question
- Votes across 5 checkpoints: 5 total votes per question

### Self-Consistency Ensemble (New)
- **N samples per checkpoint** (sampling with temperature > 0)
- Epoch 24 can produce **different answers** to the same question
- Votes across 5 checkpoints × N samples: **5×N total votes per question**

## Why This Helps

1. **Diversity**: Sampling introduces diversity even within the same checkpoint
2. **Error Correction**: If one sample makes a mistake, other samples can correct it
3. **Consensus**: Majority voting finds the most consistent answer across all samples
4. **Better Accuracy**: Often improves accuracy beyond single-sample ensemble

## Usage

### Basic Usage (Single Sample Per Checkpoint - Default)

```bash
# Traditional ensemble: 1 sample per checkpoint (greedy)
bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh \
    /workspace/outputs/grpo/checkpoints/root/experiment/trial0/default \
    "4 9 14 19 24"
```

### Self-Consistency Voting (Multiple Samples Per Checkpoint)

```bash
# Generate 5 samples per checkpoint with temperature=0.7
export N_SAMPLES=5
export TEMPERATURE=0.7

bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh \
    /workspace/outputs/grpo/checkpoints/root/experiment/trial0/default \
    "4 9 14 19 24"
```

Or directly with Python:

```bash
python3 examples/cloud_gsm8k/test_checkpoint_ensemble.py \
    --checkpoint-dir /workspace/outputs/grpo/checkpoints/root/experiment/trial0/default \
    --epochs 4 9 14 19 24 \
    --n-samples 5 \
    --temperature 0.7
```

## Parameters

### `--n-samples` / `N_SAMPLES`
- **Default**: 1 (single sample, greedy decoding)
- **Recommended**: 5-10 for self-consistency
- **Effect**: Number of samples to generate per checkpoint per question
- **Trade-off**: More samples = better accuracy but slower (5x slower for 5 samples)

### `--temperature` / `TEMPERATURE`
- **Default**: 0.0 (greedy decoding)
- **Recommended**: 0.7-1.0 for diverse samples
- **Effect**: Controls randomness in sampling
  - `0.0`: Deterministic (always same answer)
  - `0.7-1.0`: Good diversity without too much randomness
  - `>1.0`: Very random, may hurt accuracy

## Example: Self-Consistency with 5 Samples

```bash
# Generate 5 samples from each checkpoint
export N_SAMPLES=5
export TEMPERATURE=0.7

bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh \
    /workspace/outputs/grpo/checkpoints/root/experiment/trial0/default \
    "4 9 14 19 24"
```

**What happens:**
1. For each checkpoint (4, 9, 14, 19, 24):
   - Generates 5 different answers to each question (using temperature=0.7)
   - Extracts answers from all 5 samples
2. For each question:
   - Collects all 25 answers (5 checkpoints × 5 samples)
   - Performs majority voting across all 25 answers
   - Selects the most frequent answer
3. Calculates accuracy using the voted answer

## Expected Improvements

Based on research and experiments:

- **Traditional ensemble** (5 checkpoints, 1 sample each): ~64% accuracy
- **Self-consistency ensemble** (5 checkpoints, 5 samples each): **~66-68% accuracy** (estimated)
- **Improvement**: +2-4% over traditional ensemble

## Performance Considerations

### Time Cost
- **1 sample per checkpoint**: ~50-75 minutes for 5 checkpoints
- **5 samples per checkpoint**: ~250-375 minutes (5x slower)
- **10 samples per checkpoint**: ~500-750 minutes (10x slower)

### Memory Cost
- Multiple samples require more GPU memory
- Batch size may need to be reduced if OOM occurs
- Use `TEST_BATCH_SIZE=16` or `TEST_BATCH_SIZE=8` if needed

### Recommended Settings

**For quick testing:**
```bash
export N_SAMPLES=3
export TEMPERATURE=0.7
export TEST_BATCH_SIZE=16
```

**For best accuracy:**
```bash
export N_SAMPLES=5
export TEMPERATURE=0.7
export TEST_BATCH_SIZE=16
```

**For maximum accuracy (if time allows):**
```bash
export N_SAMPLES=10
export TEMPERATURE=0.8
export TEST_BATCH_SIZE=8
```

## How Voting Works

### Example: Question with 5 Samples from Epoch 24

**Epoch 24 generates:**
- Sample 1: "18"
- Sample 2: "18"
- Sample 3: "14"
- Sample 4: "18"
- Sample 5: "18"

**Vote count:**
- "18": 4 votes
- "14": 1 vote

**Selected answer**: "18" (majority)

### Example: Cross-Checkpoint + Self-Consistency

**Epoch 4 (5 samples)**: ["18", "18", "14", "18", "18"]
**Epoch 9 (5 samples)**: ["18", "18", "18", "18", "14"]
**Epoch 14 (5 samples)**: ["18", "18", "18", "18", "18"]
**Epoch 19 (5 samples)**: ["18", "18", "18", "18", "18"]
**Epoch 24 (5 samples)**: ["18", "18", "14", "18", "18"]

**Total votes:**
- "18": 23 votes
- "14": 2 votes

**Selected answer**: "18" (overwhelming majority)

## Comparison: Traditional vs Self-Consistency

| Method | Samples | Votes/Question | Expected Accuracy | Time |
|--------|---------|----------------|-------------------|------|
| Single checkpoint | 1 | 1 | 60.50% | ~15 min |
| Traditional ensemble | 5 | 5 | 64.29% | ~75 min |
| Self-consistency (5 samples) | 25 | 25 | ~66-68% | ~375 min |

## Best Practices

1. **Start with traditional ensemble** to establish baseline
2. **Try self-consistency with 3-5 samples** for improvement
3. **Monitor accuracy vs time trade-off**
4. **Use appropriate temperature** (0.7-1.0 is usually best)
5. **Reduce batch size** if OOM occurs with multiple samples

## Troubleshooting

### Out of Memory (OOM)
- Reduce `TEST_BATCH_SIZE` to 16 or 8
- Reduce `N_SAMPLES` to 3 instead of 5

### Too Slow
- Reduce `N_SAMPLES` to 3
- Use fewer checkpoints (e.g., just epochs 14, 19, 24)

### Low Accuracy
- Increase `N_SAMPLES` to 10
- Adjust `TEMPERATURE` (try 0.8-1.0)
- Ensure checkpoints are from different training stages

## Summary

**Answer to your questions:**

1. **Will epoch 24 always produce the same answer?**
   - **With greedy decoding (temperature=0)**: Yes, always the same
   - **With sampling (temperature>0)**: No, can produce different answers

2. **Can we vote among different answers from epoch 24?**
   - **Yes!** Use `--n-samples 5 --temperature 0.7` to generate 5 samples from epoch 24
   - The script will vote among all samples from all checkpoints
   - This is called "self-consistency" and often improves accuracy

The updated script now supports both approaches and can combine them for maximum accuracy!

