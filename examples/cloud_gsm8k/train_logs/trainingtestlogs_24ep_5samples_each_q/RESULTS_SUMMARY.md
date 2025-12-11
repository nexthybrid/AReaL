# Ensemble Testing Results Summary - Epoch 24 with Self-Consistency

## Test Configuration

- **Checkpoint**: Epoch 24 (final checkpoint from full GSM8K training)
- **Training Config**: Full dataset (7473 samples), v3 conservative hyperparameters, 25 epochs
- **Ensemble Method**: Self-consistency voting (5 and 10 samples per question)
- **Temperature**: 0.7 (for diverse sampling)
- **Test Dataset**: Full GSM8K test set (1319 questions)

## Results

### Single Sample vs Self-Consistency

| Method | Accuracy | Correct/Total | Improvement |
|--------|----------|---------------|-------------|
| **Single sample (baseline)** | **59.44%** | 784/1319 | - |
| **Self-consistency (5 samples)** | **63.84%** | 842/1319 | **+4.40%** |
| **Self-consistency (10 samples)** | **66.03%** | 871/1319 | **+6.59%** |

### Key Findings

1. **Self-consistency provides significant improvement**: 
   - 5 samples: +4.40% accuracy improvement (842 vs 784 correct)
   - 10 samples: +6.59% accuracy improvement (871 vs 784 correct)
   - **10 samples provides +2.19% additional improvement over 5 samples**
2. **Stable performance**: Accuracy remained consistent throughout the test
3. **Additional correct answers**: 
   - 5 samples: 58 additional correct answers over single sample
   - 10 samples: 87 additional correct answers over single sample

## Comparison with Previous Results

### Multi-Checkpoint Ensemble (Epochs 4, 9, 14, 19, 24)

From previous testing with multiple checkpoints:
- **Multi-checkpoint ensemble**: 64.29% (848/1319)
- **Individual checkpoint range**: 56.63% - 60.50%
- **Best single checkpoint**: 60.50% (Epoch 24)

### Current vs Previous

| Method | Accuracy | Notes |
|--------|----------|-------|
| **Epoch 24 + 5 samples** | **63.84%** | Single checkpoint with self-consistency |
| **Epoch 24 + 10 samples** | **66.03%** | Single checkpoint with self-consistency (best) |
| **Multi-checkpoint ensemble (5 epochs)** | **64.29%** | Epochs 4, 9, 14, 19, 24 |
| **Difference (10 samples vs multi-checkpoint)** | **+1.74%** | Self-consistency with 10 samples outperforms multi-checkpoint |

**Analysis**: 
- Self-consistency with 10 samples achieves the **highest accuracy (66.03%)**, outperforming both the 5-sample approach and the multi-checkpoint ensemble
- The 10-sample approach provides **+1.74% improvement** over multi-checkpoint ensemble
- This demonstrates that diverse reasoning paths within a single well-trained checkpoint can be more effective than combining multiple checkpoints

## Next Steps to Improve Accuracy

### 1. Combine Multi-Checkpoint + Self-Consistency (Highest Priority) ⭐

**Recommended**: Test with multiple checkpoints (4, 9, 14, 19, 24) AND 5 samples per checkpoint.

**Expected improvement**: Should achieve **65-67%** accuracy (combining benefits of both methods)

**Command**:
```bash
export SUB_BATCH_SIZE=128 N_SAMPLES=5 TEMPERATURE=0.7
bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh \
    /workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-3gpu-full-v3-conservative/trial_20251206_214232/default \
    "4 9 14 19 24"
```

**Why this works**:
- Multi-checkpoint ensemble: 64.29% (from previous results)
- Self-consistency adds ~3.5% improvement per checkpoint
- Combined: Should see 65-67% accuracy

### 2. Increase Number of Samples

**Current**: 5 samples per question
**Try**: 10 samples per question

**Expected improvement**: +1-2% additional accuracy

**Trade-off**: 2x slower (but still manageable with optimized SUB_BATCH_SIZE)

**Command**:
```bash
export SUB_BATCH_SIZE=128 N_SAMPLES=10 TEMPERATURE=0.7
bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh \
    /workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-3gpu-full-v3-conservative/trial_20251206_214232/default \
    "24"
```

### 3. Optimize Temperature for Better Diversity

**Current**: Temperature = 0.7
**Try**: Temperature = 0.8-1.0 for more diverse samples

**Rationale**: Higher temperature may generate more diverse reasoning paths, improving self-consistency voting

**Command**:
```bash
export SUB_BATCH_SIZE=128 N_SAMPLES=5 TEMPERATURE=0.9
bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh \
    /workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-3gpu-full-v3-conservative/trial_20251206_214232/default \
    "24"
```

### 4. Train for More Epochs

**Current**: 25 epochs
**Try**: 30-35 epochs

**Rationale**: Model may still be improving. Check if accuracy continues to increase beyond epoch 24.

**Note**: Monitor for overfitting. If validation accuracy plateaus or decreases, stop training.

### 5. Fine-tune Hyperparameters

**Current**: v3 conservative settings
**Consider**:
- Slightly increase learning rate (from 1.00e-5 to 1.10e-5)
- Adjust KL control (from 0.02 to 0.015)
- Experiment with different warmup schedules

**Risk**: Previous experiments showed hyperparameter sensitivity. Test carefully.

### 6. Use Chain-of-Thought Prompting

**Current**: Standard instruction following
**Try**: Explicit chain-of-thought prompts

**Example prompt format**:
```
Let's solve this step by step:
[Question]

Step 1: [First step]
Step 2: [Second step]
...
Final Answer: [Answer]
```

### 7. Error Analysis

**Action**: Analyze which questions are consistently wrong across all 5 samples

**Method**: 
1. Identify questions where all 5 samples are incorrect
2. Categorize error types (calculation errors, reasoning errors, format errors)
3. Focus training on these error patterns

**Command** (to extract wrong answers):
```bash
grep -A 5 "❌ INCORRECT" examples/cloud_gsm8k/train_logs/trainingtestlogs_24ep_5samples_each_q/ensemble_checkpoint_epoch24.log | head -100
```

## Recommended Action Plan

### Phase 1: Quick Wins (1-2 hours)
1. ✅ **Combine multi-checkpoint + self-consistency** (highest priority)
   - Expected: 65-67% accuracy
   - Time: ~5-6 hours (5 checkpoints × 5 samples each)

### Phase 2: Optimization (2-4 hours)
2. **Increase samples to 10** (if Phase 1 shows promise)
   - Expected: +1-2% additional
   - Time: ~10-12 hours

3. **Experiment with temperature** (0.8, 0.9, 1.0)
   - Expected: +0.5-1% additional
   - Time: ~3 hours per temperature

### Phase 3: Training Improvements (Days)
4. **Train longer** (30-35 epochs) if accuracy still improving
5. **Fine-tune hyperparameters** based on error analysis
6. **Implement chain-of-thought prompting**

## Performance Notes

- **GPU Utilization**: SUB_BATCH_SIZE=128 achieved good GPU utilization
- **Speed**: ~75 minutes per checkpoint with 5 samples (optimized)
- **Memory**: No OOM issues with current settings

## Conclusion

The self-consistency approach achieved excellent results:
- **5 samples**: 63.84% accuracy (+4.40% over single sample)
- **10 samples**: 66.03% accuracy (+6.59% over single sample, **best overall**)
- **10 samples outperforms multi-checkpoint ensemble** by +1.74%

**Key Insight**: Self-consistency with 10 samples from a single well-trained checkpoint (epoch 24) achieves better performance than combining multiple checkpoints, demonstrating that diverse reasoning paths within a single model can be more effective than ensemble across training stages.

**Next step**: Combine multi-checkpoint ensemble with self-consistency (10 samples per checkpoint) for potentially even higher accuracy (expected 67-69%).

