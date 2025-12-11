# Volume Size Estimation for Full Dataset Training (3 GPUs, v3 Conservative)

## Configuration
- **Dataset**: Full GSM8K (7473 training samples)
- **GPUs**: 3x A100 80GB/H100/H200
- **Epochs**: 25
- **Training Time**: ~3-3.5 days

## Storage Breakdown

### 1. Checkpoints (~25-30 GB)
- **Model Size**: Qwen2.5-0.5B-Instruct ≈ 1 GB per checkpoint (bfloat16)
- **Checkpoints per epoch**: 1 (saved after each epoch)
- **Total epochs**: 25
- **Estimated**: 25 × 1 GB = **25 GB**
- **With safety margin**: **30 GB**

### 2. Training Logs (~8-12 GB)
- **Log files**: Detailed training logs, stdout/stderr
- **WandB sync data**: Local cache of metrics
- **Per epoch**: ~300-500 MB
- **Total**: 25 epochs × ~400 MB = **10 GB**
- **With safety margin**: **12 GB**

### 3. Test Logs (~5-8 GB)
- **Full dataset testing**: 1319 test samples
- **Interval testing**: Tests at epochs 4, 9, 14, 19, 24 (5 checkpoints)
- **Per test log**: ~1-1.5 GB (detailed per-sample results)
- **Total**: 5 tests × ~1.2 GB = **6 GB**
- **With safety margin**: **8 GB**

### 4. Generated Samples & Rollouts (~10-15 GB)
- **Rollout data**: Generated completions during training
- **Full dataset**: 7473 samples × 8 samples per prompt = ~60k generated samples
- **Per sample**: ~200-300 KB (with metadata)
- **Total**: ~12-18 GB
- **Conservative estimate**: **15 GB**

### 5. WandB Artifacts (~3-5 GB)
- **Model checkpoints uploaded**: Optional, but recommended
- **Training metrics**: Historical data
- **Total**: **5 GB**

### 6. Recovery & Metadata (~2-3 GB)
- **Recovery checkpoints**: Hourly snapshots
- **Metadata files**: Configs, trial info
- **Total**: **3 GB**

### 7. Safety Margin (~20-30%)
- **For unexpected growth**: Log rotation, additional checkpoints
- **Buffer**: **15-20 GB**

## Total Volume Size Estimate

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Checkpoints | 25 GB | 30 GB |
| Training Logs | 8 GB | 12 GB |
| Test Logs | 5 GB | 8 GB |
| Generated Samples | 10 GB | 15 GB |
| WandB Artifacts | 3 GB | 5 GB |
| Recovery & Metadata | 2 GB | 3 GB |
| Safety Margin | 15 GB | 20 GB |
| **TOTAL** | **68 GB** | **93 GB** |

## Recommendation

### Minimum Volume Size: **80 GB**
- Covers all essential components
- Minimal safety margin
- Risk of running out of space if training extends

### Recommended Volume Size: **100 GB** ⭐
- Comfortable safety margin (20-30%)
- Room for extended training (30+ epochs)
- Space for multiple test runs
- No risk of storage issues

### Optimal Volume Size: **120 GB**
- For multiple experiments
- Extended training runs (30-35 epochs)
- Multiple checkpoint versions
- Comprehensive testing

## Notes

1. **Checkpoint Frequency**: Currently set to 1 per epoch. If you reduce to every 2-3 epochs, you can save ~10-15 GB.

2. **Test Logs**: If you skip interval testing and only test the final checkpoint, you can save ~4-5 GB.

3. **Generated Samples**: These can be cleaned up after training if not needed for analysis.

4. **WandB**: Most data syncs to cloud, but local cache can grow. You can clear it periodically.

5. **Volume Resizing**: RunPod allows volume resizing, but it's better to start with adequate size to avoid interruptions.

## Comparison with Other Configs

| Config | Samples | Epochs | Recommended Size |
|--------|---------|--------|------------------|
| Fast/1-hour | 200-500 | 1-2 | 30-40 GB |
| 3-hour | 1000 | 3 | 50 GB |
| 2-GPU (1000-2000) | 1000-2000 | 3-4 | 50-60 GB |
| 3-GPU (4000) | 4000 | 25 | 70-80 GB |
| **3-GPU Full** | **7473** | **25** | **100 GB** ⭐ |

## Action Items

1. **Create RunPod Volume**: 100 GB (recommended) or 120 GB (optimal)
2. **Mount at**: `/workspace/outputs`
3. **Monitor usage**: Check periodically during training
4. **Clean up**: Remove old checkpoints/logs if needed

