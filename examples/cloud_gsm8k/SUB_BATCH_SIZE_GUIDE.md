# Sub-Batch Size Guide for A100 GPUs

## Quick Answer

For **A100 80GB** with **N_SAMPLES=5**:
- **Conservative (Safe)**: `SUB_BATCH_SIZE=32` → 160 sequences in parallel
- **Moderate (Recommended)**: `SUB_BATCH_SIZE=64` → 320 sequences in parallel  
- **Aggressive (Maximum)**: `SUB_BATCH_SIZE=128` → 640 sequences in parallel
- **Theoretical Max**: `SUB_BATCH_SIZE=256` → 1280 sequences (may cause OOM)

## Memory Calculation

### Model: Qwen2.5-0.5B-Instruct
- **Model size**: ~1GB (bfloat16)
- **Parameters**: 0.5B
- **Hidden size**: ~1024-2048 (typical for 0.5B models)

### A100 80GB Memory Breakdown
- **Total VRAM**: 80GB
- **Model weights**: ~1GB
- **Available for generation**: ~79GB

### Memory Per Sequence (Approximate)
- **KV cache per token**: ~8-16 bytes (for 0.5B model)
- **Sequence length**: ~700 tokens (200 input + 512 generated)
- **KV cache per sequence**: ~5.6-11.2 KB
- **Activation memory**: ~50-100 MB per sequence (during forward pass)

### Effective Batch Sizes

| SUB_BATCH_SIZE | Effective Batch | Total Sequences | Estimated Memory | GPU Utilization |
|----------------|-----------------|-----------------|------------------|-----------------|
| 16 (default)   | 16 × 5 = 80     | 80              | ~4-8 GB          | ~20-30%         |
| 32             | 32 × 5 = 160    | 160             | ~8-16 GB         | ~40-50%         |
| 64             | 64 × 5 = 320    | 320             | ~16-32 GB        | ~60-70%         |
| 128            | 128 × 5 = 640   | 640             | ~32-64 GB        | ~80-90%         |
| 256            | 256 × 5 = 1280  | 1280            | ~64-128 GB       | ~100% (may OOM) |

## Recommendations

### For A100 80GB

1. **Start with**: `SUB_BATCH_SIZE=64`
   - Good balance between speed and safety
   - Should utilize ~60-70% of GPU
   - Low risk of OOM

2. **If you want maximum speed**: `SUB_BATCH_SIZE=128`
   - Near-maximum GPU utilization
   - Monitor for OOM errors
   - May need to reduce if you hit memory limits

3. **If you're conservative**: `SUB_BATCH_SIZE=32`
   - Very safe, guaranteed to work
   - Lower GPU utilization (~40-50%)
   - Still much faster than sequential generation

### For A100 40GB

- **Recommended**: `SUB_BATCH_SIZE=32` (160 sequences)
- **Maximum**: `SUB_BATCH_SIZE=64` (320 sequences, may OOM)

## Usage Examples

```bash
# Conservative (safe)
SUB_BATCH_SIZE=32 N_SAMPLES=5 bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh "$CHECKPOINT_DIR" "4 9 14 19 24"

# Recommended (good balance)
SUB_BATCH_SIZE=64 N_SAMPLES=5 bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh "$CHECKPOINT_DIR" "4 9 14 19 24"

# Aggressive (maximum speed)
SUB_BATCH_SIZE=128 N_SAMPLES=5 bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh "$CHECKPOINT_DIR" "4 9 14 19 24"
```

## Monitoring GPU Usage

While running, monitor GPU memory:
```bash
watch -n 1 nvidia-smi
```

Look for:
- **Memory usage**: Should be well below 80GB (leave ~10-20GB headroom)
- **GPU utilization**: Should be 60-90% for optimal performance
- **OOM errors**: If you see "CUDA out of memory", reduce SUB_BATCH_SIZE

## Performance Impact

| SUB_BATCH_SIZE | Speedup vs Sequential | Time per Checkpoint (1319 questions) |
|----------------|----------------------|--------------------------------------|
| 1 (sequential) | 1x (baseline)        | ~75 minutes                            |
| 16 (default)   | ~8x                  | ~9-10 minutes                          |
| 32             | ~16x                 | ~4-5 minutes                           |
| 64             | ~32x                 | ~2-3 minutes                           |
| 128            | ~64x                 | ~1-2 minutes                           |

## Notes

1. **Sequence Length**: If you increase `max_new_tokens` beyond 512, reduce SUB_BATCH_SIZE accordingly.

2. **Model Size**: These recommendations are for 0.5B models. For larger models (1.5B+), reduce SUB_BATCH_SIZE.

3. **Other Processes**: If other processes are using GPU memory, reduce SUB_BATCH_SIZE.

4. **Temperature Sampling**: Higher temperature doesn't significantly affect memory, but may slow generation slightly.

## Troubleshooting

### If you get OOM errors:
1. Reduce `SUB_BATCH_SIZE` by half (e.g., 128 → 64)
2. Reduce `max_new_tokens` if it's > 512
3. Check for other processes using GPU memory

### If GPU utilization is still low:
1. Increase `SUB_BATCH_SIZE` gradually (32 → 64 → 128)
2. Monitor memory usage to ensure you're not hitting limits
3. Check that `batch_size` in the script is large enough (should be ≥ SUB_BATCH_SIZE)

