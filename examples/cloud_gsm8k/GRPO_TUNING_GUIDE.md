# GRPO Training Tuning Guide

This guide provides recommendations to improve GRPO training accuracy on GSM8K.

## Current Issues & Solutions

### 🔴 **Critical Issues (Likely Causing Degradation)**

#### 1. **Max New Tokens Too Short**
**Current**: `max_new_tokens: 256`  
**Problem**: Many math problems need longer reasoning chains. Truncation prevents complete solutions.

**Recommendation**:
```yaml
gconfig:
  max_new_tokens: 512  # For 0.5B model (or 1024 if you have memory)
```

**Why**: The reference configs use 512-1024 tokens. 256 is too restrictive for multi-step math problems.

#### 2. **No KL Penalty (KL Control = 0)**
**Current**: `kl_ctl: 0.0`  
**Problem**: Model can deviate too far from reference, causing instability and degradation.

**Recommendation**:
```yaml
actor:
  kl_ctl: 0.01  # Start small, can increase to 0.05-0.1 if needed
```

**Why**: Small KL penalty prevents the model from forgetting its base knowledge while still allowing learning.

#### 3. **Epsilon Clip Too High**
**Current**: `eps_clip: 0.4`  
**Problem**: Allows too aggressive policy updates, leading to instability.

**Recommendation**:
```yaml
actor:
  eps_clip: 0.2  # Standard PPO value, more conservative
```

**Why**: 0.2 is the standard PPO value. 0.4 allows 40% policy change per step, which is too aggressive.

#### 4. **PPO Minibatches = 1**
**Current**: `ppo_n_minibatches: 1`  
**Problem**: No data reuse, less stable updates.

**Recommendation**:
```yaml
actor:
  ppo_n_minibatches: 2  # Or 4 for better stability
```

**Why**: Multiple minibatches allow the model to learn from the same data multiple times, improving stability.

### 🟡 **Important Tuning Opportunities**

#### 5. **Learning Rate Might Be Slightly High**
**Current**: `lr: 1.70e-5`  
**Problem**: For 0.5B model, this might be slightly high (works well for 1.5B).

**Recommendation**:
```yaml
actor:
  optimizer:
    lr: 1.30e-5  # Or 1.50e-5 - slightly lower for smaller model
```

**Why**: Smaller models often need slightly lower learning rates. The 1.70e-5 was tuned for 1.5B model.

#### 6. **Reward Bias = -0.5 (Questionable)**
**Current**: `reward_bias: -0.5`  
**Problem**: Binary rewards (0/1) don't need negative bias. This might be hurting learning.

**Recommendation**:
```yaml
actor:
  reward_bias: 0.0  # Try removing the bias
  # Or keep it but understand it shifts all rewards down
```

**Why**: For binary rewards, negative bias shifts all rewards down, which might not be necessary.

#### 7. **Insufficient Warmup**
**Current**: `warmup_steps_proportion: 0.001`  
**Problem**: Very little warmup (0.1% of steps).

**Recommendation**:
```yaml
actor:
  optimizer:
    warmup_steps_proportion: 0.01  # 1% warmup (or 0.05 for 5%)
```

**Why**: More warmup helps the model adapt gradually to the new learning signal.

#### 8. **Group Size Could Be Larger**
**Current**: `n_samples: 4` (for 2GPU config)  
**Problem**: More samples per group = better group-relative normalization.

**Recommendation**:
```yaml
gconfig:
  n_samples: 8  # If memory allows, more samples = better normalization
```

**Why**: Larger groups provide better statistics for group-relative normalization.

### 🟢 **Nice-to-Have Improvements**

#### 9. **More Training Epochs**
**Current**: `total_train_epochs: 3`  
**Recommendation**: `total_train_epochs: 5` (or more for better convergence)

#### 10. **Learning Rate Schedule**
**Current**: `lr_scheduler_type: constant`  
**Recommendation**: Try `cosine` or `linear` decay for better convergence:
```yaml
actor:
  optimizer:
    lr_scheduler_type: cosine  # Or linear
```

## Recommended Config Changes (Priority Order)

### **High Priority (Do These First)**

```yaml
# 1. Increase max_new_tokens
gconfig:
  max_new_tokens: 512  # Critical for complete solutions

# 2. Add KL penalty
actor:
  kl_ctl: 0.01  # Prevents deviation from reference

# 3. Reduce epsilon clip
actor:
  eps_clip: 0.2  # More conservative updates

# 4. Increase minibatches
actor:
  ppo_n_minibatches: 2  # Better stability
```

### **Medium Priority**

```yaml
# 5. Adjust learning rate
actor:
  optimizer:
    lr: 1.30e-5  # Slightly lower for 0.5B model

# 6. Remove or adjust reward bias
actor:
  reward_bias: 0.0  # Try without bias

# 7. Increase warmup
actor:
  optimizer:
    warmup_steps_proportion: 0.01  # 1% warmup
```

### **Low Priority (If Memory Allows)**

```yaml
# 8. Increase group size
gconfig:
  n_samples: 8  # Better normalization (if memory allows)

# 9. More epochs
total_train_epochs: 5  # Better convergence
```

## Complete Improved Config Example

Here's a complete improved config for 2x A100 GPUs:

```yaml
experiment_name: gsm8k-grpo-cloud-2gpu-1000samples-improved
trial_name: trial0

max_train_samples: 1000
training_mode: "1000-SAMPLES-2GPUS-IMPROVED"
circuit_breaker_enabled: true
circuit_breaker_threshold: 50

seed: 1
total_train_epochs: 5  # Increased from 3
tokenizer_path: ${actor.path}
async_training: true

cluster:
  n_nodes: 1
  n_gpus_per_node: 2
  fileroot: /workspace/outputs/grpo
  name_resolve:
    type: nfs
    nfs_record_root: ./tmp/areal/name_resolve

allocation_mode: sglang.d1t1p1+d1t1p1

rollout:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  max_concurrent_rollouts: 24
  queue_size: null
  consumer_batch_size: ${train_dataset.batch_size}
  max_head_offpolicyness: 2
  enable_rollout_tracing: false
  request_timeout: 7200
  request_retries: 5
  setup_timeout: 300

gconfig:
  n_samples: 8  # Increased from 4 (if memory allows, else keep 4)
  min_new_tokens: 0
  max_new_tokens: 512  # Increased from 256 - CRITICAL
  greedy: false
  temperature: 1.0

actor:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  path: Qwen/Qwen2.5-0.5B-Instruct
  init_from_scratch: false
  disable_dropout: true
  gradient_checkpointing: true
  dtype: bfloat16
  mb_spec:
    max_tokens_per_mb: 4096
  optimizer:
    type: adam
    lr: 1.30e-5  # Slightly reduced from 1.70e-5
    weight_decay: 0.017
    beta1: 0.9
    beta2: 0.999
    eps: 1e-8
    lr_scheduler_type: constant  # Or try 'cosine' for decay
    gradient_clipping: 1.0
    warmup_steps_proportion: 0.01  # Increased from 0.001
  backend: fsdp
  group_size: ${gconfig.n_samples}
  eps_clip: 0.2  # Reduced from 0.4 - CRITICAL
  temperature: ${gconfig.temperature}
  reward_scaling: 10.0
  reward_bias: 0.0  # Changed from -0.5 - try this
  kl_ctl: 0.01  # Added KL penalty - CRITICAL
  ppo_n_minibatches: 2  # Increased from 1 - CRITICAL
  recompute_logprob: true
  use_decoupled_loss: true
  behav_imp_weight_cap: 5.0
  dynamic_sampling: false
  reward_norm:
    mean_level: group
    std_level: group
    group_size: ${gconfig.n_samples}
  adv_norm:
    mean_level: batch
    std_level: batch
  max_new_tokens: ${gconfig.max_new_tokens}

# ... rest of config same as before
```

## Monitoring & Validation

### Key Metrics to Watch in W&B

1. **`grpo_actor/task_reward/avg`**: Should increase over time
2. **`grpo_actor/behave_approx_kl/avg`**: Should stay small (< 0.1) - if too high, increase `kl_ctl`
3. **`grpo_actor/behave_imp_weight/avg`**: Should stay around 1.0 - if too high (> 2.0), reduce `eps_clip` or increase `kl_ctl`
4. **`grpo_actor/loss/avg`**: Should decrease over time
5. **Test accuracy**: Should improve after training

### Early Stopping Indicators

**Stop training if**:
- `behave_approx_kl` > 0.5 (model deviating too far)
- `task_reward/avg` drops below baseline for > 10 steps
- Test accuracy decreases significantly

## Testing Strategy

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
   - More epochs

## Expected Improvements

With these changes, you should see:
- **Task reward**: Increase from ~0.2-0.3 to ~0.4-0.6 (or higher)
- **Test accuracy**: Improve from baseline instead of degrading
- **Training stability**: More consistent improvements, less variance

## Troubleshooting

### If accuracy still decreases:

1. **Increase KL penalty**: Try `kl_ctl: 0.05` or `0.1`
2. **Reduce learning rate further**: Try `lr: 1.00e-5`
3. **Reduce epsilon clip more**: Try `eps_clip: 0.15`
4. **Check reward signal**: Verify rewards are being computed correctly
5. **More data**: Increase `max_train_samples` to 2000 or 4000

### If training is unstable:

1. **Increase minibatches**: `ppo_n_minibatches: 4`
2. **Reduce learning rate**: `lr: 1.00e-5`
3. **Increase KL penalty**: `kl_ctl: 0.05`
4. **Check for OOM**: Reduce batch size or max_new_tokens

## References

- Best hyperparameters from `examples/math/README.md`: lr=1.70e-5, weight_decay=0.017, group_size=4
- Training learnings from `examples/docker_gsm8k/TRAINING_LEARNINGS.md`
- AReaL documentation: `docs/algorithms/grpo.md`

