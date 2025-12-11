# Complete RunPod Guide for AReaL GRPO Training

RunPod is the most economical cloud GPU platform. This complete guide covers everything you need.

## Quick Start (TL;DR)

If you're in a hurry, here's the fastest way to get started:

1. **Create RunPod account**: https://runpod.io
2. **Create network volume**: Name `areal-outputs`, Size 50GB
3. **Deploy pod using template** (see Step 3 below)
4. **Training starts automatically!**

For detailed instructions, continue reading below.

## Why RunPod?

- 💰 **Best Pricing**: RTX 4090 at $0.29/hour (vs $0.40+ on other platforms)
- 💰 **Spot Instances**: 50-70% discount (RTX 4090 spot: ~$0.09/hour!)
- 📦 **Network Volumes**: Persistent storage for checkpoints
- 🚀 **Easy Setup**: Template system for quick deployment
- 🎯 **Good GPUs**: RTX 4090, A100 40GB/80GB, H100 available
- ⚡ **Fast Startup**: Pods start in seconds

## Step-by-Step Setup

### Step 1: Create RunPod Account

1. Go to https://runpod.io
2. Click "Sign Up"
3. Verify email
4. Add credits (minimum $10 recommended for testing)

### Step 2: Create Network Volume (CRITICAL for Model Persistence)

**⚠️ CRITICAL**: Network volumes are **REQUIRED** to ensure your trained models persist across pod restarts!

**Why**: 
- ✅ Checkpoints are saved to `/workspace/outputs/grpo/checkpoints/`
- ✅ Network volumes persist even after pod stops or is interrupted
- ✅ You can resume training from the last checkpoint
- ✅ Models are safe even if pod crashes or is terminated

1. **Go to "Volumes"** in RunPod dashboard
2. **Click "Create Volume"**
3. **Settings**:
   - **Name**: `areal-outputs`
   - **Size**: 50GB (enough for multiple training runs)
     - For reasoning models: 50GB recommended (see `REASONING_VOLUME_SIZE_GUIDE.md` for details)
     - For multi-GPU reasoning: 60GB recommended
     - For multiple experiments: 100GB+ recommended
   - **Description**: "AReaL GRPO training outputs and checkpoints"
4. **Click "Create"**

**Note**: Volume is created and ready to use. It persists across pod restarts, crashes, and interruptions.

### Step 3: Deploy Pod

#### Method A: Using Template (Recommended - Easiest)

1. **Go to "Templates"** in RunPod dashboard
2. **Click "New Template"**
3. **Fill in template**:

   **Basic Info:**
   - **Name**: `areal-grpo-training`
   - **Container Image**: `ghcr.io/inclusionai/areal-runtime:v0.3.4`
   - **Container Disk**: 15GB (enough for code and dependencies, with headroom for training artifacts)

   **Docker Command:**
   ```bash
   bash -c "set -e && pip config set global.index-url https://pypi.org/simple && pip config set global.extra-index-url '' && cd /workspace && if [ -d AReaL/.git ]; then cd AReaL && git fetch origin && git checkout -B DL4Math origin/DL4Math 2>/dev/null || git checkout -B DL4Math origin/DL4Math 2>/dev/null || (cd .. && rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git); else rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git; fi && cd /workspace/AReaL && (python3 -c 'import areal' 2>/dev/null || pip install -e .) && export WANDB_API_KEY=\$WANDB_API_KEY && bash examples/cloud_gsm8k/run_training_cloud.sh 1hour"
   ```
   
   **⚠️ CRITICAL: Disable Auto-Restart!**
   - In pod settings, set **"Restart Policy"** to **"Never"** or **"On Failure Only"**
   - This prevents RunPod from re-running the script after training completes
   - Without this, the container will restart and waste money!
   - See `RUNPOD_STOP_BEHAVIOR.md` for detailed explanation
   
   **⚠️ Important**: 
   - **Smart git handling**: If AReaL exists and is a valid git repo, it updates with `git fetch` and `git checkout` (preserves code during container restarts). Only removes/clones if invalid or missing. This prevents deleting code during training.
   - **Smart installation**: Only installs AReaL if not already installed (`python3 -c 'import areal'` check). This prevents re-installing dependencies on every pod restart or login session.
   - **Pip fix**: The Docker image (`ghcr.io/inclusionai/areal-runtime:v0.3.4`) is configured to use an internal PyPI mirror (`pypi.antfin-inc.com`) that's not accessible from RunPod. The command above overrides it to use the public PyPI (`pypi.org`).

   **Environment Variables:**
   - Click "Add Environment Variable" for each:
   - **Key**: `WANDB_API_KEY`
   - **Value**: Your WandB API key (get it from https://wandb.ai → Settings → API Keys)
   - **Key**: `PYTHONPATH`
   - **Value**: `/workspace/AReaL`
   
   **💡 Tip**: Your WandB API key is found at https://wandb.ai → Click your profile → Settings → API Keys

   **Volume Mounts:**
   - `/workspace/outputs` → Select your `areal-outputs` volume

   **GPU:**
   - Select: **RTX 4090** (recommended) or **A100** (faster)
   - **Spot Instance**: ✅ Enable (saves 50-70%!)

4. **Save Template**
5. **Deploy Pod** using the template
6. **Training starts automatically!**

#### Method B: Manual Pod Creation

1. **Go to "Pods"** → **"Deploy"**
2. **Select GPU**: 
   - **RTX 4090** (recommended, $0.29/hour)
   - **A100 40GB** (faster, $1.39/hour)
   - **A100 80GB** (most memory, $1.89/hour)
3. **Container Image**: `ghcr.io/inclusionai/areal-runtime:v0.3.4`
4. **Docker Command**: `/bin/bash`
5. **Environment Variables** (Click "Add Environment Variable" for each):
   - **Key**: `WANDB_API_KEY`
   - **Value**: Your WandB API key (get it from https://wandb.ai → Settings → API Keys)
   - **Key**: `PYTHONPATH`
   - **Value**: `/workspace/AReaL`
6. **Volume Mounts**:
   - `/workspace/outputs` → Your `areal-outputs` volume
7. **Spot Instance**: ✅ **Enable** (highly recommended!)
8. **Container Disk**: 15GB
9. **Click "Deploy"**

### Step 4: Connect to Pod

RunPod provides multiple connection methods:

#### Option 1: Web Terminal (Easiest)
1. Click on your pod in dashboard
2. Click **"Connect"** → **"Terminal"**
3. Web terminal opens in browser
4. You're now inside the pod!

#### Option 2: Jupyter (Optional)
1. Enable Jupyter in pod settings
2. Access via RunPod dashboard
3. Use Jupyter notebooks for interactive work

#### Option 3: SSH (Advanced)
1. Configure SSH in pod settings
2. Use provided SSH command from dashboard

### Step 5: Set Up (If Using Manual Pod)

If you used Method B (manual pod), run these commands in the web terminal:

```bash
# Inside pod (via web terminal)
cd /workspace

# Fix pip configuration (Docker image uses internal PyPI that's not accessible from RunPod)
pip config set global.index-url https://pypi.org/simple
pip config set global.extra-index-url ""

# Smart repository handling: update if exists, clone if not
# This prevents re-cloning on every pod restart, making restarts faster
if [ -d AReaL/.git ]; then
    cd AReaL
    git fetch origin
    git checkout -B DL4Math origin/DL4Math 2>/dev/null || git checkout -B DL4Math origin/DL4Math 2>/dev/null || (cd .. && rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git && cd AReaL)
else
    rm -rf AReaL
    git clone -b DL4Math https://github.com/nexthybrid/AReaL.git
    cd AReaL
fi

# Smart installation: only install if not already installed
# This prevents re-installing dependencies on every pod restart
if ! python3 -c "import areal" 2>/dev/null; then
    pip install -e .
fi

# Verify GPU
nvidia-smi

# Verify WandB API key
echo $WANDB_API_KEY
```

### Step 6: Run Training

#### If Using Template (Method A)
Training starts automatically! Skip to Step 7.

#### If Manual Setup (Method B)

```bash
# Inside pod
cd /workspace/AReaL

# Verify WandB API key is set
export WANDB_API_KEY=your-api-key-here  # If not set via environment

# Run training (choose one):
# Fast training (20-30 min, 200 samples)
bash examples/cloud_gsm8k/run_training_cloud.sh fast

# 1-hour training (500 samples, 2 epochs)
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour

# 3-hour training (1000 samples, 3 epochs)
bash examples/cloud_gsm8k/run_training_cloud.sh 3hour

# Full training (all samples, 5 epochs) - takes days!
bash examples/cloud_gsm8k/run_training_cloud.sh full

# 2-GPU training configs (requires 2x A100 GPUs):
bash examples/cloud_gsm8k/run_training_cloud.sh standard_2000samples_2GPUs_v3
bash examples/cloud_gsm8k/run_training_cloud.sh standard_2000samples_2GPUs_v3 6  # Override to 6 epochs
bash examples/cloud_gsm8k/run_training_cloud.sh standard_1000samples_2GPUs_v3_conservative 5  # Override to 5 epochs

# 💡 Tip: Use epoch override to experiment without modifying YAML files!
#   Format: bash run_training_cloud.sh [config_name] [epochs_override]
#   Example: bash run_training_cloud.sh standard_2000samples_2GPUs_v3 6
```

## Step 6.5: Epoch Override Feature (Optional)

**💡 New Feature**: You can now override the number of training epochs without modifying YAML files!

### Why Use Epoch Override?

- ✅ **No git push needed**: Change epochs without modifying config files
- ✅ **Quick experimentation**: Test different epoch counts easily
- ✅ **No YAML editing**: Override directly from command line

### Usage

```bash
# Normal usage (uses epochs from YAML file)
bash examples/cloud_gsm8k/run_training_cloud.sh standard_2000samples_2GPUs_v3

# Override to 6 epochs (ignores YAML setting)
bash examples/cloud_gsm8k/run_training_cloud.sh standard_2000samples_2GPUs_v3 6

# Override to 5 epochs
bash examples/cloud_gsm8k/run_training_cloud.sh standard_2000samples_2GPUs_v3 5

# Works with any config preset or YAML file path
bash examples/cloud_gsm8k/run_training_cloud.sh examples/cloud_gsm8k/gsm8k_grpo_2000samples_2GPUs_v3.yaml 6
```

### Examples

**Testing more epochs for better accuracy:**
```bash
# Original config has 4 epochs, test with 6 epochs
bash examples/cloud_gsm8k/run_training_cloud.sh standard_2000samples_2GPUs_v3 6
```

**Quick test with fewer epochs:**
```bash
# Test with 3 epochs instead of default 4
bash examples/cloud_gsm8k/run_training_cloud.sh standard_2000samples_2GPUs_v3 3
```

### Validation

The script validates that the epoch override is a positive integer:
- ✅ Valid: `5`, `6`, `10`
- ❌ Invalid: `0`, `-1`, `abc`, `5.5`

If invalid, the script will show an error and exit.

## Step 7: Monitor Training

### RunPod Dashboard

1. **Go to your pod** in RunPod dashboard
2. **View**:
   - GPU utilization (should be 80-100%)
   - Memory usage
   - Network I/O
   - **Costs** (real-time)
3. **Logs tab**: View training logs

### WandB Dashboard

1. **Go to**: https://wandb.ai
2. **Login** with your WandB account
3. **Project**: `gsm8k-grpo-local`
4. **View**:
   - Training curves (`grpo_actor/task_reward/avg`)
   - Loss curves
   - GPU utilization
   - All metrics in real-time

## Step 8: Download Results

### Via RunPod Dashboard

1. **Go to "Volumes"** in RunPod dashboard
2. **Click on** `areal-outputs` volume
3. **Browse files**:
   - Checkpoints: `outputs/grpo/checkpoints/`
   - Logs: `outputs/grpo/logs/`
4. **Download** files directly from dashboard

### Via Web Terminal

```bash
# Inside pod
ls -lh /workspace/outputs/grpo/checkpoints/
ls -lh /workspace/outputs/grpo/logs/
```

### Via Cloud Storage (S3/GCS)

```bash
# Inside pod, upload to cloud storage
aws s3 sync /workspace/outputs s3://your-bucket/areal-outputs/
# or
gsutil -m cp -r /workspace/outputs gs://your-bucket/areal-outputs/
```

## Step 9: Container Stop Behavior and Completion Markers

### ⚠️ Critical Issue: Container Auto-Restart Loop

**Problem**: RunPod has a default behavior where containers **automatically restart** when they exit. This means:

1. ✅ Training script runs and completes
2. ✅ Container exits (normal behavior)
3. ❌ **RunPod automatically restarts the container**
4. ❌ **Script runs again** (wasting money!)

This can create an **infinite loop** that wastes money!

### Solution 1: Disable Auto-Restart (Recommended)

**In RunPod Pod Settings:**
1. Go to your pod settings
2. Find **"Restart Policy"** or **"Auto-Restart"**
3. Set it to **"Never"** or **"On Failure Only"**
4. This prevents the container from restarting when it exits normally

### Solution 2: Completion Marker Protection (Implemented)

The training script includes a **completion marker** system:

1. **At start**: Script checks for completion markers
   - If found → Script exits immediately (prevents re-running)
   - If not found → Training proceeds normally

2. **At end**: Script creates a completion marker
   - Marker file: `/workspace/outputs/training_completed_YYYYMMDD_HHMMSS.marker`
   - Contains: Completion timestamp, experiment name, trial name

**If container restarts:**
- Script checks for marker → Found → Exits immediately
- No training runs → No money wasted!

### Solution 3: Manual Pod Stop

**After training completes:**
1. Go to RunPod dashboard: https://www.runpod.io/console/pods
2. Find your pod
3. Click **"Stop"** button
4. Pod stops → No more charges

**⚠️ Important**: Do this **immediately** after training completes to avoid charges!


## Step 10: Accessing Test Logs and Auto-Upload

### Where Test Logs Are Saved

All test logs are saved to:
```
/workspace/outputs/grpo/test_logs/
```

This directory is on the **network volume** (`areal-outputs`), which means:
- ✅ Logs persist after the pod stops
- ✅ You can access them from RunPod dashboard
- ✅ You can download them without starting a new pod
- ✅ No need to keep the pod running just to read results

### Log File Names

- **Reasoning Model Tests**: `test_reasoning_baseline_YYYYMMDD_HHMMSS.log` and `test_reasoning_trained_YYYYMMDD_HHMMSS.log`
- **Regular Model Tests**: `test_model_baseline_YYYYMMDD_HHMMSS.log` and `test_model_trained_YYYYMMDD_HHMMSS.log`

### Accessing Logs

**Method 1: RunPod Dashboard (Easiest)**
1. Go to RunPod Dashboard: https://www.runpod.io/console/volumes
2. Find your volume: `areal-outputs`
3. Click on the volume to view contents
4. Navigate to: `grpo/test_logs/`
5. Download the log files you want

**Method 2: Start a Temporary Pod**
If you need to access logs via command line:
```bash
# Inside pod
ls -lh /workspace/outputs/grpo/test_logs/
cat /workspace/outputs/grpo/test_logs/test_model_trained_YYYYMMDD_HHMMSS.log
```

### What's in the Logs

Each log file contains:
- Model path and configuration
- Test dataset size
- Per-sample results (question, generated answer, correct answer, correctness)
- Final accuracy percentage
- Detailed output for first few samples and incorrect answers

### Auto-Upload Test Logs

After training and testing complete, the system can automatically upload the latest test logs to:
- 📧 **Email** (SMTP)
- ☁️ **Google Drive** (via rclone)
- ☁️ **AWS S3**
- 🤗 **Hugging Face Hub**
- 📊 **Weights & Biases** (as artifacts)
- 🌐 **Generic webhook/API endpoint**

#### Quick Setup: Email (Easiest)

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

#### Other Upload Methods

**Google Drive:**
```bash
export AUTO_UPLOAD_LOGS_METHOD=gdrive
export AUTO_UPLOAD_GDRIVE_FOLDER_ID=YOUR_FOLDER_ID
# Requires rclone setup: curl https://rclone.org/install.sh | sudo bash && rclone config
```

**AWS S3:**
```bash
export AUTO_UPLOAD_LOGS_METHOD=s3
export AUTO_UPLOAD_S3_BUCKET=my-bucket-name
export AUTO_UPLOAD_S3_PREFIX=areal-training-logs
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

**Hugging Face Hub:**
```bash
export AUTO_UPLOAD_LOGS_METHOD=hf
export AUTO_UPLOAD_HF_REPO_ID=username/dataset-name
export HF_TOKEN=your-hf-token
```

**Weights & Biases:**
```bash
export AUTO_UPLOAD_LOGS_METHOD=wandb
export AUTO_UPLOAD_WANDB_PROJECT=gsm8k-grpo-cloud
export WANDB_API_KEY=your-wandb-key
```

#### Setting Environment Variables in RunPod

**Method 1: In RunPod Template Environment Variables**
1. Go to RunPod Templates
2. Edit your template
3. Add environment variables in the "Environment Variables" section

**Method 2: In RunPod Docker Command**
Add environment variables to your RunPod startup command:
```bash
bash -c "export AUTO_UPLOAD_LOGS_METHOD=email && export AUTO_UPLOAD_EMAIL_TO=your@email.com && ... (rest of command)"
```

#### Manual Upload

You can also manually upload logs after training completes:

```bash
python3 examples/cloud_gsm8k/upload_logs.py \
    --log-dir /workspace/outputs/grpo/test_logs \
    --method email \
    --email-to your@email.com \
    --latest-only
```

### Interval Testing: Evaluating Multiple Checkpoints

The `test_full_dataset.sh` script supports **interval testing**, which automatically evaluates checkpoints at 5-epoch intervals (epochs 4, 9, 14, 19, 24, etc.) instead of just the latest checkpoint. This is useful for:

- **Tracking training progression**: See how accuracy improves over epochs
- **Finding optimal checkpoint**: Identify the best-performing epoch
- **Detecting overfitting**: Observe if accuracy plateaus or decreases after a certain epoch
- **Comprehensive evaluation**: Get a complete picture of model performance throughout training

#### Usage

**Basic interval testing:**
```bash
# Test checkpoints at 5-epoch intervals (4, 9, 14, 19, etc.)
bash examples/cloud_gsm8k/test_full_dataset.sh "" examples/cloud_gsm8k/train_logs/logs_grpo_1k_v3_25epochs.txt --test-intervals
```

**With specific checkpoint path:**
```bash
# If you know the checkpoint directory
bash examples/cloud_gsm8k/test_full_dataset.sh /workspace/outputs/grpo/checkpoints/root/gsm8k-grpo-cloud-2gpu-1000samples-v3-conservative/trial_20251201_091022/default --test-intervals
```

**Auto-detect from latest checkpoint:**
```bash
# Script will find the latest checkpoint automatically
bash examples/cloud_gsm8k/test_full_dataset.sh "" "" --test-intervals
```

#### How Interval Testing Works

1. **Finds checkpoints at 5-epoch intervals**: Starting from epoch 4 (since epoch 0 is the initial state), tests epochs 4, 9, 14, 19, 24, etc.
2. **Tests each checkpoint**: Runs full dataset evaluation (1319 samples) for each interval checkpoint
3. **Skips already-tested checkpoints**: If a checkpoint was already tested (log file exists with "FINAL ACCURACY"), it skips to save time
4. **Resume capability**: If the script is interrupted (e.g., container restart), it can resume from where it left off using completion markers
5. **Uploads all logs**: All interval test logs are automatically uploaded if auto-upload is configured

#### Interval Testing Log Files

Each interval checkpoint gets its own log file:
```
test_model_epoch4_YYYYMMDD_HHMMSS.log
test_model_epoch9_YYYYMMDD_HHMMSS.log
test_model_epoch14_YYYYMMDD_HHMMSS.log
test_model_epoch19_YYYYMMDD_HHMMSS.log
test_model_epoch24_YYYYMMDD_HHMMSS.log
```

All interval logs are automatically uploaded together if auto-upload is enabled.

#### Completion Markers

The script creates completion markers to prevent re-running interval tests after container restarts:
```
/workspace/outputs/grpo/test_logs/interval_testing_completed_YYYYMMDD_HHMMSS.marker
```

If this marker exists, the script will skip interval testing to avoid redundant computation.

#### Example: Running Interval Testing in RunPod Container Starter

To run interval testing automatically after training completes, add it to your RunPod container starter code:

```bash
bash -c "set -e && pip config set global.index-url https://pypi.org/simple && pip config set global.extra-index-url '' && cd /workspace && if [ -d AReaL/.git ]; then cd AReaL && git fetch origin && git checkout -B DL4Math origin/DL4Math 2>/dev/null || git checkout -B DL4Math origin/DL4Math 2>/dev/null || (cd .. && rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git); else rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git; fi && cd /workspace/AReaL && (python3 -c 'import areal' 2>/dev/null || pip install -e .) && export WANDB_API_KEY=\$WANDB_API_KEY && bash examples/cloud_gsm8k/run_training_cloud.sh standard_1000samples_2GPUs_v3_conservative 25 && bash examples/cloud_gsm8k/test_full_dataset.sh \"\" \"\" --test-intervals"
```

**Note**: The `--test-intervals` flag must be passed to `test_full_dataset.sh`, not to `run_training_cloud.sh`.

#### Best Practices

1. **Use interval testing for long training runs**: Especially useful for 20+ epoch training to track progression
2. **Check completion markers**: If interval testing seems stuck, check if a completion marker exists
3. **Monitor log files**: Each interval test takes time (full dataset evaluation), so monitor progress
4. **Resume capability**: If interrupted, the script will resume from where it left off automatically
5. **Auto-upload all logs**: Configure auto-upload to receive all interval test results via email

### Ensemble Testing: Combining Multiple Checkpoints with Majority Voting

The `test_checkpoint_ensemble.sh` script performs **ensemble testing** by combining predictions from multiple checkpoints using majority voting. This can improve accuracy by leveraging the strengths of different training stages.

#### What is Ensemble Testing?

Instead of using a single checkpoint to answer questions, ensemble testing:
1. **Loads multiple checkpoints** (e.g., epochs 4, 9, 14, 19, 24)
2. **Generates answers** from each checkpoint for all test questions
3. **Performs majority voting** to select the most common answer
4. **Uses tie-breaking** (latest checkpoint) when answers are tied
5. **Calculates ensemble accuracy** across all questions

This approach often achieves **higher accuracy** than any single checkpoint alone.

#### Usage

**Basic ensemble testing:**
```bash
# Test with default epochs (4, 9, 14, 19, 24)
bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh /workspace/outputs/grpo/checkpoints/root/experiment/trial/default
```

**With custom epochs:**
```bash
# Test specific epochs
bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh /workspace/outputs/grpo/checkpoints/root/experiment/trial/default "4 9 14"
```

**Auto-detect checkpoint directory from training log:**
```bash
# Script will find checkpoint directory from log file
bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh "" "" examples/cloud_gsm8k/train_logs/logs_25epochs.txt
```

#### Configuration Options

**Environment Variables:**

```bash
# Number of samples per checkpoint (default: 1, use >1 for self-consistency)
export N_SAMPLES=5  # Generate 5 samples per checkpoint, then vote

# Temperature for sampling (default: 0.0 = greedy)
export TEMPERATURE=0.7  # Use with N_SAMPLES > 1 for diverse samples

# Batch size for testing (default: 32)
export TEST_BATCH_SIZE=128  # Larger batch = faster but more memory

# Sub-batch size for multi-sample generation (default: auto, ~16 for A100)
export SUB_BATCH_SIZE=128  # For A100 80GB, use 64-128 for optimal GPU utilization
```

**Performance Optimization:**

For **A100 80GB** with `N_SAMPLES=5`:
- **Recommended**: `SUB_BATCH_SIZE=64` (320 sequences in parallel)
- **Maximum**: `SUB_BATCH_SIZE=128` (640 sequences in parallel)
- **Note**: The script automatically increases `BATCH_SIZE` to match `SUB_BATCH_SIZE` if needed

See `SUB_BATCH_SIZE_GUIDE.md` for detailed recommendations.

#### Example: Running Ensemble Testing in RunPod Container Starter

To run ensemble testing automatically after training completes:

```bash
bash -c "set -e && pip config set global.index-url https://pypi.org/simple && pip config set global.extra-index-url '' && cd /workspace && if [ -d AReaL/.git ]; then cd AReaL && git fetch origin && git checkout -B DL4Math origin/DL4Math 2>/dev/null || git checkout -B DL4Math origin/DL4Math 2>/dev/null || (cd .. && rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git); else rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git; fi && cd /workspace/AReaL && (python3 -c 'import areal' 2>/dev/null || pip install -e .) && export WANDB_API_KEY=\$WANDB_API_KEY && bash examples/cloud_gsm8k/run_training_cloud.sh standard_1000samples_2GPUs_v3_conservative 25 && export SUB_BATCH_SIZE=128 N_SAMPLES=5 && bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh \"\" \"\" examples/cloud_gsm8k/train_logs/logs_25epochs.txt"
```

**With self-consistency (multiple samples per checkpoint):**
```bash
bash -c "... && export SUB_BATCH_SIZE=128 N_SAMPLES=5 TEMPERATURE=0.7 && bash examples/cloud_gsm8k/test_checkpoint_ensemble.sh \"\" \"\" examples/cloud_gsm8k/train_logs/logs_25epochs.txt"
```

#### How Ensemble Testing Works

1. **Loads each checkpoint** sequentially (e.g., epoch 4, then 9, then 14, etc.)
2. **Generates answers** for all 1319 test questions from each checkpoint
3. **Saves individual results** to log files (one per checkpoint)
4. **Performs majority voting** across all checkpoints for each question
5. **Calculates ensemble accuracy** and saves final results

**Memory-efficient approach**: Loads one checkpoint at a time, processes all questions, then moves to the next checkpoint. This prevents OOM errors even with large models.

#### Ensemble Testing Log Files

**Individual checkpoint logs:**
```
ensemble_checkpoint_epoch4.log
ensemble_checkpoint_epoch9.log
ensemble_checkpoint_epoch14.log
...
```

**Final majority voting log:**
```
ensemble_majority_voting_YYYYMMDD_HHMMSS.log
```

All logs are saved to `/workspace/outputs/grpo/test_logs/` and are automatically uploaded if auto-upload is configured.

#### Performance Considerations

**Single sample mode** (`N_SAMPLES=1`):
- Fastest: ~15 minutes per checkpoint
- Lower accuracy potential
- Recommended for quick testing

**Multi-sample mode** (`N_SAMPLES=5`):
- Slower: ~75 minutes per checkpoint (5x slower)
- Higher accuracy potential (self-consistency + ensemble)
- **Requires `SUB_BATCH_SIZE` optimization** for reasonable speed
- Recommended for final evaluation

**GPU Utilization:**
- With `SUB_BATCH_SIZE=128` and `N_SAMPLES=5`: ~80-90% GPU utilization
- With default settings: ~20-30% GPU utilization (much slower)

#### Best Practices

1. **Use ensemble testing for final evaluation**: After identifying best individual checkpoints
2. **Start with single sample mode**: Test quickly with `N_SAMPLES=1`, then use `N_SAMPLES=5` for final results
3. **Optimize SUB_BATCH_SIZE**: For A100 80GB, use 64-128 for best performance
4. **Monitor GPU utilization**: Use `nvidia-smi` to ensure GPU is well-utilized
5. **Auto-upload results**: Configure auto-upload to receive ensemble results via email
6. **Compare with individual checkpoints**: Ensemble accuracy should be ≥ best individual checkpoint

#### Example Results

Typical ensemble testing results:
- **Individual checkpoint accuracies**: 56-60% (varies by epoch)
- **Ensemble accuracy**: 64-66% (improvement from majority voting)
- **Improvement**: +4-6% over best individual checkpoint

The ensemble approach leverages the diversity of different training stages to achieve better overall performance.

## Step 11: Network Volume Size Recommendations

### Recommended Volume Sizes by Training Config

| Training Config | Samples | Epochs | Recommended Size | Minimum Size |
|----------------|---------|--------|------------------|--------------|
| Fast/1-hour | 200-500 | 1-2 | **30-40GB** | 20-30GB |
| 3-hour | 1000 | 3 | **50GB** | 40GB |
| 2-GPU (1000-2000 samples) | 1000-2000 | 3-4 | **50-60GB** | 40-50GB |
| Reasoning models | 200-2000 | 1-3 | **50-60GB** | 40-50GB |
| Multiple runs | - | - | **100GB+** | 80GB |

**Most Common Recommendation**: **50-60GB** covers all single-run scenarios with safety margin.

### Storage Components

1. **Model Checkpoints** (largest component)
   - Per epoch checkpoints (~1GB per checkpoint for Qwen2.5-0.5B)
   - Optimizer states

2. **Training Logs** (~2-5GB)
   - Training statistics
   - System logs

3. **Test Logs** (~1-3GB)
   - Full validation test results
   - Per-sample outputs

4. **Generated Samples** (~2-12GB)
   - Rollout outputs during training
   - Longer for reasoning models (up to 1024 tokens)

### Monitoring Volume Usage

```bash
# Inside pod
df -h /workspace/outputs
du -sh /workspace/outputs/grpo/checkpoints/*
du -sh /workspace/outputs/grpo/logs/*
```

### Upgrading Volume Size

If you run out of space:
1. RunPod Dashboard → Volumes
2. Select your volume → Edit/Resize
3. Increase size (RunPod allows resizing)
4. Wait for resize to complete

**Note**: RunPod volumes can be resized, but it's better to start with adequate size to avoid interruptions.

## Step 12: Resuming Training from Checkpoint

When you want to continue training (after pod restart or interruption):

### Step 12.1: Deploy New Pod (or Restart Existing)

1. **Go to "Pods"** → **"Deploy"** (or restart existing pod)
2. **Use same settings** as before:
   - Same container image: `ghcr.io/inclusionai/areal-runtime:v0.3.4`
   - **Same volume mount**: `/workspace/outputs` → `areal-outputs` (CRITICAL!)
   - Same environment variables (WandB API key, etc.)

### Step 12.2: Verify Checkpoint Exists

```bash
# Inside pod, check that checkpoints exist
ls -lh /workspace/outputs/grpo/checkpoints/

# Find your experiment and trial
# Example:
ls -lh /workspace/outputs/grpo/checkpoints/gsm8k-grpo-cloud-1hour/trial0/
```

### Step 12.3: Set Up Repository (If Needed)

```bash
# Inside pod
cd /workspace

# Fix pip configuration (Docker image uses internal PyPI that's not accessible from RunPod)
pip config set global.index-url https://pypi.org/simple
pip config set global.extra-index-url ""

# Smart repository handling: update if exists, clone if not
# This prevents re-cloning on every pod restart, making restarts faster
if [ -d AReaL/.git ]; then
    cd AReaL
    git fetch origin
    git checkout -B DL4Math origin/DL4Math 2>/dev/null || git checkout -B DL4Math origin/DL4Math 2>/dev/null || (cd .. && rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git && cd AReaL)
else
    rm -rf AReaL
    git clone -b DL4Math https://github.com/nexthybrid/AReaL.git
    cd AReaL
fi

# Smart installation: only install if not already installed
# This prevents re-installing dependencies on every pod restart
if ! python3 -c "import areal" 2>/dev/null; then
    pip install -e .
fi
```

### Step 12.4: Resume Training

```bash
# Inside pod
cd /workspace/AReaL

# Verify WandB API key is set
export WANDB_API_KEY=your-api-key-here  # If not set via environment

# Resume training with SAME experiment_name and trial_name
# The training script will automatically detect and load the last checkpoint
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour
```

**Important**: 
- ✅ Use the **same `experiment_name` and `trial_name`** as before
- ✅ The training script automatically detects checkpoints in `outputs/grpo/checkpoints/{experiment_name}/{trial_name}/`
- ✅ Training will resume from the last saved checkpoint automatically
- ✅ Progress continues seamlessly from where it left off

### Verifying Resume Works

After starting training, check the logs:

```bash
# Training should show:
# "Loading checkpoint from: outputs/grpo/checkpoints/..."
# "Resuming from epoch X, step Y"
```

If you see "Starting training from scratch", the checkpoint wasn't found. Verify:
1. Volume is mounted at `/workspace/outputs`
2. Checkpoint path is correct
3. Experiment name and trial name match previous run

## RunPod Cost Optimization

### 1. Use Spot Instances (50-70% Savings!)

**Always enable spot instances for training:**
- RTX 4090: $0.29/hour → **$0.09/hour** (spot)
- A100: $1.39/hour → **$0.42/hour** (spot)

**Trade-off**: Spot instances may be interrupted if demand is high. But:
- ✅ Checkpoints are saved to network volume
- ✅ Can resume training from checkpoint
- ✅ 50-70% cost savings!

### 2. Auto-Shutdown

**Enable in pod settings:**
- **Idle timeout**: 10-30 minutes
- **Pod stops** when idle
- **Resume** when needed
- **Saves money** when not training

### 3. Right-Size GPU

**For AReaL GRPO training:**
- **RTX 4090**: Best value ($0.29/hour, spot: $0.09/hour)
- **A100**: Faster but 5x more expensive
- **Recommendation**: Start with RTX 4090

### 4. Monitor Costs

- **Check RunPod dashboard** regularly
- **Set budget alerts** if available
- **Stop pods** when not in use

## Cost Estimates

### RTX 4090 (Recommended)

| Training | Regular | Spot | Savings |
|----------|---------|------|---------|
| Fast (30 min) | $0.15 | $0.05 | $0.10 |
| 1-hour | $0.58 | $0.18 | $0.40 |
| 3-hour | $0.87 | $0.27 | $0.60 |
| Full (5 days) | $34.80 | $10.80 | $24.00 |

### A100 40GB

| Training | Regular | Spot | Savings |
|----------|---------|------|---------|
| Fast (30 min) | $0.70 | $0.21 | $0.49 |
| 1-hour | $2.78 | $0.84 | $1.94 |
| 3-hour | $4.17 | $1.26 | $2.91 |
| Full (5 days) | $166.80 | $50.40 | $116.40 |

**Recommendation**: Use RTX 4090 spot instances for best value!

## RunPod-Specific Features

### Network Volumes

- **Persistent storage** across pod restarts
- **Shared** between pods (if needed)
- **Fast access** (network-attached)
- **Recommended size**: 
  - **Regular training**: 50GB for checkpoints
  - **Reasoning models**: 50-60GB (longer outputs require more space)
  - **Multi-GPU reasoning**: 60GB recommended
  - **Multiple experiments**: 100GB+ recommended
  - See `REASONING_VOLUME_SIZE_GUIDE.md` for detailed breakdown

### Spot Instances

- **50-70% discount** on regular pricing
- **May be interrupted** if demand is high
- **Checkpoints saved** to network volume
- **Resume training** from checkpoint

### Templates

- **Save time** on repeated deployments
- **Consistent setup** across runs
- **Share with team** members
- **One-click deployment**

### Auto-Shutdown

- **Saves costs** when pod is idle
- **Configurable timeout** (10-30 min recommended)
- **Resume** when needed
- **No data loss** (checkpoints in volume)

## Troubleshooting

### Pod Won't Start

**Check:**
- GPU availability (may need to wait)
- Container image name (must be exact)
- Environment variables (WandB API key)
- Volume mount path

### Out of Memory

**Solutions:**
- Use A100 instead of RTX 4090
- Reduce `batch_size` in config (8 → 4)
- Reduce `max_new_tokens` (512 → 256)
- Enable `gradient_checkpointing: true`

### Spot Instance Interrupted

**This is normal!** Solutions:
- Checkpoints are saved to network volume
- Resume training from last checkpoint
- Or restart training (if early in training)

### Network Volume Not Mounted

**Check:**
- Volume is created in RunPod dashboard
- Volume is attached to pod
- Mount path is `/workspace/outputs` (must be exact!)
- Permissions are correct

**Verify volume is mounted:**
```bash
# Inside pod
ls /workspace/outputs/
# Should show: grpo/ directory (not empty)
```

### Checkpoints Not Persisting

**Common causes:**
- Volume not mounted at `/workspace/outputs`
- Checkpoints saved to wrong location (not on volume)
- Volume not attached to pod

**Solution:**
1. Verify volume mount in pod settings: `/workspace/outputs` → `areal-outputs`
2. Check checkpoint location: `ls /workspace/outputs/grpo/checkpoints/`
3. If empty, check if checkpoints were saved elsewhere: `find /workspace -name "checkpoint_*" -type d`
4. Ensure config uses: `cluster.fileroot: /workspace/outputs/grpo` (absolute path to mounted volume)

### Pip Installation Fails (Connection Timeout to pypi.antfin-inc.com)

**Problem**: The Docker image is configured to use an internal PyPI mirror (`pypi.antfin-inc.com`) that's not accessible from RunPod, causing connection timeouts.

**Error messages:**
- `Connection to pypi.antfin-inc.com timed out`
- `ERROR: Could not find a version that satisfies the requirement`
- `ERROR: No matching distribution found`

**Solution**: Override pip configuration to use public PyPI before installing:

```bash
# Inside pod, before pip install
pip config set global.index-url https://pypi.org/simple
pip config set global.extra-index-url ""

# Then proceed with installation
pip install -e .
```

**For templates/pod commands**, include this fix in the Docker command (with smart installation):
```bash
bash -c "set -e && pip config set global.index-url https://pypi.org/simple && pip config set global.extra-index-url '' && cd /workspace && if [ -d AReaL/.git ]; then cd AReaL && git fetch origin && git checkout -B DL4Math origin/DL4Math 2>/dev/null || git checkout -B DL4Math origin/DL4Math 2>/dev/null || (cd .. && rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git); else rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git; fi && cd /workspace/AReaL && (python3 -c 'import areal' 2>/dev/null || pip install -e .) && export WANDB_API_KEY=\$WANDB_API_KEY && bash examples/cloud_gsm8k/run_training_cloud.sh 1hour"
```

**Why this happens**: The AReaL Docker image (`ghcr.io/inclusionai/areal-runtime:v0.3.4`) is built for internal use at Ant Group and includes pip configuration pointing to their internal PyPI mirror, which is not accessible from external cloud platforms like RunPod.

### Container Restart Loop ⚠️

**Problem**: Container keeps restarting every ~17 seconds, never actually runs training.

**Symptoms:**
- Container status shows "Restarting" repeatedly
- System logs show container starting every ~17 seconds
- Container logs show: `fatal: destination path 'AReaL' already exists and is not an empty directory`
- No training output
- GPU never gets used

**Root Cause**: Docker command tries to `git clone` AReaL, but directory already exists from previous run. `git clone` fails, container exits, RunPod auto-restarts it, creating infinite loop.

**Solution**: Updated Docker command with smart git handling and installation:
```bash
bash -c "set -e && pip config set global.index-url https://pypi.org/simple && pip config set global.extra-index-url '' && cd /workspace && if [ -d AReaL/.git ]; then cd AReaL && git fetch origin && git checkout -B DL4Math origin/DL4Math 2>/dev/null || git checkout -B DL4Math origin/DL4Math 2>/dev/null || (cd .. && rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git); else rm -rf AReaL && git clone -b DL4Math https://github.com/nexthybrid/AReaL.git; fi && cd /workspace/AReaL && (python3 -c 'import areal' 2>/dev/null || pip install -e .) && export WANDB_API_KEY=\$WANDB_API_KEY && bash examples/cloud_gsm8k/run_training_cloud.sh 1hour"
```

**Key improvements**:
- **Smart git handling**: Checks if AReaL exists and is valid before cloning
  - If AReaL exists and is a valid git repo: Updates it with `git fetch` and `git checkout` (preserves code during restarts)
  - If AReaL exists but is invalid: Removes and clones fresh
  - If AReaL doesn't exist: Clones fresh
- **Smart installation**: Only installs if not already installed (`python3 -c 'import areal'` check)
- **Prevents restart loops**: Handles existing directories gracefully

This prevents deleting code during training while still handling restart scenarios.

**Note**: Container restart loops are usually caused by Docker command errors or missing dependencies. Check the Docker command in your template and ensure all required environment variables are set.

### Training Crashes Around Step 50-60 (Repeating Pattern)

**Problem**: Training consistently crashes around step 50-60 with terminal closing and GPU stopping. This is **not** a spot interruption but an execution crash.

**Symptoms:**
- Training runs for ~50-60 steps
- Terminal closes suddenly
- GPU utilization drops to zero
- Pattern repeats across multiple runs
- No error messages in terminal

**Root Cause**: Likely a **timeout or crash during disk-based weight updates**. The weight update process has hardcoded timeouts that may be insufficient.

**Solutions** (try in order):

1. **Increase timeouts in config** (already done in A40 config):
   ```yaml
   rollout:
     request_timeout: 7200  # 2 hours (from default 3600)
     request_retries: 5  # More retries (from default 3)
     setup_timeout: 300  # 5 minutes (from default 120)
   ```

2. **Check system logs for crash details**:
   ```bash
   # Inside pod
   dmesg | tail -50
   journalctl -n 50
   dmesg | grep -i "out of memory\|killed"
   ```

3. **Check SGLang server logs**:
   ```bash
   tail -100 /workspace/AReaL/outputs/grpo/logs/*/llm_server.log
   grep -i "error\|crash\|timeout" /workspace/AReaL/outputs/grpo/logs/*/llm_server.log
   ```

4. **Check disk space and I/O**:
   ```bash
   df -h /workspace/outputs
   # Test disk speed
   dd if=/dev/zero of=/workspace/outputs/test bs=1G count=1 oflag=direct
   ```

5. **Check memory usage**:
   ```bash
   # Monitor during training
   watch -n 1 'free -h && nvidia-smi'
   ```

6. **Try regular (non-spot) instance** to rule out spot interruptions

**Note**: Training crashes can be caused by OOM, SGLang server disconnects, or configuration issues. See `H200_STEP188_DIAGNOSIS.md` for an example analysis. The circuit breaker in `gsm8k_grpo_train.py` will automatically stop training if task reward is zero for 10 consecutive steps.

### Training Too Slow

**Optimizations:**
- Use A100 instead of RTX 4090
- Check GPU utilization (`nvidia-smi`)
- Verify batch size is appropriate
- Check network volume performance

### CUDA Out of Memory (OOM) on A40 GPU

**Problem**: A40 GPU (44GB) runs out of memory when both SGLang inference server and trainer share the same GPU.

**Error message:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB. 
GPU 0 has a total capacity of 44.42 GiB of which 27.50 MiB is free.
Process 2963820 has 36.34 GiB memory in use.  # SGLang server
Process 2965022 has 8.05 GiB memory in use.   # Trainer
```

**Solutions** (apply in order):

1. **Use A40-optimized config** (recommended):
   ```bash
   # Use the A40-optimized config file
   bash examples/cloud_gsm8k/run_training_cloud.sh 1hour
   # But override config to use A40 version:
   python3 -m areal.launcher.local examples/cloud_gsm8k/gsm8k_grpo_train.py \
       --config examples/cloud_gsm8k/gsm8k_grpo_1hour.yaml \
       experiment_name=gsm8k-grpo-cloud-1hour \
       trial_name=trial0
   ```

2. **Enable gradient checkpointing** (reduces training memory by ~30-40%):
   ```yaml
   actor:
     gradient_checkpointing: true  # Change from false to true
   ```

3. **Reduce SGLang memory fraction** (leaves more memory for trainer):
   ```yaml
   sglang:
     mem_fraction_static: 0.5  # Reduce from 0.8 to 0.5
   ```

4. **Reduce batch size**:
   ```yaml
   train_dataset:
     batch_size: 4  # Reduce from 8 to 4
   ```

5. **Reduce max_new_tokens**:
   ```yaml
   gconfig:
     max_new_tokens: 256  # Reduce from 512 to 256
   ```

6. **Reduce max_tokens_per_mb**:
   ```yaml
   actor:
     mb_spec:
       max_tokens_per_mb: 4096  # Reduce from 5120
   ```

7. **Set PyTorch memory allocator** (reduces fragmentation):
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

8. **Reduce max_concurrent_rollouts**:
   ```yaml
   rollout:
     max_concurrent_rollouts: 16  # Reduce from 32
   ```

**Quick fix for A40/RTX 5090**: The script automatically detects memory-constrained GPUs and uses `gsm8k_grpo_1hour_memory_optimized.yaml` which includes all these optimizations. You can also manually specify it if needed.

**Memory breakdown on A40**:
- SGLang server: ~36GB (with `mem_fraction_static: 0.8`)
- Trainer: ~8GB
- Total: ~44GB (exceeds A40 capacity)

**With A40 optimizations**:
- SGLang server: ~22GB (with `mem_fraction_static: 0.5`)
- Trainer: ~6GB (with gradient checkpointing)
- Total: ~28GB (fits comfortably in A40)

## Best Practices

1. ✅ **Always use spot instances** (50-70% savings)
2. ✅ **ALWAYS use network volumes** for checkpoints (required for persistence!)
3. ✅ **Verify checkpoints exist** before stopping pod: `ls /workspace/outputs/grpo/checkpoints/`
4. ✅ **Enable auto-shutdown** to save costs (checkpoints are safe in volume)
5. ✅ **Monitor in WandB** for training progress
6. ✅ **Stop pod when done** (don't leave running - checkpoints persist in volume)
7. ✅ **Use same experiment_name/trial_name** when resuming training
8. ✅ **Download important checkpoints** to local machine as backup
9. ✅ **Use templates** for repeated runs (ensures consistent volume mounting)
10. ✅ **Test logs are saved to network volume** at `/workspace/outputs/grpo/test_logs/` - accessible after pod stops!

## Quick Reference

### Deploy Pod
```
RunPod Dashboard → Pods → Deploy
- GPU: RTX 4090 (spot enabled)
- Image: ghcr.io/inclusionai/areal-runtime:v0.3.4
- Volume: /workspace/outputs → areal-outputs
- Env: WANDB_API_KEY=your-key
```

### Run Training
```bash
# Inside pod
cd /workspace/AReaL
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour
```

### Monitor
- RunPod dashboard: GPU, costs, logs
- WandB: https://wandb.ai (training curves)

### Download Results
- RunPod dashboard → Volumes → areal-outputs
- Or use cloud storage (S3/GCS)

## Next Steps

1. ✅ Create RunPod account
2. ✅ Create network volume
3. ✅ Deploy pod (use template for easiest setup)
4. ✅ Monitor training in WandB
5. ✅ Download results
6. ✅ Stop pod (save costs!)

Happy training on RunPod! 🚀

---

## Additional Resources

- **Training Learnings**: See `TRAINING_LEARNINGS.md` for detailed guides, GRPO tuning, epoch analysis, and best practices
- **Main README**: See `README.md` for overview and quick reference

