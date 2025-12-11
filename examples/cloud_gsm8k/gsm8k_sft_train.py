"""
Consolidated SFT training script for GSM8K.

This script handles all training configurations (1K, 2K, full)
by reading settings from the YAML config file and command-line overrides.

Usage:
    python -m areal.launcher.local examples/cloud_gsm8k/gsm8k_sft_train.py \
        --config examples/cloud_gsm8k/gsm8k_sft_1000samples_1GPU.yaml

Config parameters:
    - max_train_samples: Limit dataset size (None for full dataset)
    - training_mode: Display name for training mode (e.g., "1K-SAMPLES-1GPU", "2K-SAMPLES-3GPUS", "FULL-3GPUS")
"""
import os
import sys
import logging
import subprocess
import tempfile
from pathlib import Path

import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo
from areal.dataset import get_custom_dataset
from areal.engine.sft.lm_engine import FSDPLMEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import (
    broadcast_tensor_container,
    pad_sequences_to_tensors,
    tensor_container_to,
)
from areal.utils.dataloader import create_dataloader
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    # Store script_dir for later use in log upload
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Extract custom training parameters from YAML before config validation
    # These parameters are not part of SFTConfig, so we read them directly from YAML
    import argparse
    from omegaconf import OmegaConf
    
    # Parse args to find config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the main configuration file", required=True)
    # Skip script path if present
    if args and args[0].endswith(".py"):
        args = args[1:]
    parsed_args, _ = parser.parse_known_args(args)
    
    # Load YAML directly to extract custom fields
    config_file = Path(parsed_args.config)
    if not config_file.is_absolute():
        # Make it absolute relative to current working directory
        config_file = Path.cwd() / config_file
    config_file_str = str(config_file)  # Store for later use
    raw_yaml = OmegaConf.load(config_file)
    
    # Extract custom training parameters BEFORE removing them
    max_train_samples = raw_yaml.get("max_train_samples", None)
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
    training_mode = raw_yaml.get("training_mode", "TRAINING")
    
    # Remove custom fields from OmegaConf object to avoid validation errors
    # These fields are not part of SFTConfig, so they must be removed before validation
    custom_keys = ["max_train_samples", "training_mode"]
    cleaned_yaml = OmegaConf.create(OmegaConf.to_container(raw_yaml, resolve=False))
    for key in custom_keys:
        if key in cleaned_yaml:
            del cleaned_yaml[key]
    
    # Write cleaned config to a temporary file
    temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    try:
        OmegaConf.save(cleaned_yaml, temp_config_file.name)
        temp_config_file.close()
        
        # Modify args to use the temporary config file
        modified_args = []
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg == "--config" and i + 1 < len(args):
                modified_args.extend(["--config", temp_config_file.name])
                skip_next = True
            else:
                modified_args.append(arg)
        
        # Now load the config using AReaL's config loader
        config, _ = load_expr_config(modified_args, SFTConfig)
        config: SFTConfig
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_config_file.name)
        except Exception:
            pass
    
    rank = int(os.getenv("RANK", "0"))
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info(f"SFT Training: {training_mode}")
        logger.info("=" * 80)
        if max_train_samples:
            logger.info(f"Training on {max_train_samples} samples")
        else:
            logger.info("Training on full dataset")
        logger.info(f"Total epochs: {config.total_train_epochs}")
        logger.info(f"Allocation mode: {config.allocation_mode}")
        logger.info("=" * 80)
    
    seeding.set_random_seed(config.seed, f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    
    engine = FSDPLMEngine(config=config.model)
    engine.create_process_group(parallel_strategy=parallel_strategy)
    
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    
    # Create dataset and dataloaders
    # Apply max_train_samples limit if specified
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    
    # Limit dataset if max_train_samples is specified
    if max_train_samples:
        original_len = len(train_dataset)
        if max_train_samples < original_len:
            if rank == 0:
                logger.info(f"Limiting dataset from {original_len} to {max_train_samples} samples")
            # HuggingFace datasets support select()
            train_dataset = train_dataset.select(range(max_train_samples))
    
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )
    
    train_dataloader = create_dataloader(
        train_dataset,
        rank=engine.data_parallel_rank,
        world_size=engine.data_parallel_world_size,
        dataset_config=config.train_dataset,
        collate_fn=pad_sequences_to_tensors,
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=engine.data_parallel_rank,
        world_size=engine.data_parallel_world_size,
        dataset_config=config.valid_dataset,
        collate_fn=pad_sequences_to_tensors,
    )
    
    # Calculate actual dataset size for FinetuneSpec
    # The dataset has already been limited if max_train_samples was specified
    actual_dataset_size = len(train_dataset)
    
    # Initialize engine
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=actual_dataset_size,
        train_batch_size=config.train_dataset.batch_size,
    )
    engine.initialize(None, ft_spec)
    
    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)
    
    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        engine,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )
    
    total_epochs = config.total_train_epochs
    
    if rank == 0:
        logger.info(f"Starting training from step {start_step}")
        logger.info(f"Total epochs: {total_epochs}")
        logger.info(f"Steps per epoch: {len(train_dataloader)}")
        logger.info(f"Total steps: {len(train_dataloader) * total_epochs}")
    
    global_step = 0
    for epoch in range(total_epochs):
        if rank == 0:
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{total_epochs}")
            logger.info(f"{'='*80}")
        
        for step, data in enumerate(train_dataloader):
            if global_step < start_step:
                global_step += 1
                continue
            step_info = StepInfo(
                global_step=global_step,
                epoch=epoch,
                epoch_step=step,
                steps_per_epoch=len(train_dataloader),
            )
            
            with stats_tracker.record_timing("to_device"):
                # NOTE: data are identical across model+context parallel group
                data = tensor_container_to(data, current_platform.current_device())
            
            with stats_tracker.record_timing("bcast"):
                data = broadcast_tensor_container(
                    data,
                    src_rank=engine.current_data_parallel_head(),
                    group=engine.context_and_model_parallel_group,
                )
            
            with (
                stats_tracker.record_timing("train_step"),
                stats_tracker.scope("sft"),
            ):
                stats = engine.train_lm(data)
                engine.step_lr_scheduler()
                stats_tracker.scalar(**stats)
            
            with stats_tracker.record_timing("save"):
                saver.save(engine, epoch, step, global_step, tokenizer=tokenizer)
            
            with stats_tracker.record_timing("checkpoint_for_recover"):
                recover_handler.dump(
                    engine,
                    step_info,
                    saver,
                    evaluator,
                    stats_logger,
                    train_dataloader,
                    tokenizer=tokenizer,
                )
            
            dist.barrier(device_ids=[engine.device.index])
            current_platform.synchronize()
            
            with stats_tracker.record_timing("eval"):
                # No need to log anything. Logging will be handled outside
                # via stats_tracker.export().
                def evaluate_fn():
                    with stats_tracker.scope("sft-eval"):
                        for data in valid_dataloader:
                            data = tensor_container_to(
                                data, current_platform.current_device()
                            )
                            data = broadcast_tensor_container(
                                data,
                                src_rank=engine.current_data_parallel_head(),
                                group=engine.context_and_model_parallel_group,
                            )
                            engine.evaluate_lm(data)
                
                evaluator.evaluate(
                    evaluate_fn,
                    epoch,
                    step,
                    global_step,
                )
            
            dist.barrier(device_ids=[engine.device.index])
            current_platform.synchronize()
            
            stats_logger.commit(
                epoch,
                step,
                global_step,
                stats_tracker.export(reduce_group=engine.data_parallel_group),
            )
            global_step += 1
            
            # Log progress every 10 steps
            if rank == 0 and step % 10 == 0:
                stats_dict = stats_tracker.export(reduce_group=engine.data_parallel_group)
                loss = stats_dict.get("sft/loss/avg", 0.0)
                logger.info(f"Epoch {epoch + 1}/{total_epochs}, Step {step}/{len(train_dataloader)}, Loss: {loss:.4f}")
    
    if rank == 0:
        logger.info("\n" + "=" * 80)
        logger.info("Training completed!")
        logger.info("=" * 80)
        
        # Auto-upload logs if configured (same as GRPO script)
        upload_method = os.environ.get("AUTO_UPLOAD_LOGS_METHOD")
        if upload_method:
            # Normalize method to lowercase (handle EMAIL -> email, etc.)
            upload_method = upload_method.lower()
            logger.info(f"\n{'='*80}")
            logger.info(f"📤 Auto-uploading training summary via {upload_method}...")
            logger.info(f"{'='*80}\n")
            try:
                upload_script = os.path.join(script_dir, "upload_logs.py")
                
                # Create a training summary file to upload
                # This is similar to what GRPO does, but for SFT we create a summary since there are no test logs
                summary_dir = os.path.join(config.cluster.fileroot, "sft", "test_logs")
                os.makedirs(summary_dir, exist_ok=True)
                
                from datetime import datetime
                summary_file = os.path.join(summary_dir, f"sft_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
                
                # Write training summary
                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write("=" * 80 + "\n")
                    f.write("SFT Training Summary\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Experiment: {config.experiment_name}\n")
                    f.write(f"Trial: {config.trial_name}\n")
                    f.write(f"Training Mode: {training_mode}\n")
                    f.write(f"Total Epochs: {total_epochs}\n")
                    f.write(f"Dataset Size: {actual_dataset_size}\n")
                    f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("\n")
                    f.write("Checkpoints saved to:\n")
                    checkpoint_base = os.path.join(config.cluster.fileroot, "sft", "checkpoints", config.experiment_name, config.trial_name)
                    latest_checkpoint = None
                    if os.path.exists(checkpoint_base):
                        # Find latest checkpoint
                        import glob
                        checkpoint_dirs = glob.glob(os.path.join(checkpoint_base, "**", "epoch*"), recursive=True)
                        if checkpoint_dirs:
                            checkpoint_dirs.sort(key=os.path.getmtime, reverse=True)
                            latest_checkpoint = checkpoint_dirs[0]
                            f.write(f"  Latest: {latest_checkpoint}\n")
                            f.write(f"  Total checkpoints: {len(checkpoint_dirs)}\n")
                        else:
                            f.write(f"  Base directory: {checkpoint_base}\n")
                    else:
                        f.write(f"  Base directory: {checkpoint_base} (not found yet)\n")
                    f.write("\n")
                    f.write("=" * 80 + "\n")
                    f.write("To test the model, run:\n")
                    f.write(f"  python examples/cloud_gsm8k/test_sft_model_cloud.py \\\n")
                    if latest_checkpoint:
                        f.write(f"    --model-path {latest_checkpoint} \\\n")
                    else:
                        f.write(f"    --model-path <checkpoint_path> \\\n")
                    f.write(f"    --test-all \\\n")
                    f.write(f"    --batch-size 32\n")
                    f.write("=" * 80 + "\n")
                
                logger.info(f"Created training summary: {summary_file}")
                
                # Upload the summary file
                upload_cmd = [
                    sys.executable,
                    upload_script,
                    "--log-dir", summary_dir,
                    "--method", upload_method,
                    "--log-files", summary_file,  # Upload specific file
                ]
                
                # Add method-specific arguments from environment
                if upload_method == "email":
                    if os.environ.get("AUTO_UPLOAD_EMAIL_TO"):
                        upload_cmd.extend(["--email-to", os.environ.get("AUTO_UPLOAD_EMAIL_TO")])
                elif upload_method == "gdrive":
                    if os.environ.get("AUTO_UPLOAD_GDRIVE_FOLDER_ID"):
                        upload_cmd.extend(["--gdrive-folder-id", os.environ.get("AUTO_UPLOAD_GDRIVE_FOLDER_ID")])
                elif upload_method == "s3":
                    if os.environ.get("AUTO_UPLOAD_S3_BUCKET"):
                        upload_cmd.extend(["--s3-bucket", os.environ.get("AUTO_UPLOAD_S3_BUCKET")])
                    if os.environ.get("AUTO_UPLOAD_S3_PREFIX"):
                        upload_cmd.extend(["--s3-prefix", os.environ.get("AUTO_UPLOAD_S3_PREFIX")])
                elif upload_method == "hf":
                    if os.environ.get("AUTO_UPLOAD_HF_REPO_ID"):
                        upload_cmd.extend(["--hf-repo-id", os.environ.get("AUTO_UPLOAD_HF_REPO_ID")])
                elif upload_method == "wandb":
                    if os.environ.get("AUTO_UPLOAD_WANDB_PROJECT"):
                        upload_cmd.extend(["--wandb-project", os.environ.get("AUTO_UPLOAD_WANDB_PROJECT")])
                    if os.environ.get("AUTO_UPLOAD_WANDB_RUN_NAME"):
                        upload_cmd.extend(["--wandb-run-name", os.environ.get("AUTO_UPLOAD_WANDB_RUN_NAME")])
                elif upload_method == "webhook":
                    if os.environ.get("AUTO_UPLOAD_WEBHOOK_URL"):
                        upload_cmd.extend(["--webhook-url", os.environ.get("AUTO_UPLOAD_WEBHOOK_URL")])
                    if os.environ.get("AUTO_UPLOAD_WEBHOOK_API_KEY"):
                        upload_cmd.extend(["--webhook-api-key", os.environ.get("AUTO_UPLOAD_WEBHOOK_API_KEY")])
                
                logger.info(f"Running upload command: {' '.join(upload_cmd)}")
                upload_result = subprocess.run(upload_cmd, check=False, capture_output=True, text=True)
                
                if upload_result.stdout:
                    logger.info(f"Upload stdout: {upload_result.stdout}")
                if upload_result.stderr:
                    logger.warning(f"Upload stderr: {upload_result.stderr}")
                
                if upload_result.returncode == 0:
                    logger.info(f"✅ Training summary uploaded successfully via {upload_method}!")
                else:
                    logger.warning(f"⚠️  Log upload failed with exit code {upload_result.returncode}")
                    if upload_result.stderr:
                        logger.warning(f"   Error: {upload_result.stderr}")
            except Exception as upload_error:
                logger.warning(f"⚠️  Failed to upload logs: {upload_error}")
                import traceback
                logger.warning(f"   Traceback: {traceback.format_exc()}")
                logger.info(f"   You can manually upload the summary using:")
                logger.info(f"   python {upload_script} --log-dir {summary_dir} --method {upload_method} --log-files {summary_file} ...")
            logger.info(f"{'='*80}\n")
    
    stats_logger.close()
    engine.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])

