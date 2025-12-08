#!/usr/bin/env python3
"""
Checkpoint Ensemble Testing Script

This script tests multiple checkpoints (epochs 4, 9, 14, 19, 24) on the full GSM8K test set
and uses majority voting to select the best answer for each question.

For efficiency on RunPod:
- Loads one checkpoint at a time
- Processes all 1319 test samples with that checkpoint
- Moves to next checkpoint
- After all checkpoints are processed, performs majority voting
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from areal.reward.math_parser import extract_answer, process_results


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model and tokenizer from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Check if it's a HuggingFace model identifier or local path
    if os.path.isdir(checkpoint_path):
        # Local checkpoint directory
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
    else:
        # HuggingFace model identifier
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
    
    return model, tokenizer


def extract_answer_from_text(text: str) -> Optional[str]:
    """
    Extract the numerical answer from generated text using AReaL's math parser.
    """
    try:
        extracted = extract_answer(text, "math", use_last_number=True)
        if extracted and extracted.strip() not in ["None", "none", ""]:
            return extracted.strip()
    except Exception:
        pass
    return None


def test_checkpoint(
    checkpoint_path: str,
    checkpoint_name: str,
    device: torch.device,
    max_new_tokens: int = 512,
    batch_size: int = 32,
    log_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Test a single checkpoint on the full GSM8K test set.
    
    Returns:
        List of dictionaries with keys: question, correct_answer, generated_text, extracted_answer
    """
    if log_dir is None:
        log_dir = os.path.join("/workspace", "outputs", "grpo", "test_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, f"ensemble_checkpoint_{checkpoint_name}.log")
    
    def _log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()
    
    _log(f"\n{'='*80}")
    _log(f"Testing checkpoint: {checkpoint_name}")
    _log(f"Checkpoint path: {checkpoint_path}")
    _log(f"{'='*80}\n")
    
    # Load model
    model, tokenizer = load_model_from_checkpoint(checkpoint_path, device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    model.eval()
    _log(f"Model loaded and set to eval mode")
    
    # Load GSM8K test set
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    num_samples = len(dataset)
    _log(f"Testing on FULL dataset: {num_samples} samples")
    
    results = []
    dataset_subset = list(dataset)
    
    _log(f"Processing {num_samples} samples in batches of {batch_size}...")
    
    # Process in batches
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_samples = dataset_subset[batch_start:batch_end]
        batch_size_actual = len(batch_samples)
        
        # Prepare batch data
        batch_questions = []
        batch_correct_answers = []
        batch_messages = []
        
        for sample in batch_samples:
            question = sample["question"]
            correct_answer = sample["answer"]
            batch_questions.append(question)
            batch_correct_answers.append(correct_answer)
            
            # Format prompt: simple user message format
            messages = [
                {"role": "user", "content": f"{question}\nPlease put your final answer within \\boxed{{}}."}
            ]
            batch_messages.append(messages)
        
        # Tokenize batch
        tokenizer.padding_side = "left"
        batch_inputs = tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        # Store input lengths for decoding
        input_lengths = (batch_inputs != tokenizer.pad_token_id).sum(dim=1).cpu().tolist()
        
        # Generate in batch
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,  # Greedy decoding
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            batch_outputs = model.generate(batch_inputs, **gen_kwargs)
        
        # Decode each output
        for batch_idx, sample in enumerate(batch_samples):
            sample_idx = batch_start + batch_idx
            question = batch_questions[batch_idx]
            correct_answer = batch_correct_answers[batch_idx]
            
            # Extract generated tokens
            input_len = input_lengths[batch_idx]
            generated_token_ids = batch_outputs[batch_idx][input_len:]
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            
            # Extract answer from generated text
            extracted_answer = extract_answer_from_text(generated_text)
            
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "generated_text": generated_text,
                "extracted_answer": extracted_answer,
            })
        
        # Log progress
        if (batch_end) % 100 == 0 or batch_end >= num_samples:
            _log(f"Progress: {batch_end}/{num_samples} samples processed")
    
    _log(f"\nCompleted testing checkpoint {checkpoint_name}")
    _log(f"Results saved to: {log_path}")
    
    # Free GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return results


def majority_vote(answers: List[Optional[str]], default_answer: Optional[str]) -> Optional[str]:
    """
    Perform majority voting on a list of answers.
    If there's a tie, returns the default_answer (from latest checkpoint).
    """
    # Filter out None values
    valid_answers = [a for a in answers if a is not None]
    
    if not valid_answers:
        return default_answer
    
    # Count occurrences
    answer_counts = Counter(valid_answers)
    
    # Find the most common answer(s)
    max_count = max(answer_counts.values())
    most_common = [answer for answer, count in answer_counts.items() if count == max_count]
    
    # If there's a clear winner (only one answer with max count), return it
    if len(most_common) == 1:
        return most_common[0]
    
    # If there's a tie, return the default (latest checkpoint answer)
    return default_answer


def evaluate_ensemble(
    all_results: Dict[int, List[Dict]],
    checkpoint_epochs: List[int],
    log_dir: Optional[str] = None,
) -> Dict:
    """
    Perform majority voting and evaluate ensemble accuracy.
    
    Args:
        all_results: Dictionary mapping epoch -> list of results
        checkpoint_epochs: List of epochs in order (latest should be last)
        log_dir: Directory for log files
    
    Returns:
        Dictionary with accuracy metrics and detailed results
    """
    if log_dir is None:
        log_dir = os.path.join("/workspace", "outputs", "grpo", "test_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"ensemble_majority_voting_{ts}.log")
    
    def _log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()
    
    _log(f"\n{'='*80}")
    _log(f"ENSEMBLE MAJORITY VOTING")
    _log(f"Checkpoints: {checkpoint_epochs}")
    _log(f"{'='*80}\n")
    
    # Verify all checkpoints have same number of results
    num_samples = len(all_results[checkpoint_epochs[0]])
    for epoch in checkpoint_epochs:
        if len(all_results[epoch]) != num_samples:
            raise ValueError(f"Checkpoint {epoch} has {len(all_results[epoch])} samples, expected {num_samples}")
    
    _log(f"Processing {num_samples} samples with majority voting...")
    
    ensemble_results = []
    correct = 0
    
    # Latest checkpoint (for tie-breaking)
    latest_epoch = checkpoint_epochs[-1]
    
    # Process each sample
    for sample_idx in range(num_samples):
        # Collect answers from all checkpoints
        answers_by_epoch = {}
        question = None
        correct_answer = None
        
        for epoch in checkpoint_epochs:
            result = all_results[epoch][sample_idx]
            if question is None:
                question = result["question"]
                correct_answer = result["correct_answer"]
            
            extracted_answer = result.get("extracted_answer")
            answers_by_epoch[epoch] = extracted_answer
        
        # Get default answer (from latest checkpoint)
        default_answer = answers_by_epoch[latest_epoch]
        
        # Collect all answers for voting
        all_answers = [answers_by_epoch[epoch] for epoch in checkpoint_epochs]
        
        # Perform majority voting
        ensemble_answer = majority_vote(all_answers, default_answer)
        
        # Check correctness using AReaL's math parser
        is_correct = False
        parser_error = None
        try:
            parser_result, _ = process_results(correct_answer, ensemble_answer if ensemble_answer else "")
            is_correct = bool(parser_result)
        except Exception as e:
            parser_error = str(e)
        
        if is_correct:
            correct += 1
        
        ensemble_results.append({
            "question": question,
            "correct_answer": correct_answer,
            "answers_by_epoch": answers_by_epoch,
            "ensemble_answer": ensemble_answer,
            "correct": is_correct,
            "parser_error": parser_error,
        })
        
        # Log progress
        if (sample_idx + 1) % 100 == 0 or (sample_idx + 1) == num_samples:
            current_acc = correct / (sample_idx + 1) * 100
            _log(f"Progress: {sample_idx + 1}/{num_samples} | Correct: {correct}/{sample_idx + 1} | Accuracy: {current_acc:.2f}%")
    
    # Calculate final accuracy
    accuracy = correct / num_samples * 100
    
    _log(f"\n{'='*80}")
    _log(f"ENSEMBLE FINAL ACCURACY: {accuracy:.2f}% ({correct}/{num_samples})")
    _log(f"Log saved to: {log_path}")
    _log(f"{'='*80}\n")
    
    # Also calculate individual checkpoint accuracies for comparison
    individual_accuracies = {}
    for epoch in checkpoint_epochs:
        epoch_correct = 0
        for sample_idx in range(num_samples):
            result = all_results[epoch][sample_idx]
            extracted_answer = result.get("extracted_answer")
            correct_answer = result["correct_answer"]
            
            try:
                parser_result, _ = process_results(correct_answer, extracted_answer if extracted_answer else "")
                if bool(parser_result):
                    epoch_correct += 1
            except Exception:
                pass
        
        epoch_accuracy = epoch_correct / num_samples * 100
        individual_accuracies[epoch] = epoch_accuracy
        _log(f"Epoch {epoch} individual accuracy: {epoch_accuracy:.2f}% ({epoch_correct}/{num_samples})")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": num_samples,
        "individual_accuracies": individual_accuracies,
        "results": ensemble_results,
        "log_path": log_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test ensemble of checkpoints using majority voting"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Base directory containing checkpoints (e.g., /workspace/outputs/grpo/checkpoints/root/experiment/trial/default/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[4, 9, 14, 19, 24],
        help="Epochs to test (default: 4 9 14 19 24)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for log files (default: /workspace/outputs/grpo/test_logs)",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="epoch{epoch}epochstep*globalstep*",
        help="Pattern to match checkpoint directories (default: epoch{epoch}epochstep*globalstep*)",
    )
    
    args = parser.parse_args()
    
    # Determine device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # Find checkpoint paths
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    checkpoint_paths = {}
    for epoch in args.epochs:
        # Try to find checkpoint matching pattern
        pattern = args.checkpoint_pattern.replace("{epoch}", str(epoch))
        # Use glob to find matching directories
        matches = list(checkpoint_dir.glob(pattern))
        if not matches:
            raise ValueError(f"Could not find checkpoint for epoch {epoch} in {checkpoint_dir}")
        if len(matches) > 1:
            # If multiple matches, use the one with highest global step
            matches.sort(key=lambda p: int(p.name.split("globalstep")[-1]) if "globalstep" in p.name else 0, reverse=True)
        checkpoint_paths[epoch] = str(matches[0])
        print(f"Epoch {epoch}: {checkpoint_paths[epoch]}")
    
    # Sort epochs to ensure latest is processed last (for tie-breaking)
    sorted_epochs = sorted(args.epochs)
    
    # Test each checkpoint
    all_results = {}
    for epoch in sorted_epochs:
        checkpoint_path = checkpoint_paths[epoch]
        print(f"\n{'='*80}")
        print(f"Testing checkpoint for epoch {epoch}")
        print(f"{'='*80}\n")
        
        results = test_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint_name=f"epoch{epoch}",
            device=device,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            log_dir=args.log_dir,
        )
        all_results[epoch] = results
    
    # Perform ensemble evaluation
    print(f"\n{'='*80}")
    print(f"Performing majority voting...")
    print(f"{'='*80}\n")
    
    ensemble_result = evaluate_ensemble(
        all_results=all_results,
        checkpoint_epochs=sorted_epochs,
        log_dir=args.log_dir,
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ENSEMBLE TESTING SUMMARY")
    print(f"{'='*80}")
    print(f"Ensemble Accuracy: {ensemble_result['accuracy']:.2f}% ({ensemble_result['correct']}/{ensemble_result['total']})")
    print(f"\nIndividual Checkpoint Accuracies:")
    for epoch, acc in ensemble_result['individual_accuracies'].items():
        print(f"  Epoch {epoch}: {acc:.2f}%")
    print(f"\nDetailed log: {ensemble_result['log_path']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

