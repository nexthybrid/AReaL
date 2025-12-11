#!/usr/bin/env python3
"""
Test script for trained SFT model on GSM8K dataset (Cloud version).
This script loads a trained SFT model checkpoint and evaluates it on the GSM8K test set.
Uses process_results from areal.reward.math_parser for consistent answer extraction.
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure AReaL is in the Python path for math_parser
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import AReaL's math parser for consistent answer extraction
from areal.reward.math_parser import process_results


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from SFT checkpoint."""
    if os.path.isdir(checkpoint_path):
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            print(f"[INFO] Loading model from checkpoint directory: {checkpoint_path}")
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if device.type == "cuda" else None,
            )
            if device.type != "cuda":
                model = model.to(device)
            return model, tokenizer
        else:
            raise ValueError(
                f"Checkpoint directory {checkpoint_path} does not contain config.json. "
                f"Expected a HuggingFace-format checkpoint."
            )
    else:
        print(f"[INFO] Loading model from HuggingFace: {checkpoint_path}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            model = model.to(device)
        return model, tokenizer


def test_model(
    model_path: str,
    max_samples: int = 50,
    max_new_tokens: int = 512,
    log_dir: str | None = None,
    test_all: bool = False,
    temperature: float = 0.0,
    model_name: str = "SFT Model",
    batch_size: int = 32,
):
    """Test the SFT model on GSM8K samples.
    
    Args:
        model_path: Path to model checkpoint or HuggingFace model ID
        max_samples: Maximum number of test samples (ignored if test_all=True)
        max_new_tokens: Maximum tokens to generate
        log_dir: Directory to save log file
        test_all: If True, test on full test set (1319 samples)
        temperature: Sampling temperature (0.0 for greedy)
        model_name: Name for logging
        batch_size: Number of samples to process in parallel
    """
    
    # Prepare logging - save to network volume so it persists after pod stops
    if log_dir is None:
        log_dir = os.path.join("/workspace", "outputs", "sft", "test_logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"test_sft_model_{model_name.lower().replace(' ', '_')}_{ts}.log")
    
    def _log(msg: str):
        # Print directly to stdout/stderr (unbuffered) so it shows in terminal immediately
        print(msg, flush=True)
        # Also write to log file
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")
            lf.flush()  # Ensure it's written immediately
    
    _log(f"\n{'='*80}")
    _log(f"Testing {model_name} model: {model_path}")
    _log(f"Max new tokens: {max_new_tokens}")
    _log(f"Temperature: {temperature}")
    _log(f"Batch size: {batch_size}")
    _log(f"Log file: {log_path}")
    _log(f"{'='*80}\n")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Using device: {device}")
    
    # Load model and tokenizer
    _log("Loading model and tokenizer...")
    model, tokenizer = load_model_from_checkpoint(model_path, device)
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    _log("Model loaded successfully!\n")
    
    # Load GSM8K test dataset
    _log("Loading GSM8K test dataset...")
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    # Determine number of samples to test
    if test_all:
        num_samples = len(test_dataset)
        _log(f"Testing on FULL test set: {num_samples} samples")
    else:
        num_samples = min(max_samples, len(test_dataset))
        _log(f"Testing on {num_samples} samples (out of {len(test_dataset)} total)")
    
    _log(f"\n{'='*80}")
    _log("Starting evaluation...")
    _log(f"{'='*80}\n")
    
    correct = 0
    total = 0
    
    # Process in batches
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_indices = list(range(batch_start, batch_end))
        batch_data = test_dataset.select(batch_indices)
        
        # Prepare batch prompts
        batch_questions = [item["question"] for item in batch_data]
        batch_correct_answers = [item["answer"] for item in batch_data]
        
        # Format prompts using chat template
        batch_prompts = []
        for question in batch_questions:
            messages = [
                {"role": "user", "content": question}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            batch_prompts.append(prompt)
        
        # Tokenize batch
        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        
        # Generate
        with torch.no_grad():
            batch_outputs = model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and evaluate
        input_lengths = batch_inputs["attention_mask"].sum(dim=1)
        
        for batch_idx, i in enumerate(batch_indices):
            total += 1
            correct_answer = batch_correct_answers[batch_idx]
            
            # Extract generated tokens (skip input tokens)
            input_len = input_lengths[batch_idx].item()
            generated_token_ids = batch_outputs[batch_idx][input_len:]
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            
            # Check correctness using AReaL's math parser
            # Note: process_results(answer, solution) where answer is ground truth, solution is generated
            is_correct = False
            parser_error = None
            extracted_answers = (None, None)
            try:
                parser_result, extracted_answers = process_results(correct_answer, generated_text)
                is_correct = bool(parser_result)
                
                _log(f"[DEBUG] Sample {i+1}: parser_result={parser_result}, extracted_answers={extracted_answers}")
            except Exception as e:
                parser_error = str(e)
                _log(f"Warning: Parser failed for sample {i+1}: {e}")
            
            if is_correct:
                correct += 1
            
            # Log result
            _log(f"\n{'─'*80}")
            _log(f"Sample {i+1}/{num_samples}")
            _log(f"Question: {batch_questions[batch_idx]}")
            _log(f"Generated Response: {generated_text}")
            _log(f"Correct Answer: {correct_answer}")
            if extracted_answers[0] is not None and extracted_answers[1] is not None:
                _log(f"Extracted Answers: correct='{extracted_answers[0]}', generated='{extracted_answers[1]}'")
            if parser_error:
                _log(f"Parser Error: {parser_error}")
            _log(f"Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
            _log(f"{'─'*80}")
            
            # Progress update every 10 samples
            if total % 10 == 0:
                accuracy = (correct / total) * 100
                _log(f"\nProgress: {total}/{num_samples} | Correct: {correct}/{total} | Accuracy: {accuracy:.2f}%\n")
    
    # Final results
    accuracy = (correct / total) * 100
    _log(f"\n{'='*80}")
    _log(f"FINAL RESULTS")
    _log(f"{'='*80}")
    _log(f"Total samples: {total}")
    _log(f"Correct: {correct}")
    _log(f"Incorrect: {total - correct}")
    _log(f"Accuracy: {accuracy:.2f}%")
    _log(f"{'='*80}\n")
    _log(f"Log saved to: {log_path}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "log_path": log_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Test SFT model on GSM8K dataset")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of test samples (ignored if --test-all is set)",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test on full test set (1319 samples)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy decoding)",
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
        help="Directory to save log file (default: /workspace/outputs/sft/test_logs)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="SFT Model",
        help="Model name for logging",
    )
    
    args = parser.parse_args()
    
    results = test_model(
        model_path=args.model_path,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        log_dir=args.log_dir,
        test_all=args.test_all,
        temperature=args.temperature,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    
    print(f"\n✅ Testing completed!")
    print(f"Accuracy: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
    print(f"Log file: {results['log_path']}")


if __name__ == "__main__":
    main()

