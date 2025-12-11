#!/usr/bin/env python3
"""
Test and compare model performance before and after training.
"""

import argparse
import json
import os
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
from areal.reward.math_parser import process_results, parse_digits

def test_model(
    model_path: str,
    max_samples: int = 10,
    max_new_tokens: int = 1024,
    log_dir: str | None = None,
    test_all: bool = False,
    batch_size: int = 32
):
    """Test the model on GSM8K samples."""
    
    # Prepare logging
    if log_dir is None:
        log_dir = os.path.join("examples", "local_gsm8k", "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"test_{ts}.log")

    def _log(msg: str):
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")

    _log(f"\n{'='*60}")
    _log(f"Testing model: {model_path}")
    _log(f"Log file: {log_path}")
    _log(f"{'='*60}\n")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use CPU for more stable inference
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        # Check for MPS (macOS only) - safe check for Windows compatibility
        try:
            mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        except (AttributeError, RuntimeError):
            mps_available = False
        
        if mps_available:
            device = torch.device("mps")
    _log(f"Using device: {device}")

    # Use bfloat16 if GPU supports it (Ampere+), else float16
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load GSM8K test set
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    # Determine how many samples to test
    if test_all or max_samples == -1:
        num_samples = len(dataset)
        _log(f"Testing on FULL dataset: {num_samples} samples")
    else:
        num_samples = min(max_samples, len(dataset))
        _log(f"Testing on {num_samples} samples (out of {len(dataset)} total)")
    
    results = []
    correct = 0

    data_list = list(dataset.select(range(num_samples)))
    pbar = tqdm(total=num_samples)
    for batch in get_batch(data_list, batch_size):
        batch_prompts = []
        batch_answers = []

        for item in batch:
            # Handle GSM8K formatting
            ans = item["answer"]
            hashes_idx = ans.find("#### ")
            if hashes_idx != -1:
                ans = ans[:hashes_idx] + "\\boxed{" + ans[hashes_idx + 5 :] + "}"
            batch_answers.append(ans)

            # Apply chat template to raw string
            messages = [{"role": "user", "content": item["question"]}]
            # We use the tokenizer's template string, not the tensor yet
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_prompts.append(formatted_prompt)

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,  # This uses the 'left' padding set earlier
            truncation=True
        ).to(model.device)
        
        # Generate with greedy decoding for stability
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 4. Decode and Process Results
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        parser_result, extracted_answers = process_results(correct_answer, generated_text)
        gt_extracted, sol_extracted = [parse_digits(ans) for ans in extracted_answers]
        # Calculate the starting index for this batch relative to the total dataset
        batch_start_index = pbar.n
        for i, generated_text in enumerate(decoded_texts):
            global_idx = batch_start_index + i + 1
            correct_answer = batch_answers[i]
            question = batch[i]["question"]

            parser_result, extracted_answers = process_results(correct_answer, generated_text)
            gt_extracted, sol_extracted = [parse_digits(ans) for ans in extracted_answers]

            is_correct = bool(parser_result)
            if is_correct:
                correct += 1
        
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "generated": generated_text,
                "gt_extracted": gt_extracted,
                "sol_extracted": sol_extracted,
                "correct": is_correct,
            })
        
            _log(f"\n--- Question {global_idx} ---")
            _log(f"Question: {question}")

            log_ready_generated_text = generated_text.replace("\n", "\n\t").strip()
            log_ready_correct_answer = correct_answer.replace("\n", "\n\t").strip()
            _log(f"Generated Answer:\n\t{log_ready_generated_text}")
            _log(f"Correct Answer (full):\n\t{log_ready_correct_answer}")

            _log(f"Extracted -> GT: {gt_extracted} | Sol: {sol_extracted}")
            _log(f"Result: {'[CORRECT]' if is_correct else '[INCORRECT]'}")

            pbar.set_postfix({
                "Correct": correct,
                "Total": global_idx,
                "Accuracy (%)": f"{(correct / (global_idx) * 100):.2f}",
            })

        pbar.update(len(batch))

    accuracy = correct / len(results) * 100
    _log(f"\n{'='*60}")
    _log(f"ACCURACY: {accuracy:.2f}% ({correct}/{len(results)})")
    _log(f"Log saved to: {log_path}")
    _log(f"{'='*60}\n")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "results": results,
    }


def compare_models(base_model: str, trained_model: str, max_samples: int = 10, test_all: bool = False):
    """Compare base model and trained model."""
    
    print(f"\n{'#'*60}")
    print("MODEL COMPARISON")
    print(f"{'#'*60}\n")
    
    # Test base model
    base_results = test_model(base_model, max_samples=max_samples, test_all=test_all)
    
    # Test trained model
    trained_results = test_model(trained_model, max_samples=max_samples, test_all=test_all)
    
    # Print comparison
    print(f"\n{'#'*60}")
    print("COMPARISON SUMMARY")
    print(f"{'#'*60}")
    print(f"Base Model Accuracy:    {base_results['accuracy']:.2f}%")
    print(f"Trained Model Accuracy: {trained_results['accuracy']:.2f}%")
    improvement = trained_results['accuracy'] - base_results['accuracy']
    print(f"Improvement:            {improvement:+.2f}%")
    print(f"{'#'*60}\n")
    
    # Save results
    comparison = {
        "base_model": base_model,
        "trained_model": trained_model,
        "base_results": base_results,
        "trained_results": trained_results,
        "improvement": improvement,
    }
    
    with open("model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print("Results saved to model_comparison.json")
    
    return comparison

def get_batch(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def main():
    parser = argparse.ArgumentParser(description="Test and compare models")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path to test",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model path",
    )
    parser.add_argument(
        "--trained-model",
        type=str,
        default="./outputs/gsm8k-training",
        help="Trained model path",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base and trained models",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of samples to test (use -1 or --all for full test set)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test on full GSM8K test set (all 1319 samples)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to write timestamped log files",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Testing batch size",
    )
    
    args = parser.parse_args()
    
    # Determine if testing all samples
    test_all = args.all or args.max_samples == -1
    
    if args.compare:
        compare_models(
            args.base_model, 
            args.trained_model, 
            max_samples=args.max_samples if not test_all else -1
        )
    elif args.model:
        test_model(
            args.model,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            log_dir=args.log_dir,
            test_all=test_all,
            batch_size=args.batch_size,
        )
    else:
        # Default: test trained model
        test_model(
            args.trained_model,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            log_dir=args.log_dir,
            test_all=test_all,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()

