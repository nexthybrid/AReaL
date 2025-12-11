#!/usr/bin/env python3
"""
Extract answers from interval test logs and perform ensemble majority voting.

This script:
1. Parses test log files from interval checkpoints (epochs 4, 9, 14, 19, 24)
2. Extracts answers for all 1319 questions from each checkpoint
3. Performs majority voting to select the best answer for each question
4. Calculates ensemble accuracy

Usage:
    python3 examples/cloud_gsm8k/ensemble_from_logs.py \
        --log-dir examples/cloud_gsm8k/train_logs/trainingtestlogs_full_grpo_25ep \
        --epochs 4 9 14 19 24
"""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import math parser, but if it fails, we'll use the extracted answers from logs
try:
    from areal.reward.math_parser import process_results
    HAS_MATH_PARSER = True
except ImportError:
    HAS_MATH_PARSER = False
    print("⚠️  Warning: math_parser not available, will use extracted answers from logs only")


def parse_log_file(log_file: str) -> Dict[int, Dict]:
    """
    Parse a test log file and extract answers for all samples.
    
    Returns:
        Dictionary mapping sample_index -> {
            'question': str,
            'correct_answer': str,
            'generated_text': str,
            'extracted_answer': str (from generated text),
            'correct': bool
        }
    """
    results = {}
    current_sample = None
    current_question = None
    current_correct_answer = None
    current_generated_text = None
    current_extracted_answer = None
    current_correct_extracted = None  # Store extracted correct answer
    current_correct = None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Match DEBUG line with extracted_answers FIRST (it appears before Sample line)
        # Format: [DEBUG] Sample 1: parser_result=0, extracted_answers=('18', '14')
        debug_match = re.match(r'\[DEBUG\] Sample (\d+):.*extracted_answers=\([\'"]?([^\'"]*)[\'"]?,\s*[\'"]?([^\'"]*)[\'"]?\)', line)
        if debug_match:
            sample_num = int(debug_match.group(1))
            correct_extracted = debug_match.group(2)
            generated_extracted = debug_match.group(3)
            
            # Initialize sample if not exists
            if sample_num not in results:
                results[sample_num] = {
                    'question': None,
                    'correct_answer': None,
                    'correct_extracted': None,
                    'generated_text': None,
                    'extracted_answer': None,
                    'correct': None,
                }
            
            # Store extracted answers
            results[sample_num]['correct_extracted'] = correct_extracted if correct_extracted else None
            results[sample_num]['extracted_answer'] = generated_extracted if generated_extracted else None
        
        # Match sample number: "Sample 1/1319" or "Sample 1319/1319"
        sample_match = re.match(r'Sample (\d+)/\d+', line)
        if sample_match:
            # Start/update current sample
            current_sample = int(sample_match.group(1))
            
            # Initialize if not already done by DEBUG line
            if current_sample not in results:
                results[current_sample] = {
                    'question': None,
                    'correct_answer': None,
                    'correct_extracted': None,
                    'generated_text': None,
                    'extracted_answer': None,
                    'correct': None,
                }
            
            # Reset other fields (but keep extracted answers from DEBUG line)
            current_question = None
            current_correct_answer = None
            current_generated_text = None
            current_correct = None
        
        # Match question
        elif line == "Question:" and current_sample is not None:
            # Collect question (may span multiple lines)
            i += 1
            question_lines = []
            while i < len(lines) and not lines[i].strip().startswith("─"):
                question_lines.append(lines[i].rstrip())
                i += 1
            current_question = "\n".join(question_lines).strip()
            if current_sample is not None:
                results[current_sample]['question'] = current_question
            i -= 1  # Back up one line
        
        # Match generated response
        elif line == "Generated Response:" and current_sample is not None:
            # Collect generated response (may span multiple lines)
            i += 1
            response_lines = []
            while i < len(lines) and not lines[i].strip().startswith("─"):
                response_lines.append(lines[i].rstrip())
                i += 1
            current_generated_text = "\n".join(response_lines).strip()
            if current_sample is not None:
                results[current_sample]['generated_text'] = current_generated_text
            i -= 1  # Back up one line
        
        # Match correct answer
        elif line == "Correct Answer:" and current_sample is not None:
            # Collect correct answer (may span multiple lines)
            i += 1
            answer_lines = []
            while i < len(lines) and not lines[i].strip().startswith("─"):
                answer_lines.append(lines[i].rstrip())
                i += 1
            current_correct_answer = "\n".join(answer_lines).strip()
            if current_sample is not None:
                results[current_sample]['correct_answer'] = current_correct_answer
            i -= 1  # Back up one line
        
        # Match result
        elif line.startswith("Result:") and current_sample is not None:
            if "✅ CORRECT" in line:
                current_correct = True
                if current_sample is not None:
                    results[current_sample]['correct'] = True
            elif "❌ INCORRECT" in line:
                current_correct = False
                if current_sample is not None:
                    results[current_sample]['correct'] = False
        
        i += 1
    
    # Debug: check if we're getting correct_extracted
    samples_with_correct = sum(1 for r in results.values() if r.get('correct_extracted') is not None)
    if samples_with_correct == 0:
        print(f"⚠️  Warning: No correct_extracted values found in log file. Check DEBUG line parsing.")
    else:
        print(f"  Found correct_extracted for {samples_with_correct}/{len(results)} samples")
    
    return results


def extract_answer_from_generated_text(text: str) -> Optional[str]:
    """Extract answer from generated text (fallback if not in log)."""
    # Try to find \boxed{} pattern
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1].strip()
    
    # Try boxed{} without backslash
    boxed_pattern2 = r'boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern2, text)
    if matches:
        return matches[-1].strip()
    
    return None


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


def main():
    parser = argparse.ArgumentParser(
        description="Extract answers from interval test logs and perform ensemble majority voting"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory containing test log files",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[4, 9, 14, 19, 24],
        help="Epochs to include in ensemble (default: 4 9 14 19 24)",
    )
    parser.add_argument(
        "--output-log",
        type=str,
        default=None,
        help="Output log file path (default: ensemble_from_logs_{timestamp}.log)",
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise ValueError(f"Log directory does not exist: {log_dir}")
    
    # Find log files for each epoch
    epoch_log_files = {}
    for epoch in args.epochs:
        # Look for log files matching pattern: *epoch{epoch}*.log
        pattern = f"*epoch{epoch}*.log"
        matches = list(log_dir.glob(pattern))
        if not matches:
            print(f"⚠️  Warning: No log file found for epoch {epoch}")
            continue
        if len(matches) > 1:
            # If multiple matches, use the most recent one
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        epoch_log_files[epoch] = matches[0]
        print(f"Epoch {epoch}: {epoch_log_files[epoch].name}")
    
    if not epoch_log_files:
        raise ValueError("No log files found for any epoch!")
    
    # Sort epochs to ensure latest is processed last (for tie-breaking)
    sorted_epochs = sorted(epoch_log_files.keys())
    latest_epoch = sorted_epochs[-1]
    
    print(f"\n{'='*80}")
    print(f"Parsing log files and extracting answers...")
    print(f"{'='*80}\n")
    
    # Parse all log files
    all_results = {}
    for epoch in sorted_epochs:
        log_file = epoch_log_files[epoch]
        print(f"Parsing epoch {epoch}: {log_file.name}...")
        results = parse_log_file(str(log_file))
        all_results[epoch] = results
        print(f"  Extracted {len(results)} samples")
    
    # Verify all epochs have same number of samples
    num_samples = len(all_results[sorted_epochs[0]])
    for epoch in sorted_epochs:
        if len(all_results[epoch]) != num_samples:
            print(f"⚠️  Warning: Epoch {epoch} has {len(all_results[epoch])} samples, expected {num_samples}")
    
    print(f"\n{'='*80}")
    print(f"Performing majority voting...")
    print(f"{'='*80}\n")
    
    # Perform ensemble evaluation
    ensemble_results = []
    correct = 0
    
    # Process each sample
    for sample_idx in range(1, num_samples + 1):
        # Collect answers from all checkpoints
        answers_by_epoch = {}
        question = None
        correct_answer = None
        
        for epoch in sorted_epochs:
            if sample_idx in all_results[epoch]:
                result = all_results[epoch][sample_idx]
                if question is None:
                    question = result.get("question")
                    correct_answer = result.get("correct_answer")
                
                extracted_answer = result.get("extracted_answer")
                # If extracted_answer is None, try to extract from generated_text
                if extracted_answer is None and result.get("generated_text"):
                    extracted_answer = extract_answer_from_generated_text(result.get("generated_text"))
                
                answers_by_epoch[epoch] = extracted_answer
        
        # Get default answer (from latest checkpoint)
        default_answer = answers_by_epoch.get(latest_epoch)
        
        # Collect all answers for voting
        all_answers = [answers_by_epoch.get(epoch) for epoch in sorted_epochs]
        
        # Perform majority voting
        ensemble_answer = majority_vote(all_answers, default_answer)
        
        # Check correctness using the extracted correct answer from logs
        # Get the correct extracted answer from the first epoch (they should all be the same)
        correct_extracted = None
        if sorted_epochs and sample_idx in all_results[sorted_epochs[0]]:
            first_result = all_results[sorted_epochs[0]][sample_idx]
            correct_extracted = first_result.get("correct_extracted")
        
        is_correct = False
        parser_error = None
        
        if correct_extracted and ensemble_answer:
            # Try to use math parser if available (handles numerical equivalence)
            if HAS_MATH_PARSER:
                try:
                    parser_result, _ = process_results(correct_extracted, ensemble_answer)
                    is_correct = bool(parser_result)
                except Exception as e:
                    parser_error = str(e)
                    # Fallback to string comparison
                    is_correct = (correct_extracted.strip() == ensemble_answer.strip())
            else:
                # Simple string comparison (exact match)
                # This is less accurate than math parser but works for most cases
                is_correct = (correct_extracted.strip() == ensemble_answer.strip())
        
        if is_correct:
            correct += 1
        
        ensemble_results.append({
            "sample_idx": sample_idx,
            "question": question,
            "correct_answer": correct_answer,
            "answers_by_epoch": answers_by_epoch,
            "ensemble_answer": ensemble_answer,
            "correct": is_correct,
            "parser_error": parser_error,
        })
        
        # Log progress
        if sample_idx % 100 == 0 or sample_idx == num_samples:
            current_acc = correct / sample_idx * 100
            print(f"Progress: {sample_idx}/{num_samples} | Correct: {correct}/{sample_idx} | Accuracy: {current_acc:.2f}%")
    
    # Calculate final accuracy
    accuracy = correct / num_samples * 100
    
    # Also calculate individual checkpoint accuracies for comparison
    individual_accuracies = {}
    for epoch in sorted_epochs:
        epoch_correct = 0
        for sample_idx in range(1, num_samples + 1):
            if sample_idx in all_results[epoch]:
                result = all_results[epoch][sample_idx]
                if result.get("correct"):
                    epoch_correct += 1
        
        epoch_accuracy = epoch_correct / num_samples * 100
        individual_accuracies[epoch] = epoch_accuracy
    
    # Prepare output log
    if args.output_log is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_log = log_dir / f"ensemble_from_logs_{timestamp}.log"
    else:
        output_log = Path(args.output_log)
    
    # Write output log
    with open(output_log, 'w', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"ENSEMBLE MAJORITY VOTING FROM LOG FILES\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Log directory: {log_dir}\n")
        f.write(f"Epochs tested: {sorted_epochs}\n")
        f.write(f"Number of samples: {num_samples}\n\n")
        
        f.write(f"{'='*80}\n")
        f.write(f"ENSEMBLE FINAL ACCURACY: {accuracy:.2f}% ({correct}/{num_samples})\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Individual Checkpoint Accuracies:\n")
        for epoch, acc in individual_accuracies.items():
            f.write(f"  Epoch {epoch}: {acc:.2f}% ({int(acc * num_samples / 100)}/{num_samples})\n")
        f.write(f"\n")
        
        f.write(f"{'='*80}\n")
        f.write(f"DETAILED RESULTS\n")
        f.write(f"{'='*80}\n\n")
        
        for result in ensemble_results:
            f.write(f"\nSample {result['sample_idx']}/{num_samples}\n")
            f.write(f"{'─'*80}\n")
            f.write(f"Question: {result['question']}\n")
            f.write(f"Correct Answer: {result['correct_answer']}\n")
            f.write(f"\nAnswers by Epoch:\n")
            for epoch in sorted_epochs:
                answer = result['answers_by_epoch'].get(epoch, "N/A")
                f.write(f"  Epoch {epoch}: {answer}\n")
            f.write(f"Ensemble Answer: {result['ensemble_answer']}\n")
            f.write(f"Result: {'✅ CORRECT' if result['correct'] else '❌ INCORRECT'}\n")
            if result['parser_error']:
                f.write(f"Parser Error: {result['parser_error']}\n")
            f.write(f"{'─'*80}\n")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ENSEMBLE TESTING SUMMARY")
    print(f"{'='*80}")
    print(f"Ensemble Accuracy: {accuracy:.2f}% ({correct}/{num_samples})")
    print(f"\nIndividual Checkpoint Accuracies:")
    for epoch, acc in individual_accuracies.items():
        print(f"  Epoch {epoch}: {acc:.2f}% ({int(acc * num_samples / 100)}/{num_samples})")
    print(f"\nDetailed log: {output_log}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

