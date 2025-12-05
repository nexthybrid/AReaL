#!/usr/bin/env python3
import os, wandb
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

# --- Configuration ---
os.environ["WANDB_PROJECT"] = "qwen2.5-gsm8k-finetune"
# os.environ["WANDB_LOG_MODEL"] = "false"
SEED = 147
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LENGTH = 512
LORA_RANK = 64  # High rank for reasoning
LORA_ALPHA = 128  # Alpha = 2 * Rank is standard for high-rank
BATCH_SIZE = 24
GRAD_ACCUM = 1
LEARNING_RATE = 2e-4  # Standard LoRA LR
EPOCH=3
RUN_NAME=f"qwen2.5-0.5b-gsm8k-lora-unsloth_{MAX_SEQ_LENGTH}_{LEARNING_RATE}_{EPOCH}"
OUTPUT_DIR = RUN_NAME


def main():
    # 1. Load Model & Tokenizer
    print(f"Loading {MODEL_ID} with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # None = Auto-detect (Float16 for T4, Bfloat16 for Ampere+)
        load_in_4bit=False,  # FALSE for 0.5B models to preserve accuracy!
    )

    # 2. Add LoRA Adapters
    # We use "Target All Linear" which is crucial for Math/Reasoning tasks
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,  # Set to 0 for Unsloth (optimized)
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. Prepare Dataset (GSM8K)
    print("Loading and Formatting Dataset...")
    raw_dataset = load_dataset("openai/gsm8k", "main")
    # GSM8K has 'train' (7.5k) and 'test' (1.3k). Use 'test' as validation.
    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["test"]

    def formatting_prompts_func(examples):
        texts = []
        questions = examples["question"]
        answers = examples["answer"]

        for q, a in zip(questions, answers):
            # Qwen uses boxed answers for math, replace with \boxed{} to match Qwen's native math training.
            # GSM8K uses "####" to mark the answer.
            if "####" in a:
                # Example: ".... #### 42" -> ".... \boxed{42}"
                parts = a.split("####")
                # We strip whitespace to ensure clean formatting
                reasoning = parts[0].strip()
                final_val = parts[1].strip()
                a = f"{reasoning}\nThe final answer is \\boxed{{{final_val}}}."

            messages = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return texts

    # 4. Training Arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=10,
        num_train_epochs=EPOCH,  # 3 is standard for GSM8K
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",  # Use 8-bit optimizer to save VRAM
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=SEED,
        report_to="wandb",
        run_name=RUN_NAME,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False, # Can set to True for speed, but requires care with masking
        # Evaluation Settings
        eval_strategy="steps",
        eval_steps=50,  # Evaluate frequently
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,  # Save space
        load_best_model_at_end=True,
    )

    # 5. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        args=training_args
    )

    # Qwen 2.5 Chat Template handling
    # We need to find the specific "instruction" part vs "response" part
    # Qwen uses <|im_start|>assistant\n to mark the start of the answer
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # 6. Train
    print("Starting Training...")
    trainer_stats = trainer.train()
    print(f"Done! Training took {trainer_stats.metrics['train_runtime']/60:.1f} minutes.")

    # 7. Inference Test (Quick sanity check)
    print("\nTraining finished. Running a test inference...")
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    inputs = tokenizer(
        [
            tokenizer.apply_chat_template([
                {"role": "user", "content": "Janet buys 3 packs of gum for $2 each. How much did she spend?"}
            ], tokenize=False, add_generation_prompt=True)
        ], return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    print(tokenizer.batch_decode(outputs))

    # 8. Save Model
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")

    # Close W&B run cleanly
    wandb.finish()


if __name__ == "__main__":
    main()