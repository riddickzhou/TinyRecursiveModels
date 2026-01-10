#!/usr/bin/env python3
"""Supervised fine-tuning for LLMs on Sudoku dataset using Transformers."""

import argparse
import json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

def load_sft_data(data_path, num_samples=-1):
    """Load SFT data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    if num_samples > 0:
        data = data[:num_samples]

    print(f"Loaded {len(data)} training samples")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to base model')
    parser.add_argument('--data_path', type=str, default='data/FT.json', help='Path to SFT data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for checkpoints')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to use (-1 for all)')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Per-device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--save_steps', type=int, default=500, help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=10, help='Log every N steps')
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'sft_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("="*60)
    print(f"SFT Configuration")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*60)

    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Enable gradient checkpointing for memory efficiency
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Load and prepare dataset
    print("\nLoading dataset...")
    raw_data = load_sft_data(args.data_path, args.num_samples)

    # Tokenize dataset
    def tokenize_function(examples):
        # Format as chat and tokenize
        texts = []
        for msg in examples['messages']:
            # Apply chat template
            text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            texts.append(text)

        # Tokenize with truncation and padding
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            return_tensors=None,
        )

        # Set labels = input_ids for causal LM
        tokenized['labels'] = tokenized['input_ids'].copy()

        return tokenized

    # Convert to HF dataset and tokenize
    dataset = Dataset.from_list(raw_data)
    print(f"Tokenizing {len(dataset)} examples...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Model saved to: {output_dir / 'final_model'}")
    print("="*60)

if __name__ == '__main__':
    main()
