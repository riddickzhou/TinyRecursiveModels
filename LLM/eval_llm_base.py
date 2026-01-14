#!/usr/bin/env python3
"""Evaluate LLM baseline performance on Sudoku using vLLM."""

# Set cache directories before any imports to avoid permission errors
import os
import sys

# Get script directory and set cache to writable location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CACHE_ROOT = os.path.join(PROJECT_ROOT, '.cache')

# Set environment variables for all cache directories
os.environ['HF_HOME'] = os.path.join(CACHE_ROOT, 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_ROOT, 'huggingface', 'transformers')
os.environ['VLLM_CACHE_ROOT'] = os.path.join(CACHE_ROOT, 'vllm')
os.environ['FLASHINFER_WORKSPACE_BASE'] = PROJECT_ROOT
os.environ['XDG_CACHE_HOME'] = CACHE_ROOT

# Disable torch.compile to avoid nvcc permission errors
os.environ['VLLM_TORCH_COMPILE_LEVEL'] = '0'
os.environ['VLLM_COMPILE_LEVEL'] = '0'

# Create cache directories
for cache_dir in [os.environ['HF_HOME'], os.environ['TRANSFORMERS_CACHE'],
                  os.environ['VLLM_CACHE_ROOT'], CACHE_ROOT]:
    os.makedirs(cache_dir, exist_ok=True)

import re
import json
import argparse
import numpy as np
from pathlib import Path
from vllm import LLM, SamplingParams

TEMPLATE = """Fill the position where the values are 0 in a 9x9 grid with digits 1-9 so that each column, each row, and each of the nine 3x3 subgrids that compose the grid contains all of the digits from 1 to 9. (Sudoku Extreme)

Input:
{input_grid}

Output:"""

def format_grid(arr, sep='|'):
    """Format 9x9 array as grid."""
    lines = []
    for row in arr:
        lines.append(sep.join(str(x) for x in row))
    return '\n'.join(lines)

def tokens_to_sudoku(tokens):
    """Convert token array to sudoku numbers (token 1->0, 2->1, ..., 10->9)."""
    return np.where(tokens == 1, 0, tokens - 1)

def parse_output(text):
    """Parse model output to extract 9x9 grid using multiple regex patterns.

    Uses LAST occurrence of valid pattern to handle models that iterate/refine.

    Returns:
        tuple: (parsed_grid, format_type) where grid is 9x9 numpy array or None
               format_type is one of: 'pipe', 'space', 'nospace', 'invalid'
    """
    # Try pattern 1: pipe-separated (e.g., "1|2|3|...")
    pattern_pipe = r'(\d)\|(\d)\|(\d)\|(\d)\|(\d)\|(\d)\|(\d)\|(\d)\|(\d)'
    matches_pipe = re.findall(pattern_pipe, text)
    if len(matches_pipe) >= 9:
        try:
            # Take LAST 9 rows (final answer after reasoning)
            grid = np.array([[int(x) for x in row] for row in matches_pipe[-9:]])
            if grid.shape == (9, 9) and np.all((grid >= 0) & (grid <= 9)):
                return grid, 'pipe'
        except:
            pass

    # Try pattern 2: space-separated (e.g., "1 2 3 ...")
    pattern_space = r'(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)'
    matches_space = re.findall(pattern_space, text)
    if len(matches_space) >= 9:
        try:
            # Take LAST 9 rows
            grid = np.array([[int(x) for x in row] for row in matches_space[-9:]])
            if grid.shape == (9, 9) and np.all((grid >= 0) & (grid <= 9)):
                return grid, 'space'
        except:
            pass

    # Try pattern 3: no separator (e.g., "123456789")
    pattern_nospace = r'(\d)(\d)(\d)(\d)(\d)(\d)(\d)(\d)(\d)'
    matches_nospace = re.findall(pattern_nospace, text)
    if len(matches_nospace) >= 9:
        try:
            # Take LAST 9 rows
            grid = np.array([[int(x) for x in row] for row in matches_nospace[-9:]])
            if grid.shape == (9, 9) and np.all((grid >= 0) & (grid <= 9)):
                return grid, 'nospace'
        except:
            pass

    # No valid pattern found
    return None, 'invalid'

def check_accuracy(pred_grid, target_grid):
    """Check if prediction matches target."""
    if pred_grid is None:
        return False
    return np.array_equal(pred_grid, target_grid)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate (use -1 for all)')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (0 for greedy)')
    parser.add_argument('--max_tokens', type=int, default=8192, help='Max tokens to generate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for vLLM inference')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPUs for tensor parallelism')
    args = parser.parse_args()

    # Setup output directory
    model_name = Path(args.model_path).name
    output_dir = Path(f'outputs/LLM/{model_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Surgical fix: active log file for append-only saving
    active_log_file = output_dir / 'conversations.jsonl'

    # Load previous conversations if exist (for crash recovery)
    conv_file = output_dir / 'conversations.json'
    existing_conversations = {}
    if conv_file.exists():
        print(f"Loading existing conversations from {conv_file}...")
        with open(conv_file) as f:
            existing_convs = json.load(f)
            existing_conversations = {c['question_id']: c for c in existing_convs}
        print(f"Loaded {len(existing_conversations)} existing conversations")

    # Load test data
    print("Loading test data...")
    data_dir = Path('data/sudoku-extreme-1k-aug-1000/test')
    inputs = np.load(data_dir / 'all__inputs.npy', mmap_mode='r')
    labels = np.load(data_dir / 'all__labels.npy', mmap_mode='r')

    # Handle num_samples (-1 means all)
    if args.num_samples > 0:
        inputs = inputs[:args.num_samples]
        labels = labels[:args.num_samples]

    print(f"Evaluating on {len(inputs)} samples")

    # Build prompts (skip already processed ones)
    print("Building prompts...")
    prompts = []
    targets = []
    indices_to_process = []
    for i in range(len(inputs)):
        if i in existing_conversations:
            continue  # Skip already processed
        input_grid = tokens_to_sudoku(inputs[i]).reshape(9, 9)
        target_grid = tokens_to_sudoku(labels[i]).reshape(9, 9)

        prompt = TEMPLATE.format(input_grid=format_grid(input_grid, sep='|'))
        prompts.append([{"role": "user", "content": prompt}])
        targets.append(target_grid)
        indices_to_process.append(i)

    print(f"Processing {len(prompts)} new samples (skipping {len(existing_conversations)} already done)")

    # Skip vLLM if nothing to process
    if len(prompts) == 0:
        print("All samples already processed, skipping vLLM")
        outputs = []
    else:
        # Load model with vLLM
        print(f"Loading model: {args.model_path}")
        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            max_model_len=8192,  # Allow long reasoning chains
            gpu_memory_utilization=0.9,
            tensor_parallel_size=args.tensor_parallel_size,
        )

        # Generate
        print("Generating responses...")
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=1.0 if args.temperature == 0 else 0.95,  # Greedy if temp=0
        )
        outputs = llm.chat(prompts, sampling_params)

    from tqdm import tqdm
    # Evaluate
    print("Evaluating...", flush=True)
    conversations = list(existing_conversations.values())  # Start with existing

    for i, output in enumerate(tqdm(outputs, desc="Evaluating samples")):
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason  # 'stop' or 'length'

        pred_grid, format_type = parse_output(generated_text)
        is_correct = check_accuracy(pred_grid, targets[i])

        # Record conversation
        question_id = indices_to_process[i]
        conv = {
            'question_id': question_id,
            'input': prompts[i],
            'output': generated_text,
            'target': format_grid(targets[i], sep='|'),
            'parsed_grid': pred_grid.tolist() if pred_grid is not None else None,
            'format_type': format_type,
            'finish_reason': finish_reason,
            'correct': is_correct,
        }
        conversations.append(conv)

        # Surgical fix: Save incrementally every 10 samples (crash recovery) - JSONL append only
        if (i + 1) % 10 == 0:
            with open(active_log_file, 'a') as f:
                # Write the last 10 conversations
                for c in conversations[-10:]:
                    f.write(json.dumps(c) + '\n')

    # Recompute metrics from all conversations (including loaded ones)
    correct = sum(1 for c in conversations if c['correct'])
    invalid_format = sum(1 for c in conversations if c['format_type'] == 'invalid')
    token_bottleneck = sum(1 for c in conversations if c['finish_reason'] == 'length')
    format_counts = {}
    stop_reasons = {}
    for c in conversations:
        format_counts[c['format_type']] = format_counts.get(c['format_type'], 0) + 1
        stop_reasons[c['finish_reason']] = stop_reasons.get(c['finish_reason'], 0) + 1

    accuracy = correct / len(inputs)
    invalid_pct = invalid_format / len(inputs)
    token_bottleneck_pct = token_bottleneck / len(inputs)

    metrics = {
        'model_path': args.model_path,
        'num_samples': len(inputs),
        'accuracy': accuracy,
        'invalid_format_pct': invalid_pct,
        'token_bottleneck_pct': token_bottleneck_pct,
        'format_distribution': format_counts,
        'stop_reasons': stop_reasons,
        'correct_count': correct,
        'invalid_count': invalid_format,
        'token_bottleneck_count': token_bottleneck,
    }

    # Save results
    with open(output_dir / 'data.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / 'conversations.json', 'w') as f:
        json.dump(conversations, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(inputs)}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(inputs)})")
    print(f"Invalid format: {invalid_pct:.2%} ({invalid_format}/{len(inputs)})")
    print(f"Token bottleneck: {token_bottleneck_pct:.2%} ({token_bottleneck}/{len(inputs)})")
    print(f"\nFormat distribution:")
    for fmt, count in format_counts.items():
        print(f"  {fmt}: {count} ({count/len(inputs):.2%})")
    print(f"\nStop reasons:")
    for reason, count in stop_reasons.items():
        print(f"  {reason}: {count} ({count/len(inputs):.2%})")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
