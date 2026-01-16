#!/usr/bin/env python3
"""
Halting Experiment: Test hypothesis that halting when output stabilizes yields higher accuracy.

Hypothesis: If the model reaches correct answer, it stops updating. So if we halt when
two consecutive rounds produce the same output, we might achieve near-perfect precision.

This script uses existing trained models (no retraining needed).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.functions import load_model_class


def load_checkpoint(ckpt_path, data_path):
    """Load model and config from checkpoint."""
    import yaml
    from pathlib import Path

    ckpt_path = Path(ckpt_path)
    config_path = ckpt_path.parent / 'all_config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    metadata_path = Path(data_path) / 'train' / 'dataset.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    config['arch'].update({
        'batch_size': 1,
        'seq_len': metadata['seq_len'],
        'num_puzzle_identifiers': metadata['num_puzzle_identifiers'],
        'vocab_size': metadata['vocab_size'],
    })

    model_cls = load_model_class(config['arch']['name'])
    model = model_cls(config['arch'])

    state_dict = torch.load(ckpt_path, map_location='cuda')
    state_dict = {k.replace('_orig_mod.model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    return model.cuda().eval(), config


def extract_latents_with_adaptive_halting(model, batch, stability_threshold=2, max_steps=-1):
    """
    Extract latents with adaptive halting based on output stability.

    Args:
        model: Trained TRM model
        batch: Input batch
        stability_threshold: Number of consecutive identical outputs before halting (default: 2)
        max_steps: Safety cap on maximum steps (-1 for unlimited)

    Returns:
        Dictionary with results and per-sample halting info
    """
    num_samples = batch['inputs'].shape[0]
    unlimited = (max_steps == -1)
    device = batch['inputs'].device

    with torch.no_grad():
        carry = model.initial_carry(batch)
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
        carry.halted = carry.halted.to(device)
        carry.steps = carry.steps.to(device)
        for k in carry.current_data:
            if isinstance(carry.current_data[k], torch.Tensor):
                carry.current_data[k] = carry.current_data[k].to(device)

        embed_weight = model.inner.embed_tokens.embedding_weight
        puzzle_emb_len = model.inner.puzzle_emb_len

        # Track per-sample state
        sample_halted = [False] * num_samples
        sample_halt_step = [None] * num_samples  # Will be set when halted
        sample_stable_count = [0] * num_samples
        prev_preds = [None] * num_samples
        final_preds = [None] * num_samples
        final_z_H = [None] * num_samples
        final_z_L = [None] * num_samples

        # Collect all states for visualization (adaptive per sample)
        all_states_per_sample = [[] for _ in range(num_samples)]

        sup_step = 0
        while not all(sample_halted) and (unlimited or sup_step < max_steps):
            if sup_step == 0:
                carry.inner_carry = model.inner.reset_carry(carry.halted, carry.inner_carry)

            # Run inner forward with all_states
            new_inner_carry, logits, q_logits, step_states = model.inner(
                carry.inner_carry, batch, return_all_states=True
            )

            # Get current predictions (tokenized output y)
            current_preds = torch.argmax(logits, dim=-1)  # [B, seq_len]

            # Collect states for each non-halted sample
            for state_type, h_step, l_step, tensor in step_states:
                if state_type == 'z_H':  # Only track y states
                    tokenized = torch.argmax(model.inner.lm_head(tensor), dim=-1)[:, puzzle_emb_len:]
                    for idx in range(num_samples):
                        if not sample_halted[idx]:
                            all_states_per_sample[idx].append({
                                'sup_step': sup_step,
                                'h_step': h_step,
                                'tokenized': tokenized[idx].cpu(),
                            })

            # Check stability for each sample
            for idx in range(num_samples):
                if sample_halted[idx]:
                    continue

                curr_pred = current_preds[idx]

                if prev_preds[idx] is not None:
                    if torch.equal(curr_pred, prev_preds[idx]):
                        sample_stable_count[idx] += 1
                    else:
                        sample_stable_count[idx] = 0

                # Check if we should halt this sample
                if sample_stable_count[idx] >= stability_threshold - 1:
                    sample_halted[idx] = True
                    sample_halt_step[idx] = sup_step + 1  # 1-indexed step count
                    final_preds[idx] = curr_pred.cpu()
                    final_z_H[idx] = new_inner_carry.z_H[idx].cpu()
                    final_z_L[idx] = new_inner_carry.z_L[idx].cpu()
                else:
                    prev_preds[idx] = curr_pred.clone()

            # Update carry
            carry.inner_carry = new_inner_carry

            # Increment step counter
            sup_step += 1

        # For samples that never halted (hit safety cap), use final state
        for idx in range(num_samples):
            if not sample_halted[idx]:
                sample_halt_step[idx] = sup_step  # Hit the safety cap
                final_preds[idx] = torch.argmax(logits, dim=-1)[idx].cpu()
                final_z_H[idx] = carry.inner_carry.z_H[idx].cpu()
                final_z_L[idx] = carry.inner_carry.z_L[idx].cpu()

        # Reverse embed final states
        final_z_H_stacked = torch.stack(final_z_H).to(device)
        final_z_L_stacked = torch.stack(final_z_L).to(device)

        tokenized_zH = reverse_embed(final_z_H_stacked[:, puzzle_emb_len:], embed_weight)
        tokenized_zL = reverse_embed(final_z_L_stacked[:, puzzle_emb_len:], embed_weight)

    result = {
        'latent_y_raw': final_z_H_stacked.cpu(),
        'latent_z_raw': final_z_L_stacked.cpu(),
        'latent_y': tokenized_zH.cpu(),
        'latent_z': tokenized_zL.cpu(),
        'predictions': torch.stack(final_preds),
        'inputs': batch['inputs'].cpu(),
        'targets': batch['targets'].cpu(),
        'halt_steps': sample_halt_step,
        'all_states_per_sample': all_states_per_sample,
    }
    return result


def reverse_embed(latent, embed_weight):
    """Convert latent vectors to tokens via nearest neighbor in embedding space."""
    similarities = torch.matmul(latent.float(), embed_weight.float().T)
    return torch.argmax(similarities, dim=-1)


def tokens_to_sudoku(arr):
    """Convert token IDs to Sudoku numbers."""
    arr = arr.astype(int)
    arr = np.where(arr == 1, 0, arr - 1)
    return arr.tolist()


def visualize_sudoku(data, idx, save_path):
    """Create 6-panel visualization for Sudoku (same as visualize_latents.py)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    def to_sudoku(arr):
        arr = arr.astype(int)
        return np.where(arr == 1, 0, arr - 1)

    grids = [
        to_sudoku(data['inputs'][idx].numpy().reshape(9, 9)),
        to_sudoku(data['targets'][idx].numpy().reshape(9, 9)),
        to_sudoku(data['predictions'][idx].numpy().reshape(9, 9)),
        to_sudoku(data['latent_y'][idx].numpy().reshape(9, 9)),
        to_sudoku(data['latent_z'][idx].numpy().reshape(9, 9)),
        None
    ]
    titles = ['Input x', 'Target y', 'Prediction ŷ', 'Latent y\n(before lm_head)', 'Latent z\n(reasoning)', '']
    target_grid = grids[1]

    for grid_idx, (ax, title, grid) in enumerate(zip(axes.flat, titles, grids)):
        if grid is not None:
            ax.set_xlim(-0.5, 8.5)
            ax.set_ylim(-0.5, 8.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xlabel(title, fontsize=24, fontweight='normal', labelpad=10)

            for i in range(10):
                lw = 2 if i % 3 == 0 else 0.5
                ax.axhline(i - 0.5, color='black', linewidth=lw)
                ax.axvline(i - 0.5, color='black', linewidth=lw)

            for i in range(9):
                for j in range(9):
                    if grid[i, j] != 0:
                        if grid_idx == 2 and grid[i, j] != target_grid[i, j]:
                            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='red', alpha=0.3))
                        ax.text(j, i, str(int(grid[i, j])), ha='center', va='center',
                               fontsize=20, fontweight='normal')

            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    plt.tight_layout(h_pad=3.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_adaptive_states(data, idx, save_path, max_plot_rows=1000):
    """Visualize intermediate states with adaptive sizing based on actual halt step."""
    states = data['all_states_per_sample'][idx]
    halt_step = data['halt_steps'][idx]
    target_grid = np.array(tokens_to_sudoku(data['targets'][idx].numpy().reshape(9, 9)))

    if not states:
        return

    # Determine actual number of supervision steps used, cap at max_plot_rows
    actual_sup_steps = max(s['sup_step'] for s in states) + 1
    n_cols = 3  # T (H_cycles)
    n_rows = min(actual_sup_steps, max_plot_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Initialize all axes
    for row in range(n_rows):
        for col in range(n_cols):
            axes[row, col].axis('off')

    for state in states:
        sup_step = state['sup_step']
        if sup_step >= max_plot_rows:
            continue  # Skip states beyond plot limit
        h_step = state['h_step']

        grid = tokens_to_sudoku(state['tokenized'].numpy().reshape(9, 9))
        grid = np.array(grid)

        row = sup_step
        col = h_step
        title = f'y (s={sup_step}, T={h_step})'

        ax = axes[row, col]
        ax.axis('on')
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.5, 8.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel(title, fontsize=7)

        for i in range(10):
            lw = 1.0 if i % 3 == 0 else 0.2
            ax.axhline(i - 0.5, color='black', linewidth=lw)
            ax.axvline(i - 0.5, color='black', linewidth=lw)

        for i in range(9):
            for j in range(9):
                if grid[i, j] != 0:
                    if grid[i, j] != target_grid[i, j]:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='red', alpha=0.3))
                    ax.text(j, i, str(int(grid[i, j])), ha='center', va='center',
                           fontsize=5, fontweight='normal')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'Sample {idx}: Halted at step {halt_step} (adaptive)\n(rows=supervision steps, cols=T cycles)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Halting experiment: test adaptive halting hypothesis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/halting')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to test')
    parser.add_argument('--stability_threshold', type=int, default=2,
                        help='Consecutive identical outputs before halting (default: 2)')
    parser.add_argument('--max_steps', type=int, default=16, help='Maximum supervision steps')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output suffix based on max_steps
    suffix = "unlimited" if args.max_steps == -1 else str(args.max_steps)

    # Setup multi-GPU
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs")

    print(f"=== Halting Experiment ===")
    print(f"Stability threshold: {args.stability_threshold} consecutive identical outputs")
    print(f"Max steps: {'unlimited' if args.max_steps == -1 else args.max_steps}")
    print(f"Num samples: {'all' if args.num_samples == -1 else args.num_samples}")
    print(f"Num GPUs: {num_gpus}")
    print()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, args.data_path)

    # Load data
    print(f"Loading dataset: {args.data_path}")
    data_dir = Path(args.data_path) / 'test'
    all_inputs = np.load(data_dir / 'all__inputs.npy', mmap_mode='r')
    all_labels = np.load(data_dir / 'all__labels.npy', mmap_mode='r')
    all_puzzle_ids = np.load(data_dir / 'all__puzzle_identifiers.npy', mmap_mode='r')

    # -1 means all samples
    num_samples = len(all_inputs) if args.num_samples == -1 else args.num_samples
    inputs = all_inputs[:num_samples]
    labels = all_labels[:num_samples]
    puzzle_ids = all_puzzle_ids[:num_samples]
    print(f"Using {num_samples} samples")

    print("Running adaptive halting extraction...")
    if num_gpus > 1:
        # Multi-GPU: split samples and process in parallel
        from concurrent.futures import ThreadPoolExecutor
        import copy

        chunk_size = (num_samples + num_gpus - 1) // num_gpus
        results = [None] * num_gpus

        def process_chunk(gpu_id, start_idx, end_idx):
            torch.cuda.set_device(gpu_id)
            # Load model on this GPU
            chunk_model, _ = load_checkpoint(args.checkpoint, args.data_path)
            chunk_model = chunk_model.to(f'cuda:{gpu_id}')

            chunk_batch = {
                'inputs': torch.from_numpy(np.array(inputs[start_idx:end_idx])).to(f'cuda:{gpu_id}'),
                'targets': torch.from_numpy(np.array(labels[start_idx:end_idx])).to(f'cuda:{gpu_id}'),
                'puzzle_identifiers': torch.from_numpy(np.array(puzzle_ids[start_idx:end_idx])).to(f'cuda:{gpu_id}'),
            }
            return extract_latents_with_adaptive_halting(
                chunk_model, chunk_batch,
                stability_threshold=args.stability_threshold,
                max_steps=args.max_steps
            )

        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id in range(num_gpus):
                start_idx = gpu_id * chunk_size
                end_idx = min(start_idx + chunk_size, num_samples)
                if start_idx < end_idx:
                    futures.append(executor.submit(process_chunk, gpu_id, start_idx, end_idx))

            chunk_results = [f.result() for f in futures]

        # Merge results
        data = {
            'predictions': torch.cat([r['predictions'] for r in chunk_results]),
            'inputs': torch.cat([r['inputs'] for r in chunk_results]),
            'targets': torch.cat([r['targets'] for r in chunk_results]),
            'latent_y': torch.cat([r['latent_y'] for r in chunk_results]),
            'latent_z': torch.cat([r['latent_z'] for r in chunk_results]),
            'latent_y_raw': torch.cat([r['latent_y_raw'] for r in chunk_results]),
            'latent_z_raw': torch.cat([r['latent_z_raw'] for r in chunk_results]),
            'halt_steps': sum([r['halt_steps'] for r in chunk_results], []),
            'all_states_per_sample': sum([r['all_states_per_sample'] for r in chunk_results], []),
        }
    else:
        batch = {
            'inputs': torch.from_numpy(np.array(inputs)).cuda(),
            'targets': torch.from_numpy(np.array(labels)).cuda(),
            'puzzle_identifiers': torch.from_numpy(np.array(puzzle_ids)).cuda(),
        }
        data = extract_latents_with_adaptive_halting(
            model, batch,
            stability_threshold=args.stability_threshold,
            max_steps=args.max_steps
        )

    # Compute accuracy
    predictions = data['predictions'].numpy()
    targets = data['targets'].numpy()

    per_sample_correct = []
    for idx in tqdm(range(num_samples), desc="Computing accuracy"):
        pred = predictions[idx]
        target = targets[idx]
        is_correct = np.array_equal(pred, target)
        per_sample_correct.append(is_correct)

    accuracy = sum(per_sample_correct) / len(per_sample_correct)
    avg_halt_step = sum(data['halt_steps']) / len(data['halt_steps'])

    print()
    print(f"=== Results ===")
    print(f"Accuracy: {accuracy * 100:.2f}% ({sum(per_sample_correct)}/{len(per_sample_correct)})")
    print(f"Average halt step: {avg_halt_step:.2f}")
    print()

    # Save experiment results (accuracy and step counts)
    experiment_results = {
        'config': {
            'stability_threshold': args.stability_threshold,
            'max_steps': args.max_steps,
            'num_samples': num_samples,
            'checkpoint': args.checkpoint,
        },
        'summary': {
            'accuracy': accuracy,
            'accuracy_percent': f"{accuracy * 100:.2f}%",
            'total_correct': sum(per_sample_correct),
            'total_samples': len(per_sample_correct),
            'average_halt_step': avg_halt_step,
        },
        'per_sample': [
            {
                'sample_id': idx,
                'correct': per_sample_correct[idx],
                'halt_step': data['halt_steps'][idx],
            }
            for idx in range(num_samples)
        ]
    }

    results_filename = f'experiment_results_{suffix}.json'
    with open(output_dir / results_filename, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print(f"Saved: {output_dir / results_filename}")

    # Save visualization JSON (same format as visualize_latents.py)
    json_data = []
    for idx in range(num_samples):
        sample_data = {
            'sample_id': idx,
            'input_x': tokens_to_sudoku(data['inputs'][idx].numpy().reshape(9, 9)),
            'target_y': tokens_to_sudoku(data['targets'][idx].numpy().reshape(9, 9)),
            'prediction_y_hat': tokens_to_sudoku(data['predictions'][idx].numpy().reshape(9, 9)),
            'latent_y': tokens_to_sudoku(data['latent_y'][idx].numpy().reshape(9, 9)),
            'latent_z': tokens_to_sudoku(data['latent_z'][idx].numpy().reshape(9, 9)),
            'halt_step': data['halt_steps'][idx],
            'correct': per_sample_correct[idx],
        }
        if data['all_states_per_sample'][idx]:
            sample_data['intermediate_y'] = [
                {'sup_step': s['sup_step'], 't_step': s['h_step'],
                 'grid': tokens_to_sudoku(s['tokenized'].numpy().reshape(9, 9))}
                for s in data['all_states_per_sample'][idx]
            ]
        json_data.append(sample_data)

    vis_filename = f'vis_out_{suffix}.json'
    with open(output_dir / vis_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved: {output_dir / vis_filename}")

    # Save raw data
    latent_filename = f'latent_data_{suffix}.pt'
    torch.save(data, output_dir / latent_filename)
    print(f"Saved: {output_dir / latent_filename}")

    # Create visualizations (only first 10 samples to save time)
    print("Creating visualizations...")
    vis_samples = min(10, num_samples)
    for idx in tqdm(range(vis_samples), desc="Plotting samples"):
        visualize_sudoku(data, idx, output_dir / f'sample_{idx}.png')
        visualize_adaptive_states(data, idx, output_dir / f'sample_{idx}_all_states.png')

    print(f"\nDone! Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
