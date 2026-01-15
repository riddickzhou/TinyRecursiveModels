#!/usr/bin/env python3
"""Minimal script to visualize z_H and z_L latent features from trained TRM models."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.functions import load_model_class
from puzzle_dataset import PuzzleDataset


def load_checkpoint(ckpt_path, data_path):
    """Load model and config from checkpoint."""
    import yaml
    import json
    from pathlib import Path

    ckpt_path = Path(ckpt_path)
    config_path = ckpt_path.parent / 'all_config.yaml'

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset metadata
    metadata_path = Path(data_path) / 'train' / 'dataset.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Merge metadata into arch config
    config['arch'].update({
        'batch_size': 1,
        'seq_len': metadata['seq_len'],
        'num_puzzle_identifiers': metadata['num_puzzle_identifiers'],
        'vocab_size': metadata['vocab_size'],
    })

    # Load model
    model_cls = load_model_class(config['arch']['name'])
    model = model_cls(config['arch'])

    # Load state dict
    state_dict = torch.load(ckpt_path, map_location='cuda')
    state_dict = {k.replace('_orig_mod.model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    return model.cuda().eval(), config


def reverse_embed(latent, embed_weight):
    """Convert latent vectors to tokens via nearest neighbor in embedding space."""
    similarities = torch.matmul(latent.float(), embed_weight.float().T)
    return torch.argmax(similarities, dim=-1)


def extract_latents(model, batch, return_all_states=False):
    """Extract z_H and z_L from model forward pass."""
    with torch.no_grad():
        carry = model.initial_carry(batch)
        # Move carry tensors to CUDA
        carry.inner_carry.z_H = carry.inner_carry.z_H.cuda()
        carry.inner_carry.z_L = carry.inner_carry.z_L.cuda()
        carry.halted = carry.halted.cuda()
        carry.steps = carry.steps.cuda()
        for k in carry.current_data:
            if isinstance(carry.current_data[k], torch.Tensor):
                carry.current_data[k] = carry.current_data[k].cuda()

        # Reset carry for fresh start (simulates halted=True scenario)
        carry.inner_carry = model.inner.reset_carry(carry.halted, carry.inner_carry)

        # Run inner forward with optional all_states
        if return_all_states:
            new_inner_carry, logits, q_logits, all_states = model.inner(
                carry.inner_carry, batch, return_all_states=True
            )
        else:
            new_inner_carry, logits, q_logits = model.inner(carry.inner_carry, batch)
            all_states = None

        z_H = new_inner_carry.z_H  # [batch, seq+puzzle_emb_len, hidden]
        z_L = new_inner_carry.z_L

        # Get embedding weight
        embed_weight = model.inner.embed_tokens.embedding_weight

        # Reverse embed (skip puzzle_emb positions)
        puzzle_emb_len = model.inner.puzzle_emb_len
        tokenized_zH = reverse_embed(z_H[:, puzzle_emb_len:], embed_weight)
        tokenized_zL = reverse_embed(z_L[:, puzzle_emb_len:], embed_weight)

        # Get predictions
        preds = torch.argmax(logits, dim=-1)

        # Process all_states if requested
        all_states_tokenized = None
        if return_all_states and all_states:
            all_states_tokenized = []
            for state_type, h_step, l_step, tensor in all_states:
                tokenized = reverse_embed(tensor[:, puzzle_emb_len:], embed_weight)
                all_states_tokenized.append({
                    'type': state_type,
                    'h_step': h_step,
                    'l_step': l_step,
                    'tensor': tensor.cpu(),
                    'tokenized': tokenized.cpu(),
                })

    result = {
        'z_H': z_H.cpu(),
        'z_L': z_L.cpu(),
        'tokenized_zH': tokenized_zH.cpu(),
        'tokenized_zL': tokenized_zL.cpu(),
        'predictions': preds.cpu(),
        'inputs': batch['inputs'].cpu(),
        'targets': batch['targets'].cpu(),
    }
    if return_all_states:
        result['all_states'] = all_states_tokenized
    return result


def visualize_single_grid(grid, title, save_path):
    """Visualize a single Sudoku grid."""
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=16, fontweight='normal', pad=10)

    # Draw grid lines
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i - 0.5, color='black', linewidth=lw)
        ax.axvline(i - 0.5, color='black', linewidth=lw)

    # Fill in numbers
    for i in range(9):
        for j in range(9):
            if grid[i, j] != 0:
                ax.text(j, i, str(int(grid[i, j])), ha='center', va='center',
                       fontsize=16, fontweight='normal')

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_sudoku(data, idx, save_path):
    """Create Figure 6 style visualization for Sudoku."""
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert tokens to Sudoku numbers (1->0, 2->1, ..., 10->9)
    def to_sudoku(arr):
        arr = arr.astype(int)
        return np.where(arr == 1, 0, arr - 1)

    grids = [
        to_sudoku(data['inputs'][idx].numpy().reshape(9, 9)),
        to_sudoku(data['targets'][idx].numpy().reshape(9, 9)),
        to_sudoku(data['predictions'][idx].numpy().reshape(9, 9)),
        to_sudoku(data['tokenized_zH'][idx].numpy().reshape(9, 9)),
        to_sudoku(data['tokenized_zL'][idx].numpy().reshape(9, 9)),
        None
    ]
    titles = ['Input x', 'Output y', 'Prediction ŷ', 'Latent y\n(before lm_head)', 'Latent z\n(reasoning feature)', '']

    for ax, title, grid in zip(axes.flat, titles, grids):
        if grid is not None:
            ax.set_xlim(-0.5, 8.5)
            ax.set_ylim(-0.5, 8.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xlabel(title, fontsize=24, fontweight='normal', labelpad=10)

            # Draw grid lines
            for i in range(10):
                lw = 2 if i % 3 == 0 else 0.5
                ax.axhline(i - 0.5, color='black', linewidth=lw)
                ax.axvline(i - 0.5, color='black', linewidth=lw)

            # Fill in numbers (skip if value is 0 - blank)
            for i in range(9):
                for j in range(9):
                    if grid[i, j] != 0:
                        ax.text(j, i, str(int(grid[i, j])), ha='center', va='center',
                               fontsize=20, fontweight='normal')

            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    plt.tight_layout(h_pad=3.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/visualization')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--all_states', action='store_true', help='Visualize all intermediate states')
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, args.data_path)

    # Load data
    print(f"Loading dataset: {args.data_path}")
    import numpy as np
    data_dir = Path(args.data_path) / 'test'
    inputs = np.load(data_dir / 'all__inputs.npy', mmap_mode='r')[:args.num_samples]
    labels = np.load(data_dir / 'all__labels.npy', mmap_mode='r')[:args.num_samples]
    puzzle_ids = np.load(data_dir / 'all__puzzle_identifiers.npy', mmap_mode='r')[:args.num_samples]

    # Create batch
    batch = {
        'inputs': torch.from_numpy(np.array(inputs)).cuda(),
        'targets': torch.from_numpy(np.array(labels)).cuda(),
        'puzzle_identifiers': torch.from_numpy(np.array(puzzle_ids)).cuda(),
    }

    print("Extracting latents...")
    data = extract_latents(model, batch, return_all_states=args.all_states)

    # Save raw data
    torch.save(data, output_dir / 'latent_data.pt')
    print(f"Saved raw data: {output_dir / 'latent_data.pt'}")

    # Prepare JSON output (convert tokens to Sudoku numbers: 1->0, 2->1, ..., 10->9)
    import json
    import numpy as np

    def tokens_to_sudoku(arr):
        """Convert token IDs to Sudoku numbers."""
        arr = arr.astype(int)
        arr = np.where(arr == 1, 0, arr - 1)  # Token 1 -> 0 (blank), others -> subtract 1
        return arr.tolist()

    json_data = []
    for idx in range(args.num_samples):
        json_data.append({
            'sample_id': idx,
            'input': tokens_to_sudoku(data['inputs'][idx].numpy().reshape(9, 9)),
            'target': tokens_to_sudoku(data['targets'][idx].numpy().reshape(9, 9)),
            'prediction': tokens_to_sudoku(data['predictions'][idx].numpy().reshape(9, 9)),
            'tokenized_zH': tokens_to_sudoku(data['tokenized_zH'][idx].numpy().reshape(9, 9)),
            'tokenized_zL': tokens_to_sudoku(data['tokenized_zL'][idx].numpy().reshape(9, 9)),
        })

    # Save JSON
    with open(output_dir / 'vis_out.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON: {output_dir / 'vis_out.json'}")

    # Visualize
    print("Creating visualizations...")
    for idx in range(args.num_samples):
        visualize_sudoku(data, idx, output_dir / f'sample_{idx}.png')

    # Visualize all intermediate states if requested
    if args.all_states and 'all_states' in data:
        print("Creating intermediate state visualizations...")
        states_dir = output_dir / 'states'
        states_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(args.num_samples):
            sample_dir = states_dir / f'sample_{idx}'
            sample_dir.mkdir(parents=True, exist_ok=True)

            for state_idx, state in enumerate(data['all_states']):
                state_type = state['type']
                h_step = state['h_step']
                l_step = state['l_step']

                grid = tokens_to_sudoku(state['tokenized'][idx].numpy().reshape(9, 9))
                grid = np.array(grid)

                if state_type == 'z_L':
                    title = f'z_L (H={h_step}, L={l_step})'
                    filename = f'state_{state_idx:02d}_zL_H{h_step}_L{l_step}.png'
                else:
                    title = f'z_H (H={h_step})'
                    filename = f'state_{state_idx:02d}_zH_H{h_step}.png'

                visualize_single_grid(grid, title, sample_dir / filename)

            print(f"  Saved {len(data['all_states'])} states for sample {idx}")

    print(f"\nDone! Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
