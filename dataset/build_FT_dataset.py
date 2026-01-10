#!/usr/bin/env python3
"""Convert sudoku dataset to finetuning format."""

import json
import numpy as np
from pathlib import Path

TEMPLATE = """Fill the position where the values are 0 in a 9x9 grid with digits 1-9 so that each column, each row, and each of the nine 3x3 subgrids that compose the grid contains all of the digits from 1 to 9. (Sudoku Extreme)

Input:
{input_grid}

Output:"""

def format_grid(arr):
    """Format 9x9 array as grid with | separators."""
    lines = []
    for row in arr:
        lines.append('|'.join(str(x) for x in row))
    return '\n'.join(lines)

def tokens_to_sudoku(tokens):
    """Convert token array to sudoku numbers (token 1->0, 2->1, ..., 10->9)."""
    return np.where(tokens == 1, 0, tokens - 1)

def main():
    # Load training data
    data_dir = Path('data/sudoku-extreme-1k-aug-1000/train')
    inputs = np.load(data_dir / 'all__inputs.npy', mmap_mode='r')
    labels = np.load(data_dir / 'all__labels.npy', mmap_mode='r')

    # Process samples
    samples = []
    for i in range(len(inputs)):
        # Convert tokens to sudoku grids
        input_grid = tokens_to_sudoku(inputs[i]).reshape(9, 9)
        output_grid = tokens_to_sudoku(labels[i]).reshape(9, 9)

        # Format prompt
        user_content = TEMPLATE.format(input_grid=format_grid(input_grid))
        assistant_content = format_grid(output_grid)

        samples.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        })

        if (i + 1) % 100000 == 0:
            print(f"Processed {i + 1}/{len(inputs)} samples")

    # Save
    output_path = Path('data/FT.json')
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} samples to {output_path}")

if __name__ == '__main__':
    main()
