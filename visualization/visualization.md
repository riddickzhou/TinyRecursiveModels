# Latent Visualization Documentation

## Data Source & Selection

**Dataset:** Test set (`data/*/test/`)
**Selection:** Sequential (first N samples)
**Toggle settings:**
- Train vs Test: Line 233 in `visualize_latents.py` - change `'test'` to `'train'`
- Random selection: Lines 234-236 - replace `[:args.num_samples]` with random indexing
- Number of samples: `--num_samples` argument (default: 5)

---

## Files and Functions

### `visualize_latents.py`
Python script to extract and visualize z_H and z_L latent features from trained TRM models.

**Functions:**
- `load_checkpoint(ckpt_path)`: Load model and config from checkpoint file
- `reverse_embed(latent, embed_weight)`: Convert latent vectors to tokens via nearest neighbor in embedding space
- `extract_latents(model, batch, return_all_states)`: Extract z_H, z_L from model forward pass and reverse-embed to tokens. Optionally returns all intermediate states.
- `visualize_single_grid(grid, title, save_path)`: Visualize a single Sudoku grid (used for intermediate states)
- `visualize_sudoku(data, idx, save_path)`: Create Figure 6 style visualization (6-panel grid showing input/target/prediction/tokenized latents)
- `main()`: CLI entry point with argument parsing

### `run_visualization.sh`
SLURM job script to run visualization on GPU (MLP model only).

---

## Usage Guide

### Run with SLURM:

**Activate conda environment and submit in same terminal:**
```bash
module load conda/new
conda activate trm-sudoku
sbatch visualization/run_visualization.sh [DATA_PATH] [NUM_SAMPLES] [ALL_STATES]
```

**Note:** The conda environment must be activated in your terminal before running `sbatch` so the job inherits the Python environment.

**Default arguments:**
- `DATA_PATH`: `data/sudoku-extreme-1k-aug-1000`
- `NUM_SAMPLES`: `5`
- `ALL_STATES`: `""` (pass `"true"` to enable intermediate state visualization)

**Examples:**
```bash
# Basic visualization (final states only)
sbatch visualization/run_visualization.sh

# With all intermediate states (21 states per sample)
sbatch visualization/run_visualization.sh data/sudoku-extreme-1k-aug-1000 5 true
```

### Direct Python call (for interactive use):
```bash
module load conda/new
conda activate trm-sudoku

# Basic visualization
python visualization/visualize_latents.py \
    --checkpoint checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_65100 \
    --data_path data/sudoku-extreme-1k-aug-1000 \
    --output_dir outputs/visualization/mlp \
    --num_samples 5

# With all intermediate states
python visualization/visualize_latents.py \
    --checkpoint checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_65100 \
    --data_path data/sudoku-extreme-1k-aug-1000 \
    --output_dir outputs/visualization/mlp \
    --num_samples 5 \
    --all_states
```

---

## Input/Output

**Input:**
- Trained model checkpoint (`.pt` file)
- Test dataset directory
- Number of samples to visualize
- `--all_states` flag (optional): Visualize all intermediate loop states

**Output** (saved to `outputs/visualization/mlp/`):
- `sample_0.png, sample_1.png, ...`: 6-panel visualizations showing:
  - Input x (original puzzle)
  - Target y (ground truth solution)
  - Prediction ŷ (model output)
  - Tokenized z_H (decoded high-level latent)
  - Tokenized z_L (decoded low-level latent)
- `latent_data.pt`: Raw tensors (z_H, z_L, tokenized versions, predictions, inputs, targets)
- `vis_out.json`: JSON output with all grids
- `slurm-{job_id}.out`: SLURM job log

**Additional output with `--all_states`:**
- `states/sample_N/`: Directory containing intermediate state visualizations
  - `state_00_zL_H0_L0.png` through `state_20_zH_H2.png`: 21 states per sample
  - Naming: `state_{idx}_{type}_H{h_step}_L{l_step}.png`

---

## Intermediate States

With default config (H_cycles=3, L_cycles=6), the model produces **21 intermediate states**:

| H_step | States |
|--------|--------|
| 0 | z_L (L=0..5), then z_H |
| 1 | z_L (L=0..5), then z_H |
| 2 | z_L (L=0..5), then z_H |

Total: 3 × 6 = 18 z_L states + 3 z_H states = 21 states

---

## Implementation Details

**How it works:**
- Loads trained TRM model and extracts `inner_carry` containing z_H and z_L tensors
- Performs reverse embedding: computes cosine similarity between latent vectors and embedding weight matrix, then takes argmax to find nearest token
- Creates Figure 6 style visualizations matching the paper (page 12)
- With `--all_states`, captures all intermediate z_H and z_L during the reasoning loop
- Minimal dependencies: only adds `matplotlib` to existing environment

**Key code paths:**
- Model forward pass: `model.inner(carry, batch, return_all_states=True)` returns all intermediate states
- Embedding weight: `model.inner.embed_tokens.embedding_weight` ([vocab_size, hidden_size])
- Reverse embed formula: `argmax(z @ embed_weight.T, dim=-1)`
