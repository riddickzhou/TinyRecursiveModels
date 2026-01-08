# Latent Visualization Documentation

## Data Source & Selection

**Dataset:** Test set (`data/*/test/`)
**Selection:** Sequential (first N samples)
**Toggle settings:**
- Train vs Test: Line 116 in `visualize_latents.py` - change `'test'` to `'train'`
- Random selection: Lines 117-119 - replace `[:args.num_samples]` with random indexing
- Number of samples: `--num_samples` argument (default: 5)

---

## Files and Functions

### `visualize_latents.py`
Python script to extract and visualize z_H and z_L latent features from trained TRM models.

**Functions:**
- `load_checkpoint(ckpt_path)`: Load model and config from checkpoint file
- `reverse_embed(latent, embed_weight)`: Convert latent vectors to tokens via nearest neighbor in embedding space
- `extract_latents(model, batch)`: Extract z_H, z_L from model forward pass and reverse-embed to tokens
- `visualize_sudoku(data, idx, save_path)`: Create Figure 6 style visualization (6-panel grid showing input/target/prediction/tokenized latents)
- `main()`: CLI entry point with argument parsing

### `run_visualization.sh`
SLURM job script to run visualization on GPU.

---

## Usage Guide

### Run with SLURM:

**Activate conda environment and submit in same terminal:**
```bash
module load conda/new
conda activate trm-sudoku
sbatch visualization/run_visualization.sh [CHECKPOINT] [DATA_PATH] [OUTPUT_DIR] [NUM_SAMPLES]
```

**Note:** The conda environment must be activated in your terminal before running `sbatch` so the job inherits the Python environment.

**Default arguments:**
- `CHECKPOINT`: `checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_att_sudoku/step_65100`
- `DATA_PATH`: `data/sudoku-extreme-1k-aug-1000`
- `OUTPUT_DIR`: `outputs/visualization`
- `NUM_SAMPLES`: `5`

**Example:**
```bash
module load conda/new
conda activate trm-sudoku
sbatch visualization/run_visualization.sh
```

### Direct Python call (for interactive use):
```bash
module load conda/new
conda activate trm-sudoku

python visualization/visualize_latents.py \
    --checkpoint checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_att_sudoku/step_65100 \
    --data_path data/sudoku-extreme-1k-aug-1000 \
    --output_dir outputs/visualization \
    --num_samples 5
```

---

## Input/Output

**Input:**
- Trained model checkpoint (`.pt` file)
- Test dataset directory
- Number of samples to visualize

**Output** (saved to `outputs/visualization/`):
- `sample_0.png, sample_1.png, ...`: 6-panel visualizations showing:
  - Input x (original puzzle)
  - Target y (ground truth solution)
  - Prediction ŷ (model output)
  - Tokenized z_H (decoded high-level latent)
  - Tokenized z_L (decoded low-level latent)
- `latent_data.pt`: Raw tensors (z_H, z_L, tokenized versions, predictions, inputs, targets)
- `slurm-{job_id}.out`: SLURM job log

---

## Implementation Details

**How it works:**
- Loads trained TRM model and extracts `inner_carry` containing z_H and z_L tensors
- Performs reverse embedding: computes cosine similarity between latent vectors and embedding weight matrix, then takes argmax to find nearest token
- Creates Figure 6 style visualizations matching the paper (page 12)
- Minimal dependencies: only adds `matplotlib` to existing environment

**Key code paths:**
- Model forward pass: `model(carry, batch)` returns `new_carry` with `inner_carry.z_H` and `inner_carry.z_L`
- Embedding weight: `model.inner.embed_tokens.embedding_weight` ([vocab_size, hidden_size])
- Reverse embed formula: `argmax(z @ embed_weight.T, dim=-1)`
