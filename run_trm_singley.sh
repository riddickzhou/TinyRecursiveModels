#!/bin/bash

#SBATCH --job-name=trm_singley
#SBATCH --account=rl
#SBATCH --partition=compute
#SBATCH --qos=high
#SBATCH --nodelist=lux-2-node-10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=1000G
#SBATCH --time=7-00:00:00
#SBATCH --output=outputs/trm_singley/slurm-%j.out

set -e

# Load environment variables from .env file
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found at $SCRIPT_DIR/.env"
fi

echo "=================================="
echo "TRM Singley Training"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "=================================="

# Create output directory
OUTPUT_DIR="outputs/trm_singley_experiments"
mkdir -p "$OUTPUT_DIR"

# Configuration
# Using trm_singley.yaml which has: H_cycles=3, L_cycles=6, mlp_t=False, puzzle_emb_len=16
DATA_PATH="data/sudoku-extreme-1k-aug-1000"
EPOCHS=50000
EVAL_INTERVAL=5000
MIN_EVAL_INTERVAL=0

# Hyperparameters (from run_n_t_experiments.sh)
LR=0.0003
PUZZLE_EMB_LR=0.003
WEIGHT_DECAY=0.01
PUZZLE_EMB_WEIGHT_DECAY=0.001

# Use existing conda environment
PYTHON_BIN="/pm/conda/envs/users/trm-sudoku/bin/python"

echo "Configuration:"
echo "  Architecture: trm_singley (from config/arch/trm_singley.yaml)"
echo "  Data path: $DATA_PATH"
echo "  Epochs: $EPOCHS"
echo "  Eval interval: $EVAL_INTERVAL"
echo "  Learning rate: $LR"
echo "  Puzzle emb LR: $PUZZLE_EMB_LR"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Puzzle emb weight decay: $PUZZLE_EMB_WEIGHT_DECAY"
echo "=================================="

# Run training using trm_singley architecture
WANDB_MODE=online $PYTHON_BIN -m torch.distributed.run --nproc_per_node=8 pretrain.py \
    arch=trm_singley \
    data_paths="[$DATA_PATH]" \
    evaluators="[]" \
    epochs=$EPOCHS \
    eval_interval=$EVAL_INTERVAL \
    min_eval_interval=$MIN_EVAL_INTERVAL \
    lr=$LR \
    puzzle_emb_lr=$PUZZLE_EMB_LR \
    weight_decay=$WEIGHT_DECAY \
    puzzle_emb_weight_decay=$PUZZLE_EMB_WEIGHT_DECAY \
    global_batch_size=256 \
    +run_name=trm_singley_training \
    ema=True

echo ""
echo "=================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=================================="

# Generate summary if checkpoint exists
if [ -f "$OUTPUT_DIR/latest.pt" ]; then
    echo ""
    echo "Latest checkpoint saved at: $OUTPUT_DIR/latest.pt"
fi

echo ""
echo "All outputs saved to: $OUTPUT_DIR"
echo "=================================="
