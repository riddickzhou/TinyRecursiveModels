#!/bin/bash

#SBATCH --job-name=replicate_arc1
#SBATCH --account=rl
#SBATCH --partition=compute
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=outputs/replicate_arc1/slurm-%j.out

set -e

# Load environment variables from .env file
# Use $SLURM_SUBMIT_DIR if available (when running via sbatch), otherwise use current directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a  # automatically export all variables
    source "$SCRIPT_DIR/.env"
    set +a
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found at $SCRIPT_DIR/.env"
fi

echo "=================================="
echo "Replicate ARC-AGI-1 Baseline with Pretrained Weights"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "=================================="

# Create output directory
OUTPUT_DIR="outputs/replicate_arc1"
mkdir -p "$OUTPUT_DIR"

# Configuration from Sanjin2024/TinyRecursiveModels-ARC-AGI-1 (all_config.yaml)
DATA_PATH="data/arc1concept-aug-1000"
CHECKPOINT_PATH="checkpoints/arc-agi-1/step_155718"

# Architecture params (from pretrained model config)
H_CYCLES=3
L_CYCLES=4
L_LAYERS=2
HIDDEN_SIZE=512

# Training params (matching pretrained config)
EPOCHS=200000
EVAL_INTERVAL=10000
LR=1e-4
PUZZLE_EMB_LR=1e-2
WEIGHT_DECAY=0.1
PUZZLE_EMB_WEIGHT_DECAY=0.1
GLOBAL_BATCH_SIZE=4608

# Wandb project name
PROJECT_NAME="TinyRecursiveModels"

run_name="replicate_arc1_baseline"

echo ""
echo "=========================================="
echo "Starting: $run_name"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Data: $DATA_PATH"
echo "  Architecture: L_layers=$L_LAYERS, H_cycles=$H_CYCLES, L_cycles=$L_CYCLES"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Start time: $(date)"
echo "=========================================="

# Run evaluation with pretrained weights
# Using 8 GPUs with torch.distributed.run
WANDB_MODE=online /pm/conda/envs/users/trm-sudoku/bin/python -m torch.distributed.run --nproc_per_node=8 pretrain.py \
    arch=trm \
    data_paths="[$DATA_PATH]" \
    evaluators="[arc@ARC]" \
    epochs=$EPOCHS \
    eval_interval=$EVAL_INTERVAL \
    lr=$LR \
    puzzle_emb_lr=$PUZZLE_EMB_LR \
    weight_decay=$WEIGHT_DECAY \
    puzzle_emb_weight_decay=$PUZZLE_EMB_WEIGHT_DECAY \
    global_batch_size=$GLOBAL_BATCH_SIZE \
    arch.mlp_t=False \
    arch.pos_encodings=rope \
    arch.L_layers=$L_LAYERS \
    arch.hidden_size=$HIDDEN_SIZE \
    arch.H_cycles=$H_CYCLES \
    arch.L_cycles=$L_CYCLES \
    load_checkpoint=$CHECKPOINT_PATH \
    +run_name=$run_name \
    ema=True

exit_code=$?
echo "Run finished with exit code: $exit_code"
echo "End time: $(date)"

echo "=========================================="
echo "Replicate ARC-1 Completed"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
