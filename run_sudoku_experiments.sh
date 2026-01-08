#!/bin/bash

#SBATCH --job-name=sudoku_pretrain
#SBATCH --account=rl
#SBATCH --partition=compute
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out

set -e

echo "--- Starting Experiment: pretrain_mlp_t_sudoku ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
date

# Use torchrun to launch script across all allocated GPUs
# WANDB_MODE=online torchrun --nproc_per_node=8 pretrain.py \
# arch=trm \
# data_paths="[data/sudoku-extreme-1k-aug-1000]" \
# evaluators="[]" \
# epochs=50000 \
# eval_interval=5000 \
# lr=1e-4 \
# puzzle_emb_lr=1e-4 \
# weight_decay=1.0 \
# puzzle_emb_weight_decay=1.0 \
# arch.mlp_t=True \
# arch.pos_encodings=none \
# arch.L_layers=2 \
# arch.H_cycles=3 \
# arch.L_cycles=6 \
# +run_name=pretrain_mlp_t_sudoku \
# ema=True

echo "--- MLP run finished. Starting ATT run. ---"
date

WANDB_MODE=online torchrun --nproc_per_node=8 pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 \d
eval_interval=5000 \
lr=1e-4 \
puzzle_emb_lr=1e-4 \
weight_decay=1.0 \
puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 \
arch.L_cycles=6 \
+run_name=pretrain_att_sudoku \
ema=True

echo "--- Both runs finished. ---"
date