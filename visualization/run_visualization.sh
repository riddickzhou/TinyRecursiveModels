#!/bin/bash
#SBATCH --job-name=viz_latent
#SBATCH --account=rl
#SBATCH --partition=compute
#SBATCH --qos=high
#SBATCH --nodelist=lux-2-node-09
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=outputs/visualization/slurm-%j.out

set -e

echo "--- Starting Visualization ---"
echo "Job ID: $SLURM_JOB_ID"
date

DATA_PATH=${1:-"data/sudoku-extreme-1k-aug-1000"}
NUM_SAMPLES=${2:-10}
ALL_STATES=${3:-""}  # Pass "true" to enable --all_states

ALL_STATES_FLAG=""
if [ "$ALL_STATES" = "true" ]; then
    ALL_STATES_FLAG="--all_states"
fi

echo "--- Running MLP model ---"
/pm/conda/envs/users/trm-sudoku/bin/python visualization/visualize_latents.py \
    --checkpoint "checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_65100" \
    --data_path "$DATA_PATH" \
    --output_dir "outputs/visualization/mlp" \
    --num_samples "$NUM_SAMPLES" \
    $ALL_STATES_FLAG

echo "--- Running ATT model ---"
/pm/conda/envs/users/trm-sudoku/bin/python visualization/visualize_latents.py \
    --checkpoint "checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_att_sudoku/step_65100" \
    --data_path "$DATA_PATH" \
    --output_dir "outputs/visualization/att" \
    --num_samples "$NUM_SAMPLES" \
    $ALL_STATES_FLAG

echo "--- Visualization complete! ---"
date
