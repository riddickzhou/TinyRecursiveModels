#!/bin/bash
#SBATCH --job-name=halting_exp
#SBATCH --account=rl
#SBATCH --partition=compute
#SBATCH --qos=high
#SBATCH --nodelist=lux-2-node-09
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=outputs/halting/slurm-%j.out

set -e

echo "=== Halting Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
date
echo ""

# Arguments with defaults
# MAX_STEPS: -1 for unlimited (run until all samples stabilize)
DATA_PATH=${1:-"data/sudoku-extreme-1k-aug-1000"}
NUM_SAMPLES=${2:-10000}
STABILITY_THRESHOLD=${3:-2}
MAX_STEPS=${4:--1}
NUM_GPUS=${5:-8}

echo "Data path: $DATA_PATH"
echo "Num samples: $( [ "$NUM_SAMPLES" -eq -1 ] && echo 'all' || echo "$NUM_SAMPLES" )"
echo "Stability threshold: $STABILITY_THRESHOLD"
echo "Max steps: $( [ "$MAX_STEPS" -eq -1 ] && echo 'unlimited' || echo "$MAX_STEPS" )"
echo "Num GPUs: $NUM_GPUS"
echo ""

# Create output directory
mkdir -p outputs/halting

echo "--- Running MLP model with adaptive halting ---"
/pm/conda/envs/users/trm-sudoku/bin/python halting/halting_experiment.py \
    --checkpoint "checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_65100" \
    --data_path "$DATA_PATH" \
    --output_dir "outputs/halting" \
    --num_samples "$NUM_SAMPLES" \
    --stability_threshold "$STABILITY_THRESHOLD" \
    --max_steps "$MAX_STEPS" \
    --num_gpus "$NUM_GPUS"

echo ""
echo "=== Experiment complete! ==="
date
