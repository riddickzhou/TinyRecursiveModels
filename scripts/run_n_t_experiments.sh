#!/bin/bash

#SBATCH --job-name=n_t_exp
#SBATCH --account=rl
#SBATCH --partition=compute
#SBATCH --qos=high
#SBATCH --nodelist=lux-2-node-10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --output=outputs/n_t_experiments/slurm-%j.out

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
echo "n and T Scaling Experiments"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "=================================="

# Create output directory
OUTPUT_DIR="outputs/n_t_experiments"
mkdir -p "$OUTPUT_DIR"

# Base configuration - EXACT copy from run_sudoku_experiments.sh (MLP variant)
# Paper notation: n = L_cycles, T = H_cycles
# Paper optimal: n=6, T=3 (87.4% accuracy, depth=42)
DATA_PATH="data/sudoku-extreme-1k-aug-1000"
EPOCHS=50000
EVAL_INTERVAL=5000
LR=1e-4
PUZZLE_EMB_LR=1e-4
WEIGHT_DECAY=1.0
PUZZLE_EMB_WEIGHT_DECAY=1.0

# Wandb project name
PROJECT_NAME="TinyRecursiveModels"

# Results JSON file
RESULTS_FILE="$OUTPUT_DIR/results.json"
echo "[]" > "$RESULTS_FILE"

# Function to run experiment and save results
# NOTE: All experiments use 2 layers, 512 hidden, MLP-T (paper's optimal architecture)
# Only vary: H_cycles (T) and L_cycles (n)
run_experiment() {
    local h_cycles=$1  # T in paper
    local l_cycles=$2  # n in paper
    local run_suffix=$3

    # Calculate effective depth: (T + n) * layers = (h_cycles + l_cycles) * 2
    local depth=$((($h_cycles + $l_cycles) * 2))

    local run_name="n_t_T${h_cycles}_n${l_cycles}${run_suffix}"

    echo ""
    echo "=========================================="
    echo "Starting: $run_name"
    echo "  T (H_cycles): $h_cycles"
    echo "  n (L_cycles): $l_cycles"
    echo "  Effective depth: $depth"
    echo "  Start time: $(date)"
    echo "=========================================="

    # Run training - EXACT args from run_sudoku_experiments.sh MLP variant
    # Only change: arch.H_cycles and arch.L_cycles
    WANDB_MODE=online /pm/conda/envs/users/trm-sudoku/bin/python -m torch.distributed.run --nproc_per_node=2 pretrain.py \
        arch=trm \
        data_paths="[$DATA_PATH]" \
        evaluators="[]" \
        epochs=$EPOCHS \
        eval_interval=$EVAL_INTERVAL \
        lr=$LR \
        puzzle_emb_lr=$PUZZLE_EMB_LR \
        weight_decay=$WEIGHT_DECAY \
        puzzle_emb_weight_decay=$PUZZLE_EMB_WEIGHT_DECAY \
        arch.mlp_t=True \
        arch.pos_encodings=none \
        arch.L_layers=2 \
        arch.hidden_size=512 \
        arch.H_cycles=$h_cycles \
        arch.L_cycles=$l_cycles \
        +run_name=$run_name \
        ema=True

    local exit_code=$?
    echo "Training finished with exit code: $exit_code"
    echo "End time: $(date)"

    # Extraction logic removed as per request (view wandb directly)
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Training failed for $run_name"
    fi

    echo "=========================================="
    echo ""
}

# ==========================================
# Experiment Set: n and T Scaling
# ==========================================
echo ""
echo "######################################"
echo "# n & T SCALING EXPERIMENTS          #"
echo "# Architecture: 2L, 512H, MLP-T      #"
echo "# Baseline: T=3, n=6 (87.4%)         #"
echo "######################################"
echo ""

# Baseline: T=3, n=6 (depth=42, 87.4% from paper)

# echo "### Experiment 1: 2x n - Double L_cycles (T=3, n=12) ###"
# run_experiment 3 12 "_2xn"

# echo "### Experiment 2: 2x T - Double H_cycles (T=6, n=6) ###"
# run_experiment 6 6 "_2xT"

# T=3, n=9, 3
run_experiment 3 9 ""
run_experiment 3 3 ""

# n=6, T=1, 9, 12
run_experiment 1 6 ""
run_experiment 9 6 ""
run_experiment 12 6 ""

# ==========================================
# Generate Final Summary Report
# ==========================================
echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=========================================="
echo "End time: $(date)"
echo ""

# Generate summary report
python -c "
import json
from pathlib import Path

results_file = Path('$RESULTS_FILE')
if not results_file.exists():
    print('No results file found!')
    exit(1)

with open(results_file) as f:
    results = json.load(f)

print()
print('=' * 90)
print('n AND T SCALING EXPERIMENTS - RESULTS SUMMARY')
print('=' * 90)
print()
print('Architecture: 2 layers, 512 hidden, MLP-T')
print('Baseline: T=3, n=6 (depth=42, 87.4% from paper)')
print()

# Sort by exact accuracy descending
results_sorted = sorted(results, key=lambda x: x.get('test_metrics', {}).get('exact_accuracy', 0), reverse=True)

# Table header
print(f'{'Rank':<5} {'Run Name':<35} {'T':<5} {'n':<5} {'Depth':<8} {'Test Acc %':<12}')
print('-' * 90)

for i, result in enumerate(results_sorted, 1):
    run_name = result.get('run_name', 'unknown')
    config = result.get('experiment_config', {})
    h_cycles = config.get('H_cycles', '?')
    l_cycles = config.get('L_cycles', '?')
    depth = config.get('effective_depth', '?')

    test_metrics = result.get('test_metrics', {})
    exact_acc = test_metrics.get('exact_accuracy', 0)

    # Convert to percentage if needed
    if exact_acc > 0 and exact_acc < 1:
        exact_acc *= 100

    print(f'{i:<5} {run_name:<35} {h_cycles:<5} {l_cycles:<5} {depth:<8} {exact_acc:<12.2f}')

print()
print('=' * 90)
print()
print('Experiments:')
print('  1. T=3, n=12 (2x n) - Double L_cycles, depth=60')
print('  2. T=6, n=6  (2x T) - Double H_cycles, depth=48')
print()
print('Expected: Likely diminishing returns or overfitting vs baseline (87.4%)')
print()
print(f'Full results saved to: {results_file}')
print(f'Individual run results: $OUTPUT_DIR/*_metrics.json')
print()
"

echo "=========================================="
echo "All results saved to: $OUTPUT_DIR"
echo "Master results file: $RESULTS_FILE"
echo "=========================================="
