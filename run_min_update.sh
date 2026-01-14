#!/bin/bash

#SBATCH --job-name=min_update
#SBATCH --account=rl
#SBATCH --partition=compute
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --output=outputs/min_update_experiments/slurm-%j.out

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
echo "Minimal Update Experiments (n=0 vs n=1)"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "=================================="

# Create output directory
OUTPUT_DIR="outputs/min_update_experiments"
mkdir -p "$OUTPUT_DIR"

# Base configuration - EXACT copy from run_n_t_experiments.sh
# Paper baseline: n=6, T=3 (87.4% accuracy, depth=42)
# Testing: Does recursion matter? n=0 vs n=1
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
# Only vary: L_cycles (n) - testing n=0 and n=1
# Keep T=3 same as baseline
run_experiment() {
    local h_cycles=$1  # T in paper (fixed at 3)
    local l_cycles=$2  # n in paper (testing 0 and 1)
    local run_suffix=$3

    # Calculate effective depth: (T + n) * layers = (h_cycles + l_cycles) * 2
    local depth=$((($h_cycles + $l_cycles) * 2))

    local run_name="min_update_T${h_cycles}_n${l_cycles}${run_suffix}"

    echo ""
    echo "=========================================="
    echo "Starting: $run_name"
    echo "  T (H_cycles): $h_cycles"
    echo "  n (L_cycles): $l_cycles"
    echo "  Effective depth: $depth"
    echo "  Start time: $(date)"
    echo "=========================================="

    # Run training - EXACT args from run_n_t_experiments.sh
    # Only change: arch.L_cycles
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
# Experiment Set: Minimal Update (n=0 vs n=1)
# ==========================================
echo ""
echo "######################################"
echo "# MINIMAL UPDATE EXPERIMENTS         #"
echo "# Architecture: 2L, 512H, MLP-T      #"
echo "# Baseline: T=3, n=6 (87.4%)         #"
echo "# Testing: Does latent recursion     #"
echo "#          matter at all?            #"
echo "######################################"
echo ""

echo "### Experiment 1: n=0 - No latent updates (T=3, n=0) ###"
echo "Expected: z_L never updated in latent_recursion loop"
echo "Only y updated using stale z_L from previous supervision step"
run_experiment 3 0 "_no_latent_updates"

echo "### Experiment 2: n=1 - Minimal latent updates (T=3, n=1) ###"
echo "Expected: z_L updated once, then y updated"
echo "Minimal recursion within each supervision step"
run_experiment 3 1 "_minimal_latent_updates"

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
print('MINIMAL UPDATE EXPERIMENTS - RESULTS SUMMARY')
print('=' * 90)
print()
print('Architecture: 2 layers, 512 hidden, MLP-T')
print('Baseline: T=3, n=6 (depth=42, 87.4% from paper)')
print()

# Sort by exact accuracy descending
results_sorted = sorted(results, key=lambda x: x.get('test_metrics', {}).get('exact_accuracy', 0), reverse=True)

# Table header
print(f'{'Rank':<5} {'Run Name':<45} {'T':<5} {'n':<5} {'Depth':<8} {'Test Acc %':<12}')
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

    print(f'{i:<5} {run_name:<45} {h_cycles:<5} {l_cycles:<5} {depth:<8} {exact_acc:<12.2f}')

print()
print('=' * 90)
print()
print('Hypothesis Test:')
print('  If n=0 or n=1 achieve similar accuracy to n=6 baseline (87.4%),')
print('  then latent recursion within each supervision step is NOT critical.')
print('  Instead, deep supervision across N_sup=16 steps is the main driver.')
print()
print('Expected from Paper Table 3:')
print('  n=1 (with T=1): 63.2% accuracy')
print('  Baseline n=6 (with T=3): 87.4% accuracy')
print()
print(f'Full results saved to: {results_file}')
print(f'Individual run results: $OUTPUT_DIR/*_metrics.json')
print()
"

echo "=========================================="
echo "All results saved to: $OUTPUT_DIR"
echo "Master results file: $RESULTS_FILE"
echo "=========================================="
