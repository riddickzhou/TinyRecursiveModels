#!/bin/bash

#SBATCH --job-name=scale_exp
#SBATCH --account=rl
#SBATCH --partition=compute
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=1000G
#SBATCH --time=7-00:00:00
#SBATCH --output=outputs/scale_experiments/slurm-%j.out

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
echo "Section 4.4: Less is More - Scale Experiments"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "=================================="

# Create output directory
OUTPUT_DIR="outputs/scale_experiments"
mkdir -p "$OUTPUT_DIR"

# Base configuration - EXACT copy from run_sudoku_experiments.sh (MLP variant, line 23-39)
# For Sudoku, paper uses MLP-T by default (achieves 87.4% vs 75% for attention)
DATA_PATH="data/sudoku-extreme-1k-aug-1000"
EPOCHS=50000
EVAL_INTERVAL=5000
LR=1e-4
PUZZLE_EMB_LR=1e-4
WEIGHT_DECAY=1.0
PUZZLE_EMB_WEIGHT_DECAY=1.0
H_CYCLES=3
L_CYCLES=6

# Wandb project name
PROJECT_NAME="TinyRecursiveModels"

# Results JSON file
RESULTS_FILE="$OUTPUT_DIR/results.json"
echo "[]" > "$RESULTS_FILE"

# Function to run experiment and save results
# NOTE: All experiments use MLP-T (arch.mlp_t=True) to match paper's Sudoku baseline
run_experiment() {
    local layers=$1
    local hidden_size=$2
    local run_suffix=$3

    local run_name="scale_${layers}L_${hidden_size}H${run_suffix}"

    echo ""
    echo "=========================================="
    echo "Starting: $run_name"
    echo "  Layers: $layers"
    echo "  Hidden size: $hidden_size"
    echo "  Start time: $(date)"
    echo "=========================================="

    # Run training - EXACT args from run_sudoku_experiments.sh MLP variant (line 23-39)
    # Only change: arch.L_layers and arch.hidden_size
    # Use explicit Python path from conda environment
    WANDB_MODE=online /pm/conda/envs/users/trm-sudoku/bin/python -m torch.distributed.run --nproc_per_node=8 pretrain.py \
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
        arch.L_layers=$layers \
        arch.hidden_size=$hidden_size \
        arch.H_cycles=$H_CYCLES \
        arch.L_cycles=$L_CYCLES \
        +run_name=$run_name \
        ema=True

    local exit_code=$?
    echo "Training finished with exit code: $exit_code"
    echo "End time: $(date)"

    # Extract and save metrics
    if [ $exit_code -eq 0 ]; then
        echo "Extracting metrics for $run_name..."

        # Create individual result file
        local result_file="$OUTPUT_DIR/${run_name}_metrics.json"

        # Try to extract from wandb (with timeout)
        python extract_final_metrics.py \
            --run_name "$run_name" \
            --project_name "$PROJECT_NAME" \
            --output_file "$result_file" \
            --max_wait_hours 1 || echo "Warning: Could not extract wandb metrics for $run_name"

        # Add configuration info to result
        if [ -f "$result_file" ]; then
            # Create a comprehensive result entry
            python -c "
import json
import sys

try:
    with open('$result_file', 'r') as f:
        data = json.load(f)

    # Add experiment configuration
    data['experiment_config'] = {
        'L_layers': $layers,
        'hidden_size': $hidden_size,
        'mlp_t': True,
        'pos_encodings': 'none',
        'H_cycles': $H_CYCLES,
        'L_cycles': $L_CYCLES,
        'epochs': $EPOCHS,
        'lr': $LR,
        'weight_decay': $WEIGHT_DECAY,
    }

    # Save back
    with open('$result_file', 'w') as f:
        json.dump(data, f, indent=2)

    # Append to master results
    try:
        with open('$RESULTS_FILE', 'r') as f:
            results = json.load(f)
    except:
        results = []

    results.append(data)

    with open('$RESULTS_FILE', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    test_acc = data.get('test_metrics', {}).get('exact_accuracy', 'N/A')
    print(f'✓ {data[\"run_name\"]}: Test Exact Accuracy = {test_acc}')

except Exception as e:
    print(f'Error processing results: {e}', file=sys.stderr)
    sys.exit(1)
"
        fi
    else
        echo "ERROR: Training failed for $run_name"
    fi

    echo "=========================================="
    echo ""
}

# ==========================================
# Experiment 1: Layer Ablation (hidden_size=512)
# ==========================================
echo ""
echo "######################################"
echo "# EXPERIMENT SET 1: LAYER ABLATION   #"
echo "# Testing: 2, 4, 8 layers            #"
echo "# Hidden size: 512 (baseline)        #"
echo "# All use MLP-T (paper's Sudoku cfg) #"
echo "######################################"
echo ""

# # 2 layers, 512 dim - BASELINE (paper's 87.4% result)
# run_experiment 2 512 "_baseline"

# # 4 layers, 512 dim (paper tested: 79.5% in Table 1)
# run_experiment 4 512 "_4layer"

# # 8 layers, 512 dim (new exploration)
# run_experiment 8 512 "_8layer"

# ==========================================
# Experiment 2: Hidden Size Ablation (2 layers)
# ==========================================
echo ""
echo "######################################"
echo "# EXPERIMENT SET 2: WIDTH ABLATION   #"
echo "# Testing: 2x, 4x hidden size        #"
echo "# Layers: 2 (baseline)               #"
echo "# All use MLP-T                      #"
echo "######################################"
echo ""

# 2 layers, 1024 dim (2x width)
run_experiment 2 1024 "_2xwidth"

# 2 layers, 2048 dim (4x width)
run_experiment 2 2048 "_4xwidth"

# ==========================================
# Experiment 3: Combined Ablation
# ==========================================
echo ""
echo "######################################"
echo "# EXPERIMENT SET 3: COMBINED         #"
echo "# Testing 4 layers with wider width  #"
echo "# All use MLP-T                      #"
echo "######################################"
echo ""

# 4 layers, 1024 dim (double both)
run_experiment 4 1024 "_4L_2xwidth"

# 4 layers, 2048 dim (4 layers + 4x width)
run_experiment 4 2048 "_4L_4xwidth"

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
print('=' * 80)
print('SECTION 4.4 \"LESS IS MORE\" - EXPERIMENTAL RESULTS SUMMARY')
print('=' * 80)
print()
print('All experiments use MLP-T architecture (paper default for Sudoku)')
print()

# Sort by exact accuracy descending
results_sorted = sorted(results, key=lambda x: x.get('test_metrics', {}).get('exact_accuracy', 0), reverse=True)

# Table header
print(f'{'Rank':<5} {'Run Name':<40} {'Layers':<8} {'Hidden':<8} {'Test Acc %':<12}')
print('-' * 80)

for i, result in enumerate(results_sorted, 1):
    run_name = result.get('run_name', 'unknown')
    config = result.get('experiment_config', {})
    layers = config.get('L_layers', '?')
    hidden = config.get('hidden_size', '?')

    test_metrics = result.get('test_metrics', {})
    exact_acc = test_metrics.get('exact_accuracy', 0)

    # Convert to percentage if needed
    if exact_acc > 0 and exact_acc < 1:
        exact_acc *= 100

    print(f'{i:<5} {run_name:<40} {layers:<8} {hidden:<8} {exact_acc:<12.2f}')

print()
print('=' * 80)
print()
print('Expected Results (from paper):')
print('  - 2L, 512H (baseline): ~87.4% (paper best result)')
print('  - 4L, 512H: ~79.5% (paper Table 1, line 5)')
print()
print('Paper Finding: \"Less is More\"')
print('  → 2 layers generalize better than 4 layers (87.4% vs 79.5%)')
print('  → Smaller networks avoid overfitting on small data')
print()
print(f'Full results saved to: {results_file}')
print(f'Individual run results: $OUTPUT_DIR/*_metrics.json')
print()
"

echo "=========================================="
echo "All results saved to: $OUTPUT_DIR"
echo "Master results file: $RESULTS_FILE"
echo "=========================================="
