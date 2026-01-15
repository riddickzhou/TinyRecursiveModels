#!/bin/bash

#SBATCH --job-name=llm_sft_eval
#SBATCH --account=rl
#SBATCH --partition=compute
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/LLM/sft-%j.out

# Don't exit on error - we handle crashes with retry logic
set +e

# Parse command line arguments
# Usage: sbatch run_llm_sft_eval.sh [MODEL_PATH]
# If MODEL_PATH is provided, only eval that model
# Otherwise, eval all models sequentially
SELECTED_MODEL="${1:-}"

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

# Set cache directories to writable locations to avoid permission errors
export HF_HOME="${SCRIPT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${SCRIPT_DIR}/.cache/huggingface/transformers"
export VLLM_CACHE_ROOT="${SCRIPT_DIR}/.cache/vllm"
export FLASHINFER_WORKSPACE_BASE="${SCRIPT_DIR}"
export XDG_CACHE_HOME="${SCRIPT_DIR}/.cache"
# Disable torch compile to avoid nvcc permission errors
export TORCH_COMPILE_DISABLE=1

# Create cache directories
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"

echo "=================================="
echo "LLM SFT Evaluation on Sudoku"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "=================================="

# Use explicit python path from trm-vllm conda environment
# Avoids conda activation issues when submitted from active conda env
PYTHON_BIN="/pm/conda/envs/users/linbo/trm-vllm/bin/python"

# All available SFT models (final_model from SFT training)
ALL_MODELS=(
    "outputs/LLM/SFT/Qwen--Qwen2.5-1.5B-Instruct/final_model"
    "outputs/LLM/SFT/Qwen--Qwen2.5-3B-Instruct/final_model"
    "outputs/LLM/SFT/Qwen--Qwen2.5-7B-Instruct/final_model"
    "outputs/LLM/SFT/allenai--Olmo-3-7B-Instruct/final_model"
    "outputs/LLM/SFT/allenai--Olmo-3-7B-Think/final_model"
)

# If SELECTED_MODEL is provided, only eval that model
if [ -n "$SELECTED_MODEL" ]; then
    MODELS=("$SELECTED_MODEL")
    echo "Evaluating only: $SELECTED_MODEL"
else
    MODELS=("${ALL_MODELS[@]}")
    echo "Evaluating all models sequentially"
fi

# Run evaluation on all test samples (422786 total, -1 means all)
NUM_SAMPLES=-1
# Allow overriding TP size from second argument, default to 1
TENSOR_PARALLEL_SIZE="${2:-1}"

# Run each model
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
    OUTPUT_DIR="outputs/LLM/sft-${MODEL_NAME}"
    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "=========================================="
    echo "Evaluating: $MODEL_NAME"
    echo "Model path: $MODEL_PATH"
    echo "Output dir: $OUTPUT_DIR"
    echo "Samples: $NUM_SAMPLES"
    echo "Start time: $(date)"
    echo "=========================================="

    # Auto re-launch on crash (up to 5 attempts)
    MAX_ATTEMPTS=5
    ATTEMPT=1
    while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
        echo "Attempt $ATTEMPT/$MAX_ATTEMPTS..."

        $PYTHON_BIN LLM/eval_llm_base.py \
            --model_path "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --num_samples "$NUM_SAMPLES" \
            --temperature 0.0 \
            --max_tokens 8192 \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ Successfully evaluated $MODEL_NAME"
            break
        else
            echo "✗ Attempt $ATTEMPT failed (exit code: $EXIT_CODE)"
            if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
                echo "Retrying in 10 seconds..."
                sleep 10
            else
                echo "✗ Failed to evaluate $MODEL_NAME after $MAX_ATTEMPTS attempts"
            fi
            ATTEMPT=$((ATTEMPT + 1))
        fi
    done

    echo "End time: $(date)"
    echo "=========================================="
    echo ""
done

# Generate summary report
echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETED"
echo "=========================================="
echo "End time: $(date)"
echo ""

$PYTHON_BIN -c "
import json
from pathlib import Path

print()
print('=' * 80)
print('LLM SFT EVALUATION SUMMARY')
print('=' * 80)
print()

results = []
for model_dir in Path('outputs/LLM').glob('sft-*'):
    data_file = model_dir / 'data.json'
    if data_file.exists():
        with open(data_file) as f:
            data = json.load(f)
            results.append({
                'model': model_dir.name,
                'accuracy': data.get('accuracy', 0),
                'invalid_pct': data.get('invalid_format_pct', 0),
                'num_samples': data.get('num_samples', 0),
            })

# Sort by accuracy descending
results.sort(key=lambda x: x['accuracy'], reverse=True)

# Print table
print(f\"{'Model':<40} {'Samples':<10} {'Accuracy':<12} {'Invalid %':<12}\")
print('-' * 80)

for r in results:
    print(f\"{r['model']:<40} {r['num_samples']:<10} {r['accuracy']*100:<12.2f} {r['invalid_pct']*100:<12.2f}\")

print()
print('=' * 80)
print()
"

echo "=========================================="
echo "All results saved to: outputs/LLM/sft-*"
echo "=========================================="
