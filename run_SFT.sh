#!/bin/bash

#SBATCH --job-name=llm_sft
#SBATCH --account=rl
#SBATCH --partition=compute
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/LLM/SFT/slurm-%j.out

set -e

# Parse command line arguments
# Usage: sbatch run_SFT.sh [MODEL_NAME]
# If MODEL_NAME is provided, only train that model
# Otherwise, train all 5 models sequentially
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

echo "=================================="
echo "LLM Supervised Fine-tuning on Sudoku"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "=================================="

# Create output directory
OUTPUT_DIR="outputs/LLM/SFT"
mkdir -p "$OUTPUT_DIR"

# Use explicit python path from trm-vllm conda environment
# Avoids conda activation issues when submitted from active conda env
PYTHON_BIN="/pm/conda/envs/users/linbo/trm-vllm/bin/python"

# Training configuration
DATA_PATH="data/FT.json"
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4
LEARNING_RATE=2e-5
MAX_SEQ_LENGTH=2048

# All available models
ALL_MODELS=(
    "data/LLM/Qwen--Qwen2.5-1.5B-Instruct"
    "data/LLM/Qwen--Qwen2.5-3B-Instruct"
    "data/LLM/Qwen--Qwen2.5-7B-Instruct"
    "data/LLM/allenai--Olmo-3-7B-Instruct"
    "data/LLM/allenai--Olmo-3-7B-Think"
)

# If SELECTED_MODEL is provided, only train that model
if [ -n "$SELECTED_MODEL" ]; then
    MODELS=("$SELECTED_MODEL")
    echo "Training only: $SELECTED_MODEL"
else
    MODELS=("${ALL_MODELS[@]}")
    echo "Training all models sequentially"
fi

# Fine-tune each model
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    SFT_OUTPUT_DIR="$OUTPUT_DIR/$MODEL_NAME"

    echo ""
    echo "=========================================="
    echo "Fine-tuning: $MODEL_NAME"
    echo "Base model: $MODEL_PATH"
    echo "Output dir: $SFT_OUTPUT_DIR"
    echo "Start time: $(date)"
    echo "=========================================="

    $PYTHON_BIN LLM/SFT/run_SFT.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$SFT_OUTPUT_DIR" \
        --num_samples -1 \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --learning_rate $LEARNING_RATE \
        --max_seq_length $MAX_SEQ_LENGTH \
        --save_steps 500 \
        --logging_steps 10

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Successfully fine-tuned $MODEL_NAME"
    else
        echo "✗ Failed to fine-tune $MODEL_NAME (exit code: $EXIT_CODE)"
    fi

    echo "End time: $(date)"
    echo "=========================================="
    echo ""
done

echo ""
echo "=========================================="
echo "ALL FINE-TUNING COMPLETED"
echo "=========================================="
echo "End time: $(date)"
echo ""

# Generate summary
python -c "
import json
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
model_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])

print()
print('=' * 80)
print('SFT SUMMARY')
print('=' * 80)
print()

for model_dir in model_dirs:
    config_file = model_dir / 'sft_config.json'
    final_model = model_dir / 'final_model'

    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

        status = '✓ Completed' if final_model.exists() else '✗ Failed'
        print(f'{status}: {model_dir.name}')
        print(f'  Epochs: {config.get(\"num_epochs\", \"?\")}')
        print(f'  Samples: {config.get(\"num_samples\", \"?\")}')
        print(f'  LR: {config.get(\"learning_rate\", \"?\")}')
        print()

print('=' * 80)
print(f'All models saved to: {output_dir}')
print('=' * 80)
"

echo "=========================================="
echo "All results saved to: $OUTPUT_DIR"
echo "=========================================="
