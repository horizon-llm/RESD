#!/usr/bin/env bash
set -xeuo pipefail

python -m pip install matplotlib

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

########################### Data Preprocess ###########################
# Manufactoria data needs to be generated from HF datasets first.

NUM_DATA=${NUM_DATA:--1}

python selfevolve/resd/data/format/manufactoria.py \
    --train_data_source manufactoria/has_train \
    --test_data_source manufactoria/has_test \
    --num_data ${NUM_DATA} \
    --data_source_suffix "has"

########################### Configuration ###########################

MODEL=${MODEL:-"Qwen/Qwen3-4B-Thinking-2507"}
DATA=${DATA:-"selfevolve/resd/datasets/manufactoria/train_${NUM_DATA}.parquet"}
OUTPUT=${OUTPUT:-"selfevolve/resd/datasets/manufactoria/train_hard.parquet"}
K=${K:-8}
TP=${TP:-4}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
MAX_TOKENS=${MAX_TOKENS:-20480}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-83968}
GPU_MEM=${GPU_MEM:-0.9}
RESULTS=${RESULTS:-"selfevolve/resd/datasets/manufactoria/eval_results.jsonl"}

########################### Run ###########################

python -m selfevolve.resd.filter_hard_examples \
    --model "$MODEL" \
    --data "$DATA" \
    --output "$OUTPUT" \
    --k "$K" \
    --tensor_parallel_size "$TP" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_tokens "$MAX_TOKENS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_MEM" \
    --enable_thinking \
    --reward_function_path selfevolve/resd/feedback/manufactoria.py \
    --reward_function_name compute_score \
    --save_results "$RESULTS"

echo ""
echo "Hard examples saved to: $OUTPUT"
echo "Detailed results saved to: $RESULTS"
