#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

########################### Configuration ###########################

MODEL=${MODEL:-"Qwen/Qwen3-4B-Thinking-2507"}
DATA=${DATA:-"selfevolve/resd/datasets/sokoban/train.parquet"}
OUTPUT=${OUTPUT:-"selfevolve/resd/datasets/sokoban/train_hard.parquet"}
K=${K:-8}
TP=${TP:-4}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
MAX_TOKENS=${MAX_TOKENS:-32768}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-83968}
GPU_MEM=${GPU_MEM:-0.9}
RESULTS=${RESULTS:-"selfevolve/resd/datasets/sokoban/eval_results.jsonl"}

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
    --save_results "$RESULTS"

echo ""
echo "Hard examples saved to: $OUTPUT"
echo "Detailed results saved to: $RESULTS"
echo ""
echo "To train on hard examples only:"
echo "  DATA=selfevolve/resd/datasets/sokoban/train_hard.parquet bash selfevolve/resd/run_sokoban_grpo_qwen3_4b_fsdp.sh"
