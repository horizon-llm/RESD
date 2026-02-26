#!/bin/bash

CHECKPOINT_PATH=${1:-${CHECKPOINT_PATH:-"checkpoints"}}
TOKENIZER_NAME=${TOKENIZER_NAME:-"Qwen/Qwen3-4B"}
DUMP_DIR="$CHECKPOINT_PATH/token_loss_dumps"

# Plot each step individually
for pt_file in $(ls "$DUMP_DIR"/*.pt 2>/dev/null | sort -t_ -k2 -n); do
    echo "Processing $(basename "$pt_file")..."
    python selfevolve/sdpo_migrate/visualize_token_loss.py "$pt_file" --tokenizer "$TOKENIZER_NAME"
done
