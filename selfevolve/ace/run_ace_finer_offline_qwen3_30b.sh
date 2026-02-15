#!/usr/bin/env bash

########################### Quick Config ###########################

MODEL=Qwen/Qwen3-30B-A3B-Thinking-2507
MODEL_SLUG=qwen3-30b-a3b-thinking-2507
BATCH_SIZE=${BATCH_SIZE:-32} # controls update frequency
BATCH_WORKERS=${BATCH_WORKERS:-32} # controls concurrency
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
NUM_EPOCHS=${NUM_EPOCHS:-3}
EVAL_STEPS=${EVAL_STEPS:-4} # number of batches between evaluations

PYTHON="$CONDA_PREFIX/bin/python"
echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
wandb login cde3bf4dce4d89d49519e73eabf0196c798f8ee8

########################### Start Servers ###########################

nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --max-model-len 131072 \
    --enable-expert-parallel \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --port 8000 \
    > vllm_serve.log 2>&1 &
VLLM_PID=$!
# Set up trap to kill the server processes on script exit (normal or error)
trap "pkill -P $VLLM_PID 2>/dev/null; kill $VLLM_PID 2>/dev/null || true" EXIT

wait_for_servers() {
    local urls=("$@")
    local max_wait=${MAX_WAIT:-3000}  # Default 50 minutes
    local check_interval=5

    echo "Waiting for servers to be ready..."
    echo "URLs: ${urls[@]}"

    for url in "${urls[@]}"; do
        local elapsed=0
        echo -n "Checking $url... "

        while [ $elapsed -lt $max_wait ]; do
            if curl -s -f "$url" > /dev/null 2>&1; then
                echo "✓ Ready (${elapsed}s)"
                break
            fi

            sleep $check_interval
            elapsed=$((elapsed + check_interval))

            if [ $elapsed -ge $max_wait ]; then
                echo "✗ Timeout after ${max_wait}s"
                return 1
            fi
        done
    done

    echo "All servers are ready!"
    return 0
}

wait_for_servers "http://localhost:8000/v1/models"

########################### Sync Results ###########################

nohup bash scripts/sync_ace_results.sh --interval 600 --verbose >"sync_s3.out" 2>&1 | tee sync_s3.out &
SYNC_PID=$!
# Set up trap to kill the sync process on script exit (normal or error)
trap "echo 'Killing sync process (PID: $SYNC_PID)...'; kill $SYNC_PID 2>/dev/null || true" EXIT

########################### Run ACE ###########################

project_name='ace_finer'
exp_name="${MODEL_SLUG}_b${BATCH_SIZE}_e${NUM_EPOCHS}_maxlen${MAX_RESPONSE_LENGTH}"
result_dir="results_${MODEL_SLUG}_b${BATCH_SIZE}_e${NUM_EPOCHS}_maxlen${MAX_RESPONSE_LENGTH}"
"$PYTHON" -m selfevolve.ace.run \
    --task_name finer \
    --mode offline \
    --save_path $result_dir \
    --num_epochs $NUM_EPOCHS \
    --eval_steps $EVAL_STEPS \
    --max_tokens $MAX_RESPONSE_LENGTH \
    --api_provider vllm \
    --generator_model $MODEL \
    --reflector_model $MODEL \
    --curator_model $MODEL \
    --batch_mode \
    --batch_size $BATCH_SIZE \
    --batch_workers $BATCH_WORKERS \
    --wandb_project $project_name \
    --wandb_run $exp_name