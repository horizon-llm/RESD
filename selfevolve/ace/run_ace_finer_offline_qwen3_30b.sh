#!/usr/bin/env bash

PYTHON="$CONDA_PREFIX/bin/python"
echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --max-model-len 131072 \
    --enable-expert-parallel \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --port 8000 \
    > vllm_serve.log 2>&1 &
VLLM_PID=$!
# Set up trap to kill the server processes on script exit (normal or error)
trap "pkill -P $VLLM_PID 2>/dev/null; kill $VLLM_PID 2>/dev/null || true" EXIT

# ==== start servers ==== #
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

# ==== sync results to S3 in the background ==== #
nohup bash scripts/sync_ace_results.sh --interval 600 --verbose >"sync_s3.out" 2>&1 | tee sync_s3.out &
SYNC_PID=$!
# Set up trap to kill the sync process on script exit (normal or error)
trap "echo 'Killing sync process (PID: $SYNC_PID)...'; kill $SYNC_PID 2>/dev/null || true" EXIT
# ==== sync results to S3 in the background ==== #

"$PYTHON" -m selfevolve.ace.run \
    --task_name finer \
    --mode offline \
    --save_path results \
    --num_epochs 3 \
    --eval_steps 100 \
    --max_tokens 8192 \
    --api_provider vllm \
    --generator_model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --reflector_model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --curator_model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --batch_mode \
    --batch_size 32 \
    --batch_workers 32