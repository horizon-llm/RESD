
export CUDA_VISIBLE_DEVICES=2,3
#MODEL_DIR=/data/hal245/hf_model/Qwen3-30B-A3B-Thinking-2507
MODEL_DIR=/data/hal245/hf_model/Qwen3-32B
#MODEL_DIR=/data/hal245/hf_model/Qwen3-4B-Thinking-2507
vllm serve $MODEL_DIR \
  --host 0.0.0.0 \
  --port 8004 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.5 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --served-model-name Qwen3-32B \
  --max-model-len 32768 \
  --enforce-eager \
  --disable-custom-all-reduce

