#!/usr/bin/env bash
set -xeuo pipefail

# ---- Usage ----
# Test the base model (no checkpoint):
#   bash selfevolve/sdpo_fewshot/test_lcbv6.sh
#
# Test a local checkpoint:
#   CHECKPOINT_PATH=checkpoints/sdpo_lcb_v6/my_exp/global_step_100 bash selfevolve/sdpo_fewshot/test_lcbv6.sh
#
# Download from S3 then test:
#   S3_CHECKPOINT=s3://shopqa-users/yuwzhan/iterative-opd/checkpoints/sdpo_lcb_v6/my_exp/global_step_100 \
#     bash selfevolve/sdpo_fewshot/test_lcbv6.sh
# ---------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
ulimit -c 0

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

########################### Download Checkpoint from S3 ###########################

S3_CHECKPOINT="${S3_CHECKPOINT:-}"

if [[ -n "$S3_CHECKPOINT" ]]; then
    # Derive local path: strip the s3 prefix to get a relative local directory
    # e.g. s3://shopqa-users/yuwzhan/iterative-opd/checkpoints/... -> checkpoints/...
    LOCAL_CHECKPOINT_DIR="${S3_CHECKPOINT#*iterative-opd/}"
    mkdir -p "$LOCAL_CHECKPOINT_DIR"
    echo "[bootstrap] Syncing ${S3_CHECKPOINT}/ -> ${LOCAL_CHECKPOINT_DIR}/"
    aws s3 sync "${S3_CHECKPOINT}/" "${LOCAL_CHECKPOINT_DIR}/" --region us-east-1
    CHECKPOINT_PATH="$LOCAL_CHECKPOINT_DIR"
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

########################### Data Preprocess ###########################

val_path=selfevolve/sdpo/datasets/lcb_v6_resplit/test.parquet

if [[ ! -f "${val_path}" ]]; then
    echo "[data] Downloading LCBv6 dataset from HuggingFace..."
    hf download YWZBrandon/lcb_v6_resplit_v2 \
        --local-dir selfevolve/sdpo/datasets/lcb_v6_resplit \
        --repo-type dataset
fi

########################### Quick Config ###########################

TASK=lcb_v6
export TASK

MAX_WORKERS=${MAX_WORKERS:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-20480}
ROLLOUT_N=${ROLLOUT_N:-4}

exp_name="test_lcbv6"

########################### Launch ###########################

# Build resume args only when a checkpoint is provided
RESUME_ARGS=()
if [[ -n "$CHECKPOINT_PATH" ]]; then
    VAL_OUTPUT_DIR="${CHECKPOINT_PATH}/val_generations"
    RESUME_ARGS+=(
        trainer.resume_mode=resume_path
        trainer.resume_from_path=${CHECKPOINT_PATH}
    )
else
    VAL_OUTPUT_DIR="val_generations/${exp_name}"
fi

"$PYTHON" -m selfevolve.sdpo_fewshot.trainer.main_ppo \
    --config-name=sdpo \
    data.train_files=${val_path} \
    data.val_files=${val_path} \
    data.train_batch_size=32 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.truncation='error' \
    data.filter_overlong_prompts=True \
    data.shuffle=False \
    "data.apply_chat_template_kwargs={enable_thinking: True}" \
    custom_reward_function.path=selfevolve/sdpo_fewshot/feedback/code.py \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.max_workers=${MAX_WORKERS} \
    +custom_reward_function.reward_kwargs.report_response_length=True \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_model_len=69632 \
    actor_rollout_ref.rollout.enforce_eager=True \
    trainer.total_epochs=1 \
    trainer.project_name=deepseek_lcb_v6 \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.logger='["console"]' \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.validation_data_dir="${VAL_OUTPUT_DIR}" \
    "${RESUME_ARGS[@]}" \
    "$@"

########################### Sync Results to S3 ###########################

# if [[ -n "$S3_CHECKPOINT" ]]; then
#     echo "[upload] Syncing val_generations back to S3..."
#     aws s3 sync "${CHECKPOINT_PATH}/val_generations/" "${S3_CHECKPOINT}/val_generations/" --region us-east-1
# fi
