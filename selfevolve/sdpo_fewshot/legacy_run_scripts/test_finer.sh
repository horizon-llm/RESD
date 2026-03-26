#!/usr/bin/env bash
set -xeuo pipefail

# ---- Usage ----
# Test a local checkpoint:
#   CHECKPOINT_PATH=checkpoints/sdpo_finer/my_exp/global_step_100 bash selfevolve/sdpo_fewshot/test_finer.sh
#
# Download from S3 then test:
#   S3_CHECKPOINT=s3://shopqa-users/yuwzhan/iterative-opd/checkpoints/sdpo_finer/my_exp/global_step_100 \
#     bash selfevolve/sdpo_fewshot/test_finer.sh
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

CHECKPOINT_PATH="${CHECKPOINT_PATH:?Please set CHECKPOINT_PATH or S3_CHECKPOINT}"

########################### Data Preprocess ###########################

python selfevolve/sdpo_fewshot/prepare_finer_dataset.py \
        --task_name finer \
        --input selfevolve/ace/data/finer_val_batched_500_samples.jsonl \
        --output data/finer/val.parquet

val_path=data/finer/val.parquet

########################### Quick Config ###########################

TASK=finer
export TASK

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
ROLLOUT_N=${ROLLOUT_N:-4}

exp_name="test_finer"

########################### Launch ###########################

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
    custom_reward_function.path=selfevolve/sdpo_fewshot/feedback/finer.py \
    custom_reward_function.name=compute_score_count \
    +custom_reward_function.reward_kwargs.correctness_feedback=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Thinking-2507 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_model_len=65536 \
    actor_rollout_ref.rollout.enforce_eager=True \
    trainer.total_epochs=1 \
    trainer.project_name=sdpo_finer \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.logger='["console"]' \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=${CHECKPOINT_PATH} \
    trainer.validation_data_dir="${CHECKPOINT_PATH}/val_generations" \
    "$@"

########################### Sync Results to S3 ###########################

if [[ -n "$S3_CHECKPOINT" ]]; then
    echo "[upload] Syncing val_generations back to S3..."
    aws s3 sync "${CHECKPOINT_PATH}/val_generations/" "${S3_CHECKPOINT}/val_generations/" --region us-east-1
fi
