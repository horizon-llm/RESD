#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
# Add repo root to PYTHONPATH so `selfevolve.sdpo` is importable as a package.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
# export RAY_DEBUG=1
ulimit -c 0

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
wandb login cde3bf4dce4d89d49519e73eabf0196c798f8ee8

########################### Quick Config ###########################

CONFIG_NAME="sdpo"
NUM_DATA=${NUM_DATA:--1} # Use -1 to indicate using the full dataset; otherwise, specify the number of samples to use for training (e.g., 1000)

python selfevolve/iterative_opd/prepare_finer_dataset.py \
        --task_name finer \
        --input selfevolve/ace/data/finer_train_batched_1000_samples.jsonl \
        --num_data $NUM_DATA \
        --output data/finer/train_${NUM_DATA}.parquet
python selfevolve/iterative_opd/prepare_finer_dataset.py \
        --task_name finer \
        --input selfevolve/ace/data/finer_val_batched_500_samples.jsonl \
        --output data/finer/val.parquet

finer_train_path=data/finer/train_${NUM_DATA}.parquet
finer_val_path=data/finer/val.parquet

# Hyperparameters (from experiments/run_sdpo_all.sh)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
LR=${LR:-1e-5}
LAMBDA=${LAMBDA:-0.0}
CLIP_ADV_HIGH=${CLIP_ADV_HIGH:-null}
DONTS_REPROMPT_ON_SELF_SUCCESS=${DONTS_REPROMPT_ON_SELF_SUCCESS:-True}
ALPHA=${ALPHA:-0.5}
EMA_WEIGHT=${EMA_WEIGHT:-0.05}
TASK=finer
export TASK
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
NUM_EPOCHS=${NUM_EPOCHS:-3}
MAX_REPROMPT_LENGTH=${MAX_REPROMPT_LENGTH:-49152}
ENV_ONLY_WHEN_NO_SOLUTION=${ENV_ONLY_WHEN_NO_SOLUTION:-True}
CONCISE_FREQUENCY=${CONCISE_FREQUENCY:-4}
MAX_BULLETS=${MAX_BULLETS:-null}
CONCISE_METHOD=${CONCISE_METHOD:-reset}
use_reflection_in_teacher_prompt=${use_reflection_in_teacher_prompt:-True}
use_playbook_in_teacher_prompt=${use_playbook_in_teacher_prompt:-True}

project_name='iterative_opd_finer'
exp_name="qwen3_4b_fsdp_ndata${NUM_DATA}_trbs${TRAIN_BATCH_SIZE}_rbs${ROLLOUT_BATCH_SIZE}_maxlen${MAX_RESPONSE_LENGTH}_maxreprompt${MAX_REPROMPT_LENGTH}_alpha${ALPHA}_lr${LR}_ema${EMA_WEIGHT}_envonly${ENV_ONLY_WHEN_NO_SOLUTION}_concise${CONCISE_FREQUENCY}_maxb${MAX_BULLETS}_cmethod${CONCISE_METHOD}_reflection${use_reflection_in_teacher_prompt}_playbook${use_playbook_in_teacher_prompt}"

########################### Sync Results ###########################

nohup bash scripts/sync_checkpoints.sh --verbose >"sync_s3.out" 2>&1 | tee sync_s3.out &
SYNC_PID=$!
# Set up trap to kill the sync process on script exit (normal or error)
trap "echo 'Killing sync process (PID: $SYNC_PID)...'; kill $SYNC_PID 2>/dev/null || true" EXIT

########################### Download Existing Checkpoints ###########################

CHECKPOINT_BASE_S3="s3://shopqa-users/yuwzhan/iterative-opd/checkpoints"
LOCAL_CHECKPOINT_DIR="checkpoints/${project_name}/${exp_name}"
S3_CHECKPOINT_PREFIX="${CHECKPOINT_BASE_S3}/${project_name}/${exp_name}"
MARKER_FILE="latest_checkpointed_iteration.txt"
mkdir -p "$LOCAL_CHECKPOINT_DIR"

# ---- new: check if the prefix exists / has any objects ----
SHOULD_SYNC=true
if ! LS_OUT="$(aws s3 ls "${S3_CHECKPOINT_PREFIX}/" 2>&1)"; then
    echo "[bootstrap] Can't access ${S3_CHECKPOINT_PREFIX}/ (aws error below); skipping sync"
    echo "[bootstrap] ${LS_OUT}"
    SHOULD_SYNC=false
elif [[ -z "${LS_OUT//[[:space:]]/}" ]]; then
    echo "[bootstrap] No objects found under ${S3_CHECKPOINT_PREFIX}/ yet; skipping sync"
    SHOULD_SYNC=false
fi
# -----------------------------------------------------------

if [[ "$SHOULD_SYNC" == "true" ]]; then
    STEP="$(aws s3 cp "${S3_CHECKPOINT_PREFIX}/${MARKER_FILE}" --region us-east-1 - 2>/dev/null | head -n1 | tr -d '\r\n[:space:]')"
    echo "[bootstrap] Syncing global_step_${STEP}/ from ${S3_CHECKPOINT_PREFIX} -> ${LOCAL_CHECKPOINT_DIR}"
    aws s3 sync "${S3_CHECKPOINT_PREFIX}/global_step_${STEP}/" "${LOCAL_CHECKPOINT_DIR}/global_step_${STEP}/" --region us-east-1 \
    || echo "[bootstrap] checkpoint sync failed"
    aws s3 cp "${S3_CHECKPOINT_PREFIX}/${MARKER_FILE}" "${LOCAL_CHECKPOINT_DIR}/${MARKER_FILE}" --region us-east-1 \
    || echo "[bootstrap] marker file sync failed"
fi

########################### Parameter Arrays ###########################

DATA=(
    data.train_files=${finer_train_path}
    data.val_files=${finer_val_path}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=49152
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.truncation='error'
    data.filter_overlong_prompts=True
    data.shuffle=False
    custom_reward_function.path=selfevolve/iterative_opd/feedback/finer.py
    custom_reward_function.name=compute_score_count
    +custom_reward_function.reward_kwargs.correctness_feedback=True
)

MODEL=(
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Thinking-2507
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=$LR
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.self_distillation.distillation_topk=100
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS}
    actor_rollout_ref.actor.self_distillation.alpha=$ALPHA
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$EMA_WEIGHT
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=${MAX_REPROMPT_LENGTH}
    actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution=${ENV_ONLY_WHEN_NO_SOLUTION}
    actor_rollout_ref.actor.self_distillation.concise_frequency=${CONCISE_FREQUENCY}
    actor_rollout_ref.actor.self_distillation.max_bullets=${MAX_BULLETS}
    actor_rollout_ref.actor.self_distillation.concise_method=${CONCISE_METHOD}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45
    actor_rollout_ref.rollout.max_model_len=65536
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=0.95
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

ALGORITHM=(
    algorithm.lam=${LAMBDA}
    algorithm.rollout_correction.rollout_is=token
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.total_epochs=${NUM_EPOCHS}
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.max_actor_ckpt_to_keep=1
    trainer.save_freq=4
    trainer.test_freq=4
    trainer.val_before_train=True
)

########################### Launch ###########################

"$PYTHON" -m selfevolve.iterative_opd.trainer.main_ppo \
    --config-name=${CONFIG_NAME} \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"