#!/usr/bin/env bash
set -xeuo pipefail

python -m pip install matplotlib

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
# Add repo root to PYTHONPATH so `selfevolve.sdpo` is importable as a package.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
ulimit -c 0

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
wandb login cde3bf4dce4d89d49519e73eabf0196c798f8ee8

########################### Data Preprocess ###########################

CONFIG_NAME="ppo"
NUM_DATA=${NUM_DATA:--1}
USE_HARD_DATA=${USE_HARD_DATA:-False}

if [[ "$USE_HARD_DATA" == "True" ]]; then
    python selfevolve/sdpo_fewshot/preprocess.py --truncate_parquet selfevolve/sdpo_fewshot/datasets/manufactoria/train_hard.parquet --num_data $NUM_DATA
    train_path=selfevolve/sdpo_fewshot/datasets/manufactoria/train_hard_${NUM_DATA}.parquet
else
    python selfevolve/sdpo_fewshot/data/format/manufactoria.py \
        --train_data_source manufactoria/has_train \
        --test_data_source manufactoria/has_test \
        --num_data ${NUM_DATA} \
        --data_source_suffix "has"
    train_path=selfevolve/sdpo_fewshot/datasets/manufactoria/train_${NUM_DATA}.parquet
fi

val_path=selfevolve/sdpo_fewshot/datasets/manufactoria/test.parquet

########################### Quick Config ###########################

TASK=manufactoria
export TASK

# === optim ===
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
LR=${LR:-1e-6}
GAMMA=${GAMMA:-1.0}
LAMBDA=${LAMBDA:-1.0}
CLIP_ADV_HIGH=${CLIP_ADV_HIGH:-null}
# === model ===
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp"}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-20480}
ENABLE_THINKING=True
# === stream trainer ===
max_updates_per_batch=${max_updates_per_batch:-8}
min_updates_per_batch=${min_updates_per_batch:-8}
early_stop_improvement_threshold=${early_stop_improvement_threshold:-0.0}
# === critic ===
CRITIC_LR=${CRITIC_LR:-1e-5}
CRITIC_WARMUP=${CRITIC_WARMUP:-0}
# === reward function ===
sparse_rewards=${sparse_rewards:-True}

project_name='ppo_stream_manufactoria'

# Build exp_name: only include non-default args to keep the name short.
_add() { local tag=$1 val=$2 def=${3:-}; [[ -n "$def" && "$val" == "$def" ]] || exp_name+="_${tag}${val}"; }

exp_name="qwen3_4b_$FSDP_STRATEGY"
_add ndata   "$NUM_DATA"
_add hard    "$USE_HARD_DATA"              False
_add trbs    "$TRAIN_BATCH_SIZE"           32
_add rbs     "$ROLLOUT_BATCH_SIZE"         8
_add maxpl   "$MAX_PROMPT_LENGTH"          4096
_add maxlen  "$MAX_RESPONSE_LENGTH"        20480
_add gamma   "$GAMMA"                      1.0
_add lam     "$LAMBDA"                     1.0
_add lr      "$LR"                         1e-6
_add clr     "$CRITIC_LR"                  1e-5
_add cwarm   "$CRITIC_WARMUP"              0
_add think   "$ENABLE_THINKING"            True
_add mupb    "$max_updates_per_batch"      4
_add minupb  "$min_updates_per_batch"      1
_add esith   "$early_stop_improvement_threshold" 0.0
_add sparse  "$sparse_rewards"             False

########################### Sync Results ###########################

nohup bash scripts/sync_checkpoints.sh --verbose >"sync_s3.out" 2>&1 | tee sync_s3.out &
SYNC_PID=$!
trap "echo 'Killing sync process (PID: $SYNC_PID)...'; kill $SYNC_PID 2>/dev/null || true" EXIT

########################### Download Existing Checkpoints ###########################

CHECKPOINT_BASE_S3="s3://shopqa-users/yuwzhan/iterative-opd/checkpoints"
LOCAL_CHECKPOINT_DIR="checkpoints/${project_name}/${exp_name}"
S3_CHECKPOINT_PREFIX="${CHECKPOINT_BASE_S3}/${project_name}/${exp_name}"
MARKER_FILE="latest_checkpointed_iteration.txt"
mkdir -p "$LOCAL_CHECKPOINT_DIR"

SHOULD_SYNC=true
if ! LS_OUT="$(aws s3 ls "${S3_CHECKPOINT_PREFIX}/" 2>&1)"; then
    echo "[bootstrap] Can't access ${S3_CHECKPOINT_PREFIX}/ (aws error below); skipping sync"
    echo "[bootstrap] ${LS_OUT}"
    SHOULD_SYNC=false
elif [[ -z "${LS_OUT//[[:space:]]/}" ]]; then
    echo "[bootstrap] No objects found under ${S3_CHECKPOINT_PREFIX}/ yet; skipping sync"
    SHOULD_SYNC=false
fi

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
    data.train_files=${train_path}
    data.val_files=${val_path}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.truncation='error'
    data.filter_overlong_prompts=True
    data.shuffle=False
    "data.apply_chat_template_kwargs={enable_thinking: ${ENABLE_THINKING}}"
    custom_reward_function.path=selfevolve/sdpo_fewshot/feedback/manufactoria.py
    custom_reward_function.name=compute_score
    +custom_reward_function.reward_kwargs.sparse_rewards=${sparse_rewards}
)

MODEL=(
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Thinking-2507
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=$FSDP_STRATEGY
    actor_rollout_ref.actor.optim.lr=$LR
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=69632
)

CRITIC=(
    critic.model.path=Qwen/Qwen3-4B-Thinking-2507
    critic.model.enable_gradient_checkpointing=True
    critic.strategy=$FSDP_STRATEGY
    critic.optim.lr=$CRITIC_LR
    critic.optim.lr_warmup_steps=10
    critic.ppo_micro_batch_size_per_gpu=1
    critic.model.fsdp_config.param_offload=True
    critic.model.fsdp_config.optimizer_offload=False
    critic.ppo_max_token_len_per_gpu=69632
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE
    actor_rollout_ref.rollout.val_kwargs.n=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35
    actor_rollout_ref.rollout.max_model_len=69632
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
    algorithm.gamma=${GAMMA}
    algorithm.lam=${LAMBDA}
    algorithm.adv_estimator=gae
    algorithm.rollout_correction.rollout_is=token
)

TRAINER=(
    trainer.use_stream_trainer=True
    trainer.max_updates_per_batch=${max_updates_per_batch}
    trainer.min_updates_per_batch=${min_updates_per_batch}
    trainer.early_stop_improvement_threshold=${early_stop_improvement_threshold}
    trainer.critic_warmup=${CRITIC_WARMUP}
    trainer.logger='["console","wandb"]'
    trainer.total_epochs=1
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.max_actor_ckpt_to_keep=1
    trainer.max_critic_ckpt_to_keep=1
    trainer.save_freq=1
    trainer.test_freq=1
    trainer.forget_eval.eval_freq=0
    trainer.val_before_train=True
    trainer.rollout_data_dir="checkpoints/${project_name}/${exp_name}/rollouts"
    trainer.validation_data_dir="checkpoints/${project_name}/${exp_name}/val_generations"
)

########################### Launch ###########################

"$PYTHON" -m selfevolve.sdpo_fewshot.trainer.main_ppo \
    --config-name=${CONFIG_NAME} \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${CRITIC[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
