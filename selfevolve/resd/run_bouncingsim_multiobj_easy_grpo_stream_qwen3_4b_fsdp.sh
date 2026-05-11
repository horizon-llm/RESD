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

CONFIG_NAME="grpo"
NUM_DATA=${NUM_DATA:--1}

python selfevolve/resd/data/format/bouncingsim.py \
    --data_source bouncingsim/bouncingsim-MULTIOBJ-easy \
    --num_data ${NUM_DATA} \
    --data_source_suffix "multiobj_easy"

train_path=selfevolve/resd/datasets/bouncingsim_multiobj_easy/train_${NUM_DATA}.parquet
val_path=selfevolve/resd/datasets/bouncingsim_multiobj_easy/test.parquet

########################### Quick Config ###########################

TASK=bouncingsim_multiobj_easy
export TASK

# === optim ===
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
LR=${LR:-1e-6}
LAMBDA=${LAMBDA:-0.0}
CLIP_ADV_HIGH=${CLIP_ADV_HIGH:-null}
# === model ===
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp"}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-58368}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-25600}
# === stream trainer ===
max_updates_per_batch=${max_updates_per_batch:-4}
min_updates_per_batch=${min_updates_per_batch:-4}
early_stop_improvement_threshold=${early_stop_improvement_threshold:-0.0}
# === reward function ===
sparse_rewards=${sparse_rewards:-True}

project_name='grpo_stream_bouncingsim_easy'

# Build exp_name: only include non-default args to keep the name short.
_add() { local tag=$1 val=$2 def=${3:-}; [[ -n "$def" && "$val" == "$def" ]] || exp_name+="_${tag}${val}"; }

exp_name="qwen3_4b_$FSDP_STRATEGY"
_add ndata   "$NUM_DATA"
_add trbs    "$TRAIN_BATCH_SIZE"           32
_add rbs     "$ROLLOUT_BATCH_SIZE"         8
_add maxpl   "$MAX_PROMPT_LENGTH"          58368
_add maxlen  "$MAX_RESPONSE_LENGTH"        25600
_add lam     "$LAMBDA"                     0.0
_add lr      "$LR"                         1e-6
_add mupb    "$max_updates_per_batch"      4
_add minupb  "$min_updates_per_batch"      4
_add esith   "$early_stop_improvement_threshold" 0.0
_add sparse  "$sparse_rewards"             False

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
    "data.apply_chat_template_kwargs={enable_thinking: True}"
    custom_reward_function.path=selfevolve/resd/feedback/bouncingsim.py
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=83968
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE
    actor_rollout_ref.rollout.val_kwargs.n=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45
    actor_rollout_ref.rollout.max_model_len=83968
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
    trainer.use_stream_trainer=True
    trainer.max_updates_per_batch=${max_updates_per_batch}
    trainer.min_updates_per_batch=${min_updates_per_batch}
    trainer.early_stop_improvement_threshold=${early_stop_improvement_threshold}
    trainer.logger='["console","wandb"]'
    trainer.total_epochs=1
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.max_actor_ckpt_to_keep=1
    trainer.save_freq=1
    trainer.test_freq=1
    trainer.forget_eval.eval_freq=0
    trainer.val_before_train=True
    trainer.rollout_data_dir="checkpoints/${project_name}/${exp_name}/rollouts"
    trainer.validation_data_dir="checkpoints/${project_name}/${exp_name}/val_generations"
)

########################### Launch ###########################

"$PYTHON" -m selfevolve.resd.trainer.main_ppo \
    --config-name=${CONFIG_NAME} \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
