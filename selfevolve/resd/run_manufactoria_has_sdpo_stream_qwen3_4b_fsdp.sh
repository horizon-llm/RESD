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
# export RAY_DEBUG="legacy"
ulimit -c 0

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
wandb login $WANDB_API_KEY

########################### Data Preprocess ###########################

CONFIG_NAME="sdpo"
NUM_DATA=${NUM_DATA:--1}

python selfevolve/resd/data/format/manufactoria.py \
    --train_data_source manufactoria/has_train \
    --test_data_source manufactoria/has_test \
    --num_data ${NUM_DATA} \
    --data_source_suffix "has"
train_path=selfevolve/resd/datasets/manufactoria/train_${NUM_DATA}.parquet

val_path=selfevolve/resd/datasets/manufactoria/test.parquet

########################### Quick Config ###########################

TASK=manufactoria
export TASK

# === optim ===
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
LR=${LR:-1e-6}
LAMBDA=${LAMBDA:-0.0}
CLIP_ADV_HIGH=${CLIP_ADV_HIGH:-null}
# === model ===
EMA_WEIGHT=${EMA_WEIGHT:-0.0001} # 0.0 means no EMA, higher means more weight on updated student
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-49152}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-20480}
# === distillation feedback ===
MAX_REPROMPT_LENGTH=${MAX_REPROMPT_LENGTH:-49152}
DONTS_REPROMPT_ON_SELF_SUCCESS=${DONTS_REPROMPT_ON_SELF_SUCCESS:-False} # whether to skip reprompting when the model's own generation is already successful
# === distillation objective ===
ALPHA=${ALPHA:-1.0} # 0.5 means JSD, 0.0 means forward KL, 1.0 means reverse KL
DISTILLATION_TOPK=${DISTILLATION_TOPK:-100}
teacher_prob_min_ratio=${teacher_prob_min_ratio:-0.2} # Clamp teacher prob to be at least this proportion of student prob; null disables
teacher_prob_max_ratio=${teacher_prob_max_ratio:-null} # Clamp teacher prob to be at most this proportion of student prob; null disables
success_rate_weighting=${success_rate_weighting:-False} # whether to weight distillation loss by group success rate
success_rate_alpha=${success_rate_alpha:-1.0} # exponent for success sample weights: (1-sr)^alpha
success_rate_beta=${success_rate_beta:-1.0} # exponent for failure sample weights: sr^beta
# === context updater ===
use_context_updater=${use_context_updater:-False}
playbook_mode=${playbook_mode:-"global"} # how to manage playbook: "global" means one shared playbook for all examples; "per_example" means a separate playbook for each example
concise_frequency=${concise_frequency:-4} # how often to concise the context
max_bullets=${max_bullets:-null} # maximum number of feedback bullets to include in the context; null means no limit
concise_method=${concise_method:-"reset"} # method for concising context, choose from "reset", "prioritized" and "staleness"
concise_after_curation=${concise_after_curation:-True} # whether to run concise again after curator adds bullets to enforce max_bullets
tag_correct_samples=${tag_correct_samples:-True} # whether to run success tagging on correct samples to reinforce playbook bullet counts
use_solution_buffer=${use_solution_buffer:-False} # whether to cache successful trials across steps (useful when batch_size=1)
deduplicate_rollouts=${deduplicate_rollouts:-True} # whether to deduplicate rollouts per example_id in curator/success-tagging (useful when rollout.n > 1)
use_reflection_in_teacher_prompt=${use_reflection_in_teacher_prompt:-True} # whether to include model's own reflection in the teacher prompt
use_playbook_in_teacher_prompt=${use_playbook_in_teacher_prompt:-True} # whether to include playbook in the teacher prompt
use_feedback_in_teacher_prompt=${use_feedback_in_teacher_prompt:-True} # whether to include teacher feedback in the teacher prompt
use_previous_trial_in_teacher_prompt=${use_previous_trial_in_teacher_prompt:-True} # whether to include previous trial in the teacher prompt; only applies if use_context_updater is True
use_solution_in_teacher_prompt=${use_solution_in_teacher_prompt:-False} # whether to include successful solutions in the teacher prompt; requires {solution} placeholder in template
reflector_prompt_file=${reflector_prompt_file:-null} # path to a .txt file with custom reflector prompt; null uses built-in default
curator_prompt_file=${curator_prompt_file:-null} # path to a .txt file with custom curator prompt; null uses built-in default
cu_teacher_prompt_file=${cu_teacher_prompt_file:-"selfevolve/resd/context_updater/prompts/manufactoria_generator_v1.txt"} # path to a .txt file with custom context-updater teacher prompt; null uses built-in default
# === stream trainer ===
max_updates_per_batch=${max_updates_per_batch:-4}
min_updates_per_batch=${min_updates_per_batch:-4}
early_stop_improvement_threshold=${early_stop_improvement_threshold:-0.0}
# === reward function ===
sparse_rewards=${sparse_rewards:-True} # whether to only provide rewards on the final answer (i.e., after all test cases) instead of per test case

project_name='sdpo_stream_manufactoria'

# Build exp_name: only include non-default args to keep the name short.
# Usage: _add <tag> <value> [<default>]
#   If value != default (or no default given), appends _<tag><value> to exp_name.
_add() { local tag=$1 val=$2 def=${3:-}; [[ -n "$def" && "$val" == "$def" ]] || exp_name+="_${tag}${val}"; }

exp_name="qwen3_4b_fsdp"
_add ndata   "$NUM_DATA"
_add trbs    "$TRAIN_BATCH_SIZE"           32
_add rbs     "$ROLLOUT_BATCH_SIZE"         8
_add maxpl   "$MAX_PROMPT_LENGTH"          49152
_add maxlen  "$MAX_RESPONSE_LENGTH"        20480
_add maxrp   "$MAX_REPROMPT_LENGTH"        49152
_add alpha   "$ALPHA"                      0.5
_add lam     "$LAMBDA"                     0.0
_add lr      "$LR"                         1e-6
_add ema     "$EMA_WEIGHT"                 0.0001
_add distk   "$DISTILLATION_TOPK"          100
_add tpmin   "$teacher_prob_min_ratio"     null
_add tpmax   "$teacher_prob_max_ratio"     null
_add srw     "$success_rate_weighting"   False
_add sra     "$success_rate_alpha"       1.0
_add srb     "$success_rate_beta"        1.0
_add dontrep "$DONTS_REPROMPT_ON_SELF_SUCCESS" True
_add ctxupd  "$use_context_updater"        False
_add pbmode  "$playbook_mode"              global
_add cfreq   "$concise_frequency"          4
_add mbull   "$max_bullets"                null
_add cmeth   "$concise_method"             reset
_add cacur  "$concise_after_curation"    False
_add tagcor  "$tag_correct_samples"       False
_add solbuf  "$use_solution_buffer"      False
_add dedup  "$deduplicate_rollouts"      False
_add ureftp  "$use_reflection_in_teacher_prompt" True
_add uplaybp "$use_playbook_in_teacher_prompt" True
_add ufbttp  "$use_feedback_in_teacher_prompt" True
_add uprevttp "$use_previous_trial_in_teacher_prompt" True
_add usoltp  "$use_solution_in_teacher_prompt" False
_add rpf     "$(basename "${reflector_prompt_file}" .txt)"    null
_add cpf     "$(basename "${curator_prompt_file}" .txt)"      null
_add ctpf    "$(basename "${cu_teacher_prompt_file}" .txt)"   null
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
    custom_reward_function.path=selfevolve/resd/feedback/manufactoria.py
    custom_reward_function.name=compute_score
    +custom_reward_function.reward_kwargs.sparse_rewards=${sparse_rewards}
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=69632
    actor_rollout_ref.actor.token_loss_dump_n=2
)

DISTILLATION=(
    actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILLATION_TOPK
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS}
    actor_rollout_ref.actor.self_distillation.alpha=$ALPHA
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$EMA_WEIGHT
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=${MAX_REPROMPT_LENGTH}
    actor_rollout_ref.actor.self_distillation.teacher_prob_min_ratio=${teacher_prob_min_ratio}
    actor_rollout_ref.actor.self_distillation.teacher_prob_max_ratio=${teacher_prob_max_ratio}
    actor_rollout_ref.actor.self_distillation.success_rate_weighting=${success_rate_weighting}
    actor_rollout_ref.actor.self_distillation.success_rate_alpha=${success_rate_alpha}
    actor_rollout_ref.actor.self_distillation.success_rate_beta=${success_rate_beta}
)

CONTEXT_UPDATER=(
    actor_rollout_ref.actor.self_distillation.context_updater.enabled=${use_context_updater}
    actor_rollout_ref.actor.self_distillation.context_updater.playbook_mode=${playbook_mode}
    actor_rollout_ref.actor.self_distillation.context_updater.concise_frequency=${concise_frequency}
    actor_rollout_ref.actor.self_distillation.context_updater.max_bullets=${max_bullets}
    actor_rollout_ref.actor.self_distillation.context_updater.concise_method=${concise_method}
    actor_rollout_ref.actor.self_distillation.context_updater.concise_after_curation=${concise_after_curation}
    actor_rollout_ref.actor.self_distillation.context_updater.tag_correct_samples=${tag_correct_samples}
    actor_rollout_ref.actor.self_distillation.context_updater.use_solution_buffer=${use_solution_buffer}
    actor_rollout_ref.actor.self_distillation.context_updater.deduplicate_rollouts=${deduplicate_rollouts}
    actor_rollout_ref.actor.self_distillation.context_updater.use_reflection_in_teacher_prompt=${use_reflection_in_teacher_prompt}
    actor_rollout_ref.actor.self_distillation.context_updater.use_playbook_in_teacher_prompt=${use_playbook_in_teacher_prompt}
    actor_rollout_ref.actor.self_distillation.context_updater.use_feedback_in_teacher_prompt=${use_feedback_in_teacher_prompt}
    actor_rollout_ref.actor.self_distillation.context_updater.use_previous_trial_in_teacher_prompt=${use_previous_trial_in_teacher_prompt}
    actor_rollout_ref.actor.self_distillation.context_updater.use_solution_in_teacher_prompt=${use_solution_in_teacher_prompt}
    actor_rollout_ref.actor.self_distillation.context_updater.reflector_prompt_file=${reflector_prompt_file}
    actor_rollout_ref.actor.self_distillation.context_updater.curator_prompt_file=${curator_prompt_file}
    actor_rollout_ref.actor.self_distillation.context_updater.cu_teacher_prompt_file=${cu_teacher_prompt_file}
)

TEACHER=(
    actor_rollout_ref.actor.self_distillation.teacher.server_ip="127.0.0.1"
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE
    actor_rollout_ref.rollout.val_kwargs.n=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45
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
    trainer.reprompt_data_dir="checkpoints/${project_name}/${exp_name}/reprompts"
)

########################### Launch ###########################

"$PYTHON" -m selfevolve.resd.trainer.main_ppo \
    --config-name=${CONFIG_NAME} \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${DISTILLATION[@]}" \
    "${CONTEXT_UPDATER[@]}" \
    "${TEACHER[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"