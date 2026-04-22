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
# export NCCL_SOCKET_IFNAME=eth0
ulimit -c 0

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
wandb login cde3bf4dce4d89d49519e73eabf0196c798f8ee8

########################### Data Preprocess ###########################

CONFIG_NAME="srpo"
NUM_DATA=${NUM_DATA:--1}

python selfevolve/sdpo_fewshot/data/format/manufactoria.py \
    --train_data_source manufactoria/has_train \
    --test_data_source manufactoria/has_test \
    --num_data ${NUM_DATA} \
    --data_source_suffix "has"

train_path=selfevolve/sdpo_fewshot/datasets/manufactoria/train_${NUM_DATA}.parquet
val_path=selfevolve/sdpo_fewshot/datasets/manufactoria/test.parquet

########################### Quick Config ###########################

TASK=manufactoria
export TASK

# === optim ===
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
LR=${LR:-1e-6}
LAMBDA=${LAMBDA:-0.0}
CLIP_ADV_HIGH=${CLIP_ADV_HIGH:-null}
NUM_EPOCHS=${NUM_EPOCHS:-4}
# === model ===
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp"} # "fsdp" or "fsdp2"
EMA_WEIGHT=${EMA_WEIGHT:-0.01} # 0.0 means no EMA, higher means more weight on updated student
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-20480}
ENABLE_THINKING=True
# === SRPO-specific ===
SRPO_ENTROPY_BETA=${SRPO_ENTROPY_BETA:-1.0}  # β for DW-SDPO entropy weighting: w = exp(-β * H_teacher)
# === distillation feedback ===
MAX_REPROMPT_LENGTH=${MAX_REPROMPT_LENGTH:-49152}
ENV_ONLY_WHEN_NO_SOLUTION=${ENV_ONLY_WHEN_NO_SOLUTION:-True} # whether to only use environment feedback when none of the rollouts is successful
DONTS_REPROMPT_ON_SELF_SUCCESS=${DONTS_REPROMPT_ON_SELF_SUCCESS:-True} # whether to skip reprompting when the model's own generation is already successful
remove_thinking_from_demonstration=${remove_thinking_from_demonstration:-False} # whether to remove <think>...</think> tokens from demonstration in the feedback prompt
include_previous_attempt=${include_previous_attempt:-False} # whether to include previous attempt when feedbacks are used
# === context updater ===
use_context_updater=${use_context_updater:-False}
playbook_mode=${playbook_mode:-"global"} # how to manage playbook: "global" means one shared playbook for all examples; "per_example" means a separate playbook for each example
concise_frequency=${concise_frequency:-4} # how often to concise the context
max_bullets=${max_bullets:-null} # maximum number of feedback bullets to include in the context; null means no limit
concise_method=${concise_method:-"reset"} # method for concising context, choose from "reset" or "prioritized"
concise_after_curation=${concise_after_curation:-False} # whether to run concise again after curator adds bullets to enforce max_bullets
tag_correct_samples=${tag_correct_samples:-False} # whether to run success tagging on correct samples to reinforce playbook bullet counts
use_solution_buffer=${use_solution_buffer:-False} # whether to cache successful trials across steps (useful when batch_size=1)
deduplicate_rollouts=${deduplicate_rollouts:-False} # whether to deduplicate rollouts per example_id in curator/success-tagging (useful when rollout.n > 1)
use_reflection_in_teacher_prompt=${use_reflection_in_teacher_prompt:-True} # whether to include model's own reflection in the teacher prompt
use_playbook_in_teacher_prompt=${use_playbook_in_teacher_prompt:-True} # whether to include playbook in the teacher prompt
use_feedback_in_teacher_prompt=${use_feedback_in_teacher_prompt:-True} # whether to include teacher feedback in the teacher prompt
use_previous_trial_in_teacher_prompt=${use_previous_trial_in_teacher_prompt:-True} # whether to include previous trial in the teacher prompt; only applies if use_context_updater is True
use_solution_in_teacher_prompt=${use_solution_in_teacher_prompt:-False} # whether to include successful solutions in the teacher prompt; requires {solution} placeholder in template
reflector_prompt_file=${reflector_prompt_file:-null} # path to a .txt file with custom reflector prompt; null uses built-in default
curator_prompt_file=${curator_prompt_file:-null} # path to a .txt file with custom curator prompt; null uses built-in default
cu_teacher_prompt_file=${cu_teacher_prompt_file:-"selfevolve/sdpo_fewshot/context_updater/prompts/manufactoria_generator_v1.txt"} # path to a .txt file with custom context-updater teacher prompt; null uses built-in default
use_playbook_in_student_rollout=${use_playbook_in_student_rollout:-False} # whether to inject playbook snapshot into the student prompt during first rollout
student_playbook_sync_frequency=${student_playbook_sync_frequency:-null} # how often to sync the student playbook snapshot; null defaults to concise_frequency
student_prompt_file=${student_prompt_file:-null} # path to a .txt file with custom student prompt template; null uses built-in default
# === teacher ===
teacher_enabled=${teacher_enabled:-False}
feedback_on_correct=${feedback_on_correct:-False} # whether to provide teacher feedback even when the model output is already correct
# === reward function ===
sparse_rewards=${sparse_rewards:-True} # whether to only provide rewards on the final answer (i.e., after all test cases) instead of per test case

project_name='srpo_manufactoria'

# Build exp_name: only include non-default args to keep the name short.
# Usage: _add <tag> <value> [<default>]
#   If value != default (or no default given), appends _<tag><value> to exp_name.
_add() { local tag=$1 val=$2 def=${3:-}; [[ -n "$def" && "$val" == "$def" ]] || exp_name+="_${tag}${val}"; }

exp_name="qwen3_4b_${FSDP_STRATEGY}_getsolutionv2"
_add ndata   "$NUM_DATA"
_add trbs    "$TRAIN_BATCH_SIZE"           32
_add rbs     "$ROLLOUT_BATCH_SIZE"         8
_add maxpl   "$MAX_PROMPT_LENGTH"          4096
_add maxlen  "$MAX_RESPONSE_LENGTH"        20480
_add maxrp   "$MAX_REPROMPT_LENGTH"        49152
_add lam     "$LAMBDA"                     0.0
_add lr      "$LR"                         5e-6
_add ema     "$EMA_WEIGHT"                 0.05
_add envonly "$ENV_ONLY_WHEN_NO_SOLUTION"  True
_add think   "$ENABLE_THINKING"            True
_add rmthd   "$remove_thinking_from_demonstration" False
_add prevatt "$include_previous_attempt"   False
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
_add sturollpb "$use_playbook_in_student_rollout" False
_add stusync "$student_playbook_sync_frequency"  null
_add stupf   "$(basename "${student_prompt_file}" .txt)"   null
_add teachfb "$teacher_enabled"   False
_add foc     "$feedback_on_correct"        False
_add sparse  "$sparse_rewards"             False
_add srpobeta "$SRPO_ENTROPY_BETA"         1.0

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

DISTILLATION=(
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS}
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$EMA_WEIGHT
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=${MAX_REPROMPT_LENGTH}
    actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution=${ENV_ONLY_WHEN_NO_SOLUTION}
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=${remove_thinking_from_demonstration}
    actor_rollout_ref.actor.self_distillation.include_previous_attempt=${include_previous_attempt}
    # SRPO-specific
    actor_rollout_ref.actor.self_distillation.srpo_entropy_beta=${SRPO_ENTROPY_BETA}
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
    actor_rollout_ref.actor.self_distillation.context_updater.use_playbook_in_student_rollout=${use_playbook_in_student_rollout}
    actor_rollout_ref.actor.self_distillation.context_updater.student_playbook_sync_frequency=${student_playbook_sync_frequency}
    actor_rollout_ref.actor.self_distillation.context_updater.student_prompt_file=${student_prompt_file}
)

TEACHER=(
    actor_rollout_ref.actor.self_distillation.teacher.enabled=${teacher_enabled}
    actor_rollout_ref.actor.self_distillation.teacher.server_ip="127.0.0.1"
    actor_rollout_ref.actor.self_distillation.teacher.feedback_on_correct=${feedback_on_correct}
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
    trainer.logger='["console","wandb"]'
    trainer.total_epochs=${NUM_EPOCHS}
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.max_actor_ckpt_to_keep=1
    trainer.save_freq=8
    trainer.test_freq=8
    trainer.val_before_train=True
    trainer.rollout_data_dir="checkpoints/${project_name}/${exp_name}/rollouts"
    trainer.validation_data_dir="checkpoints/${project_name}/${exp_name}/val_generations"
    trainer.reprompt_data_dir="checkpoints/${project_name}/${exp_name}/reprompts"
)

########################### Launch ###########################

"$PYTHON" -m selfevolve.sdpo_fewshot.trainer.main_ppo \
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
