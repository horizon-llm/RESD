#!/usr/bin/env bash
set -euo pipefail

# Run test_ifeval.sh against every global_step checkpoint (that contains an actor/ folder)
# under each S3 experiment path listed in EXPERIMENTS.

EXPERIMENTS=(
    # "s3://shopqa-users/yuwzhan/iterative-opd/checkpoints/sdpo_stream_manufactoria/qwen3_4b_fsdp_getsolutionv2_ndata-1_rbs1_maxpl49152_maxrp49152_alpha1.0_lr1e-6_ema0.0001_tpmin0.2_tpmax5_ctxupdTrue_mbull201_cmethprioritized_tagcorTrue_solbufTrue_usoltpTrue_ctpfmanufactoria_generator_v4_minupb4_sparseTrue"
    # "s3://shopqa-users/yuwzhan/iterative-opd/checkpoints/sdpo_stream_bouncingsim_easy/qwen3_4b_fsdp_getsolutionv3_ndata-1_rbs1_tpmin0.2_srwTrue_ctxupdTrue_mbull120_cmethstaleness_cacurTrue_tagcorTrue_solbufTrue_dedupTrue_usoltpTrue_ctpfbouncingsim_generator_v4_sparseTrue"
    "s3://shopqa-users/yuwzhan/iterative-opd/checkpoints/sdpo_stream_finer/qwen3_4b_fsdp_ndata-1_rbs1_maxpl49152_maxlen20480_maxrp49152_ema0.0001_tpmin0.2_srwTrue_dontrepFalse_ctxupdTrue_mbull120_cmethstaleness_cacurTrue_tagcorTrue_solbufTrue_dedupTrue_usoltpTrue_ctpffiner_generator_v4"
)

LOCAL_BASE="checkpoints/ifeval_eval"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

clear_gpus() {
    echo "[cleanup] Stopping Ray cluster..."
    ray stop --force 2>/dev/null || true

    echo "[cleanup] Killing lingering GPU processes..."
    if command -v fuser &>/dev/null; then
        fuser -k /dev/nvidia* 2>/dev/null || true
    else
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
            | xargs -r kill -9 2>/dev/null || true
    fi

    sleep 5

    echo "[cleanup] GPU memory after cleanup:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv 2>/dev/null || true
}

for s3_exp in "${EXPERIMENTS[@]}"; do
    echo "########################################"
    echo "[exp] ${s3_exp}"
    echo "########################################"

    # Discover all global_step_* prefixes under this experiment
    all_steps=$(aws s3 ls "${s3_exp}/" --region us-east-1 \
        | grep -oP 'PRE \Kglobal_step_\d+' | sort -t_ -k3 -n)

    if [[ -z "$all_steps" ]]; then
        echo "[skip] No global_step_* folders found."
        continue
    fi

    # Filter to only steps that contain an actor/ subfolder
    checkpoints=()
    for step in $all_steps; do
        has_actor=$(aws s3 ls "${s3_exp}/${step}/actor/" --region us-east-1 2>/dev/null || true)
        if [[ -n "$has_actor" ]]; then
            checkpoints+=("$step")
        fi
    done

    if [[ ${#checkpoints[@]} -eq 0 ]]; then
        echo "[skip] No checkpoints with actor/ found."
        continue
    fi

    echo "[info] Found ${#checkpoints[@]} checkpoint(s) with actor/: ${checkpoints[*]}"

    # Derive experiment name including the task prefix for uniqueness
    # e.g. s3://.../checkpoints/sdpo_stream_manufactoria/qwen3_4b_... -> sdpo_stream_manufactoria/qwen3_4b_...
    task_name=$(basename "$(dirname "${s3_exp}")")
    exp_name="${task_name}/$(basename "${s3_exp}")"

    for ckpt in "${checkpoints[@]}"; do
        echo "========================================"
        echo "[loop] Testing ${exp_name} / ${ckpt}"
        echo "========================================"

        local_path="${LOCAL_BASE}/${exp_name}/${ckpt}"
        mkdir -p "${local_path}"

        echo "[sync] ${s3_exp}/${ckpt}/ -> ${local_path}/"
        aws s3 sync "${s3_exp}/${ckpt}/" "${local_path}/" --region us-east-1

        val_dir="${LOCAL_BASE}/${exp_name}/${ckpt}/val_ifeval"
        mkdir -p "${val_dir}"

        # These are FSDP sharded checkpoints — use resume_path to load them
        CHECKPOINT_PATH="${local_path}" \
        bash "${SCRIPT_DIR}/test_ifeval.sh" \
            trainer.validation_data_dir="${val_dir}" \
            "$@" \
            2>&1 | tee "${val_dir}/test_output.log"

        grep -E "^val-(core|aux)/" "${val_dir}/test_output.log" > "${val_dir}/metrics_summary.txt" 2>/dev/null || true

        # Upload results back into the same S3 checkpoint folder
        echo "[upload] Syncing ${val_dir}/ -> ${s3_exp}/${ckpt}/val_ifeval/"
        aws s3 sync "${val_dir}/" "${s3_exp}/${ckpt}/val_ifeval/" --region us-east-1

        clear_gpus
    done
done
