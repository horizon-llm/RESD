#!/usr/bin/env bash
set -euo pipefail

# Run test_lcbv6.sh against every checkpoint under
# s3://shopqa-users/kayleexl/models_livecodebench/

# S3_BASE="s3://shopqa-users/kayleexl/models_livecodebench"
# S3_BASE="s3://shopqa-users/kayleexl/final_models/base-model-7b"
S3_BASE="s3://shopqa-users/kayleexl/final_models/eval-model-bran-no-don"
S3_RESULTS="s3://shopqa-users/yuwzhan/kayleexl_models_livecodebench"
LOCAL_BASE="checkpoints/kayleexl_models_livecodebench"

CHECKPOINTS=(
    "const0.01_w0.5_beta1_theta0.1_280"
    "const5e-3_w0.5_beta1_theta0.2_300"
    "deepseek-7b_pen_beta1_theta0.05_230"
    "deepseek-7b_pen_beta1_theta0.1_410"
    "deepseek-7b_pen_beta1_theta0.2_580"
    "deepseek-7b_pen_beta1_theta0.3_470"
    "neg2_penty1_w0.5-0.3_max0.1_beta1_theta0.2_540"
    "neg_penty1_w0.5-0.2_max0.1_beta1_theta0.2_580"
    "outcome_pen_0.3_0.7_1.5_340"
    "pos_w0.5_beta1_theta0.2_340"
    "pos_w0.5_beta1_theta0.5_700"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

clear_gpus() {
    echo "[cleanup] Stopping Ray cluster..."
    ray stop --force 2>/dev/null || true

    echo "[cleanup] Killing lingering GPU processes..."
    # Kill any processes still holding NVIDIA device handles
    if command -v fuser &>/dev/null; then
        fuser -k /dev/nvidia* 2>/dev/null || true
    else
        # Fallback: kill via nvidia-smi
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
            | xargs -r kill -9 2>/dev/null || true
    fi

    # Brief pause to let the driver release memory
    sleep 5

    echo "[cleanup] GPU memory after cleanup:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv 2>/dev/null || true
}

for ckpt in "${CHECKPOINTS[@]}"; do
    echo "========================================"
    echo "[loop] Testing checkpoint: ${ckpt}"
    echo "========================================"

    local_path="${LOCAL_BASE}/${ckpt}"
    mkdir -p "${local_path}"

    echo "[sync] ${S3_BASE}/${ckpt}/ -> ${local_path}/"
    aws s3 sync "${S3_BASE}/${ckpt}/" "${local_path}/" --region us-east-1

    val_dir="kayleexl_models_livecodebench/${ckpt}"
    mkdir -p "${val_dir}"

    # These are plain HuggingFace weights, not training checkpoints.
    # Load via model.path override; do NOT set CHECKPOINT_PATH (that triggers
    # trainer.resume_mode=resume_path which requires "global_step_" in the path).
    bash "${SCRIPT_DIR}/test_lcbv6.sh" \
        actor_rollout_ref.model.path="${local_path}" \
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
        trainer.validation_data_dir="${val_dir}" \
        "$@" \
        2>&1 | tee "${val_dir}/test_output.log"

    # Extract val-core / val-aux metric lines into a summary file
    grep -E "^val-(core|aux)/" "${val_dir}/test_output.log" > "${val_dir}/metrics_summary.txt" 2>/dev/null || true

    # Upload results to S3
    echo "[upload] Syncing ${val_dir}/ -> ${S3_RESULTS}/${ckpt}/"
    aws s3 sync "${val_dir}/" "${S3_RESULTS}/${ckpt}/" --region us-east-1

    clear_gpus
done
